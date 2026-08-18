"""Episode recording driven from the left controller's face buttons.

X toggles: a press starts an episode, the next press stops and saves it (a
cancelled record_episode goal is the recorder's operator-stop). Holding Y
finishes the session: the recorder finalizes and mirrors the dataset and opens
a fresh one in place. The recorder owns every refusal; this side only relays
them to the log and the status panel.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from xr_commander.bus import (
    GOAL_CANCEL_TIMEOUT_S,
    CancellationToken,
    log,
    select_producer,
    ticks,
)
from xr_commander.config import Settings
from xr_commander.devices import FrameSource, fresh_sample
from xr_commander.publish import fire_goal

# Every episode's task label.
TASK = "VR Task"

# How long an outcome (saved, finished, refused) stays on the panel row.
_NOTE_S = 3.0


@dataclass(frozen=True)
class Timing:
    """The button loop's deadlines and holds, injectable so tests can
    tighten one without waiting out the production value."""

    # Y must be held this long to finish the session: finishing rolls the
    # dataset, so a graze must not trigger it.
    finish_hold_s: float = 1.0
    # The result is requested only after the goal's feedback stream has
    # closed, when it is already terminal on the recorder; the finish call is
    # one round trip that may sit behind an in-flight save.
    result_timeout_s: float = 30.0
    finish_timeout_s: float = 30.0
    cancel_timeout_s: float = GOAL_CANCEL_TIMEOUT_S


class RecorderStatus:
    """The recorder as the panel shows it; every writer runs on the event loop."""

    def __init__(self) -> None:
        self.recording = False
        self.frames = 0
        # True from the operator's stop until the recorder's result: the
        # save, which scales with episode length.
        self.saving = False
        # True once Y has been held to the finish threshold.
        self.finish_held = False
        self._note = ""
        self._note_until_monotonic = 0.0

    def set_note(self, text: str) -> None:
        """Flash an outcome on the panel row for the next few seconds."""
        self._note = text
        self._note_until_monotonic = time.monotonic() + _NOTE_S

    def note(self, now_monotonic: float | None = None) -> str:
        now = time.monotonic() if now_monotonic is None else now_monotonic
        return self._note if now < self._note_until_monotonic else ""


@dataclass(frozen=True)
class _Episode:
    """One in-flight goal: its handle for cancelling, its watcher for done."""

    handle: object
    watcher: asyncio.Task


async def run_recorder_buttons(
    node_runner,
    *,
    action_module,
    finish_module,
    status: RecorderStatus,
    session: FrameSource,
    settings: Settings,
    token: CancellationToken,
    timing: Timing = Timing(),
) -> None:
    """Drive the bound recorder from the left controller's face buttons.

    One goal at a time: X while one is in flight cancels it (stop-and-save)
    rather than firing another. The goal's watcher owns `status`, so the
    panel's REC line tracks the recorder's answers, not this side's intent.
    """
    target = select_producer(action_module, node_runner, "recorder")
    if target is None:
        return

    # None until X has been observed at all: a thumb already down on the
    # first tracked frame is not a press, and neither is one held across a
    # stale gap. Rising edges exist only against an observed release.
    prev_x: bool | None = None
    held_y_since: float | None = None
    finish_fired = False
    episode: _Episode | None = None

    try:
        async for _ in ticks(settings.tick_period_s, token):
            left = fresh_sample(session, "left", settings.stale_timeout_s)
            x = left.primary_button if left is not None else prev_x
            rising_x = bool(x) and prev_x is False
            y = bool(left and left.secondary_button)
            loop_now = asyncio.get_running_loop().time()

            if episode is not None and episode.watcher.done():
                episode = None

            if rising_x and episode is None:
                episode = await _start_episode(
                    node_runner, action_module, target, status, timing
                )
            elif rising_x:
                await _stop_episode(episode, status, timing)

            if left is None:
                # Unknown frames cannot extend a hold, but one physical hold
                # must not finish twice: the fired latch survives the gap.
                held_y_since = None
            elif not y:
                held_y_since, finish_fired = None, False
            elif held_y_since is None:
                held_y_since = loop_now
            elif not finish_fired and loop_now - held_y_since >= timing.finish_hold_s:
                finish_fired = True
                await _finish_session(node_runner, finish_module, target, status, timing)
            # Shown only once the hold is long enough: a graze that will not
            # finish anything must not say it is finishing.
            status.finish_held = (
                held_y_since is not None
                and loop_now - held_y_since >= timing.finish_hold_s
            )
            prev_x = x
    finally:
        if episode is not None:
            episode.watcher.cancel()


async def _start_episode(
    node_runner, action_module, target, status: RecorderStatus, timing: Timing
) -> _Episode | None:
    """Fire one record_episode goal; its watcher runs it to its end."""
    handle = await fire_goal(
        node_runner, action_module, target, "record_episode", task=TASK
    )
    if handle is None:
        return None
    log(f"recording: {TASK!r}")
    return _Episode(handle, asyncio.create_task(_watch_episode(handle, status, timing)))


async def _stop_episode(
    episode: _Episode, status: RecorderStatus, timing: Timing = Timing()
) -> None:
    """Ask the recorder to stop and save; the watcher reports the outcome.
    Only a delivered stop flips the row to saving: a failed cancel leaves
    the episode recording."""
    try:
        await episode.handle.cancel_goal(timing.cancel_timeout_s)
    except Exception as e:
        log(f"record_episode cancel failed: {e!r}")
        return
    # A watcher that finished while the cancel was in flight already cleared
    # the row; claiming saving now would wedge it with nothing left to reset.
    if not episode.watcher.done():
        status.saving = True


async def _finish_session(
    node_runner, finish_module, target, status: RecorderStatus, timing: Timing
) -> None:
    """Roll the session over; the recorder owns every refusal (it refuses
    mid-episode) and the answer is relayed either way."""
    try:
        response = await finish_module.poll(node_runner, target, timing.finish_timeout_s)
    except Exception as e:
        log(f"finish_session failed: {e!r}")
        status.set_note("finish failed")
        return
    data = response.data
    if data.error:
        log(f"finish_session refused: {data.error}")
        status.set_note("finish refused")
    else:
        log(f"session finished: {data.session}")
        status.set_note("session finished")


async def _watch_episode(
    handle, status: RecorderStatus, timing: Timing = Timing()
) -> None:
    """Track one goal from acceptance to its result, feeding the panel."""
    status.recording, status.frames = True, 0
    try:
        while True:
            try:
                feedback = await handle.on_next_feedback_message()
            except asyncio.CancelledError:
                raise
            except ConnectionError:
                log("record_episode: recorder died mid-episode")
                break
            except Exception:
                break
            status.frames = feedback.frames_recorded
        try:
            result = await handle.get_result(timing.result_timeout_s)
        except Exception as e:
            log(f"record_episode result: {e!r}")
            return
        data = result.data
        if data is None:
            log(f"record_episode {result.status.name.lower()}")
            status.set_note(f"episode {result.status.name.lower()}")
        elif data.discarded:
            log(f"episode discarded: {data.error or 'recorder gave no reason'}")
            status.set_note("episode discarded")
        else:
            ended = f": {data.error}" if data.error else ""
            log(
                f"episode {data.episode_index} saved, "
                f"{data.frames_recorded} frames{ended}"
            )
            status.set_note(f"saved episode {data.episode_index}")
    finally:
        status.recording, status.frames, status.saving = False, 0, False
