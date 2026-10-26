"""Episode recording driven from the left controller's face buttons.

X toggles: a press starts an episode, the next press stops and saves it (a
cancelled record_episode goal is the recorder's operator-stop). Holding Y
finishes the session: the recorder finalizes and mirrors the dataset and opens
a fresh one in place. The recorder owns every refusal and every error reply;
this side only relays them to the log and the status panel.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

from xr_commander.bus import (
    GOAL_CANCEL_TIMEOUT_S,
    REAL_CLOCK,
    CancellationToken,
    Clock,
    FlashNote,
    Latch,
    cause,
    log,
    ticks,
)
from xr_commander.config import Settings
from xr_commander.devices import FrameSource, fresh_sample
from xr_commander.publish import fire_goal
from xr_commander.robots import PairedRobots


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

    def __init__(self, clock: Clock = REAL_CLOCK) -> None:
        self.recording = False
        self.frames = 0
        # True from the operator's stop until the recorder's result: the
        # save, which scales with episode length.
        self.saving = False
        # True once Y has been held to the finish threshold.
        self.finish_held = False
        self._note = FlashNote(clock)

    def set_note(self, text: str) -> None:
        """Flash an outcome on the panel row for the next few seconds."""
        self._note.set(text)

    def note(self, now_monotonic: float | None = None) -> str:
        return self._note.text(now_monotonic)


def shown_recorder(
    status: RecorderStatus, chosen_has_recorder: bool
) -> RecorderStatus | None:
    """The recorder the panel's REC row and task line show: while the chosen
    robot has one, and while an episode is in flight on any robot or its
    outcome is on the row."""
    if chosen_has_recorder or status.recording or status.saving or status.note():
        return status
    return None


@dataclass(frozen=True)
class _Episode:
    """One in-flight goal: its handle for cancelling, its watcher for done,
    and the robot whose recorder records it."""

    handle: object
    watcher: asyncio.Task
    robot: str | None


async def run_recorder_buttons(
    node_runner,
    *,
    action_module,
    finish_module,
    status: RecorderStatus,
    session: FrameSource,
    robots: PairedRobots,
    settings: Settings,
    token: CancellationToken,
    read_task: Callable[[], str],
    timing: Timing = Timing(),
    clock: Clock = REAL_CLOCK,
) -> None:
    """Drive the chosen robot's recorder from the left controller's face
    buttons.

    One goal at a time: X while one is in flight cancels it (stop-and-save)
    and Y finishes its session, whichever robot it records. The goal's
    watcher owns `status`, so the panel's REC line tracks the recorder's
    answers. With no goal in flight, a press while the chosen robot has no
    recorder is inert, said once.

    `read_task` is read per goal, so an episode carries the label that was live
    when it started.
    """
    unbound = Latch()

    # None until X has been observed at all: a thumb already down on the
    # first tracked frame is not a press, and neither is one held across a
    # stale gap. Rising edges exist only against an observed release.
    prev_x: bool | None = None
    held_y_since: float | None = None
    finish_fired = False
    episode: _Episode | None = None

    try:
        async for _ in ticks(settings.tick_period_s, token, clock):
            left = fresh_sample(session, "left", settings.stale_timeout_s, clock)
            x = left.primary_button if left is not None else prev_x
            rising_x = bool(x) and prev_x is False
            y = bool(left and left.secondary_button)
            loop_now = clock.monotonic()

            if episode is not None and episode.watcher.done():
                episode = None

            if rising_x and episode is None:
                target = robots.producer_for(action_module)
                if target is None:
                    unbound.trip(
                        "record_episode: the chosen robot has no recorder, or no robot is paired"
                    )
                else:
                    unbound.clear()
                    episode = await _start_episode(
                        node_runner,
                        action_module,
                        target,
                        robots.chosen(),
                        status,
                        read_task(),
                        timing,
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
                target = finish_target(robots, finish_module, episode)
                if target is None:
                    unbound.trip(
                        "finish_session: the chosen robot has no recorder, or no robot is paired"
                    )
                else:
                    unbound.clear()
                    await _finish_session(
                        node_runner, finish_module, target, status, timing
                    )
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


def finish_target(robots: PairedRobots, finish_module, episode: _Episode | None):
    """The recorder a Y hold finishes: the one recording the episode in
    flight, else the chosen robot's; None while there is none."""
    if episode is not None:
        return robots.producer_of(finish_module, episode.robot)
    return robots.producer_for(finish_module)


async def _start_episode(
    node_runner,
    action_module,
    target,
    robot: str | None,
    status: RecorderStatus,
    task: str,
    timing: Timing,
) -> _Episode | None:
    """Fire one record_episode goal at `robot`'s recorder; its watcher runs
    it to its end."""
    handle = await fire_goal(
        node_runner, action_module, target, "record_episode", task=task
    )
    if handle is None:
        return None
    if not handle.accepted:
        status.set_note(f"record refused: {cause(handle.refusal.message)}")
        return None
    log(f"recording: {task!r}")
    watcher = asyncio.create_task(_watch_episode(handle, status, timing))
    return _Episode(handle, watcher, robot)


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
    """Roll the session over; the recorder owns every error reply (it refuses
    mid-episode) and the answer is relayed either way: an error reply that
    the finish failed as a failure, any other as a refusal."""
    try:
        response = await finish_module.poll(
            node_runner, target, timing.finish_timeout_s
        )
    except finish_module.Failure as failure:
        log(f"finish_session error reply: {failure}")
        failed = failure.kind is finish_module.FailureKind.ATTEMPT_FAILED
        status.set_note("finish failed" if failed else "finish refused")
        return
    except Exception as e:
        log(f"finish_session failed: {e!r}")
        status.set_note("finish failed")
        return
    log(f"session finished: {response.data.session}")
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
        # A saved episode ends its goal cancelled, with its result, for the
        # reason of the operator's stop or of the recorder's own; an episode
        # the recorder could not save fails its goal.
        data = result.data
        if result.failure is not None:
            log(f"episode discarded: {result.failure}")
            status.set_note("episode discarded")
        elif data is None:
            log(f"record_episode {result.status.name.lower()}")
            status.set_note(f"episode {result.status.name.lower()}")
        else:
            ended = f": {result.cancel_reason}" if result.cancel_reason else ""
            saved = (
                f"episode {data.episode_index} saved, "
                f"{data.frames_recorded} frames{ended}"
            )
            note = f"saved episode {data.episode_index}"
            # A video that misses frames does not line up with the state
            # rows: the operator must know before training on the episode.
            if data.video_frames_missing > 0:
                missing = f"{data.video_frames_missing} video frame(s) missing"
                saved += f"; {missing}, do not train on it"
                note += f", {missing}"
            log(saved)
            status.set_note(note)
    finally:
        status.recording, status.frames, status.saving = False, 0, False
