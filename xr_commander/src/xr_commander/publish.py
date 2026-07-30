"""The command streams, one task per hand per pairing slot.

Silence is the deadman: a disengaged hand publishes nothing at all and the
follower holds. One task per stream.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

import peppygen.clock
from peppygen import QoSProfile

from xr_commander.bus import CancellationToken, Latch, log, messages
from xr_commander.clutch import HandClutch
from xr_commander.config import Settings
from xr_commander.devices import FrameSource, fresh_sample, gripper_opening
from xr_commander.frames import Pose

# Bus deadlines for the ready action: sending and cancelling are near-instant
# round trips.
_GOAL_SEND_TIMEOUT_S = 2.0
_GOAL_CANCEL_TIMEOUT_S = 2.0
_GOAL_RESULT_FLOOR_S = 120.0


def _result_timeout_s(ready_move_duration_s: float) -> float:
    """Wait out the whole move: the follower stretches short requests to its
    velocity limits and grants a grace multiple of the nominal before failing,
    so triple the request under a generous floor."""
    return max(_GOAL_RESULT_FLOOR_S, 3.0 * ready_move_duration_s)


class LatestPose:
    """Most recent measured pose; one writer, one reader, both on the loop."""

    def __init__(self) -> None:
        self._value: Pose | None = None
        self._received_monotonic_s = 0.0

    def set(self, pose: Pose) -> None:
        self._value = pose
        self._received_monotonic_s = time.monotonic()

    def fresh(self, stale_timeout_s: float) -> Pose | None:
        """The measured pose, or None when none has arrived recently: a dead
        follower's last report is not a pose the arm is still at."""
        if self._value is None:
            return None
        if time.monotonic() - self._received_monotonic_s > stale_timeout_s:
            return None
        return self._value


def _stamp_seconds() -> float:
    """Daemon-resolved clock (sim time under a sim clock)."""
    return peppygen.clock.now_ns() / 1e9


async def drain_pose_states(
    node_runner,
    topic_module,
    holder: LatestPose,
    label: str,
    token: CancellationToken,
) -> None:
    """Keep `holder` at the follower's newest measured pose."""
    subscription = await topic_module.subscribe(node_runner)
    # Latched: a malformed pose would repeat every tick.
    unusable = Latch()
    async for _producer, message in messages(
        subscription, token, f"{label} pose_states"
    ):
        try:
            holder.set(Pose.from_xyz_quat(message.position, message.orientation))
            unusable.clear()
        except Exception as e:
            unusable.trip(f"{label} pose_states unusable: {e!r}")


async def _run_stream(
    node_runner,
    topic_module,
    period: float,
    label: str,
    token: CancellationToken,
    build: Callable[[], object | None],
) -> None:
    """Publish `build()` every `period` seconds, skipping ticks that return None.

    Deadline-paced so the cadence does not drift; a slipped tick resyncs
    instead of bursting.
    """
    try:
        publisher = await topic_module.declare_publisher(node_runner)
    except Exception as e:
        log(f"failed to declare the {label} publisher: {e!r}")
        return

    deadline = time.monotonic()
    building = Latch()
    publishing = Latch()

    while not token.is_cancelled():
        deadline += period
        delay = deadline - time.monotonic()
        if delay > 0.0:
            await asyncio.sleep(delay)
        else:
            deadline = time.monotonic()
            # A saturated stream must still yield to the rest of the loop.
            await asyncio.sleep(0)

        try:
            payload = build()
            building.clear()
        except Exception as e:
            building.trip(f"{label} could not build a message: {e!r}")
            continue
        if payload is None:
            continue

        try:
            await publisher.publish(payload)
            publishing.clear()
        except Exception as e:
            publishing.trip(f"{label} publish failing: {e!r}")


async def stream_pose(
    node_runner,
    *,
    topic_module,
    handedness: str,
    clutch: HandClutch,
    session: FrameSource,
    measured: LatestPose,
    settings: Settings,
    token: CancellationToken,
) -> None:
    """Drive one hand's pose_link slot from the clutch."""
    # The one refusal an operator cannot see coming: squeezing while the
    # follower's measured pose is absent or stale does nothing. Say so once.
    waiting = Latch()

    def build():
        sample = fresh_sample(session, handedness, settings.stale_timeout_s)
        measured_ee = measured.fresh(settings.stale_timeout_s)
        if measured_ee is not None:
            waiting.clear()
        elif sample is not None and sample.squeezing:
            waiting.trip(
                f"{handedness} arm: squeeze ignored until the follower "
                "reports a fresh measured pose"
            )
        target = clutch.step(
            squeezing=sample.squeezing if sample else False,
            hand=sample.pose if sample else None,
            measured_ee=measured_ee,
        )
        if target is None:
            return None
        position, orientation = target.as_wire()
        return topic_module.build_message(_stamp_seconds(), position, orientation)

    await _run_stream(
        node_runner,
        topic_module,
        settings.tick_period_s,
        f"{handedness} arm",
        token,
        build,
    )


async def stream_gripper(
    node_runner,
    *,
    topic_module,
    handedness: str,
    session: FrameSource,
    settings: Settings,
    token: CancellationToken,
) -> None:
    """Drive one hand's gripper_link slot from its trigger.

    Gated on the grip deadman, not the arm clutch: a side with a gripper but
    no pose_link must still grip.
    """

    def build():
        sample = fresh_sample(session, handedness, settings.stale_timeout_s)
        if sample is None or not sample.squeezing:
            return None
        opening = gripper_opening(sample.trigger, settings.gripper_open_fraction)
        # A trigger carries no effort intent, so max_effort rides 0 and the
        # follower's ceiling applies.
        return topic_module.build_message(_stamp_seconds(), opening, 0.0)

    await _run_stream(
        node_runner,
        topic_module,
        settings.tick_period_s,
        f"{handedness} gripper",
        token,
        build,
    )


async def run_ready_button(
    node_runner,
    *,
    action_module,
    session: FrameSource,
    settings: Settings,
    token: CancellationToken,
) -> None:
    """Fire the follower's move_to_ready on the A button's rising edge.

    Squeezing either grip cancels the move in flight: taking manual control
    back is the cancel gesture, so a squeeze also vetoes a same-tick press.
    Zero bound producers leaves the button inert.
    """
    producers = action_module.bound_producers(node_runner)
    if not producers:
        return
    target = producers[0]
    if len(producers) > 1:
        log(f"several ready_posture producers bound; using {target.instance_id}")

    prev_primary = False
    prev_squeeze = False
    handle = None
    waiter: asyncio.Task | None = None

    def disown_waiter() -> None:
        nonlocal handle, waiter
        if waiter is not None:
            waiter.cancel()
        handle, waiter = None, None

    try:
        while not token.is_cancelled():
            await asyncio.sleep(settings.tick_period_s)
            right = fresh_sample(session, "right", settings.stale_timeout_s)
            left = fresh_sample(session, "left", settings.stale_timeout_s)
            primary = bool(right and right.primary_button)
            squeeze = bool(right and right.squeezing) or bool(left and left.squeezing)

            if waiter is not None and waiter.done():
                handle, waiter = None, None
            if handle is not None and squeeze and not prev_squeeze:
                try:
                    await handle.cancel_goal(_GOAL_CANCEL_TIMEOUT_S)
                except Exception as e:
                    # The goal may still finish unwatched; re-arm the button.
                    log(f"move_to_ready cancel failed: {e!r}")
                    disown_waiter()
            elif handle is None and primary and not prev_primary and not squeeze:
                fired = None
                try:
                    request = action_module.GoalRequest(
                        duration_s=settings.ready_move_duration_s
                    )
                    fired = await action_module.ActionHandle.fire_goal(
                        node_runner,
                        target,
                        request,
                        _GOAL_SEND_TIMEOUT_S,
                        QoSProfile.Standard,
                    )
                except Exception as e:
                    log(f"move_to_ready failed to send: {e!r}")
                if fired is not None and not fired.accepted:
                    log(f"move_to_ready rejected: {fired.reason}")
                elif fired is not None:
                    handle = fired
                    waiter = asyncio.create_task(
                        _log_ready_result(
                            fired, _result_timeout_s(settings.ready_move_duration_s)
                        )
                    )
            prev_primary, prev_squeeze = primary, squeeze
    finally:
        if waiter is not None:
            waiter.cancel()


async def _log_ready_result(handle, timeout_s: float) -> None:
    try:
        result = await handle.get_result(timeout_s)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        log(f"move_to_ready result: {e!r}")
        return
    detail = result.data.message if result.data else ""
    log(f"move_to_ready {result.status.name.lower()}: {detail}")
