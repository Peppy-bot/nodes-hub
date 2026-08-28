"""The command streams, one task per hand per pairing slot.

Silence is the deadman: a disengaged hand stops commanding and the follower
holds. Releasing the arm clutch first streams a short burst of the measured
pose, so the follower stops where the robot is rather than at the last hand
target. One task per stream.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

import peppygen.clock
from peppygen import QoSProfile

from xr_commander.bus import (
    GOAL_CANCEL_TIMEOUT_S,
    GOAL_SEND_TIMEOUT_S,
    CancellationToken,
    Latch,
    log,
    messages,
    select_producer,
    ticks,
)
from xr_commander.clutch import HandClutch
from xr_commander.config import Settings
from xr_commander.devices import (
    FrameSource,
    Handedness,
    HandSample,
    fresh_sample,
    gripper_opening,
)
from xr_commander.frames import Pose

_GOAL_RESULT_FLOOR_S = 120.0
# A follower may stretch a short requested duration to its own limits, so the
# result wait scales the request and keeps a generous floor.
_RESULT_GRACE_FACTOR = 3.0


def _result_timeout_s(posture_move_duration_s: float) -> float:
    """How long to wait for a posture move's result before giving up on it."""
    return max(_GOAL_RESULT_FLOOR_S, _RESULT_GRACE_FACTOR * posture_move_duration_s)


class LatestPose:
    """Most recent measured pose; one writer, one reader, both on the loop."""

    def __init__(self) -> None:
        self._value: Pose | None = None
        self._received_monotonic_s = 0.0

    def set(self, pose: Pose) -> None:
        self._value = pose
        self._received_monotonic_s = time.monotonic()

    def fresh(
        self, stale_timeout_s: float, now_monotonic: float | None = None
    ) -> Pose | None:
        """The measured pose, or None when none has arrived recently: a dead
        follower's last report is not a pose the arm is still at."""
        if self._value is None:
            return None
        now = time.monotonic() if now_monotonic is None else now_monotonic
        if now - self._received_monotonic_s > stale_timeout_s:
            return None
        return self._value


def _timestamp_seconds() -> float:
    """Daemon-resolved clock (sim time under a sim clock)."""
    return peppygen.clock.now_ns() / 1e9


# How long the release freeze streams the measured pose after the clutch
# disengages. The follower's motion authority may keep converging to the
# last command it heard (openarm's backbone documents that choice), so the
# release explicitly commands "stay where the robot is"; a burst rather
# than a single message, so one dropped sample cannot lose the stop.
RELEASE_HOLD_S = 0.25
# The burst's own floor in messages, so a slow command rate cannot reduce it
# to the single sample the burst exists to avoid depending on.
RELEASE_HOLD_MESSAGES = 3


class ReleaseHold:
    """The release freeze: when a driving hand disengages, the last fresh
    measured pose becomes the streamed target for a short burst, stopping
    the follower where the robot is instead of at the last hand pose."""

    def __init__(self, tick_period_s: float, hold_s: float = RELEASE_HOLD_S):
        self._hold_s = max(hold_s, RELEASE_HOLD_MESSAGES * tick_period_s)
        self._driving = False
        self._pose: Pose | None = None
        self._until = 0.0

    def step(
        self, *, target: Pose | None, measured_ee: Pose | None, now_s: float
    ) -> Pose | None:
        """This tick's pose to stream, or None for silence: the clutch's
        target while driving, then the latched measured pose for the burst
        that follows release. One call per tick, so no caller can arm the
        burst without also being able to void it. Without a fresh measured
        pose at release there is nothing truthful to freeze on, so silence."""
        if target is not None:
            self._driving = True
            self._until = 0.0
            return target
        if self._driving:
            self._driving = False
            if measured_ee is not None:
                self._pose = measured_ee
                self._until = now_s + self._hold_s
        if now_s < self._until:
            return self._pose
        return None


async def drain_pose_states(
    node_runner,
    topic_module,
    holder: LatestPose,
    label: str,
    token: CancellationToken,
) -> None:
    """Keep `holder` at the follower's newest measured pose."""
    try:
        subscription = await topic_module.subscribe(node_runner)
    except Exception as e:
        # Loud and fail-safe: with no measured pose the hand never engages.
        log(f"{label} pose_states subscribe failed: {e!r}")
        return
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
    """Publish `build()` every `period` seconds, skipping ticks that return None."""
    try:
        publisher = await topic_module.declare_publisher(node_runner)
    except Exception as e:
        log(f"failed to declare the {label} publisher: {e!r}")
        return

    building = Latch()
    publishing = Latch()

    async for _ in ticks(period, token):
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
    handedness: Handedness,
    clutch: HandClutch,
    session: FrameSource,
    measured: LatestPose,
    settings: Settings,
    token: CancellationToken,
) -> None:
    """Drive one hand's pose_link slot from the clutch. Releasing the
    clutch streams a short freeze burst of the measured pose, so the
    follower stops where the robot is instead of walking on to the last
    hand pose."""
    # The one refusal an operator cannot see coming: squeezing while the
    # follower's measured pose is absent or stale does nothing. Say so once.
    waiting = Latch()
    hold = ReleaseHold(settings.tick_period_s)

    def build():
        # Stamped first: a clock that cannot stamp must fail the tick before
        # the clutch mutates its engagement state.
        timestamp = _timestamp_seconds()
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
        commanded = hold.step(
            target=target, measured_ee=measured_ee, now_s=time.monotonic()
        )
        if commanded is None:
            return None
        position, orientation = commanded.as_wire()
        return topic_module.build_message(timestamp, position, orientation)

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
    handedness: Handedness,
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
        return topic_module.build_message(_timestamp_seconds(), opening, 0.0)

    await _run_stream(
        node_runner,
        topic_module,
        settings.tick_period_s,
        f"{handedness} gripper",
        token,
        build,
    )


async def run_posture_button(
    node_runner,
    *,
    action_module,
    pressed: Callable[[HandSample], bool],
    session: FrameSource,
    settings: Settings,
    token: CancellationToken,
) -> None:
    """Fire the module's posture move on the button's rising edge.

    `pressed` reads the right hand's button for this posture. Squeezing either
    grip cancels the move in flight: taking manual control back is the cancel
    gesture, so a squeeze also vetoes a same-tick press. Zero bound producers
    leaves the button inert; a surplus is logged here.
    """
    name = action_module.TARGET_ACTION_NAME
    target = select_producer(action_module, node_runner, name)
    if target is None:
        return

    # None until the button has been observed at all: a thumb already down on
    # the first tracked frame is not a press, and neither is one held across
    # a stale gap. Rising edges exist only against an observed release.
    prev_button: bool | None = None
    prev_squeeze = False
    handle = None
    waiter: asyncio.Task | None = None

    def disown_waiter() -> None:
        nonlocal handle, waiter
        if waiter is not None:
            waiter.cancel()
        handle, waiter = None, None

    try:
        async for _ in ticks(settings.tick_period_s, token):
            right = fresh_sample(session, "right", settings.stale_timeout_s)
            left = fresh_sample(session, "left", settings.stale_timeout_s)
            squeeze = bool(right and right.squeezing) or bool(left and left.squeezing)
            button = pressed(right) if right is not None else prev_button
            rising = bool(button) and prev_button is False

            if waiter is not None and waiter.done():
                handle, waiter = None, None
            if handle is not None and squeeze and not prev_squeeze:
                try:
                    await handle.cancel_goal(GOAL_CANCEL_TIMEOUT_S)
                except Exception as e:
                    # The goal may still finish unwatched; re-arm the button.
                    log(f"{name} cancel failed: {e!r}")
                    disown_waiter()
            elif handle is None and rising and not squeeze:
                fired = await fire_goal(
                    node_runner,
                    action_module,
                    target,
                    name,
                    duration_s=settings.posture_move_duration_s,
                )
                if fired is not None:
                    handle = fired
                    timeout = _result_timeout_s(settings.posture_move_duration_s)
                    waiter = asyncio.create_task(_log_result(name, fired, timeout))
            prev_button, prev_squeeze = button, squeeze
    finally:
        if waiter is not None:
            waiter.cancel()


async def fire_goal(node_runner, action_module, target, name: str, **request_fields):
    """The accepted goal handle, or None with the failure or refusal logged.
    Builds the request too, so a raising constructor is a logged send failure
    rather than a dead button task."""
    try:
        request = action_module.GoalRequest(**request_fields)
        fired = await action_module.ActionHandle.fire_goal(
            node_runner, target, request, GOAL_SEND_TIMEOUT_S, QoSProfile.Standard
        )
    except Exception as e:
        log(f"{name} failed to send: {e!r}")
        return None
    if not fired.accepted:
        log(f"{name} rejected: {fired.reason}")
        return None
    return fired


async def _log_result(name: str, handle, timeout_s: float) -> None:
    try:
        result = await handle.get_result(timeout_s)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        log(f"{name} result: {e!r}")
        return
    detail = result.data.message if result.data else ""
    log(f"{name} {result.status.name.lower()}: {detail}")
