"""The command streams, one task per hand per pairing slot.

Each slot holds a pair per robot; a stream commands the chosen robot's pair
and is silent toward the others. Silence is the deadman: a disengaged hand
stops commanding and the follower holds. Releasing the arm clutch first
streams a short burst of the measured pose, so the follower stops where the
robot is; a change of robot releases the same way toward the robot left
behind, and the new robot engages on the next squeeze. One task per stream.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

import peppygen.clock

from xr_commander.bus import (
    GOAL_CANCEL_TIMEOUT_S,
    GOAL_SEND_TIMEOUT_S,
    CancellationToken,
    FlashNote,
    Latch,
    log,
    messages,
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
from xr_commander.robots import PairedRobots

_GOAL_RESULT_FLOOR_S = 120.0
# A follower may stretch a short requested duration to its own limits, so the
# result wait scales the request and keeps a generous floor.
_RESULT_GRACE_FACTOR = 3.0


def _result_timeout_s(posture_move_duration_s: float) -> float:
    """How long to wait for a posture move's result before giving up on it."""
    return max(_GOAL_RESULT_FLOOR_S, _RESULT_GRACE_FACTOR * posture_move_duration_s)


class MeasuredPoses:
    """Each paired follower's most recent measured pose, by the peer that
    reported it; one writer, one reader, both on the loop."""

    def __init__(self) -> None:
        self._latest: dict[object, tuple[Pose, float]] = {}

    def set(self, peer, pose: Pose) -> None:
        self._latest[peer] = (pose, time.monotonic())

    def fresh(
        self, peer, stale_timeout_s: float, now_monotonic: float | None = None
    ) -> Pose | None:
        """`peer`'s measured pose, or None when it has reported none
        recently: a dead follower's last report is not a pose the arm is
        still at, and no peer has no pose."""
        if peer is None or peer not in self._latest:
            return None
        pose, received_monotonic_s = self._latest[peer]
        now = time.monotonic() if now_monotonic is None else now_monotonic
        if now - received_monotonic_s > stale_timeout_s:
            return None
        return pose


def timestamp_seconds() -> float:
    """This instance's bound clock."""
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


class HeldOverSqueeze:
    """A squeeze held across a change of the chosen robot drives nothing
    until a fresh frame shows the grip released: the hand engages the new
    robot on its next squeeze. The first robot chosen engages a squeeze
    already held."""

    def __init__(self) -> None:
        self._chosen: str | None = None
        self._had_chosen = False
        self._held_over = False

    def note_chosen(self, chosen: str | None) -> None:
        """Note this tick's chosen robot; each stream of the hand notes it,
        and the first to see a change trips the gate for both."""
        if chosen == self._chosen:
            return
        self._held_over = self._held_over or self._had_chosen
        self._had_chosen = self._had_chosen or chosen is not None
        self._chosen = chosen

    def held_over(self) -> bool:
        """Whether a squeeze must be released before it drives again."""
        return self._held_over

    def engages(self, sample: HandSample | None) -> bool:
        """Whether this tick's squeeze drives the chosen robot."""
        if sample is not None and not sample.squeezing:
            self._held_over = False
        return bool(sample and sample.squeezing) and not self._held_over


async def drain_pose_states(
    node_runner,
    topic_module,
    holder: MeasuredPoses,
    label: str,
    token: CancellationToken,
) -> None:
    """Keep `holder` at every paired follower's newest measured pose."""
    try:
        subscription = await topic_module.subscribe(node_runner)
    except Exception as e:
        # Loud and fail-safe: with no measured pose the hand never engages.
        log(f"{label} pose_states subscribe failed: {e!r}")
        return
    # Latched: a malformed pose would repeat every tick.
    unusable = Latch()
    async for peer, message in messages(subscription, token, f"{label} pose_states"):
        try:
            holder.set(peer, Pose.from_xyz_quat(message.position, message.orientation))
            unusable.clear()
        except Exception as e:
            unusable.trip(f"{label} pose_states unusable: {e!r}")


async def _run_stream(
    node_runner,
    topic_module,
    period: float,
    label: str,
    token: CancellationToken,
    build: Callable[[], tuple[object, bytes] | None],
) -> None:
    """Every `period` seconds publish the `(peer, payload)` that `build()`
    returns to that peer, skipping ticks that return None. A peer that
    left between the build and the send is a logged refusal."""
    try:
        publisher = await topic_module.declare_publisher(node_runner)
    except Exception as e:
        log(f"failed to declare the {label} publisher: {e!r}")
        return

    building = Latch()
    publishing = Latch()

    async for _ in ticks(period, token):
        try:
            built = build()
            building.clear()
        except Exception as e:
            building.trip(f"{label} could not build a message: {e!r}")
            continue
        if built is None:
            continue

        peer, payload = built
        try:
            await publisher.publish_to(peer, payload)
            publishing.clear()
        except Exception as e:
            publishing.trip(f"{label} publish failing: {e!r}")


async def stream_pose(
    node_runner,
    *,
    topic_module,
    handedness: Handedness,
    clutch: HandClutch,
    squeeze: HeldOverSqueeze,
    session: FrameSource,
    measured: MeasuredPoses,
    robots: PairedRobots,
    settings: Settings,
    token: CancellationToken,
) -> None:
    """Drive the chosen robot's pair on one hand's pose_link slot from the
    clutch, gated by `squeeze`. Releasing the clutch streams a short freeze burst of the measured
    pose, so the follower stops where the robot is. A change of robot is a
    release: the robot last driven gets the burst at its own measured pose,
    and the new robot engages on the next squeeze."""
    # The one refusal an operator cannot see coming: squeezing while the
    # follower's measured pose is absent or stale does nothing. Say so once.
    waiting = Latch()
    hold = ReleaseHold(settings.tick_period_s)
    # The pair the clutch last drove: every message goes to it, the release
    # burst included, whichever robot is chosen since.
    driven = None
    # This hand's pair last tick: a new pair (a robot change on this hand)
    # re-anchors the clutch.
    last_peer = None

    def build():
        nonlocal driven, last_peer
        # Stamped first: a clock that cannot stamp must fail the tick before
        # the clutch mutates its engagement state.
        timestamp = timestamp_seconds()
        sample = fresh_sample(session, handedness, settings.stale_timeout_s)
        peer = robots.peer_for(topic_module)
        squeeze.note_chosen(robots.chosen())
        if peer != last_peer:
            clutch.release()
        last_peer = peer
        squeezing = squeeze.engages(sample)
        measured_ee = measured.fresh(peer, settings.stale_timeout_s)
        if measured_ee is not None:
            waiting.clear()
        elif squeezing and peer is not None:
            waiting.trip(
                f"{handedness} arm: squeeze ignored until the follower "
                "reports a fresh measured pose"
            )
        target = clutch.step(
            squeezing=squeezing,
            hand=sample.pose if sample else None,
            measured_ee=measured_ee,
        )
        if target is not None:
            driven = peer
        commanded = hold.step(
            target=target,
            measured_ee=measured.fresh(driven, settings.stale_timeout_s),
            now_s=time.monotonic(),
        )
        if commanded is None or driven is None:
            return None
        position, orientation = commanded.as_wire()
        return driven, topic_module.build_message(timestamp, position, orientation)

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
    squeeze: HeldOverSqueeze,
    session: FrameSource,
    robots: PairedRobots,
    settings: Settings,
    token: CancellationToken,
) -> None:
    """Drive the chosen robot's pair on one hand's gripper_link slot from its
    trigger, gated by `squeeze`, the hand's gate the arm stream shares.

    Gated on the grip deadman, not the arm clutch: a side with a gripper but
    no pose_link must still grip. A robot left behind keeps its last
    opening: silence holds the grasp. The new robot's gripper follows from
    the next squeeze.
    """

    def build():
        sample = fresh_sample(session, handedness, settings.stale_timeout_s)
        peer = robots.peer_for(topic_module)
        squeeze.note_chosen(robots.chosen())
        if sample is None or not squeeze.engages(sample) or peer is None:
            return None
        opening = gripper_opening(sample.trigger, settings.gripper_open_fraction)
        # A trigger carries no effort intent, so max_effort rides 0 and the
        # follower's ceiling applies.
        return peer, topic_module.build_message(timestamp_seconds(), opening, 0.0)

    await _run_stream(
        node_runner,
        topic_module,
        settings.tick_period_s,
        f"{handedness} gripper",
        token,
        build,
    )


def cause(message: str) -> str:
    """The clause of the message of a refusal or a failure that names the
    cause, ahead of the remedy a backbone appends after a semicolon."""
    return (message or "no reason given").split(";", 1)[0]


class PostureStatus:
    """The posture buttons as the panel shows them, on the right hand's row:
    the move in flight, then its outcome flashed for a few seconds. Every
    writer runs on the event loop."""

    def __init__(self) -> None:
        self.moving: str | None = None
        self._note = FlashNote()

    def set_note(self, text: str) -> None:
        self._note.set(text)

    def state(self, now_monotonic: float | None = None) -> str | None:
        """What the row says: a fresh note, else the move in flight, else
        None."""
        note = self._note.text(now_monotonic)
        if note:
            return note
        if self.moving is not None:
            return f"moving to {self.moving}..."
        return None

    def end_move(self, label: str) -> None:
        """The move to `label` is over; another button's move stays."""
        if self.moving == label:
            self.moving = None


async def run_posture_button(
    node_runner,
    *,
    action_module,
    pressed: Callable[[HandSample], bool],
    session: FrameSource,
    robots: PairedRobots,
    status: PostureStatus,
    settings: Settings,
    token: CancellationToken,
) -> None:
    """Fire the module's posture move at the chosen robot on the button's
    rising edge.

    `pressed` reads the right hand's button for this posture. Squeezing either
    grip cancels the move in flight: taking manual control back is the cancel
    gesture, so a squeeze also vetoes a press, which the row then says, as
    does a press while a posture move runs. A press while no robot has a
    postures action does nothing, logged once and said on the row of a
    hand with a binding. The move in flight and its outcome go to `status`.
    """
    name = action_module.TARGET_ACTION_NAME
    label = name.removeprefix("move_to_")
    unoffered = Latch()

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
        status.end_move(label)

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
                    cancel = await handle.cancel_goal(GOAL_CANCEL_TIMEOUT_S)
                except Exception as e:
                    # The goal may still finish unwatched; re-arm the button.
                    log(f"move to {label}: cancel failed: {e!r}")
                    status.set_note(f"move to {label}: cancel failed (see log)")
                    disown_waiter()
                else:
                    log(f"move to {label}: cancel {cancel.state.name.lower()}")
                    if cancel.state.name == "UNKNOWN":
                        # The server lost the goal; a result may never come.
                        status.set_note(
                            f"move to {label}: cancel state unknown (see log)"
                        )
                        disown_waiter()
                    else:
                        # The row stops claiming a move; the result's note follows.
                        status.end_move(label)
            elif rising and squeeze:
                note = f"move to {label} ignored: release the grip first"
                status.set_note(note)
                log(note)
            elif rising and status.moving is not None:
                note = f"move to {label} ignored: still moving to {status.moving}"
                status.set_note(note)
                log(note)
            elif handle is None and rising:
                target = robots.producer_for(action_module)
                if target is None:
                    chosen = robots.chosen()
                    note = (
                        f"move to {label}: no robot paired"
                        if chosen is None
                        else f"move to {label}: {chosen} has no postures"
                    )
                    status.set_note(note)
                    unoffered.trip(note)
                    prev_button, prev_squeeze = button, squeeze
                    continue
                unoffered.clear()
                fired = await fire_goal(
                    node_runner,
                    action_module,
                    target,
                    name,
                    duration_s=settings.posture_move_duration_s,
                )
                if fired is None:
                    status.set_note(f"move to {label}: send failed (see log)")
                elif not fired.accepted:
                    status.set_note(f"move to {label} refused: {cause(fired.refusal.message)}")
                else:
                    handle = fired
                    status.moving = label
                    timeout = _result_timeout_s(settings.posture_move_duration_s)
                    waiter = asyncio.create_task(
                        _watch_result(label, fired, timeout, status)
                    )
            prev_button, prev_squeeze = button, squeeze
    finally:
        if waiter is not None:
            waiter.cancel()


async def fire_goal(node_runner, action_module, target, name: str, **request_fields):
    """The goal handle, accepted or refused with the refusal logged, or None
    with the send failure logged. Builds the request too, so a constructor
    that raises is logged as a send failure and the button loop goes on."""
    try:
        request = action_module.GoalRequest(**request_fields)
        fired = await action_module.ActionHandle.fire_goal(
            node_runner, target, request, GOAL_SEND_TIMEOUT_S
        )
    except Exception as e:
        log(f"{name} failed to send: {e!r}")
        return None
    if not fired.accepted:
        log(f"{name} refused: {fired.refusal}")
    return fired


async def _watch_result(
    label: str, handle, timeout_s: float, status: PostureStatus
) -> None:
    """Log the move's result and flash it on the row; the move is over
    either way."""
    try:
        result = await handle.get_result(timeout_s)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        note = f"move to {label}: result unknown"
        log(f"{note}: {e!r}")
        status.set_note(note)
        return
    finally:
        status.end_move(label)
    note = _outcome_note(label, result)
    log(note if result.cancel_reason is None else f"{note}: {result.cancel_reason}")
    status.set_note(note)


def _outcome_note(label: str, result) -> str:
    """What the row says of how the move to `label` ended: completed, failed
    with the cause of its failure, or cancelled, abandoned or expired."""
    status = result.status.name
    if status == "FAILED":
        return f"move to {label} failed: {cause(result.failure.message)}"
    return f"move to {label} {status.lower()}"
