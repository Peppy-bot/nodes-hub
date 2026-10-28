"""The command streams, one task per hand per pairing slot.

Each slot holds a pair per robot; a stream commands the chosen robot's pair
and is silent toward the others. Silence is the deadman: a disengaged hand
stops commanding and the follower holds. A squeeze engages a hand only once
the headset holds the robot (`drive`): its lease taken, or its open lease
found to allow the headset, and its moves in flight stopped. Releasing the
arm clutch first streams a short burst of the measured pose, so the follower
stops where the robot is, and the hand holds the robot until that burst
ends; a change of robot releases the same way toward the robot left behind,
and the new robot engages on the next squeeze. One task per stream.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import peppygen.clock

from xr_commander.bus import (
    GOAL_CANCEL_TIMEOUT_S,
    GOAL_SEND_TIMEOUT_S,
    REAL_CLOCK,
    CancellationToken,
    Clock,
    FlashNote,
    Latch,
    cause,
    log,
    messages,
    spawn,
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
from xr_commander.drive import (
    GRIP_HELD,
    POSTURE,
    Driver,
    HandState,
    Lost,
    Refusal,
    Withdrawn,
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

    def __init__(self, clock: Clock = REAL_CLOCK) -> None:
        self._clock = clock
        self._latest: dict[object, tuple[Pose, float]] = {}

    def set(self, peer, pose: Pose) -> None:
        self._latest[peer] = (pose, self._clock.monotonic())

    def fresh(
        self, peer, stale_timeout_s: float, now_monotonic: float | None = None
    ) -> Pose | None:
        """`peer`'s measured pose, or None when it has reported none
        recently: a dead follower's last report is not a pose the arm is
        still at, and no peer has no pose."""
        if peer is None or peer not in self._latest:
            return None
        pose, received_monotonic_s = self._latest[peer]
        now = self._clock.monotonic() if now_monotonic is None else now_monotonic
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
    build: Callable[[], Awaitable[tuple[object, bytes] | None]],
    clock: Clock,
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

    async for _ in ticks(period, token, clock):
        try:
            built = await build()
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
    gripper_module,
    handedness: Handedness,
    hand: HandState,
    driver: Driver,
    clutch: HandClutch,
    session: FrameSource,
    measured: MeasuredPoses,
    robots: PairedRobots,
    settings: Settings,
    token: CancellationToken,
    clock: Clock = REAL_CLOCK,
) -> None:
    """Drive the chosen robot's pair on one hand's pose_link slot from the
    clutch, and decide for both streams of the hand when it engages.

    A squeeze engages once the hand has a pair on the chosen robot, a fresh
    measured pose of its arm, and the headset holds the robot. Releasing
    the clutch streams a short freeze burst of the measured pose, so the
    follower stops where the robot is, and the hand keeps the robot until
    the burst ends. A change of robot is a release: the robot last driven
    gets the burst at its own measured pose, and the new robot engages on
    the next squeeze."""
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
    # The hand was engaged at the end of the last tick.
    was_engaged = False
    # The hand keeps its robot until its release burst ends.
    releasing = False

    async def engage(robot: str) -> None:
        def still_squeezing() -> bool:
            """The take runs over ticks: the grip may be released, or
            another robot chosen, since the squeeze."""
            latest = fresh_sample(session, handedness, settings.stale_timeout_s, clock)
            return latest is not None and latest.squeezing and robots.chosen() == robot

        match await driver.acquire(robot, handedness, still_squeezing):
            case Refusal(cause=refused):
                hand.note.set(f"squeeze refused: {refused}")
                log(f"{handedness} squeeze on {robot} refused: {refused}")
                hand.wait_release = True
                return
            case Withdrawn():
                return
        # A robot left behind in its release burst is released now.
        if hand.robot not in (None, robot):
            driver.drop(hand.robot, handedness)
        hand.engaged, hand.robot = True, robot

    async def build():
        nonlocal driven, last_peer, was_engaged, releasing
        # Stamped first: a clock that cannot stamp must fail the tick before
        # the clutch mutates its engagement state.
        timestamp = timestamp_seconds()
        sample = fresh_sample(session, handedness, settings.stale_timeout_s, clock)
        squeezing = bool(sample and sample.squeezing)
        chosen = robots.chosen()
        hand.observe(chosen, sample)
        peer = robots.peer_for(topic_module)
        if peer != last_peer:
            clutch.release()
        last_peer = peer
        measured_ee = measured.fresh(peer, settings.stale_timeout_s)
        if measured_ee is not None:
            waiting.clear()
        elif squeezing and peer is not None:
            waiting.trip(
                f"{handedness} arm: squeeze ignored until the follower "
                "reports a fresh measured pose"
            )
        paired = peer is not None or robots.peer_for(gripper_module) is not None
        ready = paired and (peer is None or measured_ee is not None)
        if ready and hand.may_engage(sample):
            await engage(chosen)
            if hand.engaged:
                releasing = False
        if was_engaged and not hand.engaged:
            releasing = True
        was_engaged = hand.engaged
        target = clutch.step(
            squeezing=hand.engaged,
            hand=sample.pose if sample else None,
            measured_ee=measured_ee,
        )
        if target is not None:
            driven = peer
        commanded = hold.step(
            target=target,
            measured_ee=measured.fresh(driven, settings.stale_timeout_s),
            now_s=clock.monotonic(),
        )
        if releasing and commanded is None:
            releasing = False
            driver.drop(hand.robot, handedness)
            hand.robot = None
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
        clock,
    )


async def stream_gripper(
    node_runner,
    *,
    topic_module,
    handedness: Handedness,
    hand: HandState,
    session: FrameSource,
    robots: PairedRobots,
    settings: Settings,
    token: CancellationToken,
    clock: Clock = REAL_CLOCK,
) -> None:
    """Drive the chosen robot's pair on one hand's gripper_link slot from its
    trigger while the hand is engaged on that robot, which its pose stream
    decides.

    A robot left behind keeps its last opening: silence holds the grasp. The
    new robot's gripper follows from the next squeeze.
    """

    async def build():
        sample = fresh_sample(session, handedness, settings.stale_timeout_s, clock)
        peer = robots.peer_for(topic_module)
        engaged = hand.engaged and hand.robot == robots.chosen()
        if sample is None or not engaged or peer is None:
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
        clock,
    )


class PostureStatus:
    """The posture buttons as the panel shows them, on the right hand's row:
    the move in flight, then its outcome flashed for a few seconds. Every
    writer runs on the event loop."""

    def __init__(self, clock: Clock = REAL_CLOCK) -> None:
        self.moving: str | None = None
        self._note = FlashNote(clock)

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
    driver: Driver,
    status: PostureStatus,
    settings: Settings,
    token: CancellationToken,
    clock: Clock = REAL_CLOCK,
) -> None:
    """Fire the module's posture move at the chosen robot on the button's
    rising edge.

    `pressed` reads the right hand's button for this posture. A press takes
    the robot as a squeeze does, asks the daemon whether the headset may
    move it, and keeps it until the move ends. A press while a grip is held,
    while a hand still keeps the chosen robot after its release, or while a
    posture move runs does nothing, and the row says why. A squeeze during
    the move stops it: the squeeze stops the moves in flight on the robot.
    A move whose send or result is lost ends as
    `Driver.end_lost_posture_move` says. The move in flight and its outcome
    go to `status`.
    """
    name = action_module.TARGET_ACTION_NAME
    label = name.removeprefix("move_to_")
    unoffered = Latch()

    # None until the button has been observed at all: a thumb already down on
    # the first tracked frame is not a press, and neither is one held across
    # a stale gap. Rising edges exist only against an observed release.
    prev_button: bool | None = None
    waiter: asyncio.Task | None = None

    def ignored(note: str) -> None:
        status.set_note(note)
        log(note)

    async def press() -> asyncio.Task | None:
        """The press on the chosen robot: the task that watches the move it
        started, or None with the reason on the row. The move counts as in
        flight from the press on, so the other posture button waits for it."""
        target = robots.producer_for(action_module)
        robot = robots.chosen()
        if target is None:
            note = (
                f"move to {label}: no robot paired"
                if robot is None
                else f"move to {label}: {robot} has no postures"
            )
            status.set_note(note)
            unoffered.trip(note)
            return None
        unoffered.clear()
        status.moving = label

        async def start_move():
            """The accepted goal of the move, else why it did not start."""
            refused = await driver.check_move(target.instance_id)
            if refused is not None:
                return Refusal(refused)
            fired = await fire_goal(
                node_runner,
                action_module,
                target,
                name,
                duration_s=settings.posture_move_duration_s,
            )
            if fired is None:
                return Lost("the goal could not be sent (see log)")
            if not fired.accepted:
                return Refusal(cause(fired.refusal.message))
            driver.on_posture_cancel(robot, lambda: _cancel_soon(fired, label))
            return fired

        started = await driver.start_posture_move(robot, start_move)
        if isinstance(started, Refusal):
            status.end_move(label)
            ignored(f"move to {label} refused: {started.cause}")
            return None
        timeout = _result_timeout_s(settings.posture_move_duration_s)
        return asyncio.create_task(
            _watch_result(label, started, timeout, status, driver, robot)
        )

    try:
        async for _ in ticks(settings.tick_period_s, token, clock):
            stale_s = settings.stale_timeout_s
            right = fresh_sample(session, "right", stale_s, clock)
            left = fresh_sample(session, "left", stale_s, clock)
            squeeze = bool(right and right.squeezing) or bool(left and left.squeezing)
            button = pressed(right) if right is not None else prev_button
            rising = bool(button) and prev_button is False
            prev_button = button
            if waiter is not None and waiter.done():
                waiter = None
            if not rising:
                continue
            held = driver.press_refusal(robots.chosen())
            if squeeze:
                ignored(f"move to {label} ignored: {GRIP_HELD}")
            elif held is not None:
                ignored(f"move to {label} ignored: {held}")
            elif status.moving is not None:
                ignored(f"move to {label} ignored: still moving to {status.moving}")
            elif waiter is None:
                waiter = await press()
    finally:
        if waiter is not None:
            waiter.cancel()


def _cancel_soon(handle, label: str) -> None:
    """Cancel the posture move to `label` without waiting on it: its result
    watcher reports how it ended."""

    async def cancel() -> None:
        try:
            outcome = await handle.cancel_goal(GOAL_CANCEL_TIMEOUT_S)
        except Exception as e:
            log(f"move to {label}: cancel failed: {e!r}")
            return
        log(f"move to {label}: cancel {outcome.state.name.lower()}")

    spawn(cancel())


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
    label: str,
    handle,
    timeout_s: float,
    status: PostureStatus,
    driver: Driver,
    robot: str,
) -> None:
    """Log the move's result and flash it on the row. A move whose result
    does not come ends as `Driver.end_lost_posture_move` says. The move is
    over either way, so its hold of `robot` drops."""
    try:
        result = await handle.get_result(timeout_s)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        note = f"move to {label}: result unknown"
        log(f"{note}: {e!r}")
        status.set_note(note)
        await driver.end_lost_posture_move(robot)
        return
    finally:
        status.end_move(label)
        driver.drop(robot, POSTURE)
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
