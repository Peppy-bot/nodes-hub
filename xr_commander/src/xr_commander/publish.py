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

from collections.abc import Awaitable, Callable

import peppygen.clock

from xr_commander.bus import (
    REAL_CLOCK,
    CancellationToken,
    Clock,
    Latch,
    log,
    messages,
    subscribe_or_log,
    ticks,
)
from xr_commander.clutch import HandClutch
from xr_commander.config import Settings
from xr_commander.devices import (
    FrameSource,
    Hand,
    fresh_sample,
    gripper_opening,
)
from xr_commander.drive import (
    Driver,
    HandState,
    Refusal,
    Withdrawn,
)
from xr_commander.frames import Pose
from xr_commander.robots import PairedRobots


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


def _timestamp_seconds() -> float:
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
    subscription = await subscribe_or_log(
        node_runner, topic_module, f"{label} pose_states"
    )
    if subscription is None:
        # Loud and fail-safe: with no measured pose the hand never engages.
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
    handedness: Hand,
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
        timestamp = _timestamp_seconds()
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
    handedness: Hand,
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
        return peer, topic_module.build_message(_timestamp_seconds(), opening, 0.0)

    await _run_stream(
        node_runner,
        topic_module,
        settings.tick_period_s,
        f"{handedness} gripper",
        token,
        build,
        clock,
    )
