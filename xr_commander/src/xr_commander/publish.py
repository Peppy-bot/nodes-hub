"""The command streams of the hands, one task per hand per pairing slot,
and what every stream of a grip shares, the base's included (`base`): the
loop that publishes a stream, the release burst and the grip's hold of its
robot through it (`GripStream`), and the drain that keeps each paired
follower's newest report.

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
from typing import Generic, TypeVar

import peppygen.clock
from control_core_py.geometry import Pose, Stamped

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
from xr_commander.clutch import HandClutch, Setpoint
from xr_commander.config import Settings
from xr_commander.devices import (
    FrameSource,
    Hand,
    fresh_sample,
    gripper_opening,
)
from xr_commander.drive import Driver, Grip, GripState
from xr_commander.robots import PairedRobots

# What a release burst holds: an arm's pose, or a base's velocity.
Held = TypeVar("Held")


class MeasuredPoses:
    """Each paired follower's most recent measured pose, by the peer that
    reported it; one writer, one reader, both on the loop."""

    def __init__(self, clock: Clock = REAL_CLOCK) -> None:
        self._clock = clock
        self._latest: dict[object, tuple[Stamped[Pose], float]] = {}

    def set(self, peer, pose: Stamped[Pose]) -> None:
        self._latest[peer] = (pose, self._clock.monotonic())

    def fresh(
        self, peer, stale_timeout_s: float, now_monotonic: float | None = None
    ) -> Stamped[Pose] | None:
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


# How long the release burst streams after a stream stops driving. An arm's
# follower may keep converging to the last command it heard (openarm's
# backbone documents that choice), so an arm's release commands "stay where
# the robot is"; a base's commands zero velocity, which drive_link says
# brings the base to rest sooner. A burst of several messages, so one
# dropped sample cannot lose the stop.
RELEASE_HOLD_S = 0.25
# The burst's own floor in messages, so a slow command rate cannot reduce it
# to the single sample the burst exists to avoid depending on.
RELEASE_HOLD_MESSAGES = 3


class ReleaseHold(Generic[Held]):
    """The release burst: when a stream stops driving, the value it holds at
    that tick becomes the streamed value for a short burst. An arm holds its
    last fresh measured pose, so the follower stops where the robot is; a
    base holds zero velocity."""

    def __init__(self, tick_period_s: float, hold_s: float = RELEASE_HOLD_S):
        self._hold_s = max(hold_s, RELEASE_HOLD_MESSAGES * tick_period_s)
        self._driving = False
        self._held: Held | None = None
        self._until = 0.0

    def step(
        self, *, target: Held | None, held: Held | None, now_s: float
    ) -> Held | None:
        """This tick's value to stream, or None for silence: `target` while
        driving, then `held` as latched at the tick the driving stopped, for
        the burst that follows. One call per tick, so no caller can arm the
        burst without also being able to void it. With nothing to hold at
        that tick, such as no fresh measured pose, silence."""
        if target is not None:
            self._driving = True
            self._until = 0.0
            return target
        if self._driving:
            self._driving = False
            if held is not None:
                self._held = held
                self._until = now_s + self._hold_s
        if now_s < self._until:
            return self._held
        return None


def _held_pose(measured_ee: Stamped[Pose] | None) -> Setpoint | None:
    """What an arm's release burst holds: the measured pose in its frame,
    None without a fresh one."""
    if measured_ee is None:
        return None
    return Setpoint(measured_ee.frame, measured_ee.value)


class GripStream(Generic[Held]):
    """What a stream that a grip drives sends at each tick, and the grip's
    hold on its robot: every value goes to the pair the stream last drove,
    the release burst included, whichever robot is chosen since, and the
    grip keeps its robot until that burst ends."""

    def __init__(self, driver: Driver, grip: Grip, tick_period_s: float) -> None:
        self._driver = driver
        self._grip = grip
        self._state = driver.grip(grip)
        self._burst: ReleaseHold[Held] = ReleaseHold(tick_period_s)
        # The pair the stream last drove.
        self._driven = None
        # The grip was engaged at the last step.
        self._was_engaged = False
        # The grip keeps its robot until its release burst ends.
        self._releasing = False

    def step(
        self,
        peer,
        target: Held | None,
        held_for: Callable[[object], Held | None],
        now_s: float,
    ) -> tuple[object, Held] | None:
        """This tick's pair and value, None for silence: `target` to `peer`
        while the grip drives, then, for the release burst, what `held_for`
        gives for the pair last driven. A grip that disengaged releases its
        robot once its burst ends."""
        engaged = self._state.engaged
        if engaged:
            self._releasing = False
        elif self._was_engaged:
            self._releasing = True
        self._was_engaged = engaged
        if target is not None:
            self._driven = peer
        streamed = self._burst.step(
            target=target, held=held_for(self._driven), now_s=now_s
        )
        if self._releasing and streamed is None:
            self._releasing = False
            self._driver.drop(self._state.robot, self._grip)
            self._state.robot = None
        if streamed is None or self._driven is None:
            return None
        return self._driven, streamed


async def drain(
    node_runner,
    topic_module,
    label: str,
    keep: Callable[[object, object], None],
    token: CancellationToken,
) -> None:
    """Hand each (peer, message) of `topic_module` to `keep` until the token
    cancels. A message that `keep` refuses by raising is logged once until
    one is kept: a malformed message repeats at its stream's rate. With no
    subscription, which is logged, nothing is kept, and the stream that
    reads it never engages."""
    subscription = await subscribe_or_log(node_runner, topic_module, label)
    if subscription is None:
        return
    unusable = Latch()
    async for peer, message in messages(subscription, token, label):
        try:
            keep(peer, message)
            unusable.clear()
        except Exception as e:
            unusable.trip(f"{label} unusable: {e!r}")


async def drain_pose_states(
    node_runner,
    topic_module,
    holder: MeasuredPoses,
    label: str,
    token: CancellationToken,
) -> None:
    """Keep `holder` at every paired follower's newest measured pose."""

    def keep(peer, message) -> None:
        pose = Pose.parse(message.position, message.orientation)
        holder.set(peer, Stamped.parse(message.frame, message.timestamp, pose))

    await drain(node_runner, topic_module, f"{label} pose_states", keep, token)


async def run_stream(
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
    the next squeeze. Each setpoint gives the time of its tick on this
    instance's clock, the clock of transforms, so the follower drops by its
    age a setpoint that a stall held back."""
    # The one refusal an operator cannot see coming: squeezing while the
    # follower's measured pose is absent or stale does nothing. Say so once.
    waiting = Latch()
    hand = driver.hands[handedness]
    stream = GripStream[Setpoint](driver, handedness, settings.tick_period_s)
    # This hand's pair last tick: a new pair (a robot change on this hand)
    # re-anchors the clutch.
    last_peer = None

    def still_squeezing(robot: str) -> Callable[[], bool]:
        def wanted() -> bool:
            """The take runs over ticks: the grip may be released, or
            another robot chosen, since the squeeze."""
            latest = fresh_sample(session, handedness, settings.stale_timeout_s, clock)
            return latest is not None and latest.squeezing and robots.chosen() == robot

        return wanted

    def held_pose(driven) -> Setpoint | None:
        return _held_pose(measured.fresh(driven, settings.stale_timeout_s))

    async def build():
        nonlocal last_peer
        # Stamped first: a clock that cannot stamp must fail the tick before
        # the clutch mutates its engagement state.
        timestamp = _timestamp_seconds()
        sample = fresh_sample(session, handedness, settings.stale_timeout_s, clock)
        grip = None if sample is None else sample.squeezing
        squeezing = grip is True
        chosen = robots.chosen()
        hand.observe(chosen, grip)
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
        if ready and hand.may_engage(grip):
            await driver.engage(chosen, handedness, still_squeezing(chosen))
        target = clutch.step(
            squeezing=hand.engaged,
            hand=sample.pose if sample else None,
            measured_ee=measured_ee,
        )
        sent = stream.step(peer, target, held_pose, clock.monotonic())
        if sent is None:
            return None
        driven, commanded = sent
        pose = commanded.pose
        return driven, topic_module.build_message(
            commanded.frame,
            timestamp,
            list(pose.position.position),
            list(pose.orientation.xyzw),
        )

    await run_stream(
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
    hand: GripState,
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

    await run_stream(
        node_runner,
        topic_module,
        settings.tick_period_s,
        f"{handedness} gripper",
        token,
        build,
        clock,
    )
