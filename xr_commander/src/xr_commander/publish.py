"""The command streams of the hands, one task per hand per pairing slot,
and what every stream of a grip shares, the base's included (`base`): the
loop that publishes a stream, the release burst (`ReleaseHold`) and the pins
each grip stream sends to (`GripStream`), and the drain that keeps each
paired follower's newest report.

Each slot holds a pair per robot; a stream commands the pins of its grip's
phase and is silent toward every other pair. Silence is the deadman. A
squeeze engages a hand only once the headset holds the robot
(`grip_world`): its lease taken, or its open lease found to allow the
headset, and its moves in flight stopped. When a stream stops driving it
streams a release burst of the measured pose to the pair it last drove, so
the follower stops where the robot is, and the hand holds the robot until
that burst ends. Each tick reads its inputs once and steps the world before
it streams; no tick awaits a call.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, replace
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
    HandSample,
    fresh_sample,
    gripper_opening,
)
from xr_commander.drive import Driver
from xr_commander.grip_world import (
    Engaged,
    NotReady,
    Phase,
    Pins,
    Ready,
    Readiness,
    Slot,
    Streamed,
    Tick,
    TickInputs,
    driven_pins,
    pin_on,
    reading_of,
    slot_pins,
)
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


# Why a hand's squeeze requests nothing.
NO_PAIR = "no pair for this hand"
NO_MEASURED_POSE = "no measured pose"


def arm_ready(pins: Pins, poses: Mapping[Hashable, Stamped[Pose] | None]) -> Readiness:
    """A hand's readiness: a pair on its arm or gripper slot, and a fresh
    measured pose of its arm pair when it has one."""
    arm = pin_on(pins, Slot.ARM)
    if arm is None and pin_on(pins, Slot.GRIPPER) is None:
        return NotReady(NO_PAIR)
    if arm is not None and poses.get(arm) is None:
        return NotReady(NO_MEASURED_POSE)
    return Ready()


# How long the release burst streams after a stream stops driving. An arm's
# follower may keep converging to the last command it heard (openarm's
# backbone documents that choice), so an arm's release commands "stay where
# the robot is"; a base's commands zero velocity, which drive_link says
# brings the base to rest sooner.
RELEASE_HOLD_S = 0.25
# The burst's floor in messages: it ends once it sent this many and
# RELEASE_HOLD_S passed, so one dropped sample cannot lose the stop at any
# command rate.
RELEASE_HOLD_MESSAGES = 3


@dataclass(frozen=True)
class _Burst(Generic[Held]):
    value: Held
    started_s: float
    sent: int = 0


class ReleaseHold(Generic[Held]):
    """The release burst: when a stream stops driving, the value it holds at
    that tick becomes the streamed value. Each tick of the burst sends it;
    after the send the burst ends once it has sent `RELEASE_HOLD_MESSAGES`
    values and `RELEASE_HOLD_S` passed since its start. It also ends at a
    tick its pair has left the slot. With nothing to hold, no burst
    starts."""

    def __init__(self) -> None:
        self._driving = False
        self._burst: _Burst[Held] | None = None

    @property
    def running(self) -> bool:
        return self._burst is not None

    def step(
        self,
        *,
        target: Held | None,
        held: Held | None,
        now_s: float,
        pair_present: bool,
    ) -> Held | None:
        """This tick's value, None for silence: `target` while driving, then
        `held`, latched at the tick the driving stopped, for the burst."""
        if target is not None:
            self._driving = True
            self._burst = None
            return target
        if self._driving:
            self._driving = False
            if held is not None:
                self._burst = _Burst(held, now_s)
        if self._burst is None:
            return None
        if not pair_present:
            self._burst = None
            return None
        burst = replace(self._burst, sent=self._burst.sent + 1)
        ended = (
            burst.sent >= RELEASE_HOLD_MESSAGES
            and now_s - burst.started_s >= RELEASE_HOLD_S
        )
        self._burst = None if ended else burst
        return burst.value


def _held_pose(measured_ee: Stamped[Pose] | None) -> Setpoint | None:
    """What an arm's release burst holds: the measured pose in its frame,
    None without a fresh one."""
    if measured_ee is None:
        return None
    return Setpoint(measured_ee.frame, measured_ee.value)


@dataclass(frozen=True)
class StreamStep(Generic[Held]):
    """One tick of a grip stream: the pair and value it sends, None for
    silence, and whether the burst runs, after this tick or in this tick's
    send."""

    send: tuple[Hashable, Held] | None
    burst_running: bool


class GripStream(Generic[Held]):
    """A grip's stream on one slot: every value goes to the phase's pin on
    the slot, the burst to the pin the stream last drove."""

    def __init__(self, slot: Slot) -> None:
        self._slot = slot
        self._burst: ReleaseHold[Held] = ReleaseHold()
        self._driven: Hashable | None = None

    def step(
        self,
        phase: Phase,
        *,
        target: Held | None,
        held: Held | None,
        peers: frozenset[Hashable],
        now_s: float,
    ) -> StreamStep[Held]:
        """This tick's send from the phase after its `Tick` step: `target`
        to an `Engaged` phase's pin, else the burst's `held` value."""
        assert target is None or isinstance(phase, Engaged), (
            "a target drives an engaged grip"
        )
        pin = pin_on(driven_pins(phase), self._slot)
        if self._burst.running:
            assert pin == self._driven, "the burst goes to the pin last driven"
        if target is not None:
            assert pin is not None, "a target goes to a pin"
            self._driven = pin
        value = self._burst.step(
            target=target, held=held, now_s=now_s, pair_present=pin in peers
        )
        send = None if value is None else (pin, value)
        in_burst = target is None and value is not None
        return StreamStep(send, self._burst.running or in_burst)


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
    build: Callable[[], tuple[object, bytes] | None],
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
    await run_stream(
        node_runner,
        topic_module,
        settings.tick_period_s,
        f"{handedness} arm",
        token,
        pose_tick(
            topic_module=topic_module,
            gripper_module=gripper_module,
            handedness=handedness,
            driver=driver,
            clutch=clutch,
            session=session,
            measured=measured,
            robots=robots,
            settings=settings,
            clock=clock,
        ),
        clock,
    )


def pose_tick(
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
    clock: Clock,
    timestamp_s: Callable[[], float] = _timestamp_seconds,
) -> Callable[[], tuple[object, bytes] | None]:
    """One tick of a hand's pose stream: its grip's `Tick`, the clutch, the
    burst, its `Streamed`, then the setpoint to send. `timestamp_s` stamps
    each setpoint: this instance's bound clock."""
    stream = GripStream[Setpoint](Slot.ARM)

    def tick() -> tuple[object, bytes] | None:
        # Stamped first: a clock that cannot stamp fails the tick before any
        # step.
        timestamp = timestamp_s()
        sample = fresh_sample(session, handedness, settings.stale_timeout_s, clock)
        _names, chosen = robots.standing()
        arms = robots.pairs(topic_module)
        peers, pins = slot_pins(
            {Slot.ARM: arms, Slot.GRIPPER: robots.pairs(gripper_module)}, chosen
        )
        now_s = clock.monotonic()
        poses = {
            peer: measured.fresh(peer, settings.stale_timeout_s, now_s)
            for peer in arms.peers
        }
        squeezing = None if sample is None else sample.squeezing
        readings = {handedness: reading_of(squeezing)}
        before = driver.world.grips.of(handedness).phase
        inputs = TickInputs(
            readings,
            chosen,
            peers,
            pins,
            arm_ready(pins, poses),
            arm_ready(driven_pins(before), poses),
        )
        phase = driver.step(Tick(handedness, inputs)).grips.of(handedness).phase
        measured_ee = poses.get(pin_on(driven_pins(phase), Slot.ARM))
        target = clutch.step(
            squeezing=isinstance(phase, Engaged),
            hand=sample.pose if sample is not None else None,
            measured_ee=measured_ee,
        )
        streamed = stream.step(
            phase,
            target=target,
            held=_held_pose(measured_ee),
            peers=arms.peers,
            now_s=now_s,
        )
        driver.step(Streamed(handedness, streamed.burst_running))
        if streamed.send is None:
            return None
        pin, commanded = streamed.send
        pose = commanded.pose
        return pin, topic_module.build_message(
            commanded.frame,
            timestamp,
            list(pose.position.position),
            list(pose.orientation.xyzw),
        )

    return tick


def gripper_tick(
    sample: HandSample | None,
    chosen: str | None,
    must_release: bool,
    phase: Phase,
    open_fraction: float,
) -> tuple[Hashable, float] | None:
    """The gripper stream's send at one tick: the trigger's opening to the
    gripper pin of a phase engaged on the chosen robot, while the hand's own
    fresh sample squeezes and the hand has no `must_release`; None
    otherwise."""
    if sample is None or not sample.squeezing or must_release:
        return None
    if not isinstance(phase, Engaged) or phase.robot != chosen:
        return None
    pin = pin_on(phase.pins, Slot.GRIPPER)
    if pin is None:
        return None
    return pin, gripper_opening(sample.trigger, open_fraction)


async def stream_gripper(
    node_runner,
    *,
    topic_module,
    handedness: Hand,
    driver: Driver,
    session: FrameSource,
    robots: PairedRobots,
    settings: Settings,
    token: CancellationToken,
    clock: Clock = REAL_CLOCK,
    timestamp_s: Callable[[], float] = _timestamp_seconds,
) -> None:
    """Drive the gripper pin of the hand's engaged phase from its trigger, at
    each tick whose own fresh sample squeezes with no release owed, while the
    phase is on the chosen robot. `timestamp_s` stamps each setpoint, as it
    stamps the pose stream's: this instance's bound clock.

    A robot left behind keeps its last opening: silence holds the grasp. The
    new robot's gripper follows from the next squeeze.
    """

    def build() -> tuple[object, bytes] | None:
        sample = fresh_sample(session, handedness, settings.stale_timeout_s, clock)
        _names, chosen = robots.standing()
        world = driver.world
        send = gripper_tick(
            sample,
            chosen,
            world.hands.of(handedness).must_release,
            world.grips.of(handedness).phase,
            settings.gripper_open_fraction,
        )
        if send is None:
            return None
        pin, opening = send
        # A trigger carries no effort intent, so max_effort rides 0 and the
        # follower's ceiling applies.
        return pin, topic_module.build_message(timestamp_s(), opening, 0.0)

    await run_stream(
        node_runner,
        topic_module,
        settings.tick_period_s,
        f"{handedness} gripper",
        token,
        build,
        clock,
    )
