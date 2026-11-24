"""The base stream: the thumbsticks drive the chosen robot's base over
drive_link while a grip of either hand is held.

A squeeze takes the robot as it does for an arm (`grip_world`): the lease,
then robot_stop's stop. The left thumbstick gives the velocity forward and
back along x and, on a base that drives sideways, along y; the right
thumbstick's left and right give the turn. A full deflection gives the limit
in its direction that the newest drive_states gives, as base_state's
describe_base gives it, and a partial one that fraction, so every setpoint
lies within the base's limits. Each setpoint is in the robot frame at the
latest time: it carries no frame and no timestamp.

The stream streams the commanded twist while the base grip is engaged, the
base follows now and the twist is not zero. The base follows now while its
newest drive_states, recorded within 0.5 s, reports following, no stop of
its robot is in flight, and that report arrived more than 0.5 s after the
newest stop of the robot that was answered or refused. When the stream stops
driving it streams zero velocity for the release burst, then nothing, and
the grip holds the robot through that burst; a hold the base reports does
not end the burst. A change of robot sends the burst to the base left
behind; the new base engages on the next squeeze.

The times of drive_states count by their arrival on the instance's clock,
the clock of transforms, never by their timestamps; a step back of that
clock counts as no time passing.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from enum import StrEnum

import peppygen.clock

from xr_commander.bus import REAL_CLOCK, CancellationToken, Clock
from xr_commander.config import Settings
from xr_commander.devices import (
    RESTING,
    FrameSource,
    Hand,
    Thumbstick,
    XrFrame,
    fresh_frame,
    hand_squeezing,
)
from xr_commander.drive import Driver, Stops
from xr_commander.grip_world import (
    BASE,
    Engaged,
    NotReady,
    Phase,
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
from xr_commander.publish import GripStream, run_stream
from xr_commander.robots import PairedRobots

# The thumbstick that gives the velocity along x and y, and the one whose
# left and right give the turn.
LINEAR_HAND = Hand.LEFT
TURN_HAND = Hand.RIGHT

# A base that has sent no drive_states for longer than this (ns) on the
# instance's clock does not report: base_state's bound of silence, 0.5 s.
REPORT_SILENCE_NS = 500_000_000
# How long after a stop's answer or refusal a following starts the stream
# (ns): a robot stop holds the base until more than 0.5 s pass, from the stop
# on, with no setpoint, as drive_link states.
STOP_HOLD_NS = 500_000_000


class FollowState(StrEnum):
    """Whether the base follows the headset, as drive_states gives it."""

    FOLLOWING = "following"
    BASE_MODE = "base_mode"
    BASE_SILENCE = "base_silence"
    ROBOT_STOP = "robot_stop"
    FOLLOWER_GOAL = "follower_goal"


def _follow_state(value: str) -> FollowState:
    """ValueError naming a follow_state that drive_link does not give."""
    try:
        return FollowState(value)
    except ValueError:
        raise ValueError(f"follow_state {value!r} is none of drive_link's") from None


def _lowest(name: str, value: float) -> float:
    """A most negative velocity or turn rate; ValueError naming one that
    base_state does not allow."""
    if not (math.isfinite(value) and value <= 0.0):
        raise ValueError(f"{name} {value!r} is not 0 or below")
    return value


def _highest(name: str, value: float) -> float:
    """A most positive velocity or turn rate; ValueError naming one that
    base_state does not allow."""
    if not (math.isfinite(value) and value >= 0.0):
        raise ValueError(f"{name} {value!r} is not 0 or above")
    return value


@dataclass(frozen=True)
class AxisLimits:
    """The most negative and the most positive velocity of one axis of the
    base: `lowest` is 0 or below, `highest` 0 or above."""

    lowest: float
    highest: float

    def scaled(self, deflection: float) -> float:
        """The velocity a deflection in [-1, 1] commands: that fraction of
        the limit in its direction."""
        if deflection >= 0.0:
            return deflection * self.highest
        return -deflection * self.lowest


@dataclass(frozen=True)
class BaseLimits:
    """The base's limits, as the sticks scale them: along x, along y and of
    the turn. A base whose y range is 0 to 0 takes 0 along y."""

    x: AxisLimits
    y: AxisLimits
    turn: AxisLimits

    @classmethod
    def parse(cls, states) -> BaseLimits:
        """The limits that a drive_states gives, each axis's most negative
        and most positive velocity alone: finite, with the lowest at most 0
        and the highest at least 0, as base_state allows. ValueError names a
        limit outside that; the other limits are the follower's."""
        linear = states.linear_limits
        angular = states.angular_limits
        return cls(
            x=AxisLimits(
                _lowest("min_velocity along x", linear.min_velocity[0]),
                _highest("max_velocity along x", linear.max_velocity[0]),
            ),
            y=AxisLimits(
                _lowest("min_velocity along y", linear.min_velocity[1]),
                _highest("max_velocity along y", linear.max_velocity[1]),
            ),
            turn=AxisLimits(
                _lowest("min_rate", angular.min_rate),
                _highest("max_rate", angular.max_rate),
            ),
        )


@dataclass(frozen=True)
class BaseReport:
    """The newest drive_states of a base, as the headset reads it."""

    follow_state: FollowState
    # The base's mode, which the panel names while the mode holds the base.
    mode: str
    limits: BaseLimits
    # When it arrived, on the elapsed instance clock of `BaseReports` (ns).
    arrived_ns: int

    def follows_after(self, answered_ns: int) -> bool:
        """Whether the base follows after a stop answered or refused at
        `answered_ns`: this report gives following and arrived more than
        `STOP_HOLD_NS` after it."""
        return (
            self.follow_state is FollowState.FOLLOWING
            and self.arrived_ns - answered_ns > STOP_HOLD_NS
        )


def _instance_now_ns() -> int:
    """This instance's bound clock, the clock of transforms."""
    return peppygen.clock.now_ns()


class BaseReports:
    """Each paired base's newest drive_states, by the peer that reported it.
    Every call runs on the loop: the drain records, the stream and the panel
    read, and each call that reads the instance's clock keeps count of how
    far it went back."""

    def __init__(self, now_ns: Callable[[], int] = _instance_now_ns) -> None:
        self._now_ns = now_ns
        # The first and the newest reading of the instance's clock, and how
        # far the clock went back in all before the newest (ns).
        self._first_ns: int | None = None
        self._read_ns: int | None = None
        self._back_ns = 0
        self._newest: dict[object, BaseReport] = {}
        # Why the newest drive_states of each base was dropped, until one is
        # kept.
        self._dropped: dict[object, str] = {}

    def record(self, peer, message) -> None:
        """`message`, a drive_states of `peer`, as it arrives now. One whose
        follow_state drive_link does not give, or whose limits base_state
        does not allow, is dropped: ValueError naming the value, which
        `dropped` gives until a later one is kept."""
        try:
            follow_state = _follow_state(message.follow_state)
            limits = BaseLimits.parse(message)
        except ValueError as e:
            self._dropped[peer] = str(e)
            raise
        self._newest[peer] = BaseReport(
            follow_state=follow_state,
            mode=message.mode,
            limits=limits,
            arrived_ns=self._elapsed_ns(),
        )
        self._dropped.pop(peer, None)

    def _elapsed_ns(self) -> int:
        """The time since the instance's clock was first read, each step back
        of the clock counted as no time passing."""
        now = self._now_ns()
        if self._read_ns is None:
            self._first_ns = now
        elif now < self._read_ns:
            self._back_ns += self._read_ns - now
        self._read_ns = now
        return now - self._first_ns + self._back_ns

    def mark(self) -> int:
        """Now on the elapsed instance clock (ns). Before the instance's clock
        gives its first reading, as peppygen's clock of a simulation before
        its first tick, now is that clock's start: no report arrives before
        that reading, so a time after the start counts at least as long after
        now on the shared clock."""
        try:
            return self._elapsed_ns()
        except RuntimeError:
            if self._read_ns is not None:
                raise
            return 0

    def newest(self, peer) -> BaseReport | None:
        """`peer`'s newest report while the base reports: None before its
        first, for no peer, and once more than `REPORT_SILENCE_NS` passed
        since it arrived."""
        report = self._newest.get(peer) if peer is not None else None
        if report is None:
            return None
        if self._elapsed_ns() - report.arrived_ns > REPORT_SILENCE_NS:
            return None
        return report

    def dropped(self, peer) -> str | None:
        """Why `peer`'s newest drive_states was dropped, None while it was
        kept."""
        return self._dropped.get(peer) if peer is not None else None


class NotedStops:
    """The robot stops of the headset, each noted on the elapsed instance
    clock of `BaseReports`: how many stops of a robot are in flight, and when
    the newest that was answered or refused ended. A refused stop counts: a
    backbone runs a stop it took, whatever its answer (robot_stop)."""

    def __init__(self, stops: Stops, reports: BaseReports) -> None:
        self._stops = stops
        self._reports = reports
        self._in_flight: Counter[str] = Counter()
        self._ended: dict[str, int] = {}

    async def stop(self, robot: str) -> str | None:
        self._in_flight[robot] += 1
        try:
            return await self._stops.stop(robot)
        finally:
            self._in_flight[robot] -= 1
            self._ended[robot] = self._reports.mark()

    def in_flight(self, robot: str) -> int:
        """How many stops of `robot` were sent and have neither an answer
        nor a refusal yet."""
        return self._in_flight[robot]

    def ended_at(self, robot: str) -> int | None:
        """When the newest stop of `robot` that was answered or refused
        ended (ns), None before its first."""
        return self._ended.get(robot)


@dataclass(frozen=True)
class Twist:
    """A velocity of the robot frame: x forward and y leftward (m/s), and the
    turn about +z, counterclockwise seen from above (rad/s)."""

    x: float = 0.0
    y: float = 0.0
    turn: float = 0.0

    @property
    def is_zero(self) -> bool:
        return self == STILL


STILL = Twist()


def twist_of(left: Thumbstick, right: Thumbstick, limits: BaseLimits) -> Twist:
    """The velocity the sticks command: the left stick's forward and leftward
    deflection along x and y, and the right stick's leftward deflection as
    the turn, each a fraction of the limit in its direction."""
    return Twist(
        x=limits.x.scaled(left.forward),
        y=limits.y.scaled(left.left),
        turn=limits.turn.scaled(right.left),
    )


def _stick(frame: XrFrame | None, hand: Hand) -> Thumbstick:
    """`hand`'s thumbstick in `frame`, at rest while it is not tracked."""
    sample = frame.hand(hand) if frame is not None else None
    return sample.thumbstick if sample is not None else RESTING


def _setpoint(topic_module, twist: Twist) -> bytes:
    """`twist` as a setpoint of drive_link in the robot frame at the latest
    time: no frame and no timestamp, as drive_link lets a leader give it."""
    return topic_module.build_message(
        None, None, [twist.x, twist.y, 0.0], [0.0, 0.0, twist.turn]
    )


# Why the base grip's squeeze requests nothing.
NO_BASE_PAIR = "no base pair"
NOT_REPORTING = "the base does not report"


def base_ready(pin: Hashable | None, report: BaseReport | None) -> Readiness:
    """The base grip's readiness: a base pin with a report recorded within
    0.5 s."""
    if pin is None:
        return NotReady(NO_BASE_PAIR)
    if report is None:
        return NotReady(NOT_REPORTING)
    return Ready()


def follows_now(
    report: BaseReport | None, *, stops_in_flight: int, stop_ended_ns: int | None
) -> bool:
    """Whether the base of `report` follows now: `report` is its newest
    report within `REPORT_SILENCE_NS` and gives following, no stop of its
    robot is in flight, and the report arrived more than `STOP_HOLD_NS`
    after the newest stop of the robot that was answered or refused."""
    if report is None or report.follow_state is not FollowState.FOLLOWING:
        return False
    if stops_in_flight:
        return False
    assert stop_ended_ns is not None, "an engaged base was taken with a stop"
    return report.follows_after(stop_ended_ns)


async def stream_base(
    node_runner,
    *,
    topic_module,
    driver: Driver,
    stops: NotedStops,
    reports: BaseReports,
    session: FrameSource,
    robots: PairedRobots,
    settings: Settings,
    token: CancellationToken,
    clock: Clock = REAL_CLOCK,
) -> None:
    """Drive the chosen robot's pair on the base slot from the thumbsticks
    while a grip of either hand is held, as the module states."""
    await run_stream(
        node_runner,
        topic_module,
        settings.tick_period_s,
        "base",
        token,
        base_tick(
            topic_module=topic_module,
            driver=driver,
            stops=stops,
            reports=reports,
            session=session,
            robots=robots,
            settings=settings,
            clock=clock,
        ),
        clock,
    )


def base_tick(
    *,
    topic_module,
    driver: Driver,
    stops: NotedStops,
    reports: BaseReports,
    session: FrameSource,
    robots: PairedRobots,
    settings: Settings,
    clock: Clock,
) -> Callable[[], tuple[object, bytes] | None]:
    """One tick of the base stream: the base grip's `Tick`, the target, the
    burst, its `Streamed`, then the setpoint to send."""
    stream = GripStream[Twist](Slot.BASE)

    def tick() -> tuple[object, bytes] | None:
        frame = fresh_frame(session, settings.stale_timeout_s, clock)
        _names, chosen = robots.standing()
        base = robots.pairs(topic_module)
        reported = {peer: reports.newest(peer) for peer in base.peers}
        stops_in_flight = stops.in_flight(chosen) if chosen is not None else 0
        stop_ended_ns = stops.ended_at(chosen) if chosen is not None else None
        now_s = clock.monotonic()
        peers, pins = slot_pins({Slot.BASE: base}, chosen)
        pin = pin_on(pins, Slot.BASE)
        readings = {hand: reading_of(hand_squeezing(frame, hand)) for hand in Hand}
        phase_pin = pin_on(driven_pins(driver.world.grips.base.phase), Slot.BASE)
        inputs = TickInputs(
            readings,
            chosen,
            peers,
            pins,
            base_ready(pin, reported.get(pin)),
            base_ready(phase_pin, reported.get(phase_pin)),
        )
        phase = driver.step(Tick(BASE, inputs)).grips.base.phase
        target = _target(phase, frame, reported, stops_in_flight, stop_ended_ns)
        streamed = stream.step(
            phase, target=target, held=STILL, peers=base.peers, now_s=now_s
        )
        driver.step(Streamed(BASE, streamed.burst_running))
        if streamed.send is None:
            return None
        pair, twist = streamed.send
        return pair, _setpoint(topic_module, twist)

    return tick


def _target(
    phase: Phase,
    frame: XrFrame | None,
    reported: Mapping[Hashable, BaseReport | None],
    stops_in_flight: int,
    stop_ended_ns: int | None,
) -> Twist | None:
    """The twist the engaged base streams this tick, None for none."""
    if not isinstance(phase, Engaged):
        return None
    report = reported.get(pin_on(phase.pins, Slot.BASE))
    if not follows_now(
        report, stops_in_flight=stops_in_flight, stop_ended_ns=stop_ended_ns
    ):
        return None
    left, right = _stick(frame, LINEAR_HAND), _stick(frame, TURN_HAND)
    twist = twist_of(left, right, report.limits)
    return None if twist.is_zero else twist
