"""The base stream: the thumbsticks drive the chosen robot's base over
drive_link while a grip of either hand is held.

A squeeze takes the robot as it does for an arm (`drive`): the lease, then
robot_stop's stop. The stream then sends nothing until a drive_states that
reports following arrives more than 0.5 s after the stop's answer, counted
on the instance's clock. The left thumbstick gives the velocity forward and
back along x and, on a base that drives sideways, along y; the right
thumbstick's left and right give the turn. A full deflection gives the limit
in its direction that the newest drive_states gives, as base_state's
describe_base gives it, and a partial one that fraction, so every setpoint
lies within the base's limits. Each setpoint is in the robot frame at the
latest time: it carries no frame and no timestamp.

The stream runs while the sticks give a velocity other than zero. When they
give zero, or the grip is released, it streams zero velocity for the release
burst, then nothing, and the grip holds the robot through that burst. A
drive_states that reports another follow_state ends the stream at once, and
the stream sends nothing until every stick rests. A change of robot sends the
burst to the base left behind; the new base engages on the next squeeze.

The times of drive_states count by their arrival on the instance's clock,
the clock of transforms, never by their timestamps; a step back of that
clock counts as no time passing.
"""

from __future__ import annotations

import math
from collections.abc import Callable
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
    gripping,
)
from xr_commander.drive import BASE, Driver, Stops
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
        """`peer`'s newest report, None before its first and for no peer."""
        return self._newest.get(peer) if peer is not None else None

    def fresh(self, peer) -> BaseReport | None:
        """`peer`'s newest report while the base reports: None once more
        than `REPORT_SILENCE_NS` passed since it arrived."""
        report = self.newest(peer)
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
    clock of `BaseReports`: when the newest stop of a robot that was answered
    or refused ended. A refused stop counts: a backbone runs a stop it took,
    whatever its answer (robot_stop)."""

    def __init__(self, stops: Stops, reports: BaseReports) -> None:
        self._stops = stops
        self._reports = reports
        self._ended: dict[str, int] = {}

    async def stop(self, robot: str) -> str | None:
        try:
            return await self._stops.stop(robot)
        finally:
            self._ended[robot] = self._reports.mark()

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


def _holds(report: BaseReport | None) -> bool:
    """Whether `report` gives a hold of the base."""
    return report is not None and report.follow_state is not FollowState.FOLLOWING


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
    grip = driver.base
    stream = GripStream[Twist](driver, BASE, settings.tick_period_s)

    def still_gripping(robot: str) -> Callable[[], bool]:
        def wanted() -> bool:
            """The take runs over ticks: the grips may be released, or
            another robot chosen, since the squeeze."""
            frame = fresh_frame(session, settings.stale_timeout_s, clock)
            return gripping(frame) is True and robots.chosen() == robot

        return wanted

    def held_still(driven) -> Twist | None:
        """The burst holds zero velocity; a hold ends the stream with no
        burst, as the follower drops each setpoint during it."""
        return None if _holds(reports.newest(driven)) else STILL

    def drive(peer, frame: XrFrame | None) -> Twist | None:
        """The velocity the engaged base streams this tick, None for none."""
        report = reports.newest(peer)
        if not grip.engaged or report is None:
            return None
        if not grip.followed:
            # Every hold of the headset starts with a stop of its robot.
            answered_ns = stops.ended_at(grip.robot)
            assert answered_ns is not None, f"{grip.robot} was taken with no stop"
            grip.followed = report.follows_after(answered_ns)
            if not grip.followed:
                return None
        twist = twist_of(
            _stick(frame, LINEAR_HAND), _stick(frame, TURN_HAND), report.limits
        )
        grip.wait_rest = (grip.wait_rest or _holds(report)) and not twist.is_zero
        return None if grip.wait_rest or twist.is_zero else twist

    async def build():
        frame = fresh_frame(session, settings.stale_timeout_s, clock)
        gripped = gripping(frame)
        chosen = robots.chosen()
        grip.observe(chosen, gripped)
        peer = robots.peer_of(topic_module, chosen)
        if reports.newest(peer) is not None and grip.may_engage(gripped):
            await driver.engage(chosen, BASE, still_gripping(chosen))
        sent = stream.step(peer, drive(peer, frame), held_still, clock.monotonic())
        if sent is None:
            return None
        driven, twist = sent
        # Nothing goes out while the base reports a hold.
        if _holds(reports.newest(driven)):
            return None
        return driven, _setpoint(topic_module, twist)

    await run_stream(
        node_runner,
        topic_module,
        settings.tick_period_s,
        "base",
        token,
        build,
        clock,
    )
