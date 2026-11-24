"""The thumbsticks drive the base on the wire: the base stream against the
generated mocks of the daemon target `fleet`, of the robot's `robot_stop`,
and of the follower on its `drive_link` pair, with the
controllers simulated as the page sends them, on an injected clock.

The stream ticks on a manual clock and the instance's clock is the harness's
simulated domain, so nothing depends on the host's speed. The clock moves
only once the stream waits on it, so one tick is one step of the stream, and
a driving stream sends one setpoint per tick, which each test reads in turn.
Where a test proves that nothing was streamed, it streams a distinct setpoint
next: the stream sends in order on one wire, so the first setpoint read
being that one proves that nothing went out before it.
"""

import asyncio
import contextlib
from types import SimpleNamespace

import peppygen.clock
import pytest
from peppygen.consumed_services.robot_stop import stop
from peppygen.consumed_actions.postures import move_to_home
from peppygen.paired_topics.base import drive_setpoints, drive_states

from tests.helpers import (
    CALL_S,
    HALTED,
    HELD_TICKS,
    ROBOT,
    SETTINGS,
    WIRE_TICKS,
    ManualClock,
    ScriptedHeadset,
    StopsOnCue,
    TickedRig,
    backbone_answers_stop,
    boot_clocked,
    controller,
    daemon_answers_check_move,
    daemon_answers_release,
    daemon_answers_take,
    driver_on_the_wire,
    eventually,
    fleet_view,
    following_fleet,
    posture_done,
    robots_of,
    running,
    settle,
    wire_stops,
)
from xr_commander import base, panel, postures, publish
from xr_commander.alerts import ActiveAlerts
from xr_commander.base import (
    AxisLimits,
    BaseLimits,
    BaseReport,
    BaseReports,
    FollowState,
    Twist,
    twist_of,
)
from xr_commander.devices import Thumbstick, parse_hands, primary_pressed
from xr_commander.fleet import LEASE_NAME
from xr_commander.motor_health import MotorHealthReports

# The instance's clock when each test starts (ns).
START_NS = 1_000_000_000_000
# When the base's odometry started, on the wall clock of its machine (s).
ODOMETRY_START_S = 500.0
# A full deflection, and half a deflection past joy's 0.05 rest band:
# 0.05 + 0.5 * 0.95.
FULL = 1.0
HALF = 0.525


def limits_given(*, x=(-0.4, 0.8), y=(0.0, 0.0), turn=(-1.5, 1.0)):
    """The limits a drive_states gives for a base with these ranges (m/s,
    rad/s): a differential base by default, whose backward and clockwise
    limits differ from its forward and counterclockwise ones."""
    drives_y = y != (0.0, 0.0)
    return SimpleNamespace(
        linear_limits=drive_states.MessageLinearLimits(
            min_velocity=[x[0], y[0]],
            max_velocity=[x[1], y[1]],
            max_acceleration=[1.0, 1.0 if drives_y else 0.0],
            max_deceleration=[2.0, 2.0 if drives_y else 0.0],
            deadband=[0.0, 0.0],
            rest_velocity=[0.01, 0.01 if drives_y else 0.0],
        ),
        angular_limits=drive_states.MessageAngularLimits(
            min_rate=turn[0],
            max_rate=turn[1],
            max_acceleration=2.0,
            max_deceleration=3.0,
            deadband=0.0,
            rest_rate=0.02,
        ),
    )


DIFFERENTIAL = limits_given()
HOLONOMIC = limits_given(y=(-0.3, 0.6))


def stick_axes(forward=0.0, left=0.0) -> list[float]:
    """The xr-standard axes of a controller whose thumbstick is deflected
    `forward` and `left`: the Gamepad API gives -1 forward and left."""
    return [0.0, 0.0, -left, -forward]


async def instance_clock_at(h, time_ns: int) -> None:
    """Move the harness's simulated domain to `time_ns` and wait until the
    node's clock reads it."""
    await h.clock.tick(time_ns)

    def reads() -> bool:
        try:
            return peppygen.clock.now_ns() == time_ns
        except RuntimeError:
            return False

    await eventually(reads, message=f"the instance's clock at {time_ns}")


def setpoint_of(message) -> tuple[float, float, float]:
    """A setpoint's x, y and turn."""
    return (
        message.linear_velocity[0],
        message.linear_velocity[1],
        message.angular_velocity[2],
    )


class Rig(TickedRig):
    """The headset's base stream toward the bases of the fleet listing, with
    the daemon, each robot's robot_stop and each base's follower played by
    the mocks."""

    def __init__(self, h, clock: ManualClock, robots: list[str]) -> None:
        super().__init__(clock)
        self.h = h
        self.fleet = following_fleet(*robots)
        self.robots = robots_of(h)
        self.reports = BaseReports()
        self.noted = base.NotedStops(wire_stops(h, self.robots), self.reports)
        self.driver = driver_on_the_wire(h, self.fleet, self.noted, clock)
        self.state = self.driver.base
        self.session = ScriptedHeadset(clock)
        self.daemon = h.mocks.deps.fleet
        # The harness starts the mocks of each slot in the order it is given.
        self.followers = dict(zip(robots, h.mocks.pairings.base, strict=True))
        self.stops = dict(zip(robots, h.mocks.deps.robot_stop, strict=True))
        # How far off the instance's clock the test stamps each report (s),
        # and the limits the reports give.
        self.stamp_offset_s = 0.0
        self.limits = DIFFERENTIAL

    def base_stream(self, token):
        return base.stream_base(
            self.h.node_runner,
            topic_module=drive_setpoints,
            driver=self.driver,
            stops=self.noted,
            reports=self.reports,
            session=self.session,
            robots=self.robots,
            settings=SETTINGS,
            token=token,
            clock=self.clock,
        )

    def drain(self, token):
        return publish.drain(
            self.h.node_runner,
            drive_states,
            "base drive_states",
            self.reports.record,
            token,
        )

    def wire(self, robot=ROBOT):
        """The setpoints `robot`'s follower receives."""
        return self.followers[robot].drive_setpoints

    def hold(self, *, squeeze=False, forward=0.0, sideways=0.0, turn=0.0, a=False):
        """The controllers as the page sends them: the left grip, the left
        thumbstick deflected `forward` and `sideways` (leftward), the right
        thumbstick deflected `turn` (leftward), and the right A button."""
        self.session.hands = parse_hands(
            [
                controller(
                    "left", squeezing=squeeze, axes=stick_axes(forward, sideways)
                ),
                controller("right", primary=a, axes=stick_axes(left=turn)),
            ]
        )

    async def send_report(
        self, follow_state="following", *, robot=ROBOT, mode="standing", limits=None
    ):
        """`robot`'s base sends one drive_states, stamped `stamp_offset_s` off
        the instance's clock, giving `limits`, else the rig's."""
        given = self.limits if limits is None else limits
        stamp_s = peppygen.clock.now_ns() / 1e9 + self.stamp_offset_s
        await self.followers[robot].drive_states.publish(
            drive_states.Message(
                timestamp=stamp_s,
                odometry_start_timestamp=ODOMETRY_START_S,
                frame="odom",
                child_frame="base_footprint",
                pose=drive_states.MessagePose(
                    position=[0.0, 0.0, 0.0], orientation=[0.0, 0.0, 0.0, 1.0]
                ),
                twist=drive_states.MessageTwist(
                    linear_velocity=[0.0, 0.0, 0.0], angular_velocity=[0.0, 0.0, 0.0]
                ),
                mode=mode,
                parts_within_footprint=True,
                linear_limits=given.linear_limits,
                angular_limits=given.angular_limits,
                follow_state=follow_state,
            )
        )

    async def report(
        self, follow_state="following", *, robot=ROBOT, mode="standing", limits=None
    ):
        """`robot`'s base reports once; wait until the headset keeps the
        report."""
        peer = self.peer(robot)
        before = self.reports.newest(peer)
        await self.send_report(follow_state, robot=robot, mode=mode, limits=limits)
        await eventually(
            lambda: self.reports.newest(peer) is not before,
            message=f"a report of {robot}",
        )

    def peer(self, robot=ROBOT):
        return next(
            member.info
            for member in drive_setpoints.peers(self.h.node_runner)
            if member.info.producer.instance_id == robot
        )

    async def answer_take(self, refusal=None, robot=ROBOT):
        await daemon_answers_take(self.daemon, robot, refusal)

    async def answer_stop(self, robot=ROBOT):
        await backbone_answers_stop(self.stops[robot].robot_stop_stop, ["base"])

    async def answer_release(self, robot=ROBOT):
        await daemon_answers_release(self.daemon, robot)

    async def clock_by(self, span_ns: int) -> None:
        """Move the instance's clock on by `span_ns`."""
        await instance_clock_at(self.h, peppygen.clock.now_ns() + span_ns)

    async def squeeze_and_take(self, robot=ROBOT):
        """A squeeze with the sticks at rest that takes `robot` and engages
        the base: the take, the stop, a report of the stop's hold, and a
        report of following more than 0.5 s after the stop's answer."""
        self.hold(squeeze=True)
        await self.report(robot=robot)
        await self.tick()
        await self.answer_take(robot=robot)
        await self.answer_stop(robot)
        await self.ticks_until(lambda: self.state.engaged)
        await self.report("robot_stop", robot=robot)
        await self.clock_by(base.STOP_HOLD_NS + 1)
        await self.report(robot=robot)
        await self.ticks_until(lambda: self.state.followed)

    async def burst(self, robot=ROBOT) -> list[tuple[float, float, float]]:
        """Each setpoint of the release burst toward `robot`, one per tick
        while the grips hold the robot for the base."""
        streamed = []
        for _ in range(WIRE_TICKS):
            await self.tick()
            if self.state.robot != robot:
                return streamed
            message = await asyncio.wait_for(self.wire(robot).next(), CALL_S)
            streamed.append(setpoint_of(message))
        raise AssertionError("the burst never ended")

    async def next_setpoint(self, robot=ROBOT):
        """One tick, then the next setpoint on `robot`'s wire: for a stream
        that sends at every tick."""
        await self.tick()
        return await asyncio.wait_for(self.wire(robot).next(), CALL_S)

    def row(self) -> str:
        """What the panel's base row says now."""
        source = panel.BaseSource(
            state=self.state,
            reports=self.reports,
            peer=lambda: self.robots.peer_for(drive_setpoints),
        )
        state = panel.snapshot(
            self.session,
            [],
            SETTINGS.stale_timeout_s,
            ActiveAlerts(bound_now=lambda: []),
            MotorHealthReports(),
            base=source,
            clock=self.clock,
        )
        return state.base.state


@contextlib.asynccontextmanager
async def rig(*robots: str):
    """The rig on `robots`, each a base outside any copy with its own
    robot_stop, with the drive_states drain and the base stream running."""
    robots = list(robots or (ROBOT,))
    async with boot_clocked(
        clock="consumer",
        robot_stop_instance_ids=robots,
        base_members=dict.fromkeys(robots),
    ) as h:
        await instance_clock_at(h, START_NS)
        the_rig = Rig(h, ManualClock(), robots)
        async with running(the_rig.drain), running(the_rig.base_stream) as tasks:
            the_rig.loops = tasks
            for follower in the_rig.followers.values():
                assert await follower.drive_states.wait_for_subscriber(CALL_S)
            yield the_rig


async def test_a_stop_before_the_instance_clock_ticks_takes_holds_and_releases():
    """A posture press before the instance's clock gives its first reading
    takes the robot, stops it, holds it for the move and releases its lease,
    as it does with no base slot. A base take once the clock runs follows
    by the take rule."""
    async with boot_clocked(
        clock="consumer",
        robot_stop_instance_ids=[ROBOT],
        postures_instance_ids=[ROBOT],
        base_members={ROBOT: None},
    ) as h:
        r = Rig(h, ManualClock(), [ROBOT])
        status = postures.PostureStatus(r.clock)

        def posture_button(token):
            return postures.run_posture_button(
                h.node_runner,
                action_module=move_to_home,
                pressed=primary_pressed,
                session=r.session,
                robots=r.robots,
                driver=r.driver,
                status=status,
                settings=SETTINGS,
                token=token,
                clock=r.clock,
            )

        async with running(r.drain), running(r.base_stream, posture_button) as tasks:
            r.loops = tasks
            with pytest.raises(RuntimeError):
                peppygen.clock.now_ns()
            r.hold()
            await r.tick()
            r.hold(a=True)
            await r.tick()
            await r.answer_take()
            await r.answer_stop()
            await daemon_answers_check_move(r.daemon, ROBOT)
            pending = await h.mocks.deps.postures[0].move_to_home.next_goal(CALL_S)
            goal = await pending.accept()
            assert r.driver.holds(ROBOT)
            await goal.complete(posture_done(move_to_home))
            await r.answer_release()
            await eventually(lambda: status.state() == "move to home completed")
            assert not r.driver.holds(ROBOT)
            # The clock runs: a base take waits for the stop's hold, then
            # drives.
            await instance_clock_at(h, START_NS)
            assert await r.followers[ROBOT].drive_states.wait_for_subscriber(CALL_S)
            r.hold(a=False)
            await r.squeeze_and_take()
            r.hold(squeeze=True, forward=HALF)
            assert setpoint_of(await r.next_setpoint()) == pytest.approx(
                (0.4, 0.0, 0.0)
            )


# The limits and the sticks, without a wire.


def limits_of(given=DIFFERENTIAL) -> BaseLimits:
    return BaseLimits.parse(given)


def test_each_direction_of_an_axis_scales_by_its_own_limit():
    axis = AxisLimits(-0.4, 0.8)
    assert axis.scaled(1.0) == 0.8
    assert axis.scaled(0.5) == pytest.approx(0.4)
    assert axis.scaled(-1.0) == -0.4
    assert axis.scaled(-0.5) == pytest.approx(-0.2)
    assert axis.scaled(0.0) == 0.0


def test_the_left_stick_gives_x_and_y_and_the_right_stick_gives_the_turn():
    left = Thumbstick(forward=0.5, left=-0.5)
    right = Thumbstick(forward=1.0, left=0.5)
    assert twist_of(left, right, limits_of(HOLONOMIC)) == Twist(
        x=pytest.approx(0.4), y=pytest.approx(-0.15), turn=pytest.approx(0.5)
    )
    # A differential base takes 0 along y, and the right stick's forward
    # deflection drives nothing.
    assert twist_of(Thumbstick(left=1.0), right, limits_of()) == Twist(
        turn=pytest.approx(0.5)
    )
    assert twist_of(Thumbstick(left=-1.0), Thumbstick(forward=1.0), limits_of()).is_zero


# Limits outside base_state's, and how a dropped drive_states names each.
OUTSIDE_THE_CONTRACT = [
    (limits_given(x=(0.2, 0.8)), "min_velocity along x 0.2 is not 0 or below"),
    (limits_given(x=(-0.8, -0.2)), "max_velocity along x -0.2 is not 0 or above"),
    (limits_given(y=(0.0, -0.1)), "max_velocity along y -0.1 is not 0 or above"),
    (limits_given(turn=(float("-inf"), 1.0)), "min_rate -inf is not 0 or below"),
    (limits_given(turn=(-1.0, float("nan"))), "max_rate nan is not 0 or above"),
]


@pytest.mark.parametrize(("limits", "message"), OUTSIDE_THE_CONTRACT)
def test_limits_that_base_state_does_not_allow_are_refused_by_name(limits, message):
    with pytest.raises(ValueError) as refused:
        BaseLimits.parse(limits)
    assert str(refused.value) == message


def states(follow_state="following", limits=DIFFERENTIAL):
    """A drive_states as the headset reads it."""
    return SimpleNamespace(
        follow_state=follow_state,
        mode="standing",
        linear_limits=limits.linear_limits,
        angular_limits=limits.angular_limits,
    )


def test_a_base_reports_by_arrival_on_the_instances_clock_and_counts_its_reports():
    now = {"ns": 7_000}
    reports = BaseReports(lambda: now["ns"])
    assert reports.newest("peer") is None and reports.fresh("peer") is None
    assert reports.newest(None) is None
    reports.record("peer", states("robot_stop"))
    now["ns"] += 100
    reports.record("peer", states("robot_stop", HOLONOMIC))
    newest = reports.newest("peer")
    assert newest.follow_state is FollowState.ROBOT_STOP
    # Arrival counts from the clock's first reading.
    assert (newest.arrived_ns, newest.limits) == (100, limits_of(HOLONOMIC))
    now["ns"] += base.REPORT_SILENCE_NS
    assert reports.fresh("peer") is newest
    now["ns"] += 1
    assert reports.fresh("peer") is None
    assert reports.newest("peer") is newest


@pytest.mark.parametrize(
    "dropped",
    [states("resting"), states(limits=OUTSIDE_THE_CONTRACT[0][0])],
)
def test_a_drive_states_that_drive_link_does_not_allow_is_dropped_and_named(dropped):
    reports = BaseReports(lambda: 0)
    reports.record("peer", states())
    kept = reports.newest("peer")
    with pytest.raises(ValueError) as refused:
        reports.record("peer", dropped)
    assert reports.newest("peer") is kept
    assert reports.dropped("peer") == str(refused.value)
    reports.record("peer", states())
    assert reports.dropped("peer") is None


async def test_a_stop_is_noted_at_its_answer_or_its_refusal():
    now = {"ns": 0}
    reports = BaseReports(lambda: now["ns"])
    assert reports.mark() == 0
    stops = StopsOnCue()
    noted = base.NotedStops(stops, reports)
    first = asyncio.create_task(noted.stop("alpha"))
    second = asyncio.create_task(noted.stop("alpha"))
    await settle()
    assert noted.ended_at("alpha") is None
    now["ns"] = 300
    stops.answer(0)
    stops.answer(1)
    assert await first is None
    assert await second is None
    assert noted.ended_at("alpha") == 300
    # A refused stop counts too: the backbone runs a stop it took.
    now["ns"] = 700
    refused = asyncio.create_task(noted.stop("alpha"))
    await settle()
    stops.answer(2, "alpha did not answer")
    assert await refused == "alpha did not answer"
    assert noted.ended_at("alpha") == 700
    assert noted.ended_at("bravo") is None


async def test_a_stop_cut_short_is_noted_at_its_end():
    now = {"ns": 0}
    reports = BaseReports(lambda: now["ns"])
    assert reports.mark() == 0
    noted = base.NotedStops(StopsOnCue(), reports)
    stopping = asyncio.create_task(noted.stop("alpha"))
    await settle()
    now["ns"] = 400
    stopping.cancel()
    with pytest.raises(asyncio.CancelledError):
        await stopping
    assert noted.ended_at("alpha") == 400


def test_a_following_more_than_half_a_second_after_the_answer_follows():
    reported = BaseReport(FollowState.FOLLOWING, "standing", limits_of(), 0)
    assert not reported.follows_after(-base.STOP_HOLD_NS)
    assert reported.follows_after(-base.STOP_HOLD_NS - 1)
    held = BaseReport(FollowState.ROBOT_STOP, "standing", limits_of(), 0)
    assert not held.follows_after(-base.STOP_HOLD_NS - 1)


def test_a_follow_state_outside_drive_link_is_named():
    with pytest.raises(ValueError) as refused:
        BaseReports(lambda: 0).record("peer", states("resting"))
    assert str(refused.value) == "follow_state 'resting' is none of drive_link's"


def test_the_limits_the_follower_owns_are_not_judged():
    # Only each axis's most negative and most positive velocity is read.
    owned = limits_given()
    owned.linear_limits.max_acceleration = [-1.0, float("nan")]
    owned.angular_limits.deadband = -1.0
    reports = BaseReports(lambda: 0)
    reports.record("peer", states(limits=owned))
    assert reports.newest("peer").limits == limits_of()
    assert reports.dropped("peer") is None


def test_a_stop_before_the_clocks_first_reading_stands_at_its_start():
    reading = {"ns": None}

    def now_ns() -> int:
        if reading["ns"] is None:
            raise RuntimeError("peppygen.clock: no tick of the domain yet")
        return reading["ns"]

    reports = BaseReports(now_ns)
    assert reports.mark() == 0
    reading["ns"] = 5_000_000_000
    reports.record("peer", states())
    assert reports.newest("peer").arrived_ns == 0
    reading["ns"] += 1
    assert reports.mark() == 1
    # A clock that read once and then fails is a fault, never a start.
    reading["ns"] = None
    with pytest.raises(RuntimeError):
        reports.mark()


def test_a_clock_that_goes_back_counts_as_no_time_passing():
    now = {"ns": 10_000_000_000}
    reports = BaseReports(lambda: now["ns"])
    reports.record("peer", states())
    now["ns"] += 400_000_000
    assert reports.fresh("peer") is not None
    # The instance's clock steps back 8 s: the report keeps the 0.4 s it had.
    now["ns"] = 2_000_000_000
    assert reports.fresh("peer") is not None
    now["ns"] += 100_000_000
    assert reports.fresh("peer") is not None
    now["ns"] += 1
    assert reports.fresh("peer") is None
    # A report after the step back arrives 0.5 s and 1 ns after the first on
    # the clock that counts the step as no time and starts at its first
    # reading, and the base reports again.
    reports.record("peer", states())
    report = reports.newest("peer")
    assert report.arrived_ns == 500_000_001
    assert reports.fresh("peer") is report
    now["ns"] += base.REPORT_SILENCE_NS
    assert reports.fresh("peer") is report
    now["ns"] += 1
    assert reports.fresh("peer") is None


# The stream on the wire.


async def test_the_first_setpoint_comes_after_the_lease_the_stop_and_a_late_following():
    async with rig() as r:
        r.hold(squeeze=True, forward=FULL)
        # The base follows before the squeeze; that report does not count.
        await r.report()
        await r.tick()
        await r.answer_take()
        request, responder = await r.stops[ROBOT].robot_stop_stop.next_request(CALL_S)
        assert request.reason == LEASE_NAME
        # The stream waits on the stop's answer.
        await settle()
        assert not r.state.engaged
        await responder.respond(stop.ResponseData(stopped=["base"]))
        await r.ticks_until(lambda: r.state.engaged)
        assert r.row() == "waiting for the base"
        # The stop holds the base: nothing streams, the stick held or not,
        # and a report of the hold past 0.5 s starts nothing either.
        await r.tick(HELD_TICKS)
        await r.report("robot_stop")
        await r.clock_by(base.STOP_HOLD_NS + 1)
        await r.report("robot_stop")
        await r.tick(HELD_TICKS)
        assert r.row() == (
            "a robot stop holds the base: let the sticks rest, then drive again"
        )
        # A following more than 0.5 s after the answer starts the stream, at
        # the stick held now.
        r.hold(squeeze=True, forward=HALF)
        await r.report()
        first = await r.next_setpoint()
        assert setpoint_of(first) == pytest.approx((0.4, 0.0, 0.0))
        assert r.row() == "driving"


@pytest.mark.parametrize(
    ("answered_ns", "arrives_ns", "drives"),
    [
        (0, base.STOP_HOLD_NS, False),
        (0, base.STOP_HOLD_NS + 1, True),
        # A following the base published before the stop, arriving 0.2 s
        # after the answer.
        (400_000_000, 600_000_000, False),
        (400_000_000, 400_000_000 + base.STOP_HOLD_NS, False),
        (400_000_000, 400_000_001 + base.STOP_HOLD_NS, True),
    ],
)
async def test_a_following_more_than_half_a_second_after_the_answer_starts_the_grip(
    answered_ns, arrives_ns, drives
):
    """The base's last report arrives at the start, the stop is answered
    `answered_ns` later, and a following arrives `arrives_ns` after the
    start: it starts the stream at the held stick only when it arrives more
    than 0.5 s of the instance's clock after the answer."""
    async with rig() as r:
        r.hold(squeeze=True, forward=HALF)
        await r.report()
        await r.tick()
        await r.answer_take()
        await instance_clock_at(r.h, START_NS + answered_ns)
        await r.answer_stop()
        await r.ticks_until(lambda: r.state.engaged)
        await instance_clock_at(r.h, START_NS + arrives_ns)
        await r.report()
        await r.tick()
        assert r.state.followed is drives
        if drives:
            setpoint = await asyncio.wait_for(r.wire().next(), CALL_S)
            assert setpoint_of(setpoint) == pytest.approx((0.4, 0.0, 0.0))
            return
        await r.tick(HELD_TICKS)
        # Nothing went out: the first setpoint is the stick held at the next
        # following past 0.5 s.
        await r.clock_by(base.STOP_HOLD_NS + 1)
        r.hold(squeeze=True, forward=FULL)
        await r.report()
        assert setpoint_of(await r.next_setpoint()) == pytest.approx((0.8, 0.0, 0.0))


async def test_hold_reports_before_the_answer_then_a_late_following_starts_the_grip():
    """Every report of the stop's hold arrives before its answer, and then
    the base reports following alone: the held grip drives at the first
    following that arrives more than 0.5 s after the answer."""
    async with rig() as r:
        r.hold(squeeze=True, forward=HALF)
        await r.report()
        await r.tick()
        await r.answer_take()
        await r.report("robot_stop")
        await r.answer_stop()
        await r.ticks_until(lambda: r.state.engaged)
        await r.report()
        await r.tick(HELD_TICKS)
        await r.clock_by(base.STOP_HOLD_NS)
        await r.report()
        await r.tick(HELD_TICKS)
        assert not r.state.followed
        await r.clock_by(1)
        await r.report()
        assert setpoint_of(await r.next_setpoint()) == pytest.approx((0.4, 0.0, 0.0))


async def test_each_take_waits_past_the_answer_of_its_own_stop():
    async with rig() as r:
        await r.squeeze_and_take()
        r.hold(squeeze=True, forward=HALF)
        await r.next_setpoint()
        r.hold(forward=HALF)
        await r.burst()
        await r.answer_release()
        # A new squeeze, the stick held: a following just after the new
        # stop's answer starts nothing.
        r.hold(squeeze=True, forward=HALF)
        await r.tick()
        await r.answer_take()
        await r.answer_stop()
        await r.ticks_until(lambda: r.state.engaged)
        await r.report()
        await r.tick(HELD_TICKS)
        await r.clock_by(base.STOP_HOLD_NS + 1)
        r.hold(squeeze=True, forward=FULL)
        await r.report()
        assert setpoint_of(await r.next_setpoint()) == pytest.approx((0.8, 0.0, 0.0))


async def test_half_a_deflection_gives_half_the_limit_forward_and_back_each_by_its_own():
    async with rig() as r:
        await r.squeeze_and_take()
        r.hold(squeeze=True, forward=HALF, turn=HALF)
        assert setpoint_of(await r.next_setpoint()) == pytest.approx((0.4, 0.0, 0.5))
        r.hold(squeeze=True, forward=-HALF, turn=-HALF)
        assert setpoint_of(await r.next_setpoint()) == pytest.approx(
            (-0.2, 0.0, -0.75)
        )
        r.hold(squeeze=True, forward=FULL, turn=-FULL)
        assert setpoint_of(await r.next_setpoint()) == pytest.approx((0.8, 0.0, -1.5))


async def test_sideways_gives_a_y_of_0_on_a_base_whose_y_range_is_0_to_0():
    async with rig() as r:
        await r.squeeze_and_take()
        # Sideways alone commands nothing there: it streams nothing.
        r.hold(squeeze=True, sideways=FULL)
        await r.tick(HELD_TICKS)
        r.hold(squeeze=True, forward=HALF, sideways=FULL)
        assert setpoint_of(await r.next_setpoint()) == pytest.approx((0.4, 0.0, 0.0))
        r.hold(squeeze=True, forward=FULL, sideways=-FULL)
        assert setpoint_of(await r.next_setpoint()) == pytest.approx((0.8, 0.0, 0.0))


async def test_sideways_gives_its_fraction_of_each_limit_on_a_base_that_drives_sideways():
    async with rig() as r:
        r.limits = HOLONOMIC
        await r.squeeze_and_take()
        r.hold(squeeze=True, sideways=HALF)
        assert setpoint_of(await r.next_setpoint()) == pytest.approx((0.0, 0.3, 0.0))
        r.hold(squeeze=True, sideways=-HALF)
        assert setpoint_of(await r.next_setpoint()) == pytest.approx(
            (0.0, -0.15, 0.0)
        )


async def test_a_deflection_within_the_rest_band_streams_nothing():
    async with rig() as r:
        r.limits = HOLONOMIC
        await r.squeeze_and_take()
        r.hold(squeeze=True, forward=0.05, sideways=-0.05, turn=0.04)
        await r.tick(HELD_TICKS)
        assert r.row() == "driving"
        r.hold(squeeze=True, forward=HALF)
        assert setpoint_of(await r.next_setpoint()) == pytest.approx((0.4, 0.0, 0.0))


async def test_a_release_streams_zero_velocity_for_the_burst_then_nothing():
    async with rig() as r:
        await r.squeeze_and_take()
        r.hold(squeeze=True, forward=HALF)
        await r.next_setpoint()
        r.hold(forward=HALF)
        released_at = r.clock.monotonic()
        burst = await r.burst()
        # The grip held the robot through the burst, and releases it after.
        assert len(burst) >= publish.RELEASE_HOLD_MESSAGES
        assert set(burst) == {(0.0, 0.0, 0.0)}
        assert r.clock.monotonic() - released_at >= publish.RELEASE_HOLD_S
        assert not r.driver.holds(ROBOT)
        await r.answer_release()
        await r.tick(HELD_TICKS)
        await r.squeeze_and_take()
        r.hold(squeeze=True, forward=FULL)
        assert setpoint_of(await r.next_setpoint()) == pytest.approx((0.8, 0.0, 0.0))


async def test_a_refused_take_streams_nothing_and_the_row_says_why():
    async with rig() as r:
        r.hold(squeeze=True, forward=HALF)
        await r.report()
        await r.tick()
        await r.answer_take(
            refusal="robot `backbone_inst` is leased by `director_inst`"
        )
        await r.ticks_until(lambda: r.state.note.text() != "")
        assert r.row() == (
            "squeeze refused: robot `backbone_inst` is leased by `director_inst`"
        )
        # The held grip asks nothing again until it is released.
        await r.tick(HELD_TICKS)
        assert r.state.wait_release and len(r.daemon.take.captured()) == 1
        assert r.stops[ROBOT].robot_stop_stop.captured() == []
        r.hold()
        await r.ticks_until(lambda: not r.state.wait_release)
        await r.squeeze_and_take()
        r.hold(squeeze=True, forward=FULL)
        assert setpoint_of(await r.next_setpoint()) == pytest.approx((0.8, 0.0, 0.0))


@pytest.mark.parametrize(
    ("follow_state", "mode"),
    [
        ("base_mode", "damped"),
        ("base_silence", "standing"),
        ("robot_stop", "standing"),
        ("follower_goal", "moving"),
    ],
)
async def test_each_hold_the_base_reports_ends_the_stream_until_every_stick_rests(
    follow_state, mode
):
    async with rig() as r:
        await r.squeeze_and_take()
        r.hold(squeeze=True, forward=HALF)
        assert setpoint_of(await r.next_setpoint()) == pytest.approx((0.4, 0.0, 0.0))
        await r.report(follow_state, mode=mode)
        await r.tick(HELD_TICKS)
        assert r.row() == panel._HOLD_TEXTS[FollowState(follow_state)].format(
            mode=mode
        )
        # The base follows again: the stick held through the hold drives
        # nothing until every stick rests.
        await r.report()
        await r.tick(HELD_TICKS)
        assert r.row() == "let the sticks rest"
        r.hold(squeeze=True)
        await r.tick()
        r.hold(squeeze=True, forward=FULL)
        # No burst and no stick held through the hold went out before.
        assert setpoint_of(await r.next_setpoint()) == pytest.approx((0.8, 0.0, 0.0))


async def test_a_hold_reported_in_the_release_burst_ends_the_burst():
    async with rig() as r:
        await r.squeeze_and_take()
        r.hold(squeeze=True, forward=HALF)
        await r.next_setpoint()
        r.hold(squeeze=True)
        assert setpoint_of(await r.next_setpoint()) == (0.0, 0.0, 0.0)
        await r.report("robot_stop")
        await r.tick(HELD_TICKS)
        await r.report()
        r.hold(squeeze=True, forward=FULL)
        assert setpoint_of(await r.next_setpoint()) == pytest.approx((0.8, 0.0, 0.0))


async def a_drive_stamped(stamp_offset_s: float) -> list:
    """A take, a drive through a hold, the instance's clock past the bound
    of silence, and a release, with each report stamped `stamp_offset_s`
    off the instance's clock: every setpoint and every row, in order."""
    seen = []
    async with rig() as r:
        r.stamp_offset_s = stamp_offset_s
        await r.squeeze_and_take()
        r.hold(squeeze=True, forward=HALF)
        seen += [setpoint_of(await r.next_setpoint()) for _ in range(HELD_TICKS)]
        await r.report("robot_stop")
        await r.tick()
        seen.append(r.row())
        await r.report()
        r.hold(squeeze=True)
        await r.tick()
        r.hold(squeeze=True, turn=HALF)
        seen.append(setpoint_of(await r.next_setpoint()))
        # 0.5 s of the instance's clock since the last report: it reports.
        await r.clock_by(base.REPORT_SILENCE_NS)
        seen.append(r.row())
        seen.append(setpoint_of(await r.next_setpoint()))
        await r.clock_by(1)
        seen.append(r.row())
        seen.append(setpoint_of(await r.next_setpoint()))
        r.hold()
        seen.append(await r.burst())
        await r.answer_release()
    return seen


@pytest.mark.parametrize("stamp_offset_s", [-3600.0, 3600.0])
async def test_reports_stamped_behind_or_ahead_of_the_instances_clock_change_nothing(
    stamp_offset_s,
):
    on_time = await a_drive_stamped(0.0)
    assert on_time[HELD_TICKS : HELD_TICKS + 4] == [
        "a robot stop holds the base: let the sticks rest, then drive again",
        pytest.approx((0.0, 0.0, 0.5)),
        "driving",
        pytest.approx((0.0, 0.0, 0.5)),
    ]
    assert on_time[HELD_TICKS + 4] == "the base does not report: check its base node"
    assert await a_drive_stamped(stamp_offset_s) == on_time


async def test_new_limits_in_a_drive_states_change_the_scale_of_the_next_setpoint():
    async with rig() as r:
        await r.squeeze_and_take()
        r.hold(squeeze=True, forward=HALF, sideways=HALF)
        assert setpoint_of(await r.next_setpoint()) == pytest.approx((0.4, 0.0, 0.0))
        # The base gives other limits, now with a y range: the next setpoint
        # scales to them, sideways included.
        await r.report(limits=limits_given(x=(-0.8, 1.6), y=(-0.3, 0.6)))
        assert setpoint_of(await r.next_setpoint()) == pytest.approx((0.8, 0.3, 0.0))


async def test_a_drive_states_outside_the_contract_is_dropped_logged_once_and_shown(
    capsys,
):
    async with rig() as r:
        outside, why = OUTSIDE_THE_CONTRACT[0]
        r.hold(squeeze=True, forward=FULL)
        await r.send_report(limits=outside)
        await eventually(lambda: r.reports.dropped(r.peer()) == why)
        await r.tick(HELD_TICKS)
        # Dropped, the base has not reported: the squeeze takes nothing.
        assert r.row() == f"drive_states unreadable ({why}): check its base node"
        assert r.daemon.take.captured() == []
        await r.send_report("resting")
        await eventually(
            lambda: r.reports.dropped(r.peer())
            == "follow_state 'resting' is none of drive_link's"
        )
        assert capsys.readouterr().out.count("base drive_states unusable") == 1
        # The base's next drive_states is kept, and the held grip takes it.
        await r.report()
        assert r.reports.dropped(r.peer()) is None
        await r.tick()
        await r.answer_take()
        await r.answer_stop()
        await r.ticks_until(lambda: r.state.engaged)


async def test_frames_that_go_stale_mid_drive_stream_the_burst_then_nothing():
    async with rig() as r:
        await r.squeeze_and_take()
        r.hold(squeeze=True, forward=HALF)
        await r.next_setpoint()
        # The headset's frames stop: the grip says nothing, and the base lets
        # go as at a release.
        r.session.age_s = SETTINGS.stale_timeout_s + 1.0
        burst = await r.burst()
        assert len(burst) >= publish.RELEASE_HOLD_MESSAGES
        assert set(burst) == {(0.0, 0.0, 0.0)}
        await r.answer_release()
        await r.tick(HELD_TICKS)
        # Frames again, the grip held: the base is taken anew, and its first
        # setpoint is the first on the wire since the burst.
        r.session.age_s = 0.0
        await r.squeeze_and_take()
        r.hold(squeeze=True, forward=FULL)
        assert setpoint_of(await r.next_setpoint()) == pytest.approx((0.8, 0.0, 0.0))


async def test_a_take_withdrawn_between_the_lease_and_the_stop_stops_nothing():
    async with rig() as r:
        r.hold(squeeze=True, forward=FULL)
        await r.report()
        await r.tick()
        request, responder = await r.daemon.take.next_request(CALL_S)
        assert request.robot == ROBOT
        # The grip is released while the daemon answers the take.
        r.hold(forward=FULL)
        await responder.respond()
        await r.answer_release()
        await r.tick(HELD_TICKS)
        assert r.stops[ROBOT].robot_stop_stop.captured() == []
        assert not r.state.engaged and not r.driver.holds(ROBOT)


async def test_each_setpoint_gives_no_frame_or_timestamp_and_planar_velocity_alone():
    async with rig() as r:
        r.limits = HOLONOMIC
        await r.squeeze_and_take()
        r.hold(squeeze=True, forward=FULL, sideways=HALF, turn=-HALF)
        driving = await r.next_setpoint()
        r.hold()
        burst = [await r.next_setpoint() for _ in range(publish.RELEASE_HOLD_MESSAGES)]
        assert setpoint_of(driving) == pytest.approx((0.8, 0.3, -0.75))
        for message in (driving, *burst):
            assert (message.frame, message.timestamp) == (None, None)
            assert message.linear_velocity[2] == 0.0
            assert message.angular_velocity[:2] == [0.0, 0.0]


async def test_a_grip_released_before_the_base_follows_releases_the_robot():
    async with rig() as r:
        r.hold(squeeze=True, forward=FULL)
        await r.report()
        await r.tick()
        await r.answer_take()
        await r.answer_stop()
        await r.ticks_until(lambda: r.state.engaged)
        r.hold(forward=FULL)
        await r.ticks_until(lambda: not r.driver.holds(ROBOT))
        await r.answer_release()
        await r.report()
        await r.tick(HELD_TICKS)
        await r.squeeze_and_take()
        r.hold(squeeze=True, forward=HALF)
        assert setpoint_of(await r.next_setpoint()) == pytest.approx((0.4, 0.0, 0.0))


async def test_a_squeeze_on_a_base_that_has_not_reported_takes_nothing_until_it_does():
    async with rig() as r:
        r.hold(squeeze=True)
        await r.tick(HELD_TICKS)
        assert r.row() == "the base does not report: check its base node"
        assert r.daemon.take.captured() == []
        # The held grip takes the robot once the base reports.
        await r.report()
        await r.tick()
        await r.answer_take()
        await r.answer_stop()
        await r.ticks_until(lambda: r.state.engaged)


async def test_choosing_another_robot_sends_the_burst_to_the_first_base_alone():
    async with rig("alpha", "bravo") as r:
        await r.squeeze_and_take(robot="alpha")
        r.hold(squeeze=True, forward=HALF)
        assert setpoint_of(await r.next_setpoint("alpha")) == pytest.approx(
            (0.4, 0.0, 0.0)
        )
        await r.report(robot="bravo")
        assert r.robots.choose("bravo")
        burst = await r.burst("alpha")
        assert len(burst) >= publish.RELEASE_HOLD_MESSAGES
        assert set(burst) == {(0.0, 0.0, 0.0)}
        await r.answer_release("alpha")
        # The grip held across the change drives nothing until it is
        # released; the next squeeze drives bravo.
        assert r.row() == "release the grip, then squeeze again"
        r.hold(forward=HALF)
        await r.ticks_until(lambda: not r.state.wait_release)
        await r.squeeze_and_take(robot="bravo")
        r.hold(squeeze=True, forward=FULL)
        assert setpoint_of(await r.next_setpoint("bravo")) == pytest.approx(
            (0.8, 0.0, 0.0)
        )


async def test_a_halt_lets_go_of_the_base_with_the_burst_and_a_held_grip_drives_nothing():
    async with rig() as r:
        await r.squeeze_and_take()
        r.hold(squeeze=True, forward=HALF)
        await r.next_setpoint()
        r.fleet.view = fleet_view([ROBOT], halted_by="peppy-cli")
        r.driver.fleet_changed(r.fleet.view)
        assert r.driver.note.text() == f"let go of {ROBOT}: {HALTED}"
        burst = await r.burst()
        assert len(burst) >= publish.RELEASE_HOLD_MESSAGES
        assert set(burst) == {(0.0, 0.0, 0.0)}
        assert r.row() == "release the grip, then squeeze again"
        await r.tick(HELD_TICKS)
        r.fleet.view = fleet_view([ROBOT])
        r.hold()
        await r.ticks_until(lambda: not r.state.wait_release)
        await r.squeeze_and_take()
        r.hold(squeeze=True, forward=FULL)
        assert setpoint_of(await r.next_setpoint()) == pytest.approx((0.8, 0.0, 0.0))
