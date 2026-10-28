"""The lease follows the grip on the wire: the pose stream and the posture
button against the generated mocks of the daemon target `fleet`, of the
robot's `limb_motion` and of its postures, on an injected clock.

Each test follows a fixed order of events: the clock moves only once every
loop of the test waits on it, so one tick is one step of each loop, and an
engaged hand or its release burst sends one pose per tick. The robot's
follower reports its measured pose at every tick, as a live one does. Where
a test proves that nothing was streamed, it moves the measured pose before
the next squeeze: the stream sends in order on one wire, so the first message
being the new pose proves that nothing went out before it.
"""

import asyncio
import contextlib
from itertools import groupby

import numpy as np
import pytest
from peppygen.consumed_actions.fleet import changes, lease
from peppygen.consumed_actions.postures import move_to_home
from peppygen.consumed_services.fleet import check_move, release, take
from peppygen.consumed_services.limb_motion import stop
from peppygen.paired_topics.right_arm import pose_setpoints as right_arm_pose_setpoints
from peppygen.paired_topics.right_gripper import (
    gripper_setpoints as right_gripper_setpoints,
)

from tests.helpers import (
    CALL_S,
    DIRECTOR,
    HALTED,
    HEADSET,
    POSE,
    ManualClock,
    ScriptedHeadset,
    accept_both,
    boot_clocked,
    change,
    default_parameters,
    eventually,
    fleet_view,
    lease_event,
    robots_of,
    running,
    settle,
)
from xr_commander import config, fleet, postures, publish
from xr_commander.clutch import HandClutch
from xr_commander.devices import primary_pressed
from xr_commander.drive import Driver, LimbStops
from xr_commander.fleet import LEASE_NAME, Fleet, FleetCalls, Following, Timing
from xr_commander.frames import Pose

ROBOT = "backbone_inst"
SETTINGS = config.from_parameters(
    default_parameters(command_rate_hz=100, stale_timeout_s=0.25)
)
TICK_S = SETTINGS.tick_period_s
# The most ticks a test waits for a state, and the most poses it reads.
WIRE_TICKS = 50
# Ticks a held grip gets to stream, where it must stream nothing.
HELD_TICKS = 5
ELSEWHERE = Pose(np.array([0.4, 0.0, 0.0]), POSE.orientation)
# Where the hand drives the arm in the halt test: neither the arm's pose nor
# the pose it moves to later.
HAND_AWAY = Pose(np.array([0.1, 0.0, 0.0]), POSE.orientation)


class Rig:
    """The headset's right hand on one robot of the fleet listing, with the
    daemon, the robot's limb_motion and its postures played by the mocks."""

    def __init__(self, h, clock: ManualClock) -> None:
        self.h = h
        self.clock = clock
        self.fleet = Fleet(HEADSET)
        self.fleet.view = fleet_view([ROBOT])
        self.fleet.status = Following()
        robots = robots_of(h)
        timing = Timing(call_timeout_s=CALL_S, retry_s=1.0)
        self.driver = Driver(
            self.fleet,
            FleetCalls(
                h.node_runner,
                take_module=take,
                release_module=release,
                check_move_module=check_move,
                timing=timing,
            ),
            LimbStops(h.node_runner, robots, stop, timing),
            timing=timing,
            on_change=lambda: None,
            clock=clock,
        )
        self.hand = self.driver.hands["right"]
        self.robots = robots
        self.session = ScriptedHeadset(clock)
        self.measured = publish.MeasuredPoses(clock)
        self.peer = robots.peer_for(right_arm_pose_setpoints)
        # Where the follower measures the arm; it reports at every tick.
        self.arm_at = POSE
        self.daemon = h.mocks.deps.fleet
        self.stops = h.mocks.deps.limb_motion[0].limb_motion_stop
        self.postures = h.mocks.deps.postures[0].move_to_home
        self.wire = h.mocks.pairings.right_arm[0].pose_setpoints
        self.status = postures.PostureStatus(clock)
        # The loops of the test, which each tick waits for.
        self.loops: list[asyncio.Task] = []

    def pose_stream(self, token):
        return publish.stream_pose(
            self.h.node_runner,
            topic_module=right_arm_pose_setpoints,
            gripper_module=right_gripper_setpoints,
            handedness="right",
            hand=self.hand,
            driver=self.driver,
            clutch=HandClutch(1.0),
            session=self.session,
            measured=self.measured,
            robots=self.robots,
            settings=SETTINGS,
            token=token,
            clock=self.clock,
        )

    def follow(self, token):
        """The two long goals of the fleet, with the driver as their
        listener."""
        return fleet.follow_fleet(
            self.h.node_runner,
            self.fleet,
            self.driver,
            changes_module=changes,
            lease_module=lease,
            timing=Timing(call_timeout_s=CALL_S, retry_s=1.0),
            token=token,
            clock=self.clock,
        )

    def posture_button(self, token):
        return postures.run_posture_button(
            self.h.node_runner,
            action_module=move_to_home,
            pressed=primary_pressed,
            session=self.session,
            robots=self.robots,
            driver=self.driver,
            status=self.status,
            settings=SETTINGS,
            token=token,
            clock=self.clock,
        )

    def hold(self, *, squeeze=False, a=False, hand=POSE):
        """The right hand at `hand`, its grip and its A button."""
        self.session.press(pose=hand, right=(a, False, squeeze))

    async def tick(self, count=1):
        for _ in range(count):
            self.measured.set(self.peer, self.arm_at)
            await self.clock.advance(TICK_S, parked=self.loops)

    async def ticks_until(self, predicate, limit=WIRE_TICKS):
        """Tick until `predicate()`; each tick is one step of the loops."""
        for _ in range(limit):
            if predicate():
                return
            await self.tick()
        raise AssertionError("the condition was never met")

    async def answer_take(self, refusal=None):
        request, responder = await self.daemon.take.next_request(CALL_S)
        assert (request.robot, request.name) == (ROBOT, LEASE_NAME)
        if refusal is None:
            await responder.respond()
        else:
            await responder.respond_failure(take.FailureKind.LEASED, refusal)

    async def answer_stop(self):
        request, responder = await self.stops.next_request(CALL_S)
        assert request.reason == LEASE_NAME
        await responder.respond(stop.ResponseData(stopped=["right_arm"]))

    async def answer_release(self):
        request, responder = await self.daemon.release.next_request(CALL_S)
        assert request.robot == ROBOT
        await responder.respond()

    async def answer_check_move(self, refusal=None):
        request, responder = await self.daemon.check_move.next_request(CALL_S)
        assert (request.instance, request.robot) == (ROBOT, None)
        denied = (
            None
            if refusal is None
            else check_move.ResponseDataRefusal(failure_kind="leased", message=refusal)
        )
        await responder.respond(check_move.ResponseData(refusal=denied, revision=1))

    async def press_a(self):
        """A press of A from rest: the posture button asks to take the
        robot."""
        self.hold()
        await self.tick()
        self.hold(a=True)
        await self.tick()

    async def release_grip(self):
        """Release the grip, and tick until the hand has seen the release."""
        self.hold(squeeze=False)
        await self.ticks_until(
            lambda: not self.hand.wait_release and not self.hand.engaged
        )

    async def squeeze_and_take(self):
        """A squeeze that takes the robot: the take, then the stop, then the
        stream engages at the measured pose."""
        self.hold(squeeze=True)
        await self.tick()
        await self.answer_take()
        await self.answer_stop()
        await self.ticks_until(lambda: self.hand.engaged)

    async def next_streamed(self):
        """One tick, then the next pose on the wire: for a hand that streams
        at every tick."""
        await self.tick()
        return await asyncio.wait_for(self.wire.next(), CALL_S)

    async def streamed_until(self, position: float) -> list[float]:
        """The positions on the wire before the first at `position`."""
        before = []
        for _ in range(WIRE_TICKS):
            streamed = position_of(await self.next_streamed())
            if streamed == pytest.approx(position):
                return before
            before.append(streamed)
        raise AssertionError(f"the wire never reached {position}")


@contextlib.asynccontextmanager
async def rig(*loops: str):
    """The rig with the named loops of the headset running."""
    async with boot_clocked(
        limb_motion_instance_ids=[ROBOT],
        postures_instance_ids=[ROBOT],
        right_arm_members={ROBOT: None},
        right_gripper_members={ROBOT: None},
    ) as h:
        the_rig = Rig(h, ManualClock())
        async with running(*(getattr(the_rig, loop) for loop in loops)) as tasks:
            the_rig.loops = tasks
            yield the_rig


def position_of(message) -> float:
    return message.position[0]


def runs(positions: list[float]) -> list[tuple[float, int]]:
    """Each run of one position on the wire, and its length."""
    return [
        (position, len(list(run)))
        for position, run in groupby(round(position, 6) for position in positions)
    ]


NO_RESULT = move_to_home.ResultResponseData(arm_names=[], positions=[], orientations=[])


def halt(r: "Rig") -> None:
    r.fleet.view = fleet_view([ROBOT], halted_by="peppy-cli")
    r.driver.fleet_changed(r.fleet.view)


async def start_home(r: "Rig"):
    """A press of A that takes the robot and starts its move home: the
    accepted goal."""
    await r.press_a()
    await r.answer_take()
    await r.answer_stop()
    await r.answer_check_move()
    return await (await r.postures.next_goal(CALL_S)).accept()


async def test_a_squeeze_takes_the_lease_then_stops_the_moves_then_streams():
    async with rig("pose_stream") as r:
        r.hold(squeeze=True)
        await r.tick()
        await r.answer_take()
        # The stop is asked and not answered yet: the stream waits on it, so
        # the hand does not engage.
        request, responder = await r.stops.next_request(CALL_S)
        assert request.reason == LEASE_NAME
        await settle()
        assert not r.hand.engaged
        await responder.respond(stop.ResponseData(stopped=["right_arm"]))
        first = await r.next_streamed()
        assert first.position == [0.0, 0.0, 0.0]


async def test_a_refused_take_streams_nothing_and_the_hand_says_who_holds_the_robot():
    async with rig("pose_stream") as r:
        r.hold(squeeze=True)
        await r.tick()
        await r.answer_take(
            refusal="robot `backbone_inst` is leased by `director_inst`"
        )
        await r.ticks_until(lambda: r.hand.note.text() != "")
        assert (
            r.hand.note.text()
            == "squeeze refused: robot `backbone_inst` is leased by `director_inst`"
        )
        # The held grip asks nothing again until it is released.
        await r.tick(HELD_TICKS)
        assert r.hand.wait_release
        assert len(r.daemon.take.captured()) == 1
        await r.release_grip()
        r.arm_at = ELSEWHERE
        await r.squeeze_and_take()
        assert position_of(await r.next_streamed()) == pytest.approx(0.4)


async def test_a_release_before_the_take_returns_stops_nothing_streams_nothing_and_releases_the_lease():
    async with rig("pose_stream") as r:
        r.hold(squeeze=True)
        await r.tick()
        r.hold(squeeze=False)
        await r.answer_take()
        await r.answer_release()
        assert r.stops.captured() == []
        r.arm_at = ELSEWHERE
        await r.squeeze_and_take()
        assert position_of(await r.next_streamed()) == pytest.approx(0.4)


async def test_the_lease_is_released_only_after_the_release_burst():
    async with rig("pose_stream") as r:
        await r.squeeze_and_take()
        await r.next_streamed()
        r.hold(squeeze=False)
        released_at = r.clock.monotonic()
        # The burst runs a quarter second: the headset holds the robot
        # through it.
        await r.tick()
        assert r.driver.holds(ROBOT)
        await r.ticks_until(lambda: not r.driver.holds(ROBOT))
        assert r.clock.monotonic() - released_at >= publish.RELEASE_HOLD_S
        await r.answer_release()


async def test_a_squeeze_in_the_release_burst_drives_on_and_keeps_the_lease():
    async with rig("pose_stream") as r:
        await r.squeeze_and_take()
        await r.next_streamed()
        r.hold(squeeze=False)
        await r.tick()
        assert r.driver.hand_holds(ROBOT) and not r.hand.engaged
        r.hold(squeeze=True)
        await r.ticks_until(lambda: r.hand.engaged)
        # The burst stops: no take, no stop, and the lease stays past the
        # end the burst had.
        await r.tick(int(publish.RELEASE_HOLD_S / TICK_S) + 1)
        assert r.driver.holds(ROBOT)
        assert len(r.daemon.take.captured()) == 1 and len(r.stops.captured()) == 1


async def test_a_press_in_the_release_burst_does_nothing_and_says_why():
    async with rig("pose_stream", "posture_button") as r:
        await r.squeeze_and_take()
        await r.next_streamed()
        r.hold(squeeze=False)
        await r.tick()
        assert r.driver.hand_holds(ROBOT)
        r.hold(a=True)
        await r.tick()
        assert r.status.state() == "move to home ignored: the grip was just released"


async def test_a_posture_press_takes_the_robot_asks_the_daemon_and_holds_it_for_the_move():
    async with rig("posture_button") as r:
        await r.press_a()
        await r.answer_take()
        await r.answer_stop()
        await r.answer_check_move()
        goal = await (await r.postures.next_goal(CALL_S)).accept()
        assert r.driver.holds(ROBOT)
        await goal.complete(
            move_to_home.ResultResponseData(
                arm_names=["right_arm"],
                positions=[0.3, -0.2, 0.3],
                orientations=[0.0, 0.0, 0.0, 1.0],
            )
        )
        await r.answer_release()
        await eventually(lambda: r.status.state() == "move to home completed")


async def test_a_posture_press_the_daemon_refuses_releases_the_lease():
    async with rig("posture_button") as r:
        await r.press_a()
        await r.answer_take()
        await r.answer_stop()
        await r.answer_check_move(refusal="the stack is halted (by peppy-cli)")
        await r.answer_release()
        await eventually(
            lambda: (
                r.status.state()
                == "move to home refused: the stack is halted (by peppy-cli)"
            )
        )


async def test_a_posture_goal_the_robot_refuses_releases_the_lease():
    async with rig("posture_button") as r:
        await r.press_a()
        await r.answer_take()
        await r.answer_stop()
        await r.answer_check_move()
        pending = await r.postures.next_goal(CALL_S)
        await pending.refuse(
            move_to_home.FailureKind.LEADER_STREAMING,
            "a leader is streaming arm commands; release the headset's grip",
        )
        await r.answer_release()
        await eventually(
            lambda: (
                r.status.state()
                == "move to home refused: a leader is streaming arm commands"
            )
        )


async def test_a_squeeze_during_a_posture_move_stops_it_and_keeps_the_lease_for_the_hand():
    async with rig("pose_stream", "posture_button") as r:
        await r.press_a()
        await r.answer_take()
        await r.answer_stop()
        await r.answer_check_move()
        goal = await (await r.postures.next_goal(CALL_S)).accept()
        # The squeeze stops the move in flight, and takes nothing.
        r.hold(squeeze=True)
        await r.tick()
        await r.answer_stop()
        await goal.complete_cancelled("stopped: driven from the headset", NO_RESULT)
        await eventually(lambda: r.status.state() == "move to home cancelled")
        first = await r.next_streamed()
        assert first.position == [0.0, 0.0, 0.0]
        # The end of the stopped move released nothing: the hand holds the
        # robot until its burst ends.
        assert r.driver.holds(ROBOT)
        r.hold(squeeze=False)
        await r.ticks_until(lambda: not r.driver.holds(ROBOT))
        await r.answer_release()


async def test_a_grip_held_through_a_halt_streams_nothing_until_a_new_squeeze():
    async with rig("pose_stream") as r:
        await r.squeeze_and_take()
        r.hold(squeeze=True, hand=HAND_AWAY)
        await r.streamed_until(0.1)
        halt(r)
        # The let-go is a release: the hand disengages and waits for its grip
        # to be released, and the burst holds the arm where it is measured.
        await r.ticks_until(lambda: r.hand.robot is None)
        assert not r.hand.engaged and r.hand.wait_release
        r.fleet.view = fleet_view([ROBOT])
        await r.tick(HELD_TICKS)
        # A new squeeze asks again.
        await r.release_grip()
        r.arm_at = ELSEWHERE
        await r.squeeze_and_take()
        # The hand's target streamed until the halt, then the burst at the
        # measured pose, and nothing between the burst and the new squeeze.
        (target, _), (burst, burst_length) = runs(await r.streamed_until(0.4))
        assert (target, burst) == (0.1, 0.0)
        assert burst_length >= publish.RELEASE_HOLD_MESSAGES


async def test_a_squeeze_or_a_press_during_a_halt_takes_nothing_and_says_so():
    async with rig("pose_stream", "posture_button") as r:
        halt(r)
        r.hold(squeeze=True)
        await r.ticks_until(lambda: r.hand.note.text() != "")
        assert r.hand.note.text() == f"squeeze refused: {HALTED}"
        await r.release_grip()
        await r.press_a()
        await r.ticks_until(lambda: r.status.state() is not None)
        assert r.status.state() == f"move to home refused: {HALTED}"
        assert r.daemon.take.captured() == [] and r.stops.captured() == []
        # Nothing streamed: after the halt, the first pose is the new one.
        r.fleet.view = fleet_view([ROBOT])
        r.arm_at = ELSEWHERE
        await r.squeeze_and_take()
        assert position_of(await r.next_streamed()) == pytest.approx(0.4)


async def test_a_halt_during_a_posture_move_cancels_it():
    async with rig("posture_button") as r:
        goal = await start_home(r)
        halt(r)
        await asyncio.wait_for(goal.cancel_signal(), CALL_S)
        assert not r.driver.holds(ROBOT)
        await goal.complete_cancelled("cancelled by its caller", NO_RESULT)


async def test_a_posture_goal_accepted_after_a_let_go_is_cancelled():
    async with rig("posture_button") as r:
        await r.press_a()
        await r.answer_take()
        await r.answer_stop()
        await r.answer_check_move()
        pending = await r.postures.next_goal(CALL_S)
        # The halt lands while the goal is on its way.
        halt(r)
        goal = await pending.accept()
        await asyncio.wait_for(goal.cancel_signal(), CALL_S)
        await goal.complete_cancelled("cancelled by its caller", NO_RESULT)
        await eventually(
            lambda: r.status.state() == "move to home refused: the headset let go of "
            f"{ROBOT}"
        )


async def test_a_posture_goal_whose_send_is_lost_is_stopped_before_the_release():
    async with rig("posture_button") as r:
        await r.press_a()
        await r.answer_take()
        await r.answer_stop()
        await r.answer_check_move()
        # The robot gets the goal and drops it unanswered: the send fails.
        await r.postures.next_goal(CALL_S)
        await r.answer_stop()
        await r.answer_release()
        await eventually(
            lambda: (
                r.status.state()
                == "move to home refused: the goal could not be sent (see log)"
            )
        )


async def test_a_press_landing_with_a_squeeze_takes_nothing_and_says_to_let_go():
    async with rig("posture_button") as r:
        r.hold()
        await r.tick()
        r.hold(squeeze=True, a=True)
        await r.ticks_until(lambda: r.status.state() is not None)
        assert r.status.state() == "move to home ignored: release the grip first"
        assert r.daemon.take.captured() == []


async def test_a_press_during_a_posture_move_does_nothing_and_says_why():
    async with rig("posture_button") as r:
        goal = await start_home(r)
        await r.ticks_until(lambda: r.status.state() == "moving to home...")
        r.hold()
        await r.tick()
        r.hold(a=True)
        await r.ticks_until(lambda: r.status.state() != "moving to home...")
        assert r.status.state() == "move to home ignored: still moving to home"
        assert len(r.daemon.take.captured()) == 1
        await goal.complete(
            move_to_home.ResultResponseData(
                arm_names=["right_arm"],
                positions=[0.3, -0.2, 0.3],
                orientations=[0.0, 0.0, 0.0, 1.0],
            )
        )
        await r.answer_release()


async def test_a_lease_end_that_a_listing_settles_against_the_headset_lets_go():
    async with rig("pose_stream") as r, running(r.follow):
        follow, hold = await accept_both(r.daemon)
        await follow.publish_feedback(change("listing", robots=[ROBOT]))
        # The rig's fleet follows from its start: wait for the listing of
        # the goals just started.
        await eventually(
            lambda: r.fleet.generation == 1 and r.fleet.unavailable() is None
        )
        await r.squeeze_and_take()
        await hold.publish_feedback(
            lease_event("lease_end", robot=ROBOT, cause="the stack was reset")
        )
        # The daemon allows the move, at the revision where the director took
        # the robot and opened its lease.
        request, responder = await r.daemon.check_move.next_request(CALL_S)
        assert (request.instance, request.robot) == (None, ROBOT)
        await responder.respond(check_move.ResponseData(refusal=None, revision=3))
        assert r.driver.holds(ROBOT)
        director_open = [(ROBOT, DIRECTOR, "tidy the table", True)]
        await follow.publish_feedback(
            change(
                "lease_open", ROBOT, revision=3, robots=[ROBOT], leases=director_open
            )
        )
        await eventually(lambda: not r.driver.holds(ROBOT))
        assert r.driver.note.text() == f"let go of {ROBOT}: the stack was reset"
