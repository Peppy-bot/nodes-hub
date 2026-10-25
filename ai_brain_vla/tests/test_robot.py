"""brain_core's backbone proxy on this node's generated limb_motion modules,
against the harness's mock: every way a move ends, from the goal the brain
fires to the outcome it reads."""

import asyncio

from brain_core.robot import GripResult, MoveCheck, MoveEnd, MoveResult, Robot
from conftest import PARAMS, answer_the_camera
from peppygen.consumed_actions.limb_motion import move_arm, move_gripper
from peppygen.consumed_services.limb_motion import check_arm_move
from peppygen.fixtures import harness
from peppygen.parameters import Parameters

from ai_brain_vla.__main__ import setup

WAIT = 10.0
POSITION = (0.5, 0.1, 0.7)
ORIENTATION = (0.0, 0.0, 0.0, 1.0)
# The tolerances and settle timeout the moves below name.
BARS = {
    "plan_position_tolerance_m": 0.001,
    "plan_orientation_tolerance_rad": 0.01,
    "settle_position_tolerance_m": 0.02,
    "settle_orientation_tolerance_rad": 0.15,
    "settle_timeout_s": 2.0,
}
# The fields a check names besides the pose: the moves' plan bars.
PLAN = {"duration_s": 0.0, **{key: BARS[key] for key in ("plan_position_tolerance_m", "plan_orientation_tolerance_rad")}}


def robot_on(node_runner, **kwargs) -> Robot:
    """brain_core's Robot on the generated modules this node passes it."""
    return Robot(node_runner, move_arm, move_gripper, check_arm_move, **kwargs)


class booted:
    """The node under the harness, its camera answered, for a Robot on the
    node's own runner."""

    def __init__(self) -> None:
        self._start = harness.start(setup, parameters=Parameters.from_dict(dict(PARAMS)))

    async def __aenter__(self):
        h = await self._start.__aenter__()
        answer_the_camera(h)
        return h

    async def __aexit__(self, *exc) -> None:
        await self._start.__aexit__(*exc)


async def in_flight(robot: Robot) -> None:
    while robot.moves_in_flight == 0:
        await asyncio.sleep(0)


def arm_result() -> move_arm.ResultResponseData:
    return move_arm.ResultResponseData(final_position=[0.51, 0.1, 0.69], final_orientation=[0.0, 0.0, 0.0, 1.0], action_time=0.4)


async def test_a_completed_move_returns_where_the_limb_stands():
    async with booted() as h:
        robot = robot_on(h.node_runner)
        move = asyncio.create_task(robot.move_arm("left_arm", POSITION, ORIENTATION, duration_s=1.5, **BARS))
        pending = await h.mocks.deps.limb_motion.move_arm.next_goal(WAIT)
        assert pending.request.arm_name == "left_arm"
        assert pending.request.position == [0.5, 0.1, 0.7] and pending.request.duration_s == 1.5
        active = await pending.accept()
        await active.complete(arm_result())
        assert await asyncio.wait_for(move, WAIT) == MoveResult(MoveEnd.COMPLETED, "", "", (0.51, 0.1, 0.69), (0.0, 0.0, 0.0, 1.0))

        grip = asyncio.create_task(robot.set_gripper("left_gripper", 0.2, max_effort=5.0, settle_timeout_s=2.0))
        pending = await h.mocks.deps.limb_motion.move_gripper.next_goal(WAIT)
        assert (pending.request.gripper_name, pending.request.opening, pending.request.max_effort) == ("left_gripper", 0.2, 5.0)
        active = await pending.accept()
        await active.complete(move_gripper.ResultResponseData(settled=True, final_opening=0.21, action_time=0.3))
        assert await asyncio.wait_for(grip, WAIT) == GripResult(MoveEnd.COMPLETED, "", "", True, 0.21)


async def test_a_move_the_backbone_refuses_or_fails_says_why_with_its_kind():
    async with booted() as h:
        robot = robot_on(h.node_runner)
        move = asyncio.create_task(robot.move_arm("third_arm", POSITION, ORIENTATION, duration_s=0.0, **BARS))
        pending = await h.mocks.deps.limb_motion.move_arm.next_goal(WAIT)
        await pending.refuse("rejected", "unknown arm 'third_arm'")
        assert await asyncio.wait_for(move, WAIT) == MoveResult(MoveEnd.REFUSED, "unknown arm 'third_arm'", "rejected")

        move = asyncio.create_task(robot.move_arm("left_arm", POSITION, ORIENTATION, duration_s=0.0, **BARS))
        pending = await h.mocks.deps.limb_motion.move_arm.next_goal(WAIT)
        active = await pending.accept()
        await active.fail("unreachable", "no plan reaches the target")
        # A failed goal gives no result: nothing says where the arm stands.
        assert await asyncio.wait_for(move, WAIT) == MoveResult(MoveEnd.FAILED, "no plan reaches the target", "unreachable")


async def test_a_stop_cancels_the_move_in_flight_and_reports_it_cancelled():
    async with booted() as h:
        robot = robot_on(h.node_runner)
        move = asyncio.create_task(robot.move_arm("left_arm", POSITION, ORIENTATION, duration_s=0.0, **BARS))
        pending = await h.mocks.deps.limb_motion.move_arm.next_goal(WAIT)
        active = await pending.accept()
        # The acceptance reaches the proxy after the mock sent it: the stop
        # comes once the proxy holds the accepted goal.
        await asyncio.wait_for(in_flight(robot), WAIT)
        stop = asyncio.create_task(robot.stop())
        await asyncio.wait_for(active.cancel_signal(), WAIT)
        assert active.cancel_cause() == "caller"
        await active.complete_cancelled("cancelled by its caller", arm_result())
        await asyncio.wait_for(stop, WAIT)
        result = await asyncio.wait_for(move, WAIT)
        assert (result.end, result.message, result.final_position) == (MoveEnd.CANCELLED, "cancelled by its caller", (0.51, 0.1, 0.69))
        # Nothing is left in flight: a second stop has nothing to cancel.
        await asyncio.wait_for(robot.stop(), WAIT)


async def test_a_move_the_backbones_stop_ends_reports_the_stops_reason_without_a_cancel():
    """The backbone's stop service ends a move as cancelled with no cancel
    from the brain: the move ends cancelled with the stop's reason, so a
    backend sees it did not complete and does not go on."""
    async with booted() as h:
        robot = robot_on(h.node_runner)
        move = asyncio.create_task(robot.move_arm("left_arm", POSITION, ORIENTATION, duration_s=0.0, **BARS))
        pending = await h.mocks.deps.limb_motion.move_arm.next_goal(WAIT)
        active = await pending.accept()
        await asyncio.wait_for(in_flight(robot), WAIT)
        await active.complete_cancelled("stopped: the operator asked", arm_result())
        result = await asyncio.wait_for(move, WAIT)
        assert (result.end, result.message) == (MoveEnd.CANCELLED, "stopped: the operator asked")
        assert robot.moves_in_flight == 0


async def test_a_move_without_a_result_in_time_is_cancelled_and_reported_lost():
    async with booted() as h:
        robot = robot_on(h.node_runner, result_timeout_s=0.5)
        move = asyncio.create_task(robot.move_arm("left_arm", POSITION, ORIENTATION, duration_s=0.0, **BARS))
        pending = await h.mocks.deps.limb_motion.move_arm.next_goal(WAIT)
        active = await pending.accept()
        # The backbone never answers: the proxy gives up on the result and
        # cancels the goal so the arm does not keep going unwatched.
        result = await asyncio.wait_for(move, WAIT)
        assert result.end is MoveEnd.LOST and result.message.startswith("move_arm gave no result")
        await asyncio.wait_for(active.cancel_signal(), WAIT)
        await active.complete_cancelled("cancelled by its caller", arm_result())


async def test_a_check_asks_the_backbone_and_reads_its_plan_or_its_reason():
    async with booted() as h:
        robot = robot_on(h.node_runner)
        service = h.mocks.deps.limb_motion.check_arm_move
        service.enqueue_response(check_arm_move.ResponseData(reachable=True, reason="", duration_s=1.25))
        service.enqueue_response(check_arm_move.ResponseData(reachable=False, reason="goal pose not planned", duration_s=0.0))
        assert await robot.check_arm_move("left_arm", POSITION, ORIENTATION, **PLAN) == MoveCheck(True, "", 1.25)
        assert await robot.check_arm_move("left_arm", POSITION, ORIENTATION, **PLAN) == MoveCheck(False, "goal pose not planned", 0.0)
        [first, second] = service.captured()
        assert (first.arm_name, first.position, first.orientation) == ("left_arm", [0.5, 0.1, 0.7], [0.0, 0.0, 0.0, 1.0])
        assert robot.moves_in_flight == 0
