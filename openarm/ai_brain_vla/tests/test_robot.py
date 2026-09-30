"""The backbone proxy against the harness's limb_motion mock: every way a
move ends, from the goal the brain fires to the outcome it reads."""

import asyncio

from conftest import PARAMS, answer_the_camera
from peppygen.consumed_actions.limb_motion import move_arm, move_gripper
from peppygen.fixtures import harness
from peppygen.parameters import Parameters

from openarm_ai_brain_vla.__main__ import setup
from openarm_ai_brain_vla.robot import GripResult, MoveResult, Robot

WAIT = 10.0
POSITION = (0.5, 0.1, 0.7)
ORIENTATION = (0.0, 0.0, 0.0, 1.0)


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


def arm_result(success: bool, message: str) -> move_arm.ResultResponseData:
    return move_arm.ResultResponseData(
        success=success, message=message, final_position=[0.51, 0.1, 0.69], final_orientation=[0.0, 0.0, 0.0, 1.0], action_time=0.4
    )


async def test_an_accepted_move_returns_the_backbones_result():
    async with booted() as h:
        robot = Robot(h.node_runner)
        move = asyncio.create_task(robot.move_arm("left_arm", POSITION, ORIENTATION, duration_s=1.5))
        pending = await h.mocks.deps.limb_motion.move_arm.next_goal(WAIT)
        assert pending.request.arm_name == "left_arm"
        assert pending.request.position == [0.5, 0.1, 0.7] and pending.request.duration_s == 1.5
        active = await pending.accept()
        await active.complete(arm_result(True, ""))
        assert await asyncio.wait_for(move, WAIT) == MoveResult(True, "", (0.51, 0.1, 0.69), (0.0, 0.0, 0.0, 1.0))

        grip = asyncio.create_task(robot.set_gripper("left_gripper", 0.2, max_effort=5.0))
        pending = await h.mocks.deps.limb_motion.move_gripper.next_goal(WAIT)
        assert (pending.request.gripper_name, pending.request.opening, pending.request.max_effort) == ("left_gripper", 0.2, 5.0)
        active = await pending.accept()
        await active.complete(move_gripper.ResultResponseData(success=True, message="", final_opening=0.21, action_time=0.3))
        assert await asyncio.wait_for(grip, WAIT) == GripResult(True, "", 0.21)


async def test_a_move_the_backbone_refuses_or_fails_says_why():
    async with booted() as h:
        robot = Robot(h.node_runner)
        move = asyncio.create_task(robot.move_arm("third_arm", POSITION, ORIENTATION))
        pending = await h.mocks.deps.limb_motion.move_arm.next_goal(WAIT)
        await pending.reject("unknown arm 'third_arm'")
        assert await asyncio.wait_for(move, WAIT) == MoveResult(False, "unknown arm 'third_arm'")

        move = asyncio.create_task(robot.move_arm("left_arm", POSITION, ORIENTATION))
        pending = await h.mocks.deps.limb_motion.move_arm.next_goal(WAIT)
        active = await pending.accept()
        await active.complete(arm_result(False, "no plan reaches the target"))
        result = await asyncio.wait_for(move, WAIT)
        assert (result.success, result.message) == (False, "no plan reaches the target")


async def test_a_stop_cancels_the_move_in_flight_and_reports_it_cancelled():
    async with booted() as h:
        robot = Robot(h.node_runner)
        move = asyncio.create_task(robot.move_arm("left_arm", POSITION, ORIENTATION))
        pending = await h.mocks.deps.limb_motion.move_arm.next_goal(WAIT)
        active = await pending.accept()
        # The acceptance reaches the proxy after the mock sent it: the stop
        # comes once the proxy holds the accepted goal.
        await asyncio.wait_for(in_flight(robot), WAIT)
        stop = asyncio.create_task(robot.stop())
        await asyncio.wait_for(active.cancel_signal(), WAIT)
        await active.complete_cancelled(arm_result(False, "stopped on request"))
        await asyncio.wait_for(stop, WAIT)
        result = await asyncio.wait_for(move, WAIT)
        assert (result.success, result.message, result.final_position) == (False, "stopped on request", (0.51, 0.1, 0.69))
        # Nothing is left in flight: a second stop has nothing to cancel.
        await asyncio.wait_for(robot.stop(), WAIT)


async def test_a_move_without_a_result_in_time_is_cancelled_and_reported():
    async with booted() as h:
        robot = Robot(h.node_runner, result_timeout_s=0.5)
        move = asyncio.create_task(robot.move_arm("left_arm", POSITION, ORIENTATION))
        pending = await h.mocks.deps.limb_motion.move_arm.next_goal(WAIT)
        active = await pending.accept()
        # The backbone never answers: the proxy gives up on the result and
        # cancels the goal so the arm does not keep going unwatched.
        result = await asyncio.wait_for(move, WAIT)
        assert result.success is False and result.message.startswith("move_arm gave no result")
        await asyncio.wait_for(active.cancel_signal(), WAIT)
        await active.complete_cancelled(arm_result(False, "stopped"))
