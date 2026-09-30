"""Every handler against the brain's core with fake backends: the
contract rules from goal to result, with no router and no robot."""

import asyncio

import pytest

from conftest import PARAMS, FakeCtx, FakeDetector, FakeManipulator, depth_frame, rgb_frame
from peppygen.parameters import Parameters

from openarm_ai_brain_vla.brain import Brain
from openarm_ai_brain_vla.handlers import abort, drop_item, get_state, grab_item, identify_item, place_item, scan_items
from openarm_ai_brain_vla.ports import Box, Coverage


class Ticks:
    """A clock the test advances by hand, in nanoseconds."""

    def __init__(self) -> None:
        self.now_ns = 1_000_000_000

    def __call__(self) -> int:
        return self.now_ns


def make_brain(*, detector=None, manipulator=None, ticks=None) -> Brain:
    params = Parameters.from_dict(dict(PARAMS))
    return Brain(params, node_runner=None, detector=detector, manipulator=manipulator, now=ticks or Ticks(), run_token="t0")


def with_frames(brain: Brain, depth_m: float = 1.0) -> None:
    from openarm_ai_brain_vla.perception.camera import Intrinsics

    brain.frames.color = rgb_frame(16, 12)
    brain.frames.depth = depth_frame(depth_m, 16, 12)
    brain.frames.depth_unit = 0.001
    # The camera has answered where its pixels point: a 90 degree lens.
    brain.perceiver.set_camera(brain.camera.with_intrinsics(Intrinsics.from_fovy(90.0, 16, 12)))


def grab_goal(**overrides):
    goal = dict(gripper_name="", item_id="", position=[0.0, 0.0, 0.0], orientation=None, max_effort=0.0)
    goal.update(overrides)
    return FakeCtx(**goal)


def state_of(brain: Brain):
    return get_state.handle(brain, None)


async def test_get_state_starts_idle_with_the_configured_grippers():
    brain = make_brain()
    response = state_of(brain)
    assert response.current_action == ""
    assert response.current_action_elapsed_s == 0.0
    assert response.gripper_names == ["left_gripper", "right_gripper"]
    assert response.holding == [False, False]
    assert response.held_item_ids == ["", ""]


async def test_scan_and_grab_are_refused_without_backends_but_the_rules_still_speak_first():
    brain = make_brain()
    ctx = FakeCtx(timeout_s=0.0)
    await scan_items.run(brain, ctx)
    assert ctx.completed["success"] is False
    assert ctx.completed["message"].startswith("no perception source")
    assert ctx.completed["item_ids"] == [] and ctx.completed["action_time"] == 0.0

    ctx = grab_goal(gripper_name="claw")
    await grab_item.run(brain, ctx)
    assert ctx.completed["message"] == "unknown gripper 'claw'"

    ctx = grab_goal(gripper_name="left_gripper", position=[0.5, 0.1, 0.7])
    await grab_item.run(brain, ctx)
    assert ctx.completed["success"] is False
    assert ctx.completed["message"] == "no manipulation backend: manipulation_backend is 'none'"
    assert ctx.completed["gripper_name"] == "" and ctx.completed["item_id"] == ""


async def test_scan_then_identify_then_grab_by_id_then_place():
    detector = FakeDetector([Box("cup", 0.9, 7, 5, 9, 7), Box("banana", 0.7, 1, 1, 3, 3)])
    manipulator = FakeManipulator()
    ticks = Ticks()
    brain = make_brain(detector=detector, manipulator=manipulator, ticks=ticks)
    with_frames(brain, depth_m=1.0)

    ctx = FakeCtx(timeout_s=0.0)
    await scan_items.run(brain, ctx)
    assert ctx.completed["success"] is True
    assert ctx.completed["item_ids"] == ["cup_1-t0", "banana_1-t0"]
    assert ctx.completed["labels"] == ["cup", "banana"]
    assert len(ctx.completed["positions"]) == 6
    assert ctx.completed["confidences"] == [0.9, 0.7]

    ctx = FakeCtx(description="cup", timeout_s=0.0)
    await identify_item.run(brain, ctx)
    assert ctx.completed["success"] is True
    assert ctx.completed["item_id"] == "cup_1-t0"
    assert detector.vocabulary == ["cup"]
    assert ctx.completed["orientation"] is None

    ctx = grab_goal(gripper_name="", item_id="cup_1-t0")
    ticks.now_ns += 3_000_000_000
    await grab_item.run(brain, ctx)
    assert ctx.completed["success"] is True
    assert ctx.completed["gripper_name"] == "left_gripper"
    assert ctx.completed["item_id"] == "cup_1-t0"
    assert manipulator.calls[-1][:3] == ("grab", "cup_1-t0", "left_gripper")
    assert state_of(brain).holding == [True, False]
    assert state_of(brain).held_item_ids == ["cup_1-t0", ""]

    ctx = grab_goal(gripper_name="left_gripper", item_id="banana_1-t0")
    await grab_item.run(brain, ctx)
    assert ctx.completed["message"] == "gripper 'left_gripper' already holds item 'cup_1-t0'"

    ctx = FakeCtx(gripper_name="", position=[0.6, 0.2, 0.75], orientation=None)
    await place_item.run(brain, ctx)
    assert ctx.completed["success"] is True
    assert ctx.completed["gripper_name"] == "left_gripper" and ctx.completed["item_id"] == "cup_1-t0"
    assert ctx.completed["final_position"] == [0.6, 0.2, 0.76]
    assert state_of(brain).holding == [False, False]


async def test_a_grab_by_pose_mints_an_id_and_a_failed_grab_changes_nothing():
    manipulator = FakeManipulator(fail="nothing grasped")
    brain = make_brain(manipulator=manipulator)
    ctx = grab_goal(gripper_name="right_gripper", position=[0.5, -0.1, 0.7])
    await grab_item.run(brain, ctx)
    assert ctx.completed["success"] is False and ctx.completed["message"] == "nothing grasped"
    assert state_of(brain).holding == [False, False]
    assert "item_1-t0" in brain.state.items

    manipulator.fail = ""
    ctx = grab_goal(gripper_name="right_gripper", item_id="item_1-t0")
    await grab_item.run(brain, ctx)
    assert ctx.completed["success"] is True and ctx.completed["item_id"] == "item_1-t0"

    ctx = FakeCtx(gripper_name="")
    await drop_item.run(brain, ctx)
    assert ctx.completed["success"] is True
    assert ctx.completed["gripper_name"] == "right_gripper" and ctx.completed["item_id"] == "item_1-t0"
    assert state_of(brain).holding == [False, False]


async def test_a_search_under_other_words_returns_the_scans_id():
    detector = FakeDetector([Box("mustard bottle", 0.9, 7, 5, 9, 7)])
    brain = make_brain(detector=detector)
    with_frames(brain)
    scan = FakeCtx(timeout_s=0.0)
    await scan_items.run(brain, scan)
    assert scan.completed["item_ids"] == ["mustard_bottle_1-t0"]

    # The words route names the same box with the caller's description.
    detector.boxes = [Box("yellow bottle", 0.96, 7, 5, 9, 7)]
    ctx = FakeCtx(description="yellow bottle", timeout_s=0.0)
    await identify_item.run(brain, ctx)
    assert ctx.completed["success"] is True
    assert ctx.completed["item_id"] == "mustard_bottle_1-t0"
    assert ctx.completed["label"] == "mustard bottle"
    assert ctx.completed["confidence"] == 0.96


async def test_a_scan_keeps_an_item_only_a_description_finds():
    detector = FakeDetector([Box("cup", 0.9, 7, 5, 9, 7)], coverage=Coverage(frozenset({"cup"})))
    brain = make_brain(detector=detector, manipulator=FakeManipulator())
    with_frames(brain)
    await scan_items.run(brain, FakeCtx(timeout_s=0.0))

    detector.boxes = [Box("blue ball", 0.8, 1, 1, 3, 3)]
    ctx = FakeCtx(description="blue ball", timeout_s=0.0)
    await identify_item.run(brain, ctx)
    assert ctx.completed["item_id"] == "blue_ball_1-t0"

    # The scan cannot name a blue ball, so not seeing it is no news.
    detector.boxes = [Box("cup", 0.9, 7, 5, 9, 7)]
    scan = FakeCtx(timeout_s=0.0)
    await scan_items.run(brain, scan)
    assert scan.completed["item_ids"] == ["cup_1-t0"]
    grab = grab_goal(gripper_name="", item_id="blue_ball_1-t0")
    await grab_item.run(brain, grab)
    assert grab.completed["success"] is True and grab.completed["item_id"] == "blue_ball_1-t0"


async def test_a_placed_item_keeps_its_id_in_the_next_scan():
    detector = FakeDetector([Box("cup", 0.9, 7, 5, 9, 7)])
    brain = make_brain(detector=detector, manipulator=FakeManipulator())
    with_frames(brain, depth_m=1.0)
    await scan_items.run(brain, FakeCtx(timeout_s=0.0))
    await grab_item.run(brain, grab_goal(gripper_name="", item_id="cup_1-t0"))

    # Put it where a box in the image's corner points, far from where it was.
    corner = Box("cup", 0.9, 1, 1, 3, 3)
    target = brain.perceiver.camera.deproject(*corner.centre, 1.0, 16, 12)
    ctx = FakeCtx(gripper_name="", position=list(target), orientation=None)
    await place_item.run(brain, ctx)
    assert ctx.completed["success"] is True and ctx.completed["item_id"] == "cup_1-t0"

    detector.boxes = [corner]
    scan = FakeCtx(timeout_s=0.0)
    await scan_items.run(brain, scan)
    assert scan.completed["item_ids"] == ["cup_1-t0"]


async def test_identify_reports_nothing_found_as_a_failed_result():
    brain = make_brain(detector=FakeDetector([Box("bowl", 0.9, 7, 5, 9, 7)]))
    with_frames(brain)
    ctx = FakeCtx(description="cup", timeout_s=0.0)
    await identify_item.run(brain, ctx)
    assert ctx.completed["success"] is False
    assert ctx.completed["message"] == "no item matches 'cup'"
    assert ctx.completed["item_id"] == ""


async def test_one_manipulation_at_a_time_and_abort_stops_it_without_touching_holding():
    manipulator = FakeManipulator(wait_for_stop=True)
    ticks = Ticks()
    brain = make_brain(manipulator=manipulator, ticks=ticks)
    first = grab_goal(gripper_name="left_gripper", position=[0.5, 0.1, 0.7])
    running = asyncio.create_task(grab_item.run(brain, first))
    await asyncio.sleep(0.01)
    ticks.now_ns += 2_000_000_000
    assert state_of(brain).current_action == "grab_item"
    assert state_of(brain).current_action_elapsed_s == 2.0

    second = grab_goal(gripper_name="right_gripper", position=[0.5, -0.1, 0.7])
    await grab_item.run(brain, second)
    assert second.completed["message"] == "grab_item is running"

    stop = FakeCtx(reason="operator pressed stop")
    await abort.run(brain, stop)
    await running
    assert stop.completed["success"] is True
    assert stop.completed["aborted_action"] == "grab_item"
    assert first.completed["success"] is False
    assert first.completed["message"] == "aborted: operator pressed stop"
    assert manipulator.stopped == 1
    assert state_of(brain).holding == [False, False]
    assert state_of(brain).current_action == ""

    idle = FakeCtx(reason="")
    await abort.run(brain, idle)
    assert idle.completed["success"] is True
    assert idle.completed["aborted_action"] == ""
    assert idle.completed["message"] == "no sequence was running"


async def test_a_callers_cancel_completes_the_goal_as_cancelled():
    manipulator = FakeManipulator(wait_for_stop=True)
    brain = make_brain(manipulator=manipulator)
    ctx = grab_goal(gripper_name="left_gripper", position=[0.5, 0.1, 0.7])
    running = asyncio.create_task(grab_item.run(brain, ctx))
    await asyncio.sleep(0.01)
    ctx.cancel()
    await asyncio.wait_for(running, 1.0)
    assert ctx.completed is None
    assert ctx.cancelled["success"] is False
    assert ctx.cancelled["message"] == "cancelled by the caller"
    assert manipulator.stopped == 1
    assert state_of(brain).current_action == ""


async def test_a_search_may_run_beside_a_manipulation():
    detector = FakeDetector([Box("cup", 0.9, 7, 5, 9, 7)])
    manipulator = FakeManipulator(wait_for_stop=True)
    brain = make_brain(detector=detector, manipulator=manipulator)
    with_frames(brain)
    grab = grab_goal(gripper_name="left_gripper", position=[0.5, 0.1, 0.7])
    running = asyncio.create_task(grab_item.run(brain, grab))
    await asyncio.sleep(0.01)
    scan = FakeCtx(timeout_s=0.0)
    await scan_items.run(brain, scan)
    assert scan.completed["success"] is True
    assert state_of(brain).current_action == "grab_item"
    await abort.run(brain, FakeCtx(reason=""))
    await running
