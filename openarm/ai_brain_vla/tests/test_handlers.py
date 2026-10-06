"""Every handler against the brain's core with fake backends: the
contract rules from goal to result, and the line each goal leaves in the
log, with no router and no robot."""

import asyncio
import logging

import pytest

from conftest import PARAMS, FakeCtx, FakeDetector, FakeManipulator, depth_frame, records_of, rgb_frame
from peppygen.exposed_actions.item_manipulation import abort as abort_action
from peppygen.exposed_actions.item_manipulation import drop_item as drop_action
from peppygen.exposed_actions.item_manipulation import grab_item as grab_action
from peppygen.exposed_actions.item_manipulation import place_item as place_action
from peppygen.exposed_actions.item_perception import identify_item as identify_action
from peppygen.exposed_actions.item_perception import scan_items as scan_action
from peppygen.parameters import Parameters

from openarm_ai_brain_vla.brain import Brain
from openarm_ai_brain_vla.handlers import abort, drop_item, get_state, grab_item, identify_item, place_item, scan_items
from openarm_ai_brain_vla.ports import Box, Coverage, SearchTimeout


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

    brain.frames.add_color(rgb_frame(16, 12))
    brain.frames.add_depth(depth_frame(depth_m, 16, 12))
    brain.frames.depth_unit = 0.001
    # The camera has answered where its pixels point, a 90 degree lens, and
    # the robot where it stands: at the origin, looking along +Z.
    brain.perceiver.set_camera(
        brain.camera.with_intrinsics(Intrinsics.from_fovy(90.0, 16, 12)).with_pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    )


def scan_goal(timeout_s: float = 0.0) -> FakeCtx:
    return FakeCtx(scan_action, timeout_s=timeout_s)


def identify_goal(description: str, timeout_s: float = 0.0) -> FakeCtx:
    return FakeCtx(identify_action, description=description, timeout_s=timeout_s)


def grab_goal(**overrides) -> FakeCtx:
    goal = dict(gripper_name="", item_id="", position=[0.0, 0.0, 0.0], orientation=None, max_effort=0.0)
    goal.update(overrides)
    return FakeCtx(grab_action, **goal)


def drop_goal(gripper_name: str = "") -> FakeCtx:
    return FakeCtx(drop_action, gripper_name=gripper_name)


def place_goal(position, gripper_name: str = "", orientation=None) -> FakeCtx:
    return FakeCtx(place_action, gripper_name=gripper_name, position=position, orientation=orientation)


def abort_goal(reason: str = "") -> FakeCtx:
    return FakeCtx(abort_action, reason=reason)


def state_of(brain: Brain):
    return get_state.handle(brain, None)


@pytest.fixture
def brain_log(caplog):
    """The records of the brain's core, where each goal leaves its line."""
    with records_of(caplog, "openarm_ai_brain_vla.brain") as records:
        yield records


def lines(log) -> list[str]:
    return [record.getMessage() for record in log.records]


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
    ctx = scan_goal()
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

    ctx = drop_goal()
    await drop_item.run(brain, ctx)
    assert ctx.completed["message"] == "no gripper holds an item"
    ctx = place_goal([0.5, 0.1, 0.7])
    await place_item.run(brain, ctx)
    assert ctx.completed["message"] == "no gripper holds an item"
    assert ctx.completed["final_position"] == [0.0, 0.0, 0.0]


async def test_scan_then_identify_then_grab_by_id_then_place():
    detector = FakeDetector([Box("cup", 0.9, 7, 5, 9, 7), Box("banana", 0.7, 1, 1, 3, 3)])
    manipulator = FakeManipulator()
    ticks = Ticks()
    brain = make_brain(detector=detector, manipulator=manipulator, ticks=ticks)
    with_frames(brain, depth_m=1.0)

    ctx = scan_goal()
    await scan_items.run(brain, ctx)
    assert ctx.completed["success"] is True
    assert ctx.completed["item_ids"] == ["cup_1-t0", "banana_1-t0"]
    assert ctx.completed["labels"] == ["cup", "banana"]
    assert len(ctx.completed["positions"]) == 6
    assert ctx.completed["confidences"] == [0.9, 0.7]
    # Each item's box, in the picture the scan looked at.
    assert ctx.completed["regions"] == [7.0, 5.0, 9.0, 7.0, 1.0, 1.0, 3.0, 3.0]
    assert ctx.completed["camera"] == "chest"
    assert (ctx.completed["image_width"], ctx.completed["image_height"]) == (16, 12)
    assert ctx.completed["frame_timestamp"] == 1.0

    ctx = identify_goal("cup")
    await identify_item.run(brain, ctx)
    assert ctx.completed["success"] is True
    assert ctx.completed["item_id"] == "cup_1-t0"
    assert detector.vocabulary == ["cup"]
    assert ctx.completed["orientation"] is None
    assert ctx.completed["region"] == [7.0, 5.0, 9.0, 7.0]
    assert ctx.completed["camera"] == "chest"
    assert (ctx.completed["image_width"], ctx.completed["image_height"]) == (16, 12)
    assert ctx.completed["frame_timestamp"] == 1.0

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

    ctx = place_goal([0.6, 0.2, 0.75])
    await place_item.run(brain, ctx)
    assert ctx.completed["success"] is True
    assert ctx.completed["gripper_name"] == "left_gripper" and ctx.completed["item_id"] == "cup_1-t0"
    assert ctx.completed["final_position"] == [0.6, 0.2, 0.76]
    assert state_of(brain).holding == [False, False]


async def test_a_grab_of_an_item_a_gripper_holds_is_refused_naming_the_gripper():
    detector = FakeDetector([Box("cup", 0.9, 7, 5, 9, 7)])
    manipulator = FakeManipulator()
    brain = make_brain(detector=detector, manipulator=manipulator)
    with_frames(brain)
    await scan_items.run(brain, scan_goal())
    await grab_item.run(brain, grab_goal(gripper_name="left_gripper", item_id="cup_1-t0"))
    assert state_of(brain).held_item_ids == ["cup_1-t0", ""]

    # The item is in the left jaws, not on the table: neither the right
    # gripper by name nor a free gripper by choice may be sent for it.
    for gripper_name in ("right_gripper", ""):
        ctx = grab_goal(gripper_name=gripper_name, item_id="cup_1-t0")
        await grab_item.run(brain, ctx)
        assert ctx.completed["success"] is False
        assert ctx.completed["message"] == "item 'cup_1-t0' is held by gripper 'left_gripper'"
    assert state_of(brain).held_item_ids == ["cup_1-t0", ""]
    assert len(manipulator.calls) == 1


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

    ctx = drop_goal()
    await drop_item.run(brain, ctx)
    assert ctx.completed["success"] is True
    assert ctx.completed["gripper_name"] == "right_gripper" and ctx.completed["item_id"] == "item_1-t0"
    assert state_of(brain).holding == [False, False]


async def test_a_search_under_other_words_returns_the_scans_id():
    detector = FakeDetector([Box("mustard bottle", 0.9, 7, 5, 9, 7)])
    brain = make_brain(detector=detector)
    with_frames(brain)
    scan = scan_goal()
    await scan_items.run(brain, scan)
    assert scan.completed["item_ids"] == ["mustard_bottle_1-t0"]

    # The words route names the same box with the caller's description.
    detector.boxes = [Box("yellow bottle", 0.96, 7, 5, 9, 7)]
    ctx = identify_goal("yellow bottle")
    await identify_item.run(brain, ctx)
    assert ctx.completed["success"] is True
    assert ctx.completed["item_id"] == "mustard_bottle_1-t0"
    assert ctx.completed["label"] == "mustard bottle"
    assert ctx.completed["confidence"] == 0.96


async def test_a_description_names_whole_words_only():
    # A scan under the vocabulary returns general names; a description
    # that shares letters with one of them, or only stopwords, names none.
    detector = FakeDetector([Box("thermos", 0.9, 7, 5, 9, 7), Box("candle", 0.8, 1, 1, 3, 3), Box("feather", 0.7, 4, 4, 6, 6)])
    brain = make_brain(detector=detector)
    with_frames(brain)
    for description in ("the coffee can please", "can", "the mug"):
        ctx = identify_goal(description)
        await identify_item.run(brain, ctx)
        assert ctx.completed["success"] is False
        assert ctx.completed["message"] == f"no item matches '{description}'"
    ctx = identify_goal("the candle please")
    await identify_item.run(brain, ctx)
    assert ctx.completed["success"] is True and ctx.completed["label"] == "candle"


async def test_a_scan_keeps_an_item_only_a_description_finds():
    detector = FakeDetector([Box("cup", 0.9, 7, 5, 9, 7)], coverage=Coverage(frozenset({"cup"})))
    brain = make_brain(detector=detector, manipulator=FakeManipulator())
    with_frames(brain)
    await scan_items.run(brain, scan_goal())

    detector.boxes = [Box("blue ball", 0.8, 1, 1, 3, 3)]
    ctx = identify_goal("blue ball")
    await identify_item.run(brain, ctx)
    assert ctx.completed["item_id"] == "blue_ball_1-t0"

    # The scan cannot name a blue ball, so not seeing it is no news.
    detector.boxes = [Box("cup", 0.9, 7, 5, 9, 7)]
    scan = scan_goal()
    await scan_items.run(brain, scan)
    assert scan.completed["item_ids"] == ["cup_1-t0"]
    grab = grab_goal(gripper_name="", item_id="blue_ball_1-t0")
    await grab_item.run(brain, grab)
    assert grab.completed["success"] is True and grab.completed["item_id"] == "blue_ball_1-t0"


async def test_a_placed_item_keeps_its_id_in_the_next_scan():
    detector = FakeDetector([Box("cup", 0.9, 7, 5, 9, 7)])
    brain = make_brain(detector=detector, manipulator=FakeManipulator())
    with_frames(brain, depth_m=1.0)
    await scan_items.run(brain, scan_goal())
    await grab_item.run(brain, grab_goal(gripper_name="", item_id="cup_1-t0"))

    # Put it where a box in the image's corner points, far from where it was.
    corner = Box("cup", 0.9, 1, 1, 3, 3)
    target = brain.perceiver.camera.deproject(*corner.centre, 1.0, 16, 12)
    ctx = place_goal(list(target))
    await place_item.run(brain, ctx)
    assert ctx.completed["success"] is True and ctx.completed["item_id"] == "cup_1-t0"

    detector.boxes = [corner]
    scan = scan_goal()
    await scan_items.run(brain, scan)
    assert scan.completed["item_ids"] == ["cup_1-t0"]


async def test_identify_reports_nothing_found_as_a_failed_result():
    brain = make_brain(detector=FakeDetector([Box("bowl", 0.9, 7, 5, 9, 7)]))
    with_frames(brain)
    ctx = identify_goal("cup")
    await identify_item.run(brain, ctx)
    assert ctx.completed["success"] is False
    assert ctx.completed["message"] == "no item matches 'cup'"
    assert ctx.completed["item_id"] == ""


async def test_a_search_that_stops_at_its_deadline_is_a_failed_result_and_frees_the_lane():
    class TimingOut(FakeDetector):
        def detect(self, image, deadline):
            raise SearchTimeout(deadline.budget_s)

    brain = make_brain(detector=TimingOut([Box("cup", 0.9, 7, 5, 9, 7)]))
    with_frames(brain)
    ctx = scan_goal(timeout_s=0.5)
    await scan_items.run(brain, ctx)
    assert ctx.completed["success"] is False
    assert ctx.completed["message"] == "the search did not finish within 0.5 s"
    assert ctx.completed["item_ids"] == []
    assert brain.sequencer.running("perception") is None


async def test_one_manipulation_at_a_time_and_abort_stops_it_without_touching_holding():
    manipulator = FakeManipulator(wait_for_stop=True)
    ticks = Ticks()
    brain = make_brain(manipulator=manipulator, ticks=ticks)
    first = grab_goal(gripper_name="left_gripper", position=[0.5, 0.1, 0.7])
    running = asyncio.create_task(grab_item.run(brain, first))
    await manipulator.begun()
    ticks.now_ns += 2_000_000_000
    assert state_of(brain).current_action == "grab_item"
    assert state_of(brain).current_action_elapsed_s == 2.0

    second = grab_goal(gripper_name="right_gripper", position=[0.5, -0.1, 0.7])
    await grab_item.run(brain, second)
    assert second.completed["message"] == "grab_item is running"

    stop = abort_goal("operator pressed stop")
    await abort.run(brain, stop)
    await running
    assert stop.completed["success"] is True
    assert stop.completed["aborted_action"] == "grab_item"
    assert first.completed["success"] is False
    assert first.completed["message"] == "aborted: operator pressed stop"
    assert manipulator.stopped == 1
    assert state_of(brain).holding == [False, False]
    assert state_of(brain).current_action == ""

    idle = abort_goal()
    await abort.run(brain, idle)
    assert idle.completed["success"] is True
    assert idle.completed["aborted_action"] == ""
    assert idle.completed["message"] == "no sequence was running"


async def test_a_callers_cancel_completes_the_goal_as_cancelled():
    manipulator = FakeManipulator(wait_for_stop=True)
    brain = make_brain(manipulator=manipulator)
    ctx = grab_goal(gripper_name="left_gripper", position=[0.5, 0.1, 0.7])
    running = asyncio.create_task(grab_item.run(brain, ctx))
    await manipulator.begun()
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
    await manipulator.begun()
    scan = scan_goal()
    await scan_items.run(brain, scan)
    assert scan.completed["success"] is True
    assert state_of(brain).current_action == "grab_item"
    await abort.run(brain, abort_goal())
    await running


async def test_shutdown_stops_the_running_sequence_as_aborted():
    manipulator = FakeManipulator(wait_for_stop=True)
    brain = make_brain(manipulator=manipulator)
    ctx = grab_goal(gripper_name="left_gripper", position=[0.5, 0.1, 0.7])
    running = asyncio.create_task(grab_item.run(brain, ctx))
    await manipulator.begun()
    await asyncio.wait_for(brain.shutdown(), 1.0)
    await running
    assert ctx.completed["success"] is False
    assert ctx.completed["message"] == "aborted: the node is shutting down"
    assert manipulator.stopped == 1
    assert state_of(brain).current_action == ""
    # Idle lanes make a shutdown a no-op.
    await asyncio.wait_for(brain.shutdown(), 1.0)


async def test_a_search_logs_the_frame_it_looked_at_each_box_and_the_item_it_returned(brain_log):
    ticks = Ticks()

    class Slow(FakeDetector):
        """Takes 0.84 s of the brain's clock to answer."""

        def detect(self, image, deadline):
            ticks.now_ns += 840_000_000
            return super().detect(image, deadline)

    brain = make_brain(detector=Slow([Box("cup", 0.9, 7, 5, 9, 7), Box("cup", 0.85, 7, 5, 9, 7.2)]), ticks=ticks)
    with_frames(brain)
    # The frames were taken at 1.0 s of the brain's clock.
    ticks.now_ns = 1_250_000_000
    await identify_item.run(brain, identify_goal("cup"))
    await identify_item.run(brain, identify_goal("bowl"))
    await scan_items.run(brain, scan_goal())
    boxes = "2 boxes: cup 0.90, cup 0.85 (duplicate)"
    assert lines(brain_log) == [
        f"identify_item 'cup' succeeded after 0.84 s; frame 1 taken 0.25 s before the search, {boxes}; item cup_1-t0",
        f"identify_item 'bowl' refused after 0.84 s: no item matches 'bowl'; frame 1 taken 1.09 s before the search, {boxes}",
        f"scan_items succeeded after 0.84 s; frame 1 taken 1.93 s before the search, {boxes}",
    ]
    assert {record.levelno for record in brain_log.records} == {logging.INFO}


async def test_a_search_that_found_nothing_logs_whether_the_detector_boxed_anything(brain_log):
    no_depth = make_brain(detector=FakeDetector([Box("cup", 0.9, 7, 5, 9, 7)]))
    with_frames(no_depth, depth_m=0.0)
    await identify_item.run(no_depth, identify_goal("cup"))
    no_box = make_brain(detector=FakeDetector())
    with_frames(no_box)
    await identify_item.run(no_box, identify_goal(""))
    assert lines(brain_log) == [
        "identify_item 'cup' refused after 0.00 s: no item matches 'cup'; frame 1 taken 0.00 s before the search, 1 box: cup 0.90 (no depth)",
        "identify_item '' refused after 0.00 s: no item in view; frame 1 taken 0.00 s before the search, no box",
    ]


async def test_a_search_refused_before_it_looked_logs_the_reason_alone(brain_log):
    brain = make_brain(detector=FakeDetector([Box("cup", 0.9, 7, 5, 9, 7)]))
    await identify_item.run(brain, identify_goal("  cup "))
    assert lines(brain_log) == ["identify_item 'cup' refused after 0.00 s: no perception source: no camera frame received"]


async def test_a_search_that_failed_logs_its_traceback(brain_log):
    class Broken(FakeDetector):
        def detect(self, image, deadline):
            raise RuntimeError("the GPU is gone")

    brain = make_brain(detector=Broken())
    with_frames(brain)
    ctx = scan_goal()
    await scan_items.run(brain, ctx)
    assert ctx.completed["message"] == "scan_items failed: RuntimeError('the GPU is gone')"
    [record] = brain_log.records
    assert record.levelno == logging.ERROR
    assert record.getMessage() == "scan_items failed after 0.00 s: RuntimeError('the GPU is gone')"
    assert "RuntimeError: the GPU is gone" in brain_log.text


async def test_a_manipulation_goal_logs_how_it_ended(brain_log):
    manipulator = FakeManipulator(wait_for_stop=True)
    ticks = Ticks()
    brain = make_brain(manipulator=manipulator, ticks=ticks)
    first = grab_goal(gripper_name="left_gripper", position=[0.5, 0.1, 0.7])
    running = asyncio.create_task(grab_item.run(brain, first))
    await manipulator.begun()
    ticks.now_ns += 2_000_000_000
    await grab_item.run(brain, grab_goal(gripper_name="right_gripper", position=[0.5, -0.1, 0.7]))
    await abort.run(brain, abort_goal("operator pressed stop"))
    await running
    manipulator.wait_for_stop = False
    await grab_item.run(brain, grab_goal(gripper_name="left_gripper", position=[0.5, 0.1, 0.7]))
    assert lines(brain_log) == [
        "grab_item refused after 0.00 s: grab_item is running",
        "grab_item stopped after 2.00 s: aborted: operator pressed stop",
        "grab_item succeeded after 0.00 s",
    ]
