"""The real node under the generated harness: goals and services over the
wire, the camera answered by its mocks, and every member completing the way
the contracts want, with the "none" models, with the real sam3_siglip
detector whose load cannot finish here, with fakes behind brain_core's
ports, and with the real scripted manipulator moving the backbone's
mocks."""

import asyncio
from functools import partial

from brain_core.perception import geometry
from brain_core.ports import Box
from conftest import PARAMS, FakeDetector, FakeManipulator, answer_the_camera, camera_poses, colour_intrinsics, depth_frame, rgb_frame
from peppygen.consumed_actions.limb_motion import move_gripper
from peppygen.exposed_actions.brain_model import switch_model as switch_action
from peppygen.exposed_actions.item_manipulation import abort as abort_action
from peppygen.exposed_actions.item_manipulation import drop_item as drop_action
from peppygen.exposed_actions.item_manipulation import grab_item as grab_action
from peppygen.exposed_actions.item_manipulation import place_item as place_action
from peppygen.exposed_actions.item_perception import identify_item as identify_action
from peppygen.exposed_actions.item_perception import scan_items as scan_action
from peppygen.fixtures import harness
from peppygen.fixtures.exposed_actions.brain_model import switch_model as switch_fx
from peppygen.fixtures.exposed_actions.item_manipulation import abort as abort_fx
from peppygen.fixtures.exposed_actions.item_manipulation import drop_item as drop_fx
from peppygen.fixtures.exposed_actions.item_manipulation import grab_item as grab_fx
from peppygen.fixtures.exposed_actions.item_manipulation import place_item as place_fx
from peppygen.fixtures.exposed_actions.item_perception import identify_item as identify_fx
from peppygen.fixtures.exposed_actions.item_perception import scan_items as scan_fx
from peppygen.fixtures.exposed_services.brain_model import get_model as get_model_fx
from peppygen.fixtures.exposed_services.brain_model import list_models as list_models_fx
from peppygen.fixtures.exposed_services.item_manipulation import get_state as get_state_fx
from peppygen.parameters import Parameters
from wire import (
    WAIT,
    accepted_grip,
    grab_by_pose,
    land_move,
    once_the_brain_has_limbs,
    plan,
    publish_limbs_until,
    refusal_of,
    result_of,
    stop_grip_at,
)

from ai_brain_vla.__main__ import setup

QUAT = [0.0, 0.0, 0.0, 1.0]
# Searches made before the brain has the capture and the camera's answers,
# or before a load has ended; each one is a whole goal round trip.
ATTEMPTS = 1000
SCAN = scan_action.GoalRequestData(timeout_s=0.0)


def params(**overrides) -> Parameters:
    return Parameters.from_dict({**PARAMS, **overrides})


def with_fakes(detector=None, manipulator=None):
    """The node's setup with fakes as the models the parameters of
    `faked()` name."""
    return partial(
        setup,
        detectors={"sam3_siglip": lambda gallery: detector or FakeDetector()},
        manipulators={"scripted": lambda: manipulator or FakeManipulator()},
    )


def faked() -> Parameters:
    return params(perception_model="sam3_siglip", manipulation_model="scripted")


def with_fake_detector(detector=None):
    """The node's setup with a fake as sam3_siglip and the real scripted
    manipulator, which moves the backbone's mocks."""
    return partial(setup, detectors={"sam3_siglip": lambda gallery: detector or FakeDetector()})


async def publish_a_capture(h, frame_id: int = 1) -> None:
    camera = h.mocks.deps.camera
    assert await camera.video_stream.wait_for_subscriber(WAIT)
    assert await camera.depth_stream.wait_for_subscriber(WAIT)
    await camera.video_stream.publish(rgb_frame(16, 12, frame_id=frame_id))
    await camera.depth_stream.publish(depth_frame(1.0, 16, 12, frame_id=frame_id))


def grab_request(gripper_name: str, item_id: str = "", position=(0.5, 0.1, 0.7)):
    return grab_action.GoalRequestData(gripper_name=gripper_name, item_id=item_id, position=list(position), orientation=QUAT, max_effort=0.0)


async def first_accepted_scan(h):
    """The result of the first scan the brain accepts: the capture, the
    camera's answers and a load reach the brain after its goal loops are
    up, and until then it refuses a scan with no perception source."""
    for _ in range(ATTEMPTS):
        goal = await scan_fx.send_goal(h, SCAN, 5.0)
        if goal.accepted:
            return await goal.get_result(WAIT)
        assert goal.refusal.kind is scan_fx.FailureKind.RESOURCE_UNAVAILABLE, goal.refusal
    raise AssertionError(f"no scan was accepted; the last refusal: {goal.refusal}")


async def scanned(h):
    """The result of the first scan the brain accepts, which completes."""
    result = await first_accepted_scan(h)
    assert result.status == scan_fx.ResultStatus.COMPLETED, result.failure
    return result.data


async def refused_until(h, done):
    """The first refusal of a scan that `done` accepts: a load reaches the
    brain after its goal loops are up."""
    for _ in range(ATTEMPTS):
        refusal = await refusal_of(scan_fx, h, SCAN)
        if done(refusal):
            return refusal
    raise AssertionError(f"no refusal was the one awaited; the last: {refusal}")


async def test_the_node_answers_every_member_with_the_none_models():
    async with harness.start(setup, parameters=params()) as h:
        answer_the_camera(h)
        state = await get_state_fx.poll(h, 5.0)
        assert state.gripper_names == ["left_gripper", "right_gripper"]
        assert state.holding == [False, False]
        assert state.current_action == ""

        # item_manipulation reports refusals in its result.
        grabbed = await result_of(grab_fx, h, grab_request("left_gripper"))
        assert grabbed.success is False
        assert grabbed.message == "no manipulation model: manipulation_model is 'none'"
        assert grabbed.gripper_name == "" and grabbed.item_id == ""

        dropped = await result_of(drop_fx, h, drop_action.GoalRequestData(gripper_name=""))
        assert dropped.success is False and dropped.message == "no gripper holds an item"
        placed = await result_of(place_fx, h, place_action.GoalRequestData(gripper_name="", position=[0.5, 0.1, 0.7], orientation=QUAT))
        assert placed.success is False and placed.message == "no gripper holds an item"
        assert placed.final_position == [0.0, 0.0, 0.0]

        # item_perception refuses a search of the model "none" with
        # precondition_failed, as brain_model says.
        none = "no perception source: perception_model is 'none'"
        assert await refusal_of(scan_fx, h, SCAN) == scan_fx.Failure("precondition_failed", none)
        found = await refusal_of(identify_fx, h, identify_action.GoalRequestData(description="cup", timeout_s=0.0))
        assert found == identify_fx.Failure("precondition_failed", none)

        aborted = await result_of(abort_fx, h, abort_action.GoalRequestData(reason=""))
        assert aborted.success is True
        assert aborted.aborted_action == ""
        assert aborted.message == "no sequence was running"


async def test_get_model_and_list_models_name_the_model_each_role_runs():
    async with harness.start(with_fakes(), parameters=faked()) as h:
        answer_the_camera(h)
        current = await get_model_fx.poll(h, 5.0)
        assert (current.roles, current.models) == (["perception", "manipulation"], ["sam3_siglip", "scripted"])
        offered = await list_models_fx.poll(h, 5.0)
        assert (offered.roles, offered.models) == (["perception", "manipulation"], ["sam3_siglip", "scripted"])


async def test_switch_model_accepts_the_running_model_alone():
    async with harness.start(setup, parameters=params()) as h:
        answer_the_camera(h)
        same = await result_of(switch_fx, h, switch_action.GoalRequestData(role="perception", model="none"))
        assert (same.role, same.model) == ("perception", "none")
        other = await refusal_of(switch_fx, h, switch_action.GoalRequestData(role="perception", model="sam3_siglip"))
        assert other == switch_fx.Failure("rejected", "this brain runs only 'none' for perception")
        manipulation = await refusal_of(switch_fx, h, switch_action.GoalRequestData(role="manipulation", model="policy"))
        assert manipulation == switch_fx.Failure("rejected", "this brain runs only 'none' for manipulation")
        unknown = await refusal_of(switch_fx, h, switch_action.GoalRequestData(role="gripper", model="none"))
        assert unknown == switch_fx.Failure("rejected", "this brain runs no model for the role 'gripper'")
        assert (await get_model_fx.poll(h, 5.0)).models == ["none", "none"]


async def test_a_search_while_sam3_siglip_loads_is_refused_and_then_with_the_reason_its_load_failed(staged_weights):
    # The real detector: its weights are staged, and the models cannot be
    # built here (no sam3-siglip extra), so its background load fails.
    async with harness.start(setup, parameters=params(perception_model="sam3_siglip", perception_gallery="waldo_catalogue")) as h:
        answer_the_camera(h)
        await publish_a_capture(h)
        refusal = await refused_until(h, lambda refusal: "is still loading" not in refusal.message)
        assert refusal.kind is scan_fx.FailureKind.RESOURCE_UNAVAILABLE
        assert refusal.message.startswith(
            "no perception source: perception_model 'sam3_siglip' could not load: the sam3_siglip backend needs torch and transformers"
        )


async def test_every_member_completes_over_the_wire_with_models():
    detector = FakeDetector([Box("cup", 0.9, 7, 5, 9, 7)])
    manipulator = FakeManipulator()
    async with harness.start(with_fakes(detector, manipulator), parameters=faked()) as h:
        answer_the_camera(h)
        await publish_a_capture(h)
        scan = await scanned(h)
        assert scan.labels == ["cup"] and scan.confidences == [0.9] and len(scan.positions) == 3
        # The robot placed the camera at its origin looking along +Z: the
        # box's centre, a metre deep, is a metre ahead.
        assert [round(v, 6) for v in scan.positions] == [0.0, 0.0, 1.0]
        assert scan.regions == [7.0, 5.0, 9.0, 7.0]
        assert scan.camera == "chest" and (scan.image_width, scan.image_height) == (16, 12)
        assert scan.frame_timestamp > 0.0
        [cup] = scan.item_ids

        found = await result_of(identify_fx, h, identify_action.GoalRequestData(description="the cup", timeout_s=0.0))
        assert (found.item_id, found.label, found.confidence) == (cup, "cup", 0.9)
        assert found.position == scan.positions and found.orientation is None
        assert found.region == [7.0, 5.0, 9.0, 7.0] and found.camera == "chest"
        assert detector.phrases == [[], ["the cup"]]

        # A search that finds nothing fails with not_found, and no result.
        goal = await identify_fx.send_goal(h, identify_action.GoalRequestData(description="the bowl", timeout_s=0.0), 5.0)
        assert goal.accepted, goal.refusal
        nothing = await goal.get_result(WAIT)
        assert nothing.status == identify_fx.ResultStatus.FAILED and nothing.data is None
        assert nothing.failure == identify_fx.Failure("not_found", "no item matches 'the bowl'")

        grabbed = await result_of(grab_fx, h, grab_request("left_gripper", item_id=cup))
        assert grabbed.success is True, grabbed.message
        assert (grabbed.gripper_name, grabbed.item_id) == ("left_gripper", cup)
        again = await result_of(grab_fx, h, grab_request("right_gripper", item_id=cup))
        assert again.success is False
        assert again.message == f"item '{cup}' is held by gripper 'left_gripper'"
        state = await get_state_fx.poll(h, 5.0)
        assert state.holding == [True, False] and state.held_item_ids == [cup, ""]

        placed = await result_of(place_fx, h, place_action.GoalRequestData(gripper_name="", position=[0.6, 0.2, 0.75], orientation=QUAT))
        assert placed.success is True, placed.message
        assert (placed.gripper_name, placed.item_id) == ("left_gripper", cup)
        assert placed.final_position == [0.6, 0.2, 0.76]

        by_pose = await result_of(grab_fx, h, grab_request("right_gripper", position=(0.5, -0.1, 0.7)))
        assert by_pose.success is True and by_pose.item_id.startswith("item_1-")
        dropped = await result_of(drop_fx, h, drop_action.GoalRequestData(gripper_name=""))
        assert dropped.success is True, dropped.message
        assert (dropped.gripper_name, dropped.item_id) == ("right_gripper", by_pose.item_id)
        state = await get_state_fx.poll(h, 5.0)
        assert state.holding == [False, False] and state.current_action == ""
        assert [call[0] for call in manipulator.calls] == ["grab", "place", "grab", "drop"]


async def test_a_callers_cancel_over_the_wire_ends_the_goal_cancelled():
    manipulator = FakeManipulator(wait_for_stop=True)
    async with harness.start(with_fakes(manipulator=manipulator), parameters=faked()) as h:
        answer_the_camera(h)
        goal = await grab_fx.send_goal(h, grab_request("left_gripper"), 5.0)
        assert goal.accepted, goal.refusal
        await asyncio.wait_for(manipulator.begun(), WAIT)
        state = await get_state_fx.poll(h, 5.0)
        assert state.current_action == "grab_item"
        await goal.cancel_goal(5.0)
        result = await goal.get_result(WAIT)
        assert result.status == grab_fx.ResultStatus.CANCELLED
        assert result.cancel_reason == "cancelled by its caller"
        assert result.data.success is False and result.data.message == "cancelled by its caller"
        assert result.data.gripper_name == "" and result.data.item_id == ""
        assert manipulator.stopped == 1
        state = await get_state_fx.poll(h, 5.0)
        assert state.current_action == "" and state.holding == [False, False]


async def test_an_abort_over_the_wire_stops_the_running_sequence():
    manipulator = FakeManipulator(wait_for_stop=True)
    async with harness.start(with_fakes(manipulator=manipulator), parameters=faked()) as h:
        answer_the_camera(h)
        goal = await grab_fx.send_goal(h, grab_request("left_gripper"), 5.0)
        assert goal.accepted, goal.refusal
        await asyncio.wait_for(manipulator.begun(), WAIT)
        second = await result_of(grab_fx, h, grab_request("right_gripper", position=(0.5, -0.1, 0.7)))
        assert second.success is False and second.message == "grab_item is running"
        aborted = await result_of(abort_fx, h, abort_action.GoalRequestData(reason="operator pressed stop"))
        assert aborted.success is True and aborted.aborted_action == "grab_item"
        # item_manipulation keeps its abort: the running goal completes with
        # success false.
        result = await goal.get_result(WAIT)
        assert result.status == grab_fx.ResultStatus.COMPLETED
        assert result.data.success is False and result.data.message == "aborted: operator pressed stop"
        assert manipulator.stopped == 1


async def test_the_camera_is_asked_again_until_it_knows_its_geometry(monkeypatch):
    # A sim relay gives an error reply until its simulation has spoken; the
    # brain asks again and the first scan after the answer succeeds.
    monkeypatch.setattr(geometry, "INTRINSICS_RETRY_S", 0.0)
    detector = FakeDetector([Box("cup", 0.9, 7, 5, 9, 7)])
    async with harness.start(with_fakes(detector), parameters=faked()) as h:
        answer_the_camera(h, colour=False)
        colour = h.mocks.deps.geometry.get_color_intrinsics
        not_yet = await colour.next_request(WAIT)
        await not_yet.respond_failure("precondition_failed", "no camera geometry received from the simulation yet")
        known = await colour.next_request(WAIT)
        await known.respond(colour_intrinsics())
        await publish_a_capture(h)
        scan = await scanned(h)
        assert scan.labels == ["cup"]
        assert colour.captured_count() == 2
        # The error reply about the colour stream left the depth stream
        # unasked in its round.
        assert h.mocks.deps.geometry.get_depth_intrinsics.captured_count() == 1


async def test_the_robot_is_asked_again_until_it_names_the_camera(monkeypatch):
    # The robot answers, but carries no camera by the name the brain looks
    # through: the brain keeps asking, and the first scan after the answer
    # that names it succeeds.
    monkeypatch.setattr(geometry, "INTRINSICS_RETRY_S", 0.0)
    detector = FakeDetector([Box("cup", 0.9, 7, 5, 9, 7)])
    async with harness.start(with_fakes(detector), parameters=faked()) as h:
        wrists_only = camera_poses(names=("wrist_left", "wrist_right"))
        answer_the_camera(h, mounts=(wrists_only, wrists_only, camera_poses()))
        await publish_a_capture(h)
        scan = await scanned(h)
        assert scan.labels == ["cup"]
        assert h.mocks.deps.camera_mounts.get_camera_poses.captured_count() == 3


# The scripted grab over the wire: the real scripted manipulator, the
# limb_motion mock planning every check the test scripts and landing every
# move, and the limb_state mock publishing the robot's limbs.


async def test_identify_then_a_scripted_grab_holds_the_item():
    detector = FakeDetector([Box("cup", 0.9, 7, 5, 9, 7)])
    async with harness.start(with_fake_detector(detector), parameters=faked()) as h:
        answer_the_camera(h)
        await publish_a_capture(h)
        await scanned(h)
        found = await result_of(identify_fx, h, identify_action.GoalRequestData(description="the cup", timeout_s=0.0))
        # No limb_states snapshot yet: no measured opening.
        assert (await get_state_fx.poll(h, 5.0)).gripper_openings == []
        await once_the_brain_has_limbs(h)
        plan(h, True, True, True)
        grab = await grab_fx.send_goal(
            h, grab_action.GoalRequestData(gripper_name="", item_id=found.item_id, position=[0.0] * 3, orientation=None, max_effort=15.0),
            5.0,
        )
        assert grab.accepted, grab.refusal
        opened = await stop_grip_at(h, 1.0)
        assert (opened.gripper_name, opened.opening) == ("left_gripper", 1.0)
        pre = await land_move(h)
        reach = await land_move(h)
        # The cup was seen 1 m out along +Z: the grasp point is 2.5 cm under
        # it, reached from 7.5 cm back along the approach, straight down.
        assert [round(v, 6) for v in reach.position] == [0.0, 0.0, 0.975]
        assert [round(v, 6) for v in pre.position] == [0.0, 0.0, 1.05] and reach.orientation == [0.0, 1.0, 0.0, 0.0]
        closed = await stop_grip_at(h, 0.3)
        assert (closed.opening, closed.max_effort) == (0.0, 15.0)
        done = asyncio.Event()
        stream = asyncio.create_task(publish_limbs_until(h, done, 0.3))
        lift = await land_move(h)
        assert [round(v, 6) for v in lift.position] == [0.0, 0.0, 1.05]
        result = (await grab.get_result(WAIT)).data
        done.set()
        await stream
        assert (result.success, result.gripper_name, result.item_id) == (True, "left_gripper", found.item_id), result.message
        state = await get_state_fx.poll(h, 5.0)
        assert (state.holding, state.held_item_ids) == ([True, False], [found.item_id, ""])
        # The last snapshot the backbone published: the left jaws on the cup.
        assert state.gripper_openings == [0.3, 1.0]


async def test_a_gripper_that_closes_on_nothing_retreats_fails_the_grab_and_holds_nothing():
    async with harness.start(with_fake_detector(), parameters=faked()) as h:
        answer_the_camera(h)
        await once_the_brain_has_limbs(h)
        plan(h, True, True, True)
        grab = await grab_fx.send_goal(h, grab_by_pose([0.5, 0.1, 0.75]), 5.0)
        assert grab.accepted, grab.refusal
        await stop_grip_at(h, 1.0)
        pre = await land_move(h)
        reach = await land_move(h)
        # A grab by pose reaches the pose itself, unlowered.
        assert [round(v, 6) for v in reach.position] == [0.5, 0.1, 0.75]
        await stop_grip_at(h, 0.0)
        # The retreat: the gripper opens and the arm goes back up to the
        # pre-grasp point.
        assert (await stop_grip_at(h, 1.0)).opening == 1.0
        assert (await land_move(h)).position == pre.position
        result = (await grab.get_result(WAIT)).data
        assert (result.success, result.message) == (False, "gripper 'left_gripper' closed on nothing")
        assert (result.gripper_name, result.item_id) == ("", "")
        assert (await get_state_fx.poll(h, 5.0)).holding == [False, False]


async def test_a_scripted_grab_during_a_grab_is_refused_and_abort_ends_the_first():
    async with harness.start(with_fake_detector(), parameters=faked()) as h:
        answer_the_camera(h)
        await once_the_brain_has_limbs(h)
        plan(h, True, True, True)
        first = await grab_fx.send_goal(h, grab_by_pose([0.5, 0.1, 0.75]), 5.0)
        assert first.accepted, first.refusal
        # The first grab opens the gripper; the backbone holds that move.
        _, opening = await accepted_grip(h)
        assert (await get_state_fx.poll(h, 5.0)).current_action == "grab_item"
        second = await result_of(grab_fx, h, grab_by_pose([0.4, -0.1, 0.75]))
        assert (second.success, second.message) == (False, "grab_item is running")
        stop = await abort_fx.send_goal(h, abort_action.GoalRequestData(reason="operator"), 5.0)
        assert stop.accepted, stop.refusal
        await asyncio.wait_for(opening.cancel_signal(), WAIT)
        await opening.complete_cancelled(
            "cancelled by its caller", move_gripper.ResultResponseData(settled=False, final_opening=0.6, action_time=0.2)
        )
        aborted = (await first.get_result(WAIT)).data
        assert (aborted.success, aborted.message) == (False, "aborted: operator")
        stopped = (await stop.get_result(WAIT)).data
        assert (stopped.success, stopped.aborted_action) == (True, "grab_item"), stopped.message
        state = await get_state_fx.poll(h, 5.0)
        assert (state.current_action, state.holding) == ("", [False, False])
