"""The real node under the generated harness: goals over the wire, the
camera answered by its mocks, and every member completing the way the
contracts want, with the "none" backends and with fakes behind the two
ports."""

import asyncio
import time
from functools import partial

from conftest import PARAMS, FakeDetector, FakeManipulator, answer_the_camera, colour_intrinsics, depth_frame, rgb_frame
from peppygen import QoSProfile
from peppygen.exposed_actions.item_manipulation import abort as abort_action
from peppygen.exposed_actions.item_manipulation import drop_item as drop_action
from peppygen.exposed_actions.item_manipulation import grab_item as grab_action
from peppygen.exposed_actions.item_manipulation import place_item as place_action
from peppygen.exposed_actions.item_perception import identify_item as identify_action
from peppygen.exposed_actions.item_perception import scan_items as scan_action
from peppygen.fixtures import harness
from peppygen.fixtures.exposed_actions.item_manipulation import abort as abort_fx
from peppygen.fixtures.exposed_actions.item_manipulation import drop_item as drop_fx
from peppygen.fixtures.exposed_actions.item_manipulation import grab_item as grab_fx
from peppygen.fixtures.exposed_actions.item_manipulation import place_item as place_fx
from peppygen.fixtures.exposed_actions.item_perception import identify_item as identify_fx
from peppygen.fixtures.exposed_actions.item_perception import scan_items as scan_fx
from peppygen.fixtures.exposed_services.item_manipulation import get_state as get_state_fx
from peppygen.parameters import Parameters

from openarm_ai_brain_vla.__main__ import setup
from openarm_ai_brain_vla.perception import geometry
from openarm_ai_brain_vla.ports import Box

WAIT = 10.0
QUAT = [0.0, 0.0, 0.0, 1.0]


def params() -> Parameters:
    return Parameters.from_dict(dict(PARAMS))


async def publish_a_capture(h, frame_id: int = 1) -> None:
    camera = h.mocks.deps.camera
    assert await camera.video_stream.wait_for_subscriber(WAIT)
    assert await camera.depth_stream.wait_for_subscriber(WAIT)
    await camera.video_stream.publish(rgb_frame(16, 12, frame_id=frame_id))
    await camera.depth_stream.publish(depth_frame(1.0, 16, 12, frame_id=frame_id))


async def result_of(fixture, h, request):
    goal = await fixture.send_goal(h, request, QoSProfile.Standard, 5.0)
    assert goal.accepted, goal.reason
    result = await goal.get_result(WAIT)
    assert result.data is not None
    return result.data


def grab_request(gripper_name: str, item_id: str = "", position=(0.5, 0.1, 0.7)):
    return grab_action.GoalRequestData(gripper_name=gripper_name, item_id=item_id, position=list(position), orientation=QUAT, max_effort=0.0)


async def scanned(h):
    """The first scan that succeeds. The frames and the camera's answers
    reach the brain after its goal loops are up, so the first scans can be
    refused as having no source yet; the wait is bounded by WAIT."""
    end = time.monotonic() + WAIT
    while True:
        data = await result_of(scan_fx, h, scan_action.GoalRequestData(timeout_s=0.0))
        if data.success or time.monotonic() > end:
            return data


async def test_the_node_answers_every_member_with_no_backends():
    async with harness.start(setup, parameters=params()) as h:
        answer_the_camera(h)
        state = await get_state_fx.poll(h, 5.0)
        assert state.gripper_names == ["left_gripper", "right_gripper"]
        assert state.holding == [False, False]
        assert state.current_action == ""

        grabbed = await result_of(grab_fx, h, grab_request("left_gripper"))
        assert grabbed.success is False
        assert grabbed.message == "no manipulation backend: manipulation_backend is 'none'"
        assert grabbed.gripper_name == "" and grabbed.item_id == ""

        dropped = await result_of(drop_fx, h, drop_action.GoalRequestData(gripper_name=""))
        assert dropped.success is False and dropped.message == "no gripper holds an item"
        placed = await result_of(place_fx, h, place_action.GoalRequestData(gripper_name="", position=[0.5, 0.1, 0.7], orientation=QUAT))
        assert placed.success is False and placed.message == "no gripper holds an item"
        assert placed.final_position == [0.0, 0.0, 0.0]

        scan = await result_of(scan_fx, h, scan_action.GoalRequestData(timeout_s=0.0))
        assert scan.success is False
        assert scan.message.startswith("no perception source")
        assert scan.item_ids == []
        found = await result_of(identify_fx, h, identify_action.GoalRequestData(description="cup", timeout_s=0.0))
        assert found.success is False and found.message.startswith("no perception source")
        assert found.item_id == "" and found.position == [0.0, 0.0, 0.0]

        aborted = await result_of(abort_fx, h, abort_action.GoalRequestData(reason=""))
        assert aborted.success is True
        assert aborted.aborted_action == ""
        assert aborted.message == "no sequence was running"


async def test_every_member_completes_over_the_wire_with_backends():
    detector = FakeDetector([Box("cup", 0.9, 7, 5, 9, 7)])
    manipulator = FakeManipulator()
    async with harness.start(partial(setup, detector=detector, manipulator=manipulator), parameters=params()) as h:
        answer_the_camera(h)
        await publish_a_capture(h)
        scan = await scanned(h)
        assert scan.success is True, scan.message
        assert scan.labels == ["cup"] and scan.confidences == [0.9] and len(scan.positions) == 3
        [cup] = scan.item_ids

        found = await result_of(identify_fx, h, identify_action.GoalRequestData(description="the cup", timeout_s=0.0))
        assert found.success is True, found.message
        assert (found.item_id, found.label, found.confidence) == (cup, "cup", 0.9)
        assert found.position == scan.positions and found.orientation is None

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


async def test_a_callers_cancel_over_the_wire_completes_the_goal_as_cancelled():
    manipulator = FakeManipulator(wait_for_stop=True)
    async with harness.start(partial(setup, manipulator=manipulator), parameters=params()) as h:
        answer_the_camera(h)
        goal = await grab_fx.send_goal(h, grab_request("left_gripper"), QoSProfile.Standard, 5.0)
        assert goal.accepted, goal.reason
        await asyncio.wait_for(manipulator.begun(), WAIT)
        state = await get_state_fx.poll(h, 5.0)
        assert state.current_action == "grab_item"
        await goal.cancel_goal(5.0)
        result = await goal.get_result(WAIT)
        assert result.status == grab_fx.ResultStatus.CANCELLED
        assert result.data.success is False and result.data.message == "cancelled by the caller"
        assert result.data.gripper_name == "" and result.data.item_id == ""
        assert manipulator.stopped == 1
        state = await get_state_fx.poll(h, 5.0)
        assert state.current_action == "" and state.holding == [False, False]


async def test_an_abort_over_the_wire_stops_the_running_sequence():
    manipulator = FakeManipulator(wait_for_stop=True)
    async with harness.start(partial(setup, manipulator=manipulator), parameters=params()) as h:
        answer_the_camera(h)
        goal = await grab_fx.send_goal(h, grab_request("left_gripper"), QoSProfile.Standard, 5.0)
        assert goal.accepted, goal.reason
        await asyncio.wait_for(manipulator.begun(), WAIT)
        second = await result_of(grab_fx, h, grab_request("right_gripper", position=(0.5, -0.1, 0.7)))
        assert second.success is False and second.message == "grab_item is running"
        aborted = await result_of(abort_fx, h, abort_action.GoalRequestData(reason="operator pressed stop"))
        assert aborted.success is True and aborted.aborted_action == "grab_item"
        result = await goal.get_result(WAIT)
        assert result.status == grab_fx.ResultStatus.COMPLETED
        assert result.data.success is False and result.data.message == "aborted: operator pressed stop"
        assert manipulator.stopped == 1


async def test_the_camera_is_asked_again_until_it_knows_its_geometry(monkeypatch):
    # A sim relay answers success false until its simulation has spoken;
    # the brain asks again and the first scan after the answer succeeds.
    monkeypatch.setattr(geometry, "INTRINSICS_RETRY_S", 0.0)
    detector = FakeDetector([Box("cup", 0.9, 7, 5, 9, 7)])
    async with harness.start(partial(setup, detector=detector), parameters=params()) as h:
        not_yet = colour_intrinsics(success=False, message="no camera geometry received from the simulation yet")
        answer_the_camera(h, colour=(not_yet, colour_intrinsics()))
        await publish_a_capture(h)
        scan = await scanned(h)
        assert scan.success is True, scan.message
        assert scan.labels == ["cup"]
        assert h.mocks.deps.geometry.get_color_intrinsics.captured_count() == 2
        assert h.mocks.deps.geometry.get_depth_intrinsics.captured_count() == 2
