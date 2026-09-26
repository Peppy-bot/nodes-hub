"""The real node under the generated harness: goals over the wire, the
"none" backends answering, and the camera slot standing empty."""

from conftest import PARAMS
from peppygen import QoSProfile
from peppygen.exposed_actions.item_manipulation import abort as abort_action
from peppygen.exposed_actions.item_manipulation import grab_item as grab_action
from peppygen.exposed_actions.item_perception import scan_items as scan_action
from peppygen.fixtures import harness
from peppygen.fixtures.exposed_actions.item_manipulation import abort as abort_fx
from peppygen.fixtures.exposed_actions.item_manipulation import grab_item as grab_fx
from peppygen.fixtures.exposed_actions.item_perception import scan_items as scan_fx
from peppygen.fixtures.exposed_services.item_manipulation import get_state as get_state_fx
from peppygen.consumed_services.camera import depth_stream_info
from peppygen.consumed_services.geometry import get_color_intrinsics
from peppygen.parameters import Parameters

from openarm_ai_brain_vla.__main__ import setup


async def test_the_node_answers_every_member_with_no_backends():
    params = Parameters.from_dict(dict(PARAMS))
    async with harness.start(setup, parameters=params) as h:
        # The camera slot is bound to the mock, which the frame store asks
        # for its depth unit at start; answer it so the store settles.
        h.mocks.deps.camera.depth_stream_info.enqueue_response(
            depth_stream_info.ResponseData(width=16, height=12, frames_per_second=15, encoding="z16", depth_unit=0.001)
        )
        # And the geometry slot, which the brain asks where the pixels point.
        h.mocks.deps.geometry.get_color_intrinsics.enqueue_response(
            get_color_intrinsics.ResponseData(success=True, message="", width=16, height=12, fx=6.0, fy=6.0, cx=8.0, cy=6.0, distortion_model="none", distortion=[])
        )
        state = await get_state_fx.poll(h, 5.0)
        assert state.gripper_names == ["left_gripper", "right_gripper"]
        assert state.holding == [False, False]
        assert state.current_action == ""

        goal = await grab_fx.send_goal(
            h,
            grab_action.GoalRequestData(gripper_name="left_gripper", item_id="", position=[0.5, 0.1, 0.7], orientation=[0.0, 0.0, 0.0, 1.0], max_effort=0.0),
            QoSProfile.Standard,
            5.0,
        )
        assert goal.accepted, goal.reason
        result = await goal.get_result(10.0)
        assert result.data is not None
        assert result.data.success is False
        assert result.data.message == "no manipulation backend: manipulation_backend is 'none'"
        assert result.data.gripper_name == "" and result.data.item_id == ""

        goal = await scan_fx.send_goal(h, scan_action.GoalRequestData(timeout_s=0.0), QoSProfile.Standard, 5.0)
        result = await goal.get_result(10.0)
        assert result.data.success is False
        assert result.data.message.startswith("no perception source")
        assert result.data.item_ids == []

        goal = await abort_fx.send_goal(h, abort_action.GoalRequestData(reason=""), QoSProfile.Standard, 5.0)
        result = await goal.get_result(10.0)
        assert result.data.success is True
        assert result.data.aborted_action == ""
        assert result.data.message == "no sequence was running"
