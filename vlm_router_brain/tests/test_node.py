"""The real node under the generated harness: goals and services over the
wire, the camera answered by its mocks, and the models behind a scripted
provider, so no network and no key is used. The start checks the key
before any provider is involved, so each test sets a key of its own."""

import asyncio
import logging
from functools import partial

import pytest
from conftest import FakeProvider, answer, records_of
from peppygen.consumed_actions.limb_motion import execute_joint_path, move_gripper
from peppygen.consumed_actions.motion_planning import plan_cartesian_path, plan_joint_path
from peppygen.consumed_services.motion_planning import describe_arms, get_planning_scene
from peppygen.exposed_actions.brain_model import switch_model as switch_action
from peppygen.exposed_actions.item_manipulation import abort as abort_action
from peppygen.exposed_actions.item_manipulation import grab_item as grab_action
from peppygen.exposed_actions.item_perception import identify_item as identify_action
from peppygen.exposed_actions.item_perception import scan_items as scan_action
from peppygen.fixtures import harness
from peppygen.fixtures.exposed_actions.brain_model import switch_model as switch_fx
from peppygen.fixtures.exposed_actions.item_manipulation import abort as abort_fx
from peppygen.fixtures.exposed_actions.item_manipulation import grab_item as grab_fx
from peppygen.fixtures.exposed_actions.item_perception import identify_item as identify_fx
from peppygen.fixtures.exposed_actions.item_perception import scan_items as scan_fx
from peppygen.fixtures.exposed_services.brain_model import get_model as get_model_fx
from peppygen.fixtures.exposed_services.brain_model import list_models as list_models_fx
from peppygen.fixtures.exposed_services.item_manipulation import get_state as get_state_fx
from wire import (
    ATTEMPTS,
    CAMERAS,
    ROBOT_FRAME,
    UNREAD_TARGET,
    WAIT,
    accepted_grip,
    answer_the_camera,
    failure_of,
    feedback_of,
    grab_by_pose,
    land_move,
    limbs_of,
    once_the_brain_has_limbs,
    once_the_brain_sees,
    params,
    plan,
    publish_a_capture,
    publish_limbs,
    publish_limbs_until,
    publish_wrist_until,
    refusal_of,
    result_of,
    stop_grip_at,
)

from vlm_router_brain.__main__ import setup
from vlm_router_brain.manipulators.vlm.step_model import PROBE_PROMPT as ALIGNMENT_PROBE_PROMPT
from vlm_router_brain.provider import ProviderFailure, Reply

ER = "gemini-robotics-er-2-preview"
OPUS = "claude-opus-5.5"
CUP = ("cup", [250, 250, 750, 750])


@pytest.fixture(autouse=True)
def keys(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")


def identify(description: str = "the cup", timeout_s: float = 0.0):
    return identify_action.GoalRequestData(description=description, timeout_s=timeout_s)


def scan(timeout_s: float = 0.0):
    return scan_action.GoalRequestData(timeout_s=timeout_s)


def switch(role: str, model: str):
    return switch_action.GoalRequestData(role=role, model=model)


async def test_identify_then_scan_succeed_and_the_start_made_no_call():
    provider = FakeProvider(answer(("the cup", [250, 250, 750, 750])), answer(CUP))
    async with harness.start(partial(setup, provider=provider), parameters=params()) as h:
        assert provider.calls == []
        answer_the_camera(h)
        await publish_a_capture(h)
        result = await once_the_brain_sees(identify_fx, h, identify())
        assert result.status == identify_fx.ResultStatus.COMPLETED, result.failure
        found = result.data
        assert (found.label, found.confidence, found.camera) == ("the cup", 0.0, "chest")
        # The point the chest camera saw, in the robot frame at the capture
        # time of its picture; no pose: the model resolves no turn.
        point = found.item_point
        assert (point.frame, point.timestamp, [round(v, 6) for v in point.position]) == (ROBOT_FRAME, 1.0, [0.0, 0.0, 1.0])
        assert found.item_pose is None
        assert found.region == [4.0, 3.0, 12.0, 9.0] and (found.image_width, found.image_height) == (16, 12)
        scanned = await result_of(scan_fx, h, scan())
        assert scanned.labels == ["cup"] and scanned.confidences == [0.0]
        # The identify prompt names the item; the scan asks for everything.
        assert provider.calls[0].prompt.startswith("Find the cup in this image")
        assert provider.calls[1].prompt.startswith("Detect every distinct object")
        assert {call.route.spelled for call in provider.calls} == {f"gemini:{ER}"}


async def test_a_search_with_no_capture_yet_is_refused_with_resource_unavailable():
    provider = FakeProvider()
    async with harness.start(partial(setup, provider=provider), parameters=params()) as h:
        answer_the_camera(h)
        refusal = await refusal_of(identify_fx, h, identify())
        assert refusal.kind is identify_fx.FailureKind.RESOURCE_UNAVAILABLE
        assert refusal.message.startswith("no perception source: ")
        assert provider.calls == []


async def test_a_provider_error_fails_that_search_alone_and_the_next_search_asks_again(caplog):
    failure = ProviderFailure(ER, "ProviderError", "overloaded", retryable=False)
    provider = FakeProvider(failure, answer(("the cup", [250, 250, 750, 750])))
    async with harness.start(partial(setup, provider=provider), parameters=params()) as h:
        answer_the_camera(h)
        await publish_a_capture(h)
        with records_of(caplog, "brain_core.brain") as log:
            failed = await once_the_brain_sees(identify_fx, h, identify())
        assert failed.status == identify_fx.ResultStatus.FAILED and failed.data is None
        assert failed.failure == identify_fx.Failure("attempt_failed", f"{ER} failed: ProviderError: overloaded")
        [line] = [record for record in log.records if record.levelno >= logging.WARNING]
        assert line.levelno == logging.WARNING and line.exc_info is None
        found = await result_of(identify_fx, h, identify())
        assert found.label == "the cup"
        assert len(provider.calls) == 2


async def test_an_unreadable_answer_and_a_spent_budget_each_fail_the_search_naming_why(caplog):
    provider = FakeProvider(Reply("I see a cup."))
    async with harness.start(partial(setup, provider=provider), parameters=params()) as h:
        answer_the_camera(h)
        await publish_a_capture(h)
        with records_of(caplog, "brain_core.brain") as log:
            unreadable = await once_the_brain_sees(identify_fx, h, identify())
        assert unreadable.failure == identify_fx.Failure("attempt_failed", "the model's answer could not be read")
        assert all(record.exc_info is None for record in log.records)
        spent = await failure_of(identify_fx, h, identify(timeout_s=1e-9))
        assert spent == identify_fx.Failure("timeout", "the search did not finish within 1e-09 s")
        assert len(provider.calls) == 1


async def test_a_search_that_finds_nothing_fails_with_not_found():
    provider = FakeProvider(answer(("the bowl", [250, 250, 750, 750])), answer())
    async with harness.start(partial(setup, provider=provider), parameters=params()) as h:
        answer_the_camera(h)
        await publish_a_capture(h)
        # The model names what it was asked for; a scan's labels are its own.
        found = await once_the_brain_sees(identify_fx, h, identify("the bowl"))
        assert found.status == identify_fx.ResultStatus.COMPLETED, found.failure
        nothing = await failure_of(identify_fx, h, identify("the cup"))
        assert nothing == identify_fx.Failure("not_found", "no item matches 'the cup'")


async def test_get_model_and_list_models_name_the_catalog_models_never_a_provider():
    async with harness.start(partial(setup, provider=FakeProvider()), parameters=params()) as h:
        answer_the_camera(h)
        current = await get_model_fx.poll(h, 5.0)
        assert (current.roles, current.models) == (["perception", "manipulation"], [ER, "scripted"])
        # No frame came yet, nor the limbs of the robot: both roles still start.
        assert current.starting == [
            "no perception source: no camera frame received",
            "no limb_states snapshot received from the robot yet",
        ]
        offered = await list_models_fx.poll(h, 5.0)
        catalog = [ER, "gemini-4-argon", OPUS, "gpt-6-astra"]
        assert offered.roles == ["perception"] * 5 + ["manipulation"] * 6
        assert offered.models == ["none", *catalog, "none", "scripted", *catalog]
        assert not any(":" in name or "/" in name for name in current.models + offered.models)


async def test_a_switch_probes_once_and_get_model_then_reports_the_new_model():
    provider = FakeProvider(Reply('{"boxes": []}'), answer(("the cup", [250, 250, 750, 750])))
    async with harness.start(partial(setup, provider=provider), parameters=params()) as h:
        answer_the_camera(h)
        await publish_a_capture(h)
        switched = await result_of(switch_fx, h, switch("perception", OPUS))
        assert (switched.role, switched.model) == ("perception", OPUS)
        assert (await get_model_fx.poll(h, 5.0)).models == [OPUS, "scripted"]
        found = await once_the_brain_sees(identify_fx, h, identify())
        assert found.status == identify_fx.ResultStatus.COMPLETED, found.failure
        assert [call.route.spelled for call in provider.calls] == ["openrouter:anthropic/claude-opus-5.5"] * 2


async def test_a_switch_whose_probe_fails_fails_with_attempt_failed_and_keeps_the_model():
    failure = ProviderFailure(OPUS, "ModelNotFoundError", "no such model", retryable=False)
    async with harness.start(partial(setup, provider=FakeProvider(failure)), parameters=params()) as h:
        answer_the_camera(h)
        failed = await failure_of(switch_fx, h, switch("perception", OPUS))
        assert failed == switch_fx.Failure("attempt_failed", f"{OPUS} failed: ModelNotFoundError: no such model")
        assert (await get_model_fx.poll(h, 5.0)).models == [ER, "scripted"]


async def test_a_switch_to_a_model_with_no_key_fails_with_resource_unavailable(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY")
    provider = FakeProvider()
    async with harness.start(partial(setup, provider=provider), parameters=params()) as h:
        answer_the_camera(h)
        failed = await failure_of(switch_fx, h, switch("perception", OPUS))
        assert failed.kind is switch_fx.FailureKind.RESOURCE_UNAVAILABLE
        assert failed.message.startswith(f"no key reaches '{OPUS}'")
        assert provider.calls == []


async def test_perception_switches_to_none_and_back():
    async with harness.start(partial(setup, provider=FakeProvider()), parameters=params()) as h:
        answer_the_camera(h)
        await publish_a_capture(h)
        off = await result_of(switch_fx, h, switch("perception", "none"))
        assert off.model == "none"
        # A role whose model is "none" refuses its goals with
        # precondition_failed, as brain_model says.
        refused = await refusal_of(identify_fx, h, identify())
        assert refused == identify_fx.Failure("precondition_failed", "no perception source: perception_model is 'none'")
        back = await result_of(switch_fx, h, switch("perception", ER))
        assert back.model == ER
        assert (await get_model_fx.poll(h, 5.0)).models == [ER, "scripted"]


async def test_manipulation_switches_to_none_which_refuses_every_grab_to_a_catalog_model_and_back_to_scripted():
    provider = FakeProvider(Reply('{"action": "give_up"}'))
    async with harness.start(partial(setup, provider=provider), parameters=params()) as h:
        answer_the_camera(h)
        kept = await result_of(switch_fx, h, switch("manipulation", "none"))
        assert (kept.role, kept.model) == ("manipulation", "none")
        refused = await refusal_of(switch_fx, h, switch("manipulation", "fast"))
        assert refused == switch_fx.Failure("rejected", "'fast' is not a manipulation model here; see list_models")
        # A catalog model aligns the grab: the switch probes it with the
        # alignment's question.
        aligning = await result_of(switch_fx, h, switch("manipulation", OPUS))
        assert (aligning.role, aligning.model) == ("manipulation", OPUS)
        [probe_call] = provider.calls
        assert probe_call.prompt == ALIGNMENT_PROBE_PROMPT and len(probe_call.images) == 1
        assert (await get_model_fx.poll(h, 5.0)).models == [ER, OPUS]
        kept = await result_of(switch_fx, h, switch("manipulation", "none"))
        assert (kept.role, kept.model) == ("manipulation", "none")
        unknown = await refusal_of(switch_fx, h, switch("arm", "none"))
        assert unknown == switch_fx.Failure("rejected", "this brain runs no model for the role 'arm'")
        # item_manipulation reports in its result.
        grabbed = await result_of(
            grab_fx, h,
            grab_by_pose([0.5, 0.1, 0.7], [0.0, 0.0, 0.0, 1.0]),
        )
        assert (grabbed.success, grabbed.message) == (False, "no manipulation model: manipulation_model is 'none'")
        back = await result_of(switch_fx, h, switch("manipulation", "scripted"))
        assert (back.role, back.model) == ("manipulation", "scripted")
        assert (await get_model_fx.poll(h, 5.0)).models == [ER, "scripted"]


async def test_a_switch_and_a_search_are_refused_while_a_search_of_the_role_runs():
    provider = FakeProvider(answer(("the cup", [250, 250, 750, 750])), answer(("the cup", [250, 250, 750, 750])))
    async with harness.start(partial(setup, provider=provider), parameters=params()) as h:
        answer_the_camera(h)
        await publish_a_capture(h)
        # A first search, so the brain has the capture before the gate.
        await once_the_brain_sees(identify_fx, h, identify())
        provider.gate = asyncio.Event()
        provider.called.clear()
        search = await identify_fx.send_goal(h, identify(), 5.0)
        assert search.accepted, search.refusal
        await asyncio.wait_for(provider.called.wait(), WAIT)
        refused = await refusal_of(switch_fx, h, switch("perception", OPUS))
        assert refused == switch_fx.Failure("precondition_failed", "identify_item is running")
        second = await refusal_of(scan_fx, h, scan())
        assert second == scan_fx.Failure("precondition_failed", "identify_item is running")
        provider.gate.set()
        result = await search.get_result(WAIT)
        assert result.status == identify_fx.ResultStatus.COMPLETED, result.failure
        assert (await get_model_fx.poll(h, 5.0)).models == [ER, "scripted"]


async def test_a_second_switch_is_refused_and_a_cancelled_one_keeps_the_model():
    provider = FakeProvider()
    provider.gate = asyncio.Event()
    async with harness.start(partial(setup, provider=provider), parameters=params()) as h:
        answer_the_camera(h)
        first = await switch_fx.send_goal(h, switch("perception", OPUS), 5.0)
        assert first.accepted, first.refusal
        await asyncio.wait_for(provider.called.wait(), WAIT)
        second = await refusal_of(switch_fx, h, switch("perception", "gpt-6-astra"))
        assert second == switch_fx.Failure("precondition_failed", "switch_model is running for perception")
        await first.cancel_goal(5.0)
        result = await first.get_result(WAIT)
        assert result.status == switch_fx.ResultStatus.CANCELLED
        assert result.cancel_reason == "cancelled by its caller"
        assert (result.data.role, result.data.model) == ("", "")
        assert (await get_model_fx.poll(h, 5.0)).models == [ER, "scripted"]


# The scripted grab over the wire: the limb_motion mock plans every check
# the test scripts and lands every move, and the limb_state mock publishes
# the robot's limbs.


async def test_identify_then_a_scripted_grab_holds_the_item():
    provider = FakeProvider(answer(("the cup", [250, 250, 750, 750])))
    # No motion planner: the grab moves by the backbone's lines.
    async with harness.start(partial(setup, provider=provider), parameters=params(), motion_planning_vacant=True) as h:
        answer_the_camera(h)
        await publish_a_capture(h)
        seen = await once_the_brain_sees(identify_fx, h, identify())
        assert seen.status == identify_fx.ResultStatus.COMPLETED, seen.failure
        found = seen.data
        # No limb_states snapshot yet: no measured opening.
        assert (await get_state_fx.poll(h, 5.0)).gripper_openings == []
        await once_the_brain_has_limbs(h)
        plan(h, True, True, True)
        grab = await grab_fx.send_goal(
            h, grab_action.GoalRequestData(gripper_name="", item_id=found.item_id, target=UNREAD_TARGET, max_effort=15.0),
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
        # The generated goal context publishes each stage on the feedback
        # topic: a grab with no aligner plans, moves above the item, and
        # grasps.
        reported = [(fed.stage, fed.activity, fed.number, fed.maximum) for fed in await feedback_of(grab)]
        assert reported == [
            ("plan", "The robot opens its hand for the cup and finds a way to reach it", 0, 0),
            ("move_above", "The robot moves its hand above the cup", 0, 0),
            ("grasp", "The robot lowers its hand, closes it on the cup and lifts it", 0, 0),
        ]
        state = await get_state_fx.poll(h, 5.0)
        assert (state.holding, state.held_item_ids) == ([True, False], [found.item_id, ""])
        # The last snapshot the backbone published: the left jaws on the cup.
        assert state.gripper_openings == [0.3, 1.0]


async def test_a_grab_target_is_placed_through_transform_lookup_and_one_the_tree_does_not_hold_is_refused_with_unknown_frame():
    async with harness.start(partial(setup, provider=FakeProvider()), parameters=params(), motion_planning_vacant=True) as h:
        answer_the_camera(h)
        # The brain places a target once it knows the robot frame, which
        # describe_sensors names after its goal loops are up; a goal before
        # that is accepted, and its result says why it was not placed.
        for _ in range(ATTEMPTS):
            goal = await grab_fx.send_goal(h, grab_by_pose([0.5, 0.1, 0.75], frame="odom", timestamp=2.5), 5.0)
            if not goal.accepted:
                break
            early = (await goal.get_result(WAIT)).data
            assert (early.success, early.message) == (False, "the robot's sensors have not been described yet")
        assert goal.refusal == grab_fx.Failure("unknown_frame", "the target: no transform names odom")
        # Where odom stood at the target's time, in the robot frame at the
        # latest time.
        request = h.mocks.deps.transform_lookup.lookup_transform.captured()[-1]
        assert (request.target_frame, request.source_frame, request.source_timestamp, request.target_timestamp) == (
            ROBOT_FRAME, "odom", 2.5, None,
        )


async def test_a_grab_target_with_no_frame_is_placed_from_the_robot_frame_through_transform_lookup():
    async with harness.start(partial(setup, provider=FakeProvider()), parameters=params(), motion_planning_vacant=True) as h:
        answer_the_camera(h)
        await once_the_brain_has_limbs(h)
        lookups = h.mocks.deps.transform_lookup.lookup_transform
        # The brain places a target once it knows the robot frame; a goal
        # before that is accepted, and its result says why it was not placed.
        for _ in range(ATTEMPTS):
            grab = await grab_fx.send_goal(h, grab_by_pose([0.5, 0.1, 0.75], frame=None, timestamp=2.5), 5.0)
            assert grab.accepted, grab.refusal
            placed = [request for request in lookups.captured() if request.source_timestamp == 2.5]
            if placed:
                break
            early = (await grab.get_result(WAIT)).data
            assert (early.success, early.message) == (False, "the robot's sensors have not been described yet")
        # Where the robot frame stood at the target's time, in the robot
        # frame at the latest time.
        [request] = placed
        assert (request.target_frame, request.source_frame, request.target_timestamp) == (ROBOT_FRAME, ROBOT_FRAME, None)
        _, opening = await accepted_grip(h)
        stop = await abort_fx.send_goal(h, abort_action.GoalRequestData(reason="operator"), 5.0)
        assert stop.accepted, stop.refusal
        await asyncio.wait_for(opening.cancel_signal(), WAIT)
        await opening.complete_cancelled(
            "cancelled by its caller", move_gripper.ResultResponseData(settled=False, final_opening=0.6, action_time=0.2)
        )
        result = (await grab.get_result(WAIT)).data
        assert (result.success, result.message) == (False, "aborted: operator")


async def test_a_gripper_that_closes_on_nothing_retreats_fails_the_grab_and_holds_nothing():
    async with harness.start(partial(setup, provider=FakeProvider()), parameters=params(), motion_planning_vacant=True) as h:
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


async def test_a_grab_during_a_grab_is_refused_and_abort_ends_the_first():
    async with harness.start(partial(setup, provider=FakeProvider()), parameters=params(), motion_planning_vacant=True) as h:
        answer_the_camera(h)
        await once_the_brain_has_limbs(h)
        first = await grab_fx.send_goal(h, grab_by_pose([0.5, 0.1, 0.75]), 5.0)
        assert first.accepted, first.refusal
        # The first grab opens the gripper before its checks; the backbone
        # holds that move.
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


async def test_identify_then_an_aligned_grab_moves_turns_and_holds_the_item(caplog):
    from brain_core.manipulators.scripted.grasp import jaw_line_deg

    provider = FakeProvider(
        answer(("the cup", [250, 250, 750, 750])),
        Reply('{"action": "give_up"}'),
        Reply('{"action": "move", "dx": 1.0, "dy": -2.0, "roll_deg": 45}'),
        Reply('{"action": "ready"}', 2800, 40, 30, 9.4),
    )
    async with harness.start(
        partial(setup, provider=provider), parameters=params(), wrist_cameras_instance_ids=["wrist_left"], motion_planning_vacant=True,
    ) as h:
        answer_the_camera(h)
        await publish_a_capture(h)
        seen = await once_the_brain_sees(identify_fx, h, identify())
        assert seen.status == identify_fx.ResultStatus.COMPLETED, seen.failure
        found = seen.data
        await once_the_brain_has_limbs(h)
        switched = await result_of(switch_fx, h, switch("manipulation", OPUS))
        assert (switched.role, switched.model) == ("manipulation", OPUS)
        done = asyncio.Event()
        wrist = asyncio.create_task(publish_wrist_until(h, done))
        plan(h, True, True, True)
        with records_of(caplog, "brain_core.manipulators.scripted.manipulator") as log:
            grab = await grab_fx.send_goal(
                h, grab_action.GoalRequestData(gripper_name="", item_id=found.item_id, target=UNREAD_TARGET, max_effort=15.0),
                5.0,
            )
            assert grab.accepted, grab.reason
            assert (await stop_grip_at(h, 1.0)).opening == 1.0
            pre = await land_move(h)
            assert [round(v, 6) for v in pre.position] == [0.0, 0.0, 1.05]
            # Step 1: the model moves 1 cm in x, -2 cm in y and turns 45
            # degrees; the three points are checked, then the move is sent at
            # the same height, the jaw line turned from 90 to 135 degrees.
            plan(h, True, True, True)
            aligned = await land_move(h)
            assert [round(v, 6) for v in aligned.position] == [0.01, -0.02, 1.05]
            assert jaw_line_deg(aligned.orientation) == pytest.approx(135.0, abs=0.01)
            # Step 2: ready. The descent, the close and the lift use the
            # aligned points.
            reach = await land_move(h)
            assert [round(v, 6) for v in reach.position] == [0.01, -0.02, 0.975] and reach.orientation == aligned.orientation
            closed = await stop_grip_at(h, 0.3)
            assert closed.max_effort == 15.0
            limbs_done = asyncio.Event()
            stream = asyncio.create_task(publish_limbs_until(h, limbs_done, 0.3))
            lift = await land_move(h)
            assert [round(v, 6) for v in lift.position] == [0.01, -0.02, 1.05]
            result = (await grab.get_result(WAIT)).data
            limbs_done.set()
            done.set()
            await asyncio.gather(stream, wrist)
        captured = h.mocks.deps.transform_lookup.lookup_transform.captured()
        # Each picture's frame is placed in the robot frame at the picture's
        # time, both times the same.
        assert all(request.source_timestamp == request.target_timestamp for request in captured)
        lookups = [(request.source_frame, request.source_timestamp) for request in captured]
        assert (result.success, result.gripper_name, result.item_id) == (True, "left_gripper", found.item_id), result.message
        line = next(record.getMessage() for record in log.records if "grab of item" in record.getMessage())
        assert "aligned in 2 steps: the grasp point moved 2.2 cm, the jaws close along 135 degrees" in line
        # The identify, the probe and the two steps, each with two pictures.
        assert [len(call.images) for call in provider.calls] == [1, 1, 2, 2]
        step = provider.calls[2]
        assert "Now: step 1 of 5." in step.prompt and "+x points toward 6 o'clock and +y toward 3 o'clock" in step.prompt
        assert step.schema["json_schema"]["strict"] is False
        # The identify placed the frame of the chest picture at its capture's
        # time; each step placed the frame of the wrist's new picture at its
        # time, then the frame of the chest picture at its capture's time
        # again.
        chest, wrist_left = (CAMERAS[name][0] for name in ("chest", "wrist_left"))
        assert [frame for frame, _ in lookups] == [chest, wrist_left, chest, wrist_left, chest]
        assert [time for frame, time in lookups if frame == chest] == [1.0, 1.0, 1.0]
        first_step, second_step = (time for frame, time in lookups if frame == wrist_left)
        assert 0.0 < first_step < second_step


def described_left_arm():
    """describe_arms of the OpenArm's left arm, as the motion planner names it."""
    return describe_arms.ResponseData(
        arms=[
            describe_arms.ResponseDataArmsItem(
                name="left_arm", gripper_name="left_gripper", joint_names=[f"openarm_left_joint{i}" for i in range(1, 8)],
                joint_lower_limits=[-3.0] * 7, joint_upper_limits=[3.0] * 7, link_names=[f"openarm_left_link{i}" for i in range(8)],
                grasp_link="openarm_left_tcp", approach_axis="+z", closing_axis="+y", gripper_links=LEFT_GRIPPER_LINKS,
            )
        ],
        body_link_names=["openarm_body_link0"],
    )


def a_scene(*objects):
    return get_planning_scene.ResponseData(
        frame=ROBOT_FRAME, timestamp=1.0, objects=list(objects), attached_objects=[], allowed_pairs=[], clearance_m=0.04, voxel_size_m=0.02,
    )


LEFT_GRIPPER_LINKS = ["openarm_left_ee_base_link", "openarm_left_ee_link1", "openarm_left_ee_link2"]
START = [0.0] * 7
AT_PRE = [0.1, -0.2, 0.3, 1.2, 0.4, -0.5, 0.6]


async def test_with_the_planner_bound_a_scripted_grab_plans_its_transit_and_the_held_item_is_attached():
    """The router's default manipulation model with the motion planner
    bound: the item's object before the plan, the plan of the transit, the
    reach test of the descent and the lift, the run of the plan through the
    backbone, the checked descent of stage 1, the lines down and up, then the
    object moved with the hand and attached."""
    provider = FakeProvider(answer(("the cup", [250, 250, 750, 750])))
    async with harness.start(partial(setup, provider=provider), parameters=params()) as h:
        answer_the_camera(h)
        await publish_a_capture(h)
        found = (await once_the_brain_sees(identify_fx, h, identify())).data
        await once_the_brain_has_limbs(h)
        planner = h.mocks.deps.motion_planning
        object_id = f"brain/{found.item_id}"
        # The scene at the start of the grab, the arms of its transit, and
        # the scene at the hold, the cup's object where the grab added it.
        for _ in range(3):
            planner.describe_arms.enqueue_response(described_left_arm())
        planner.get_planning_scene.enqueue_response(a_scene())
        planner.get_planning_scene.enqueue_response(a_scene(get_planning_scene.ResponseDataObjectsItem(
            id=object_id, position=list(found.item_point.position), orientation=[0.0, 0.0, 0.0, 1.0], shapes=[],
        )))
        for _ in range(2):
            planner.apply_collision_objects.enqueue_response()
        planner.set_allowed_collision.enqueue_response()
        planner.apply_attached_collision_object.enqueue_response()
        grab = await grab_fx.send_goal(
            h, grab_action.GoalRequestData(gripper_name="", item_id=found.item_id, target=UNREAD_TARGET, max_effort=15.0),
            5.0,
        )
        assert grab.accepted, grab.refusal
        assert (await stop_grip_at(h, 1.0)).opening == 1.0
        down = [0.0, 1.0, 0.0, 0.0]
        pending = await planner.plan_joint_path.next_goal(WAIT)
        planned = pending.request
        [position], [turn] = planned.goal_position_constraints, planned.goal_orientation_constraints
        assert [round(v, 6) for v in position.position] == [0.0, 0.0, 1.05]
        assert position.link_name == "openarm_left_tcp" and turn.orientation == down
        # Each constraint leaves its frame out, for the robot frame, and its
        # timestamp, for the latest time.
        assert {(c.frame, c.timestamp) for c in (position, turn)} == {(None, None)}
        assert (planned.planner, planned.refiner, planned.allowed_planning_time_s) == ("rrt_connect", "none", 0.5)
        await (await pending.accept()).complete(plan_joint_path.ResultResponseData(
            frame=ROBOT_FRAME,
            arms=[plan_joint_path.ResultResponseDataArmsItem(
                arm_name="left_arm", start_joint_positions=START,
                waypoints=[plan_joint_path.ResultResponseDataArmsItemWaypointsItem(joint_positions=AT_PRE, position=[0.0, 0.0, 1.05], orientation=down)],
            )],
            map_stamp=None,
        ))
        pending = await planner.plan_cartesian_path.next_goal(WAIT)
        reach = pending.request
        assert (reach.avoid_collisions, reach.start_joint_positions, reach.frame, reach.timestamp) == (False, AT_PRE, None, None)
        assert [[round(v, 6) for v in w.position] for w in reach.waypoints] == [[0.0, 0.0, 0.975], [0.0, 0.0, 1.05]]
        await (await pending.accept()).complete(plan_cartesian_path.ResultResponseData(
            frame=ROBOT_FRAME, start_joint_positions=AT_PRE, waypoints=[], fraction=1.0, map_stamp=None,
        ))
        pending = await h.mocks.deps.limb_motion.execute_joint_path.next_goal(WAIT)
        [run] = pending.request.arms
        assert pending.request.contact_gap_m == 0.0
        assert (run.arm_name, run.expected_start_joint_positions, [w.joint_positions for w in run.waypoints]) == ("left_arm", START, [AT_PRE])
        limbs = limbs_of(h)
        limbs.left_position, limbs.left_orientation = [0.0, 0.0, 1.05], down
        limbs.stamp += 1.0
        await publish_limbs(h)
        await (await pending.accept()).complete(execute_joint_path.ResultResponseData(
            arms=[execute_joint_path.ResultResponseDataArmsItem(arm_name="left_arm", final_joint_positions=AT_PRE, ended_on_contact=False)],
            action_time=2.0,
        ))
        pending = await planner.plan_cartesian_path.next_goal(WAIT)
        checked = pending.request
        assert (checked.avoid_collisions, checked.start_joint_positions) == (True, [])
        await (await pending.accept()).complete(plan_cartesian_path.ResultResponseData(
            frame=ROBOT_FRAME, start_joint_positions=AT_PRE, waypoints=[], fraction=0.38, map_stamp=None,
        ))
        assert [round(v, 6) for v in (await land_move(h)).position] == [0.0, 0.0, 0.975]
        await stop_grip_at(h, 0.3)
        done = asyncio.Event()
        stream = asyncio.create_task(publish_limbs_until(h, done, 0.3))
        assert [round(v, 6) for v in (await land_move(h)).position] == [0.0, 0.0, 1.05]
        result = (await grab.get_result(WAIT)).data
        done.set()
        await stream
        assert (result.success, result.item_id) == (True, found.item_id), result.message
        [added, moved] = [request.objects[0] for request in planner.apply_collision_objects.captured()]
        assert (added.id, added.operation, added.shapes[0].kind, added.shapes[0].dimensions) == (object_id, "add", "sphere", [0.05])
        seen = found.item_point.position
        assert (added.frame, added.timestamp) == (None, None)
        assert [round(v, 6) for v in added.position] == [round(v, 6) for v in seen]
        [allowed] = planner.set_allowed_collision.captured()
        assert (allowed.name, allowed.other_names, allowed.allowed) == (object_id, LEFT_GRIPPER_LINKS, True)
        # The object moved by what the grasp point moved since the close.
        assert (moved.id, moved.operation) == (object_id, "move")
        assert [round(v, 6) for v in moved.position] == [round(seen[0], 6), round(seen[1], 6), round(seen[2] + 0.075, 6)]
        [attached] = planner.apply_attached_collision_object.captured()
        assert (attached.object_id, attached.link_name, attached.touch_links, attached.operation) == (object_id, "openarm_left_tcp", LEFT_GRIPPER_LINKS, "attach")
        state = await get_state_fx.poll(h, 5.0)
        assert state.held_item_ids == [found.item_id, ""]
