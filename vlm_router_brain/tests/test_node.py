"""The real node under the generated harness: goals and services over the
wire, the camera answered by its mocks, and the models behind a scripted
provider, so no network and no key is used. The start checks the key
before any provider is involved, so each test sets a key of its own."""

import asyncio
import logging
from functools import partial

import pytest
from conftest import FakeProvider, answer, records_of
from peppygen import QoSProfile
from peppygen.consumed_actions.limb_motion import move_gripper
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
    WAIT,
    accepted_grip,
    answer_the_camera,
    grab_by_pose,
    land_move,
    once_the_brain_has_limbs,
    once_the_brain_sees,
    params,
    plan,
    publish_a_capture,
    publish_limbs_until,
    result_of,
    stop_grip_at,
)

from vlm_router_brain.__main__ import setup
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
        found = await once_the_brain_sees(identify_fx, h, identify())
        assert found.success is True, found.message
        assert (found.label, found.confidence, found.camera) == ("the cup", 0.0, "chest")
        assert [round(v, 6) for v in found.position] == [0.0, 0.0, 1.0]
        assert found.region == [4.0, 3.0, 12.0, 9.0] and (found.image_width, found.image_height) == (16, 12)
        scanned = await result_of(scan_fx, h, scan())
        assert scanned.success is True, scanned.message
        assert scanned.labels == ["cup"] and scanned.confidences == [0.0]
        # The identify prompt names the item; the scan asks for everything.
        assert provider.calls[0].prompt.startswith("Find the cup in this image")
        assert provider.calls[1].prompt.startswith("Detect every distinct object")
        assert {call.route.spelled for call in provider.calls} == {f"gemini:{ER}"}


async def test_a_provider_error_fails_that_search_alone_and_the_next_search_asks_again(caplog):
    failure = ProviderFailure(ER, "ProviderError", "overloaded", retryable=False)
    provider = FakeProvider(failure, answer(("the cup", [250, 250, 750, 750])))
    async with harness.start(partial(setup, provider=provider), parameters=params()) as h:
        answer_the_camera(h)
        await publish_a_capture(h)
        with records_of(caplog, "brain_core.brain") as log:
            failed = await once_the_brain_sees(identify_fx, h, identify())
        assert failed.success is False
        assert failed.message == f"{ER} failed: ProviderError: overloaded"
        [line] = [record for record in log.records if record.levelno >= logging.WARNING]
        assert line.levelno == logging.WARNING and line.exc_info is None
        found = await result_of(identify_fx, h, identify())
        assert found.success is True, found.message
        assert len(provider.calls) == 2


async def test_an_unreadable_answer_and_a_spent_budget_each_fail_the_search_naming_why(caplog):
    provider = FakeProvider(Reply("I see a cup."))
    async with harness.start(partial(setup, provider=provider), parameters=params()) as h:
        answer_the_camera(h)
        await publish_a_capture(h)
        with records_of(caplog, "brain_core.brain") as log:
            unreadable = await once_the_brain_sees(identify_fx, h, identify())
        assert (unreadable.success, unreadable.message) == (False, "the model's answer could not be read")
        assert all(record.exc_info is None for record in log.records)
        spent = await result_of(identify_fx, h, identify(timeout_s=1e-9))
        assert (spent.success, spent.message) == (False, "the search did not finish within 1e-09 s")
        assert len(provider.calls) == 1


async def test_get_model_and_list_models_name_the_catalog_models_never_a_provider():
    async with harness.start(partial(setup, provider=FakeProvider()), parameters=params()) as h:
        answer_the_camera(h)
        current = await get_model_fx.poll(h, 5.0)
        assert (current.roles, current.models) == (["perception", "manipulation"], [ER, "scripted"])
        offered = await list_models_fx.poll(h, 5.0)
        assert offered.roles == ["perception"] * 5 + ["manipulation"] * 2
        assert offered.models == ["none", ER, "gemini-4-argon", OPUS, "gpt-6-astra", "none", "scripted"]
        assert not any(":" in name or "/" in name for name in current.models + offered.models)


async def test_a_switch_probes_once_and_get_model_then_reports_the_new_model():
    provider = FakeProvider(Reply('{"boxes": []}'), answer(("the cup", [250, 250, 750, 750])))
    async with harness.start(partial(setup, provider=provider), parameters=params()) as h:
        answer_the_camera(h)
        await publish_a_capture(h)
        switched = await result_of(switch_fx, h, switch("perception", OPUS))
        assert (switched.success, switched.role, switched.model) == (True, "perception", OPUS), switched.message
        assert (await get_model_fx.poll(h, 5.0)).models == [OPUS, "scripted"]
        found = await once_the_brain_sees(identify_fx, h, identify())
        assert found.success is True, found.message
        assert [call.route.spelled for call in provider.calls] == ["openrouter:anthropic/claude-opus-5.5"] * 2


async def test_perception_switches_to_none_and_back():
    async with harness.start(partial(setup, provider=FakeProvider()), parameters=params()) as h:
        answer_the_camera(h)
        await publish_a_capture(h)
        off = await result_of(switch_fx, h, switch("perception", "none"))
        assert (off.success, off.model) == (True, "none"), off.message
        refused = await result_of(identify_fx, h, identify())
        assert (refused.success, refused.message) == (False, "no perception source: perception_model is 'none'")
        back = await result_of(switch_fx, h, switch("perception", ER))
        assert (back.success, back.model) == (True, ER), back.message
        assert (await get_model_fx.poll(h, 5.0)).models == [ER, "scripted"]


async def test_manipulation_switches_to_none_which_refuses_every_grab_and_back_to_scripted():
    async with harness.start(partial(setup, provider=FakeProvider()), parameters=params()) as h:
        answer_the_camera(h)
        kept = await result_of(switch_fx, h, switch("manipulation", "none"))
        assert (kept.success, kept.role, kept.model) == (True, "manipulation", "none"), kept.message
        refused = await result_of(switch_fx, h, switch("manipulation", OPUS))
        assert (refused.success, refused.role, refused.model) == (False, "", "")
        assert refused.message == f"'{OPUS}' is not a manipulation model here; see list_models"
        unknown = await result_of(switch_fx, h, switch("arm", "none"))
        assert unknown.message == "this brain runs no model for the role 'arm'"
        grabbed = await result_of(
            grab_fx, h,
            grab_action.GoalRequestData(gripper_name="left_gripper", item_id="", position=[0.5, 0.1, 0.7], orientation=[0.0, 0.0, 0.0, 1.0], max_effort=0.0),
        )
        assert (grabbed.success, grabbed.message) == (False, "no manipulation model: manipulation_model is 'none'")
        back = await result_of(switch_fx, h, switch("manipulation", "scripted"))
        assert (back.success, back.model) == (True, "scripted"), back.message
        assert (await get_model_fx.poll(h, 5.0)).models == [ER, "scripted"]


async def test_a_switch_is_refused_while_a_search_of_its_role_runs():
    provider = FakeProvider(answer(("the cup", [250, 250, 750, 750])), answer(("the cup", [250, 250, 750, 750])))
    async with harness.start(partial(setup, provider=provider), parameters=params()) as h:
        answer_the_camera(h)
        await publish_a_capture(h)
        # A first search, so the brain has the capture before the gate.
        await once_the_brain_sees(identify_fx, h, identify())
        provider.gate = asyncio.Event()
        provider.called.clear()
        search = await identify_fx.send_goal(h, identify(), QoSProfile.Standard, 5.0)
        assert search.accepted, search.reason
        await asyncio.wait_for(provider.called.wait(), WAIT)
        refused = await result_of(switch_fx, h, switch("perception", OPUS))
        assert (refused.success, refused.message) == (False, "identify_item is running")
        provider.gate.set()
        result = await search.get_result(WAIT)
        assert result.data.success is True
        assert (await get_model_fx.poll(h, 5.0)).models == [ER, "scripted"]


async def test_a_second_switch_is_refused_and_a_cancelled_one_keeps_the_model():
    provider = FakeProvider()
    provider.gate = asyncio.Event()
    async with harness.start(partial(setup, provider=provider), parameters=params()) as h:
        answer_the_camera(h)
        first = await switch_fx.send_goal(h, switch("perception", OPUS), QoSProfile.Standard, 5.0)
        assert first.accepted, first.reason
        await asyncio.wait_for(provider.called.wait(), WAIT)
        second = await result_of(switch_fx, h, switch("perception", "gpt-6-astra"))
        assert (second.success, second.message) == (False, "switch_model is running for perception")
        await first.cancel_goal(5.0)
        result = await first.get_result(WAIT)
        assert result.status == switch_fx.ResultStatus.CANCELLED
        assert (result.data.success, result.data.message, result.data.model) == (False, "cancelled by the caller", "")
        assert (await get_model_fx.poll(h, 5.0)).models == [ER, "scripted"]


# The scripted grab over the wire: the limb_motion mock plans every check
# the test scripts and lands every move, and the limb_state mock publishes
# the robot's limbs.


async def test_identify_then_a_scripted_grab_holds_the_item():
    provider = FakeProvider(answer(("the cup", [250, 250, 750, 750])))
    async with harness.start(partial(setup, provider=provider), parameters=params()) as h:
        answer_the_camera(h)
        await publish_a_capture(h)
        found = await once_the_brain_sees(identify_fx, h, identify())
        assert found.success is True, found.message
        # No limb_states snapshot yet: no measured opening.
        assert (await get_state_fx.poll(h, 5.0)).gripper_openings == []
        await once_the_brain_has_limbs(h)
        plan(h, True, True, True)
        grab = await grab_fx.send_goal(
            h, grab_action.GoalRequestData(gripper_name="", item_id=found.item_id, position=[0.0] * 3, orientation=None, max_effort=15.0),
            QoSProfile.Standard, 5.0,
        )
        assert grab.accepted, grab.reason
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
    async with harness.start(partial(setup, provider=FakeProvider()), parameters=params()) as h:
        answer_the_camera(h)
        await once_the_brain_has_limbs(h)
        plan(h, True, True, True)
        grab = await grab_fx.send_goal(h, grab_by_pose([0.5, 0.1, 0.75]), QoSProfile.Standard, 5.0)
        assert grab.accepted, grab.reason
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
    async with harness.start(partial(setup, provider=FakeProvider()), parameters=params()) as h:
        answer_the_camera(h)
        await once_the_brain_has_limbs(h)
        plan(h, True, True, True)
        first = await grab_fx.send_goal(h, grab_by_pose([0.5, 0.1, 0.75]), QoSProfile.Standard, 5.0)
        assert first.accepted, first.reason
        # The first grab opens the gripper; the backbone holds that move.
        _, opening = await accepted_grip(h)
        assert (await get_state_fx.poll(h, 5.0)).current_action == "grab_item"
        second = await result_of(grab_fx, h, grab_by_pose([0.4, -0.1, 0.75]))
        assert (second.success, second.message) == (False, "grab_item is running")
        stop = await abort_fx.send_goal(h, abort_action.GoalRequestData(reason="operator"), QoSProfile.Standard, 5.0)
        assert stop.accepted, stop.reason
        await asyncio.wait_for(opening.cancel_signal(), WAIT)
        await opening.complete_cancelled(
            move_gripper.ResultResponseData(success=False, message="goal cancelled", final_opening=0.6, action_time=0.2)
        )
        aborted = (await first.get_result(WAIT)).data
        assert (aborted.success, aborted.message) == (False, "aborted: operator")
        stopped = (await stop.get_result(WAIT)).data
        assert (stopped.success, stopped.aborted_action) == (True, "grab_item"), stopped.message
        state = await get_state_fx.poll(h, 5.0)
        assert (state.current_action, state.holding) == ("", [False, False])
