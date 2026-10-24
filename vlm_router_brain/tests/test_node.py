"""The real node under the generated harness: goals and services over the
wire, the camera answered by its mocks, and the models behind a scripted
provider, so no network and no key is used. The start checks the key
before any provider is involved, so each test sets a key of its own."""

import asyncio
import logging
from functools import partial

import pytest
from conftest import FakeProvider, answer, records_of
from peppygen.exposed_actions.brain_model import switch_model as switch_action
from peppygen.exposed_actions.item_manipulation import grab_item as grab_action
from peppygen.exposed_actions.item_perception import identify_item as identify_action
from peppygen.exposed_actions.item_perception import scan_items as scan_action
from peppygen.fixtures import harness
from peppygen.fixtures.exposed_actions.brain_model import switch_model as switch_fx
from peppygen.fixtures.exposed_actions.item_manipulation import grab_item as grab_fx
from peppygen.fixtures.exposed_actions.item_perception import identify_item as identify_fx
from peppygen.fixtures.exposed_actions.item_perception import scan_items as scan_fx
from peppygen.fixtures.exposed_services.brain_model import get_model as get_model_fx
from peppygen.fixtures.exposed_services.brain_model import list_models as list_models_fx
from wire import WAIT, answer_the_camera, failure_of, once_the_brain_sees, params, publish_a_capture, refusal_of, result_of

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
        result = await once_the_brain_sees(identify_fx, h, identify())
        assert result.status == identify_fx.ResultStatus.COMPLETED, result.failure
        found = result.data
        assert (found.label, found.confidence, found.camera) == ("the cup", 0.0, "chest")
        assert [round(v, 6) for v in found.position] == [0.0, 0.0, 1.0]
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
        assert (current.roles, current.models) == (["perception", "manipulation"], [ER, "none"])
        offered = await list_models_fx.poll(h, 5.0)
        assert offered.roles == ["perception"] * 5 + ["manipulation"]
        assert offered.models == ["none", ER, "gemini-4-argon", OPUS, "gpt-6-astra", "none"]
        assert not any(":" in name or "/" in name for name in current.models + offered.models)


async def test_a_switch_probes_once_and_get_model_then_reports_the_new_model():
    provider = FakeProvider(Reply('{"boxes": []}'), answer(("the cup", [250, 250, 750, 750])))
    async with harness.start(partial(setup, provider=provider), parameters=params()) as h:
        answer_the_camera(h)
        await publish_a_capture(h)
        switched = await result_of(switch_fx, h, switch("perception", OPUS))
        assert (switched.role, switched.model) == ("perception", OPUS)
        assert (await get_model_fx.poll(h, 5.0)).models == [OPUS, "none"]
        found = await once_the_brain_sees(identify_fx, h, identify())
        assert found.status == identify_fx.ResultStatus.COMPLETED, found.failure
        assert [call.route.spelled for call in provider.calls] == ["openrouter:anthropic/claude-opus-5.5"] * 2


async def test_a_switch_whose_probe_fails_fails_with_attempt_failed_and_keeps_the_model():
    failure = ProviderFailure(OPUS, "ModelNotFoundError", "no such model", retryable=False)
    async with harness.start(partial(setup, provider=FakeProvider(failure)), parameters=params()) as h:
        answer_the_camera(h)
        failed = await failure_of(switch_fx, h, switch("perception", OPUS))
        assert failed == switch_fx.Failure("attempt_failed", f"{OPUS} failed: ModelNotFoundError: no such model")
        assert (await get_model_fx.poll(h, 5.0)).models == [ER, "none"]


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
        assert (await get_model_fx.poll(h, 5.0)).models == [ER, "none"]


async def test_manipulation_accepts_none_alone_and_refuses_every_grab():
    async with harness.start(partial(setup, provider=FakeProvider()), parameters=params()) as h:
        answer_the_camera(h)
        kept = await result_of(switch_fx, h, switch("manipulation", "none"))
        assert (kept.role, kept.model) == ("manipulation", "none")
        refused = await refusal_of(switch_fx, h, switch("manipulation", OPUS))
        assert refused == switch_fx.Failure("rejected", f"'{OPUS}' is not a manipulation model here; see list_models")
        unknown = await refusal_of(switch_fx, h, switch("arm", "none"))
        assert unknown == switch_fx.Failure("rejected", "this brain runs no model for the role 'arm'")
        # item_manipulation reports in its result.
        grabbed = await result_of(
            grab_fx, h,
            grab_action.GoalRequestData(gripper_name="left_gripper", item_id="", position=[0.5, 0.1, 0.7], orientation=[0.0, 0.0, 0.0, 1.0], max_effort=0.0),
        )
        assert (grabbed.success, grabbed.message) == (False, "no manipulation model: manipulation_model is 'none'")


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
        assert (await get_model_fx.poll(h, 5.0)).models == [ER, "none"]


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
        assert (await get_model_fx.poll(h, 5.0)).models == [ER, "none"]
