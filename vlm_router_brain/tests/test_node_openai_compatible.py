"""The real node under the generated harness, with its real any-llm client,
calling the fake OpenAI-compatible API of fake_api.py on 127.0.0.1: a
search and an alignment probe go to the API with its key under the catalog
names, a switch to each catalog model asks the API for that name, a model
the API does not serve fails the switch with the API's message, and no
provider client is made. Without the API, a switch makes the provider's
client with the provider's key, as before."""

import json
import logging
from types import SimpleNamespace

from conftest import API_KEY, PROVIDER_KEY_VARIABLES, records_of, sentinel_of
from peppygen.exposed_actions.brain_model import switch_model as switch_action
from peppygen.exposed_actions.item_perception import identify_item as identify_action
from peppygen.fixtures import harness
from peppygen.fixtures.exposed_actions.brain_model import switch_model as switch_fx
from peppygen.fixtures.exposed_actions.item_perception import identify_item as identify_fx
from peppygen.fixtures.exposed_services.brain_model import get_model as get_model_fx
from wire import answer_the_camera, failure_of, once_the_brain_sees, params, publish_a_capture, result_of

from vlm_router_brain import provider as provider_module
from vlm_router_brain.__main__ import setup
from vlm_router_brain.manipulators.vlm.step_model import PROBE_PROMPT as ALIGNMENT_PROBE_PROMPT
from vlm_router_brain.vlm_detector import PROBE_PROMPT

ER = "gemini-robotics-er-2-preview"
OPUS = "claude-opus-5.5"
CATALOG = [ER, "gemini-4-argon", OPUS, "gpt-6-astra"]
THE_CUP = json.dumps({"boxes": [{"label": "the cup", "box_2d": [250, 250, 750, 750]}]})


def identify(description: str = "the cup"):
    return identify_action.GoalRequestData(description=description, timeout_s=0.0)


def switch(role: str, model: str):
    return switch_action.GoalRequestData(role=role, model=model)


async def test_a_search_and_an_alignment_probe_go_to_the_api_with_its_key_under_the_catalog_names(caplog, through_api):
    api = through_api.api
    # Every record of the node's own loggers, the router's and brain_core's.
    with records_of(caplog, "vlm_router_brain", logging.DEBUG), records_of(caplog, "brain_core", logging.DEBUG):
        async with harness.start(setup, parameters=params()) as h:
            answer_the_camera(h)
            await publish_a_capture(h)
            # A perception call: the search of the starting model.
            api.answer(THE_CUP)
            found = await once_the_brain_sees(identify_fx, h, identify())
            assert found.status == identify_fx.ResultStatus.COMPLETED, found.failure
            assert found.data.label == "the cup"
            # A manipulation call: the alignment probe of a switch.
            api.answer('{"action": "give_up"}')
            switched = await result_of(switch_fx, h, switch("manipulation", OPUS))
            assert (switched.role, switched.model) == ("manipulation", OPUS)
    search, probe = api.requests
    assert (search.body["model"], probe.body["model"]) == (ER, OPUS)
    assert search.question.startswith("Find the cup in this image") and probe.question == ALIGNMENT_PROBE_PROMPT
    for request in (search, probe):
        assert request.path == "/v1/chat/completions"
        assert request.headers["authorization"] == f"Bearer {API_KEY}"
        sent = json.dumps([request.headers, request.body])
        assert not any(sentinel_of(variable) in sent for variable in PROVIDER_KEY_VARIABLES)
    assert through_api.provider_clients == []
    # The log names the route through the API, never the key or the base.
    assert f"{ER} runs through openai_compatible:{ER}" in caplog.text
    assert API_KEY not in caplog.text and api.base not in caplog.text


async def test_a_switch_to_each_catalog_model_asks_the_api_for_its_catalog_name(through_api):
    order = [*CATALOG[1:], ER]
    async with harness.start(setup, parameters=params()) as h:
        answer_the_camera(h)
        for name in order:
            switched = await result_of(switch_fx, h, switch("perception", name))
            assert switched.model == name
    assert [request.body["model"] for request in through_api.api.requests] == order
    assert all(request.question == PROBE_PROMPT for request in through_api.api.requests)
    assert through_api.provider_clients == []


async def test_a_switch_to_a_model_the_api_does_not_serve_fails_with_the_apis_message_and_keeps_the_model(caplog, through_api):
    through_api.api.reset(name for name in CATALOG if name != OPUS)
    with records_of(caplog, "vlm_router_brain", logging.DEBUG), records_of(caplog, "brain_core", logging.DEBUG):
        async with harness.start(setup, parameters=params()) as h:
            answer_the_camera(h)
            failed = await failure_of(switch_fx, h, switch("perception", OPUS))
            assert failed.kind is switch_fx.FailureKind.ATTEMPT_FAILED
            assert failed.message == f"{OPUS} failed: ModelNotFoundError: [openai_compatible] Error code: 404 - model '{OPUS}' not found"
            assert (await get_model_fx.poll(h, 5.0)).models == [ER, "scripted"]
    [request] = through_api.api.requests
    assert request.body["model"] == OPUS
    assert API_KEY not in caplog.text


class ProviderClient:
    """A provider's any-llm client: records each call and answers that
    nothing is visible."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def acompletion(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content='{"boxes": []}')
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)


async def test_without_the_api_a_switch_makes_the_providers_client_with_the_providers_key(monkeypatch):
    for variable in PROVIDER_KEY_VARIABLES:
        monkeypatch.setenv(variable, sentinel_of(variable))
    made, client = [], ProviderClient()

    def create(provider, **kwargs):
        made.append((provider, kwargs))
        return client

    def create_openai_compatible(name, **kwargs):
        made.append((name, kwargs))
        raise RuntimeError("a client of an OpenAI-compatible API was made")

    monkeypatch.setattr(provider_module.AnyLLM, "create", staticmethod(create))
    monkeypatch.setattr(provider_module.AnyLLM, "create_openai_compatible", staticmethod(create_openai_compatible))
    async with harness.start(setup, parameters=params()) as h:
        answer_the_camera(h)
        switched = await result_of(switch_fx, h, switch("perception", OPUS))
        assert switched.model == OPUS
    assert made == [("openrouter", dict(api_key=sentinel_of("OPENROUTER_API_KEY"), unified_exceptions=True, max_retries=0))]
    [call] = client.calls
    assert call["model"] == "anthropic/claude-opus-5.5"
