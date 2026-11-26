"""The OpenAI-compatible API the environment names: both variables or
neither, one alone refused naming the other, the base an http address,
neither value in a message or a repr, each catalog model through the API
under its catalog name, and no provider key read on the way, through the
real any-llm client against the fake API."""

import json
import os

import pytest
from conftest import API_KEY, PROVIDER_KEY_VARIABLES, FakeProvider, catalog_of, model, records_of, sentinel_of
from wire import params

from brain_core.limb_state import LimbStateStore
from brain_core.models import Models
from brain_core.ports import Role, Senses
from brain_core.sequencer import Sequencer

from vlm_router_brain.__main__ import setup
from vlm_router_brain.catalog import Route, load_catalog
from vlm_router_brain.manipulators.vlm.step_model import PROBE_PROMPT as ALIGNMENT_PROBE_PROMPT
from vlm_router_brain.models import RouterEntry, RouterModels
from vlm_router_brain.openai_compatible import API_BASE_VARIABLE, API_KEY_VARIABLE, CompatibleApi, compatible_api, route_of
from vlm_router_brain.provider import AnyLlmProvider
from vlm_router_brain.vlm_detector import PROBE_PROMPT, VlmDetector

BASE = "http://127.0.0.1:8000/v1"
# A key value that passes peppy's value filter and is no real key.
SENTINEL = "sentinel-not-a-key-0451"
BASE_ALONE = "OPENAI_COMPATIBLE_API_BASE is set and OPENAI_COMPATIBLE_API_KEY is not: set both, or neither"
KEY_ALONE = "OPENAI_COMPATIBLE_API_KEY is set and OPENAI_COMPATIBLE_API_BASE is not: set both, or neither"
NOT_HTTP = "OPENAI_COMPATIBLE_API_BASE must be an http:// or https:// address with a host, such as http://localhost:11434/v1"


def test_neither_variable_names_no_api_whatever_provider_key_is_set():
    assert compatible_api({}) is None
    assert compatible_api({"GEMINI_API_KEY": "k", "OPENROUTER_API_KEY": "k", "OPENAI_API_KEY": "k"}) is None
    # A value of spaces counts as not set.
    assert compatible_api({API_BASE_VARIABLE: "  ", API_KEY_VARIABLE: "\n"}) is None


def test_both_variables_name_the_api_each_read_without_its_spaces():
    assert compatible_api({API_BASE_VARIABLE: f" {BASE}\n", API_KEY_VARIABLE: f" {SENTINEL} "}) == CompatibleApi(BASE, SENTINEL)
    for base in ("https://models.example/v1", "http://localhost:11434/v1", "http://user:password@models.example:8443/v1"):
        assert compatible_api({API_BASE_VARIABLE: base, API_KEY_VARIABLE: "k"}).base == base


@pytest.mark.parametrize("environ, message", [
    ({API_BASE_VARIABLE: BASE}, BASE_ALONE),
    ({API_BASE_VARIABLE: BASE, API_KEY_VARIABLE: "   "}, BASE_ALONE),
    ({API_KEY_VARIABLE: SENTINEL}, KEY_ALONE),
    ({API_KEY_VARIABLE: SENTINEL, API_BASE_VARIABLE: " "}, KEY_ALONE),
])
def test_one_variable_alone_is_refused_naming_the_other_and_no_value(environ, message):
    with pytest.raises(ValueError) as raised:
        compatible_api(environ)
    assert str(raised.value) == message


@pytest.mark.parametrize("base", [
    "localhost:11434/v1",
    "127.0.0.1:8000/v1",
    "models.example/v1",
    "ftp://models.example/v1",
    "file:///tmp/v1",
    "http:///v1",
    "http://",
    # urlsplit refuses it: the IPv6 host has no closing bracket.
    "http://[::1/v1",
])
def test_a_base_that_is_not_an_http_address_with_a_host_is_refused_without_its_value(base):
    with pytest.raises(ValueError) as raised:
        compatible_api({API_BASE_VARIABLE: base, API_KEY_VARIABLE: SENTINEL})
    assert str(raised.value) == NOT_HTTP


def test_neither_value_is_in_the_apis_repr():
    api = compatible_api({API_BASE_VARIABLE: "http://user:password@models.example/v1", API_KEY_VARIABLE: SENTINEL})
    assert SENTINEL not in repr(api) and "password" not in repr(api)


@pytest.mark.parametrize("variable, value, message", [
    (API_BASE_VARIABLE, BASE, BASE_ALONE),
    (API_KEY_VARIABLE, SENTINEL, KEY_ALONE),
])
async def test_one_variable_alone_stops_the_node_before_it_serves_anything(monkeypatch, variable, value, message):
    monkeypatch.setenv(variable, value)
    # No node runner: the refusal comes before setup uses it.
    with pytest.raises(ValueError) as raised:
        await setup(params(), None)
    assert str(raised.value) == message


# --- each catalog model through the API

API = CompatibleApi(BASE, SENTINEL)
SHIPPED = ["gemini-robotics-er-2-preview", "gemini-4-argon", "claude-opus-5.5", "gpt-6-astra"]


def test_a_model_through_the_api_is_asked_for_by_its_catalog_name_with_the_settings_of_its_first_route():
    two_routes = catalog_of({
        **model("careful", "openrouter:lab/careful", "gemini:gemini-careful"),
        "routes": [
            {"route": "openrouter:lab/careful", "min_call_timeout_s": 12.0, "reasoning_effort": "high"},
            {"route": "gemini:gemini-careful", "min_call_timeout_s": 30.0, "reasoning_effort": "low"},
        ],
    })
    assert route_of(two_routes.get("careful")) == Route("careful", "openai_compatible", "careful", 12.0, "high")


def test_each_shipped_model_through_the_api_is_asked_for_by_its_catalog_name_never_a_providers_id():
    catalog = load_catalog()
    assert [entry.name for entry in catalog.entries] == SHIPPED
    for entry in catalog.entries:
        route = route_of(entry)
        assert (route.provider, route.model, route.model_name) == ("openai_compatible", entry.name, entry.name)
        assert (route.min_call_timeout_s, route.reasoning_effort) == (entry.routes[0].min_call_timeout_s, entry.routes[0].reasoning_effort)
    assert route_of(catalog.get("gemini-4-argon")).spelled == "openai_compatible:gemini-4-argon"


def test_through_the_api_a_model_starts_with_no_provider_key_and_makes_no_call(caplog):
    provider = FakeProvider()
    catalog = catalog_of(model("careful", "openrouter:lab/careful", "gemini:gemini-careful"))
    models = RouterModels(catalog, provider, {}, api=API)
    with records_of(caplog, "vlm_router_brain.models") as log:
        detector = models.start(RouterEntry(Role.PERCEPTION, catalog.get("careful")))
    assert isinstance(detector, VlmDetector) and detector.route.spelled == "openai_compatible:careful"
    assert provider.calls == []
    assert [record.getMessage() for record in log.records] == ["careful runs through openai_compatible:careful"]


async def test_through_the_api_a_switch_probes_the_route_through_it_with_no_provider_key():
    provider = FakeProvider()
    catalog = catalog_of(model("fast", "gemini:gemini-fast"))
    detector = await RouterModels(catalog, provider, {}, api=API).prepare(RouterEntry(Role.PERCEPTION, catalog.get("fast")))
    [call] = provider.calls
    assert call.prompt == PROBE_PROMPT and call.route.spelled == "openai_compatible:fast"
    assert detector.route is call.route


async def test_through_the_api_brain_core_starts_a_model_that_has_no_provider_key():
    models = Models(RouterModels(load_catalog(), FakeProvider(), {}, api=API), Sequencer(), Senses(None, LimbStateStore(), None, None, None))
    await models.start({Role.PERCEPTION: "gemini-4-argon", Role.MANIPULATION: "gpt-6-astra"})
    assert models.current_name(Role.PERCEPTION) == "gemini-4-argon"


class EnvironReads(dict):
    """A stand-in of os.environ that records each name read: a lookup by
    name, through get, [] or in, which is how os.getenv and the libraries
    read one variable."""

    def __init__(self, values) -> None:
        super().__init__(values)
        self.read: set = set()

    def get(self, name, default=None):
        self.read.add(name)
        return super().get(name, default)

    def __getitem__(self, name):
        self.read.add(name)
        return super().__getitem__(name)

    def __contains__(self, name):
        self.read.add(name)
        return super().__contains__(name)


async def test_through_the_api_no_provider_key_is_read_and_none_reaches_the_api(monkeypatch, through_api):
    """The real any-llm client, made and called through the API for a
    perception probe and an alignment probe, with a recorder in place of
    os.environ that holds a sentinel in each provider key."""
    environ = EnvironReads(os.environ)
    monkeypatch.setattr(os, "environ", environ)
    api = compatible_api(os.environ)
    catalog = load_catalog()
    models = RouterModels(catalog, AnyLlmProvider(os.environ, api=api), os.environ, api=api)
    await models.prepare(RouterEntry(Role.PERCEPTION, catalog.get("gemini-4-argon")))
    through_api.api.answer('{"action": "give_up"}')
    await models.prepare(RouterEntry(Role.MANIPULATION, catalog.get("gpt-6-astra")))
    assert not environ.read & set(PROVIDER_KEY_VARIABLES)
    perception, manipulation = through_api.api.requests
    assert (perception.body["model"], perception.question) == ("gemini-4-argon", PROBE_PROMPT)
    assert (manipulation.body["model"], manipulation.question) == ("gpt-6-astra", ALIGNMENT_PROBE_PROMPT)
    for request in (perception, manipulation):
        assert request.headers["authorization"] == f"Bearer {API_KEY}"
        sent = json.dumps([request.headers, request.body])
        assert not any(sentinel_of(variable) in sent for variable in PROVIDER_KEY_VARIABLES)
    assert through_api.provider_clients == []
