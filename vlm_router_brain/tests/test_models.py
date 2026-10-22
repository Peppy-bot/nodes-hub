"""The router's models for brain_core: the names each role accepts, the
built-in models, the start that makes no call, the switch that probes, and
the keys that pick a route."""

import logging

import pytest
from conftest import BOTH_KEYS, TEST_CATALOG, FakeProvider, records_of

from brain_core.limb_state import LimbStateStore
from brain_core.manipulation.centre_proposer import CentreProposer
from brain_core.manipulation.none import NoneManipulator
from brain_core.manipulation.scripted import ScriptedManipulator
from brain_core.models import Models
from brain_core.perception.none import NoneDetector
from brain_core.ports import Refusal, Role
from brain_core.sequencer import Sequencer

from vlm_router_brain.catalog import CATALOG_ROLES, load_catalog
from vlm_router_brain.models import BUILDERS, BUILT_IN, BuiltIn, RouterEntry, RouterModels
from vlm_router_brain.provider import ProviderFailure
from vlm_router_brain.vlm_detector import PROBE_PROMPT, VlmDetector

ONLY_GEMINI = {"GEMINI_API_KEY": "test-gemini-key"}


def router(environ=BOTH_KEYS, provider=None) -> RouterModels:
    return RouterModels(TEST_CATALOG, provider or FakeProvider(), environ)


def perceiving(name: str) -> RouterEntry:
    """The entry `offered` gives for the test catalog's model `name` in
    perception."""
    return RouterEntry(Role.PERCEPTION, TEST_CATALOG.get(name))


def test_both_roles_accept_none_manipulation_scripted_and_perception_the_catalog_models():
    models = router()
    assert models.roles() == (Role.PERCEPTION, Role.MANIPULATION)
    assert models.names(Role.PERCEPTION) == ["none", "fast", "careful"]
    assert models.names(Role.MANIPULATION) == ["none", "scripted"]
    shipped = RouterModels(load_catalog(), FakeProvider(), BOTH_KEYS)
    assert shipped.names(Role.PERCEPTION) == ["none", "gemini-robotics-er-2-preview", "gemini-4-argon", "claude-opus-5.5", "gpt-6-astra"]


def test_an_entry_carries_the_role_it_was_offered_for():
    models = router()
    assert models.offered(Role.PERCEPTION, "none") == BuiltIn(Role.PERCEPTION, "none")
    assert models.offered(Role.MANIPULATION, "none") == BuiltIn(Role.MANIPULATION, "none")
    assert models.offered(Role.MANIPULATION, "scripted") == BuiltIn(Role.MANIPULATION, "scripted")
    offered = models.offered(Role.PERCEPTION, "fast")
    assert offered == RouterEntry(Role.PERCEPTION, TEST_CATALOG.get("fast"))
    assert offered.role is Role.PERCEPTION and offered.catalog_entry is TEST_CATALOG.get("fast")


def test_each_role_a_catalog_model_serves_has_a_builder_and_perception_builds_a_vlm_detector():
    assert BUILDERS == {Role.PERCEPTION: VlmDetector}
    assert set(BUILDERS) == CATALOG_ROLES
    assert isinstance(router().start(perceiving("fast")), VlmDetector)


def test_a_name_the_role_does_not_accept_is_refused_pointing_to_list_models():
    models = router()
    with pytest.raises(Refusal, match="^'sam9' is not a perception model here; see list_models$"):
        models.offered(Role.PERCEPTION, "sam9")
    with pytest.raises(Refusal, match="^'fast' is not a manipulation model here; see list_models$"):
        models.offered(Role.MANIPULATION, "fast")
    with pytest.raises(Refusal, match="^'scripted' is not a perception model here; see list_models$"):
        models.offered(Role.PERCEPTION, "scripted")


def test_the_start_runs_the_first_keyed_route_and_makes_no_call(caplog):
    provider = FakeProvider()
    models = router(BOTH_KEYS, provider)
    with records_of(caplog, "vlm_router_brain.models") as log:
        detector = models.start(perceiving("careful"))
    assert isinstance(detector, VlmDetector) and detector.name == "careful"
    assert detector.route.spelled == "openrouter:lab/careful"
    assert provider.calls == []
    assert [record.getMessage() for record in log.records] == ["careful runs through openrouter:lab/careful"]
    assert isinstance(models.start(BuiltIn(Role.PERCEPTION, "none")), NoneDetector)
    assert isinstance(models.start(BuiltIn(Role.MANIPULATION, "none")), NoneManipulator)


def test_the_second_route_runs_when_the_first_routes_key_is_missing():
    detector = router(ONLY_GEMINI).start(perceiving("careful"))
    assert detector.route.spelled == "gemini:gemini-careful"


def test_a_starting_model_with_no_keyed_route_stops_the_node_naming_every_variable():
    with pytest.raises(ValueError, match="^no key reaches 'careful': set OPENROUTER_API_KEY or GEMINI_API_KEY in the node's environment$"):
        router({}).start(perceiving("careful"))
    with pytest.raises(ValueError, match="^no key reaches 'fast': set GEMINI_API_KEY in the node's environment$"):
        router({"OPENROUTER_API_KEY": "set"}).start(perceiving("fast"))


async def test_a_switch_probes_the_keyed_route_once_then_runs_on_it():
    provider = FakeProvider()
    detector = await router(ONLY_GEMINI, provider).prepare(perceiving("careful"))
    [call] = provider.calls
    assert call.prompt == PROBE_PROMPT and call.route.spelled == "gemini:gemini-careful"
    assert isinstance(detector, VlmDetector) and detector.route is call.route
    assert isinstance(await router(provider=provider).prepare(BuiltIn(Role.PERCEPTION, "none")), NoneDetector)
    assert len(provider.calls) == 1


async def test_a_switch_to_a_model_with_no_keyed_route_is_refused_with_no_probe():
    provider = FakeProvider()
    with pytest.raises(Refusal, match="^no key reaches 'careful': set OPENROUTER_API_KEY or GEMINI_API_KEY in the node's environment$"):
        await router({}, provider).prepare(perceiving("careful"))
    assert provider.calls == []


async def test_a_probe_failure_fails_the_switch_with_the_providers_message():
    failure = ProviderFailure("fast", "ModelNotFoundError", "no such model", retryable=False)
    with pytest.raises(ProviderFailure, match="^fast failed: ModelNotFoundError: no such model$"):
        await router(provider=FakeProvider(failure)).prepare(perceiving("fast"))


@pytest.mark.parametrize("perception, manipulation, environ, message", [
    ("sam9", "none", BOTH_KEYS, "^perception_model: 'sam9' is not a perception model here; see list_models$"),
    ("fast", "fast", BOTH_KEYS, "^manipulation_model: 'fast' is not a manipulation model here; see list_models$"),
    ("careful", "none", {}, "^no key reaches 'careful'"),
])
async def test_brain_core_stops_the_node_on_a_starting_model_the_router_cannot_run(perception, manipulation, environ, message):
    provider = FakeProvider()
    models = Models(router(environ, provider), Sequencer(), None, LimbStateStore())
    with pytest.raises(ValueError, match=message):
        await models.start({Role.PERCEPTION: perception, Role.MANIPULATION: manipulation})
    assert provider.calls == []


async def test_a_key_value_never_reaches_the_log(caplog):
    sentinel = {"GEMINI_API_KEY": "sentinel-not-a-key-0451"}
    failure = ProviderFailure("fast", "AuthenticationError", "the key was refused", retryable=False)
    models = RouterModels(TEST_CATALOG, FakeProvider(failure), sentinel)
    with records_of(caplog, "vlm_router_brain", logging.DEBUG) as log:
        detector = models.start(perceiving("fast"))
        with pytest.raises(ProviderFailure):
            await models.prepare(perceiving("fast"))
    assert log.records and "sentinel-not-a-key-0451" not in log.text
    assert "sentinel" not in repr(detector.route)


async def test_scripted_builds_a_scripted_manipulator_on_the_centre_proposer_at_the_start_and_at_a_switch():
    provider = FakeProvider()
    models = router(provider=provider)
    started = models.start(BuiltIn(Role.MANIPULATION, "scripted"))
    switched = await models.prepare(BuiltIn(Role.MANIPULATION, "scripted"))
    for manipulator in (started, switched):
        assert isinstance(manipulator, ScriptedManipulator) and isinstance(manipulator.proposer, CentreProposer)
        assert manipulator.name == "scripted"
    # Each start and each switch builds its own; none calls a provider.
    assert started is not switched
    assert provider.calls == []


def test_each_built_in_model_runs_under_its_own_name():
    for role, models in BUILT_IN.items():
        for name, build in models.items():
            assert build().name == name, (role, name)


async def test_brain_core_starts_on_scripted_and_hands_it_the_robot_and_the_limbs():
    store = LimbStateStore()
    models = Models(router(), Sequencer(), "robot", store)
    await models.start({Role.PERCEPTION: "none", Role.MANIPULATION: "scripted"})
    manipulator = models.current(Role.MANIPULATION)
    assert isinstance(manipulator, ScriptedManipulator)
    # Started with the store: it waits for the robot's first snapshot.
    assert manipulator.why_unavailable() == "no limb_states snapshot received from the robot yet"
