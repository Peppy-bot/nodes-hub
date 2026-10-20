"""The keys: read from the environment alone, the first keyed route
taken, and no key value in any message."""

import pytest
from any_llm.exceptions import AuthenticationError, ProviderError
from conftest import TEST_CATALOG

from vlm_router_brain import provider as provider_module
from vlm_router_brain.keys import KEY_VARIABLES, first_keyed, key_for, no_key_message
from vlm_router_brain.provider import DETAIL_CHARS, AnyLlmProvider, ProviderFailure

CAREFUL = TEST_CATALOG.get("careful")
# A key value that passes peppy's value filter and is no real key.
SENTINEL = "sentinel-not-a-key-0451"


def test_each_provider_reads_its_own_variable_and_nothing_else():
    assert KEY_VARIABLES == {"openrouter": "OPENROUTER_API_KEY", "gemini": "GEMINI_API_KEY"}
    assert key_for("gemini", {"GEMINI_API_KEY": " a-key \n"}) == "a-key"
    # Gemini's other variable, which any-llm would read, is not one of ours.
    assert key_for("gemini", {"GOOGLE_API_KEY": "a-key"}) == ""
    assert key_for("openrouter", {"OPENROUTER_API_KEY": "   "}) == ""


def test_the_first_route_whose_key_is_set_is_the_one_taken():
    [openrouter, gemini] = CAREFUL.routes
    assert first_keyed(CAREFUL.routes, {"OPENROUTER_API_KEY": "k", "GEMINI_API_KEY": "k"}) is openrouter
    assert first_keyed(CAREFUL.routes, {"GEMINI_API_KEY": "k"}) is gemini
    assert first_keyed(CAREFUL.routes, {}) is None


def test_the_refusal_names_each_variable_once_in_route_order():
    assert no_key_message("careful", CAREFUL.routes) == (
        "no key reaches 'careful': set OPENROUTER_API_KEY or GEMINI_API_KEY in the node's environment"
    )
    twice = CAREFUL.routes + CAREFUL.routes
    assert no_key_message("careful", twice).count("GEMINI_API_KEY") == 1


async def failure_of(monkeypatch, error: Exception) -> ProviderFailure:
    """The failure a call through CAREFUL's OpenRouter route, keyed with
    SENTINEL, ends in when any-llm raises `error`."""

    class FailingClient:
        async def acompletion(self, **kwargs):
            raise error

    monkeypatch.setattr(provider_module.AnyLLM, "create", staticmethod(lambda provider, **kwargs: FailingClient()))
    with pytest.raises(ProviderFailure) as raised:
        await AnyLlmProvider({"OPENROUTER_API_KEY": SENTINEL}).ask(CAREFUL.routes[0], b"jpeg", "q", 10.0)
    assert SENTINEL not in raised.value.message and SENTINEL not in repr(raised.value)
    return raised.value


async def test_a_refused_key_is_never_in_the_failure(monkeypatch):
    failure = await failure_of(monkeypatch, AuthenticationError("Incorrect API key provided", status_code=401))
    assert failure.message == "careful failed: AuthenticationError: Incorrect API key provided"


async def test_a_key_the_provider_echoes_is_masked_wherever_it_is(monkeypatch):
    echoed = AuthenticationError(f"Incorrect API key provided: {SENTINEL}.\nKey {SENTINEL} is revoked.", status_code=401)
    failure = await failure_of(monkeypatch, echoed)
    assert failure.message == "careful failed: AuthenticationError: Incorrect API key provided: ***. Key *** is revoked."


async def test_a_key_across_the_length_limit_leaves_none_of_its_characters(monkeypatch):
    # Cut before the mask, the limit would keep the key's first 7 characters.
    across = ProviderError("x" * (DETAIL_CHARS - 10) + SENTINEL + "y" * 50)
    failure = await failure_of(monkeypatch, across)
    assert failure.message.endswith("x" * (DETAIL_CHARS - 10) + "***" + "y" * 4 + "...")
    assert SENTINEL[:7] not in failure.message
