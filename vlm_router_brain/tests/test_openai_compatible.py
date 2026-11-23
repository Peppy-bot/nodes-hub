"""The OpenAI-compatible API the environment names: both variables or
neither, one alone refused naming the other, the base an http address, and
neither value in a message or a repr."""

import pytest
from wire import params

from vlm_router_brain.__main__ import setup
from vlm_router_brain.openai_compatible import API_BASE_VARIABLE, API_KEY_VARIABLE, CompatibleApi, compatible_api

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
