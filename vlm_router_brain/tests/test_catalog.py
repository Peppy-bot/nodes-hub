"""The catalog: the shipped one parses, and every way a catalog can be
wrong stops the node naming the entry and the field."""

import json
import re
from pathlib import Path

import pytest
from conftest import model

from brain_core.ports import Role

from vlm_router_brain.catalog import CatalogError, load_catalog, parse_catalog

MANIFEST = Path(__file__).resolve().parents[1] / "peppy.json5"


def test_the_default_perception_model_is_gemini_robotics_er_of_the_catalog():
    # The manifest names the default; the catalog must still offer it.
    [default] = re.findall(r'perception_model: \{ \$type: "string", \$default: "([^"]*)" \}', MANIFEST.read_text())
    assert default == "gemini-robotics-er-2-preview"
    assert default in load_catalog().names_for(Role.PERCEPTION)


def test_the_shipped_catalog_parses_into_the_models_step_0_passed():
    catalog = load_catalog()
    assert [entry.name for entry in catalog.entries] == [
        "gemini-robotics-er-2-preview", "gemini-4-argon", "claude-opus-5.5", "gpt-6-astra",
    ]
    assert catalog.names_for(Role.PERCEPTION) == [entry.name for entry in catalog.entries]
    assert catalog.names_for(Role.MANIPULATION) == []
    er = catalog.get("gemini-robotics-er-2-preview")
    assert er.coordinates == "yx_0_1000" and er.roles == frozenset({Role.PERCEPTION})
    [route] = er.routes
    assert (route.provider, route.model, route.min_call_timeout_s, route.reasoning_effort) == (
        "gemini", "gemini-robotics-er-2-preview", 10.0, "medium",
    )
    assert route.model_name == "gemini-robotics-er-2-preview" and route.spelled == "gemini:gemini-robotics-er-2-preview"
    assert [entry.routes[0].provider for entry in catalog.entries[1:]] == ["openrouter"] * 3
    assert catalog.get("none") is None and catalog.get("sam3_siglip") is None


def test_an_openrouter_id_with_a_colon_splits_on_the_first_one():
    [route] = parse_catalog({"models": [model("free", "openrouter:org/model:free")]}).entries[0].routes
    assert (route.provider, route.model) == ("openrouter", "org/model:free")


def entry_with(**changes) -> dict:
    entry = model("fast", "gemini:gemini-fast")
    entry.update(changes)
    return entry


def route_with(**changes) -> dict:
    route = {"route": "gemini:gemini-fast", "min_call_timeout_s": 10.0, "reasoning_effort": "medium"}
    route.update(changes)
    return entry_with(routes=[route])


@pytest.mark.parametrize("data, message", [
    ({"model": []}, 'must be an object with one field, "models", a list'),
    ({"models": {}}, 'must be an object with one field, "models", a list'),
    ({"models": ["fast"]}, "catalog.json models[0] must be an object"),
    ({"models": [{"name": "fast", "roles": ["perception"], "coordinates": "yx_0_1000"}]}, "catalog.json models[0]: missing routes"),
    ({"models": [entry_with(cost=1)]}, "catalog.json models[0]: unknown field cost"),
    ({"models": [entry_with(name="")]}, "name must be a non-empty string"),
    ({"models": [entry_with(name=" fast")]}, "name must be a non-empty string with no spaces around it"),
    ({"models": [entry_with(name="none")]}, "model 'none': 'none' names a role's missing model, never a catalog model"),
    ({"models": [entry_with(roles=[])]}, "model 'fast': roles must be a list of at least one role"),
    ({"models": [entry_with(roles=["arm"])]}, "model 'fast': the role 'arm' is not one a catalog model serves here (perception)"),
    ({"models": [entry_with(roles=["manipulation"])]}, "the role 'manipulation' is not one a catalog model serves here"),
    ({"models": [entry_with(coordinates="xy_0_1")]}, "model 'fast': unknown coordinates 'xy_0_1'; known: yx_0_1000"),
    ({"models": [entry_with(routes=[])]}, "model 'fast': routes must be a list of at least one route"),
    ({"models": [route_with(route="gemini-fast")]}, "routes[0]: route 'gemini-fast' must be written provider:model"),
    ({"models": [route_with(route="gemini:")]}, "route 'gemini:' must be written provider:model"),
    ({"models": [route_with(route="openai:gpt")]}, "routes[0]: unknown provider 'openai'; known: gemini, openrouter"),
    ({"models": [route_with(min_call_timeout_s=0)]}, "min_call_timeout_s must be a number of seconds above 0"),
    ({"models": [route_with(min_call_timeout_s="10")]}, "min_call_timeout_s must be a number of seconds above 0"),
    ({"models": [route_with(min_call_timeout_s=True)]}, "min_call_timeout_s must be a number of seconds above 0"),
    ({"models": [route_with(reasoning_effort="lots")]}, "unknown reasoning_effort 'lots'"),
    ({"models": [route_with(response_format="json_object")]}, "routes[0]: unknown field response_format"),
    ({"models": [entry_with(), entry_with()]}, "the model 'fast' is named twice"),
])
def test_a_catalog_the_node_cannot_read_is_refused_naming_where(data, message):
    with pytest.raises(CatalogError, match=re.escape(message)):
        parse_catalog(data)


def test_a_catalog_that_is_not_json_is_refused():
    with pytest.raises(CatalogError, match="^catalog.json is not JSON: "):
        load_catalog("{models: [}")
    assert load_catalog(json.dumps({"models": []})).entries == ()
