"""The scene_control provider: what get_assets_list answers before and after
discovery, and what load_scene and clear_scene do to spawned objects."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, call

import pytest

_ROBOT_DIR = Path(__file__).resolve().parents[1] / "robots" / "openarm"
_ACTIONS = ("apply_force", "clear_scene", "load_scene", "move_object", "move_robot", "remove_object", "spawn_object")
_SERVICES = ("get_assets_list", "get_objects_list", "get_robots_list")


@pytest.fixture
def provider(monkeypatch):
    actions = ModuleType("peppygen.exposed_actions.scene")
    for name in _ACTIONS:
        setattr(actions, name, ModuleType(f"{actions.__name__}.{name}"))
    services = ModuleType("peppygen.exposed_services.scene")
    for name in _SERVICES:
        module = ModuleType(f"{services.__name__}.{name}")
        module.Response = lambda **fields: SimpleNamespace(**fields)
        setattr(services, name, module)
    for name, module in {
        "peppygen": ModuleType("peppygen"),
        "peppygen.exposed_actions": ModuleType("peppygen.exposed_actions"),
        "peppygen.exposed_actions.scene": actions,
        "peppygen.exposed_services": ModuleType("peppygen.exposed_services"),
        "peppygen.exposed_services.scene": services,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location("_scene_actions_under_test", _ROBOT_DIR / "scene_actions.py")
    module = importlib.util.module_from_spec(spec)
    # Its dataclass resolves postponed annotations through sys.modules.
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    io = module.SceneActionIO(object(), Mock(), model="openarm_v2")
    launcher = Mock()
    return SimpleNamespace(io=io, launcher=launcher)


_CATALOGUE = {
    "scene/full_warehouse": {
        "asset_id": "scene/full_warehouse", "display_name": "Full Warehouse", "kind": "scene",
        "path": "Isaac/Environments/Simple_Warehouse/full_warehouse.usd", "category": "Scenes",
    },
    "props/blocks/red_block": {
        "asset_id": "props/blocks/red_block", "display_name": "red block", "kind": "object",
        "path": "Isaac/Props/Blocks/red_block.usd", "category": "Blocks",
    },
}


def test_assets_are_refused_with_a_reason_until_discovery_hands_them_over(provider):
    response = provider.io._handle_get_assets(None)
    assert response.success is False
    assert "still discovering its asset catalogue" in response.message
    assert json.loads(response.assets_json) == []

    provider.io.set_assets({})
    response = provider.io._handle_get_assets(None)
    assert response.success is True
    assert response.message == "0 assets available"
    assert json.loads(response.assets_json) == []

    provider.io.set_assets(_CATALOGUE)
    response = provider.io._handle_get_assets(None)
    assert response.success is True
    assert [a["asset_id"] for a in json.loads(response.assets_json)] == [
        "props/blocks/red_block", "scene/full_warehouse",
    ]
    assert all("path" not in a for a in json.loads(response.assets_json))


def _spawn(provider, asset_id="props/blocks/red_block"):
    result = provider.io._execute(provider.launcher, "spawn_object", {
        "asset_id": asset_id, "position": [0.5, 0.0, 0.8], "scale": 1.0, "physics": "dynamic", "mass": 0.1,
    })
    assert result["success"]
    return result["object_id"]


def test_load_scene_removes_spawned_objects_then_replaces_the_scene(provider):
    provider.io.set_assets(_CATALOGUE)
    first, second = _spawn(provider), _spawn(provider)
    provider.launcher.reset_mock()

    result = provider.io._execute(provider.launcher, "load_scene", {"asset_id": "scene/full_warehouse", "scale": 2.0})

    assert result == {"success": True, "message": "Loaded scene scene/full_warehouse"}
    assert provider.launcher.mock_calls == [
        call._runtime_remove({"name": first}),
        call._runtime_remove({"name": second}),
        call._runtime_load_isaac_scene({
            "path": "Isaac/Environments/Simple_Warehouse/full_warehouse.usd", "scale": [2.0, 2.0, 2.0],
        }),
    ]
    assert provider.io._public_objects() == []


def test_clear_scene_removes_spawned_objects_then_the_scene(provider):
    provider.io.set_assets(_CATALOGUE)
    object_id = _spawn(provider)
    provider.launcher.reset_mock()

    result = provider.io._execute(provider.launcher, "clear_scene", {})

    assert result == {"success": True, "message": "Runtime scene cleared"}
    assert provider.launcher.mock_calls == [
        call._runtime_remove({"name": object_id}),
        call._runtime_clear_scene(),
    ]
    assert provider.io._public_objects() == []


@pytest.mark.parametrize(("asset_id", "message"), [
    ("scene/missing", "Unknown asset_id: scene/missing"),
    ("props/blocks/red_block", "Asset is not a scene: props/blocks/red_block"),
])
def test_load_scene_rejects_unknown_or_non_scene_assets_before_touching_the_stage(provider, asset_id, message):
    provider.io.set_assets(_CATALOGUE)
    _spawn(provider)
    provider.launcher.reset_mock()

    with pytest.raises(ValueError, match=message):
        provider.io._execute(provider.launcher, "load_scene", {"asset_id": asset_id, "scale": 1.0})
    assert provider.launcher.mock_calls == []
    assert len(provider.io._public_objects()) == 1


def test_the_listing_names_the_one_robot_this_simulation_stands(provider):
    listed = json.loads(provider.io._handle_get_robots(None).robots_json)

    assert listed == [
        {"robot": "openarm", "model": "openarm_v2", "position": [0.0, 0.0, 0.0], "attached": False}
    ]


def test_moving_a_robot_by_name_moves_it_and_the_listing_follows(provider):
    result = provider.io._execute(
        provider.launcher, "move_robot", {"robot": "openarm", "position": [1.0, -2.0, 0.0]}
    )

    assert result["success"] is True
    assert provider.launcher.mock_calls == [
        call._runtime_move_robot_root({"position": [1.0, -2.0, 0.0]})
    ]
    listed = json.loads(provider.io._handle_get_robots(None).robots_json)
    assert listed[0]["position"] == [1.0, -2.0, 0.0]


def test_a_robot_this_simulation_does_not_stand_is_refused_before_the_stage(provider):
    with pytest.raises(ValueError, match="no robot stands as 'bravo'"):
        provider.io._execute(
            provider.launcher, "move_robot", {"robot": "bravo", "position": [1.0, 0.0, 0.0]}
        )

    assert provider.launcher.mock_calls == []
