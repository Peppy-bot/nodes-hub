"""Runtime scene edits on a fake USD stage: replacing the scene, clearing it,
removing objects, and what each does to the cached physics views."""

import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

_ROBOT_DIR = Path(__file__).resolve().parents[1] / "robots" / "openarm"
_SCENE = "/World/RuntimeScene"
_ASSET_ROOT = "https://assets.example/Isaac/6.1"


class FakePrim:
    def __init__(self, path, valid=True):
        self.path = path
        self.valid = valid
        self.references = []
        self.scale = None

    def IsValid(self):
        return self.valid

    def GetReferences(self):
        return SimpleNamespace(AddReference=self.references.append)


class FakeStage:
    """Prims by path; removal takes descendants with it, as USD does."""

    def __init__(self):
        self.prims = {}
        self.removed = []

    def GetPrimAtPath(self, path):
        return self.prims.get(str(path)) or FakePrim(str(path), valid=False)

    def DefinePrim(self, path, kind):
        assert kind == "Xform"
        prim = FakePrim(str(path))
        self.prims[str(path)] = prim
        return prim

    def RemovePrim(self, path):
        path = str(path)
        self.removed.append(path)
        for existing in list(self.prims):
            if existing == path or existing.startswith(path + "/"):
                del self.prims[existing]


class FakeScaleOp:
    def __init__(self, prim):
        self.prim = prim

    def GetOpType(self):
        return "scale"

    def Set(self, value):
        self.prim.scale = value


@pytest.fixture
def scene(monkeypatch):
    stage = FakeStage()

    omni = ModuleType("omni")
    omni.usd = ModuleType("omni.usd")
    omni.usd.get_context = lambda: SimpleNamespace(get_stage=lambda: stage)
    pxr = ModuleType("pxr")
    pxr.Gf = SimpleNamespace(Vec3f=lambda x, y, z: (x, y, z))
    pxr.Sdf = SimpleNamespace(Path=str)
    pxr.UsdGeom = SimpleNamespace(
        XformOp=SimpleNamespace(TypeScale="scale"),
        Xformable=lambda prim: SimpleNamespace(
            GetOrderedXformOps=lambda: [],
            AddScaleOp=lambda: FakeScaleOp(prim),
        ),
    )
    storage = ModuleType("isaacsim.storage.native")
    storage.get_assets_root_path = lambda: _ASSET_ROOT + "/"
    bridge_module = ModuleType("bridge_extension")
    bridge_module.IsaacBridgeExtension = Mock()
    for name, module in {
        "omni": omni, "omni.usd": omni.usd, "pxr": pxr,
        "isaacsim": ModuleType("isaacsim"), "isaacsim.storage": ModuleType("isaacsim.storage"),
        "isaacsim.storage.native": storage, "bridge_extension": bridge_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.syspath_prepend(str(_ROBOT_DIR))
    spec = importlib.util.spec_from_file_location("_launcher_scene_under_test", _ROBOT_DIR / "_launcher.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "RuntimeCommanderServer", Mock())

    launcher = module.SimLauncher(
        Mock(), Mock(), Mock(), Mock(), Mock(), Mock(), object(), Mock(),
        cameras_enabled=False, frame_rate_hz=60,
        render_mode="RealTimePathTracing", anti_aliasing=3,
    )
    # Specced against the real bridge: a bare Mock invents any attribute, so a
    # call to a method the bridge no longer has would pass silently.
    bridge = Mock(spec=["bind", "unbind", "step", "shutdown", "is_ready"])
    launcher._extension = bridge
    launcher._runtime_robot = object()
    return SimpleNamespace(launcher=launcher, stage=stage, bridge=bridge)


def _load(scene, path, scale=None):
    command = {"path": path}
    if scale is not None:
        command["scale"] = scale
    scene.launcher._runtime_load_isaac_scene(command)


def test_first_scene_load_references_the_asset_and_keeps_the_physics_views(scene):
    _load(scene, "Isaac/Environments/Simple_Warehouse/warehouse.usd", [2.0, 2.0, 2.0])

    prim = scene.stage.prims[_SCENE]
    assert prim.references == [f"{_ASSET_ROOT}/Isaac/Environments/Simple_Warehouse/warehouse.usd"]
    assert prim.scale == (2.0, 2.0, 2.0)
    assert scene.stage.removed == []
    scene.bridge.unbind.assert_not_called()
    assert scene.launcher._runtime_robot is not None


def test_loading_another_scene_replaces_the_current_one_and_invalidates_the_views(scene, caplog):
    caplog.set_level(logging.INFO)
    _load(scene, "Isaac/Environments/Simple_Warehouse/warehouse.usd")
    _load(scene, "Isaac/Environments/Simple_Warehouse/full_warehouse.usd")

    prim = scene.stage.prims[_SCENE]
    assert prim.references == [f"{_ASSET_ROOT}/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"]
    assert prim.scale == (1.0, 1.0, 1.0)
    assert scene.stage.removed == [_SCENE]
    scene.bridge.unbind.assert_called_once_with()
    assert scene.launcher._runtime_robot is None
    assert f"Replacing runtime scene {_SCENE}" in caplog.text


def test_scene_scale_needs_three_values(scene):
    with pytest.raises(ValueError, match="exactly 3 values"):
        _load(scene, "Isaac/Environments/Grid/default_environment.usd", [1.0, 1.0])
    assert _SCENE not in scene.stage.prims


def test_clear_scene_invalidates_the_views_only_when_a_scene_was_loaded(scene, caplog):
    caplog.set_level(logging.INFO)
    scene.launcher._runtime_clear_scene()
    scene.bridge.unbind.assert_not_called()
    assert "No runtime scene to remove" in caplog.text

    _load(scene, "Isaac/Environments/Office/office.usd")
    scene.launcher._runtime_clear_scene()
    assert _SCENE not in scene.stage.prims
    scene.bridge.unbind.assert_called_once_with()
    assert scene.launcher._runtime_robot is None
    assert f"Removed runtime scene {_SCENE}" in caplog.text


def test_removing_a_runtime_object_invalidates_the_views_but_a_missing_one_does_not(scene, caplog):
    caplog.set_level(logging.INFO)
    scene.launcher._runtime_remove({"name": "obj_missing"})
    scene.bridge.unbind.assert_not_called()
    assert "Runtime object 'obj_missing' does not exist" in caplog.text

    scene.stage.DefinePrim("/World/RuntimeObjects/obj_1", "Xform")
    scene.launcher._runtime_remove({"name": "obj_1"})
    assert "/World/RuntimeObjects/obj_1" not in scene.stage.prims
    scene.bridge.unbind.assert_called_once_with()
    assert "Removed runtime object 'obj_1'" in caplog.text


def test_removals_before_the_bridge_exists_only_drop_the_commander_robot(scene):
    scene.launcher._extension = None
    _load(scene, "Isaac/Environments/Office/office.usd")
    scene.launcher._runtime_clear_scene()
    assert scene.stage.removed == [_SCENE]
    assert scene.launcher._runtime_robot is None
