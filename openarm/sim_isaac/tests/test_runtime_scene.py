"""Runtime scene edits on a fake USD stage: replacing the scene, clearing it,
removing objects, what each does to the cached physics views, and which
runtime commands may touch an object scene_control spawned."""

import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

_ROBOT_DIR = Path(__file__).resolve().parents[1] / "robots" / "openarm"
_SCENE = "/World/RuntimeScene"
_OBJECTS = "/World/RuntimeObjects"
_ASSET_ROOT = "https://assets.example/Isaac/6.1"
_ACTIONS = ("apply_force", "clear_scene", "load_scene", "move_object", "move_robot", "remove_object", "spawn_object")
_SERVICES = {"scene": "get_assets_list", "objects": "get_object_states"}
_PROPS = {
    "props/blocks/red_block": {
        "asset_id": "props/blocks/red_block", "display_name": "red block", "kind": "object",
        "path": "Isaac/Props/Blocks/red_block.usd", "category": "Blocks",
    },
}


class FakePrim:
    def __init__(self, path, valid=True):
        self.path = path
        self.valid = valid
        self.references = []
        self.translate = None
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


class FakeXformOp:
    """Sets the prim attribute of its kind: translate or scale."""

    def __init__(self, prim, kind):
        self.prim = prim
        self.kind = kind

    def GetOpType(self):
        return self.kind

    def Set(self, value):
        setattr(self.prim, self.kind, value)


@pytest.fixture
def scene(monkeypatch):
    stage = FakeStage()

    omni = ModuleType("omni")
    omni.usd = ModuleType("omni.usd")
    omni.usd.get_context = lambda: SimpleNamespace(get_stage=lambda: stage)
    pxr = ModuleType("pxr")
    pxr.Gf = SimpleNamespace(Vec3d=lambda x, y, z: (x, y, z), Vec3f=lambda x, y, z: (x, y, z))
    pxr.Sdf = SimpleNamespace(Path=str)
    pxr.UsdGeom = SimpleNamespace(
        XformOp=SimpleNamespace(TypeTranslate="translate", TypeScale="scale"),
        Xformable=lambda prim: SimpleNamespace(
            GetOrderedXformOps=lambda: [],
            AddTranslateOp=lambda: FakeXformOp(prim, "translate"),
            AddScaleOp=lambda: FakeXformOp(prim, "scale"),
        ),
    )
    # Only imported: every object spawned here is spawned without physics.
    pxr.Usd = SimpleNamespace()
    pxr.UsdPhysics = SimpleNamespace()
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

    scene_actions = Mock()
    launcher = module.SimLauncher(
        Mock(), Path("/robot.usd"), Mock(), Mock(), object(), scene_actions,
        state_rate_hz=60, cameras_enabled=False, frame_rate_hz=60,
        render_mode="RealTimePathTracing", anti_aliasing=3,
    )
    bridge = Mock()
    launcher._extension = bridge
    launcher._runtime_robot = object()
    return SimpleNamespace(launcher=launcher, stage=stage, bridge=bridge, scene_actions=scene_actions)


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
    scene.bridge.invalidate_physics_views.assert_not_called()
    assert scene.launcher._runtime_robot is not None


def test_loading_another_scene_replaces_the_current_one_and_invalidates_the_views(scene, caplog):
    caplog.set_level(logging.INFO)
    _load(scene, "Isaac/Environments/Simple_Warehouse/warehouse.usd")
    _load(scene, "Isaac/Environments/Simple_Warehouse/full_warehouse.usd")

    prim = scene.stage.prims[_SCENE]
    assert prim.references == [f"{_ASSET_ROOT}/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"]
    assert prim.scale == (1.0, 1.0, 1.0)
    assert scene.stage.removed == [_SCENE]
    scene.bridge.invalidate_physics_views.assert_called_once_with()
    assert scene.launcher._runtime_robot is None
    assert f"Replacing runtime scene {_SCENE}" in caplog.text


def test_scene_scale_needs_three_values(scene):
    with pytest.raises(ValueError, match="exactly 3 values"):
        _load(scene, "Isaac/Environments/Grid/default_environment.usd", [1.0, 1.0])
    assert _SCENE not in scene.stage.prims


def test_clear_scene_invalidates_the_views_only_when_a_scene_was_loaded(scene, caplog):
    caplog.set_level(logging.INFO)
    scene.launcher._runtime_clear_scene()
    scene.bridge.invalidate_physics_views.assert_not_called()
    assert "No runtime scene to remove" in caplog.text

    _load(scene, "Isaac/Environments/Office/office.usd")
    scene.launcher._runtime_clear_scene()
    assert _SCENE not in scene.stage.prims
    scene.bridge.invalidate_physics_views.assert_called_once_with()
    assert scene.launcher._runtime_robot is None
    assert f"Removed runtime scene {_SCENE}" in caplog.text


def test_removing_a_runtime_object_invalidates_the_views_but_a_missing_one_does_not(scene, caplog):
    caplog.set_level(logging.INFO)
    scene.launcher._runtime_remove({"name": "obj_missing"})
    scene.bridge.invalidate_physics_views.assert_not_called()
    assert "Runtime object 'obj_missing' does not exist" in caplog.text

    scene.stage.DefinePrim("/World/RuntimeObjects/obj_1", "Xform")
    scene.launcher._runtime_remove({"name": "obj_1"})
    assert "/World/RuntimeObjects/obj_1" not in scene.stage.prims
    scene.bridge.invalidate_physics_views.assert_called_once_with()
    assert "Removed runtime object 'obj_1'" in caplog.text


def test_removals_before_the_bridge_exists_only_drop_the_commander_robot(scene):
    scene.launcher._extension = None
    _load(scene, "Isaac/Environments/Office/office.usd")
    scene.launcher._runtime_clear_scene()
    assert scene.stage.removed == [_SCENE]
    assert scene.launcher._runtime_robot is None


def test_every_removal_drops_the_rigid_body_view_the_object_state_reads(scene):
    scene.stage.DefinePrim("/World/RuntimeObjects/obj_1", "Xform")
    scene.launcher._runtime_remove({"name": "obj_1"})
    scene.launcher._runtime_remove({"name": "obj_missing"})
    assert scene.scene_actions.invalidate_physics_views.call_count == 1

    scene.launcher._extension = None
    _load(scene, "Isaac/Environments/Office/office.usd")
    scene.launcher._runtime_clear_scene()
    assert scene.scene_actions.invalidate_physics_views.call_count == 2


class StageReader:
    """The object reader on the fake stage: each spawned object as the prim
    at its path has it, or a failure naming the prim a capture cannot find,
    as the real reader fails."""

    def __init__(self, stage):
        self._stage = stage

    def invalidate(self):
        pass

    def read(self, spawned):
        records = []
        for obj in spawned:
            path = f"{_OBJECTS}/{obj['object_id']}"
            prim = self._stage.GetPrimAtPath(path)
            if not prim.IsValid():
                raise RuntimeError(f"no prim at {path}")
            records.append((obj["object_id"], list(prim.references), prim.translate))
        return records


@pytest.fixture
def scene_control(scene, monkeypatch):
    """The real scene_control provider as the launcher's, reading the fake stage."""
    actions = ModuleType("peppygen.exposed_actions.scene")
    for name in _ACTIONS:
        setattr(actions, name, ModuleType(f"{actions.__name__}.{name}"))
    modules = {
        "peppygen": ModuleType("peppygen"),
        "peppygen.exposed_actions": ModuleType("peppygen.exposed_actions"),
        actions.__name__: actions,
        "peppygen.exposed_services": ModuleType("peppygen.exposed_services"),
    }
    for link, name in _SERVICES.items():
        services = ModuleType(f"peppygen.exposed_services.{link}")
        setattr(services, name, ModuleType(f"{services.__name__}.{name}"))
        modules[services.__name__] = services
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location("_scene_control_under_test", _ROBOT_DIR / "scene_actions.py")
    module = importlib.util.module_from_spec(spec)
    # Its dataclass resolves postponed annotations through sys.modules.
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)

    io = module.SceneActionIO(object(), Mock(), SimpleNamespace(capture_timestamp_s=lambda: 10.0))
    io._object_reader = StageReader(scene.stage)
    io.set_assets(_PROPS)
    scene.launcher._scene_actions = io
    return io


def _spawn(scene, scene_control):
    result = scene_control._execute(scene.launcher, "spawn_object", {
        "asset_id": "props/blocks/red_block", "position": [0.5, 0.0, 0.8], "scale": 1.0, "physics": "none", "mass": 0.1,
    })
    assert result["success"]
    return result["object_id"]


def _captured(scene_control):
    snapshot = scene_control.capture_object_states()
    assert snapshot is not None, scene_control._unavailable
    return list(snapshot.objects)


@pytest.mark.parametrize("command", [
    {"command": "remove"},
    {"command": "spawn_usd", "path": "/elsewhere/prop.usd", "position": [1.0, 0.0, 0.8]},
    {"command": "spawn_isaac_asset", "path": "Isaac/Props/Blocks/blue_block.usd", "position": [1.0, 0.0, 0.8]},
], ids=["remove", "spawn_usd", "spawn_isaac_asset"])
def test_the_runtime_commander_refuses_to_remove_or_replace_a_scene_control_object(scene, scene_control, command):
    object_id = _spawn(scene, scene_control)
    spawned = [(object_id, [f"{_ASSET_ROOT}/Isaac/Props/Blocks/red_block.usd"], (0.5, 0.0, 0.8))]
    assert _captured(scene_control) == spawned

    with pytest.raises(ValueError, match=f"'{object_id}' is a scene_control object; edit it through scene_control"):
        scene.launcher.execute_runtime_command({**command, "name": object_id})

    # Nothing left the stage or the registry, so the capture still lists it.
    assert scene.stage.removed == []
    scene.bridge.invalidate_physics_views.assert_not_called()
    assert scene_control.owns(object_id)
    assert _captured(scene_control) == spawned


def test_the_runtime_commander_still_moves_a_scene_control_object(scene, scene_control):
    object_id = _spawn(scene, scene_control)

    scene.launcher.execute_runtime_command({"command": "move_object", "name": object_id, "position": [0.2, 0.3, 0.9]})

    assert [(record[0], record[2]) for record in _captured(scene_control)] == [(object_id, (0.2, 0.3, 0.9))]


def test_the_runtime_commander_still_spawns_and_removes_other_names(scene, scene_control, tmp_path):
    object_id = _spawn(scene, scene_control)
    usd = tmp_path / "prop.usd"
    usd.write_text("#usda 1.0\n")

    scene.launcher.execute_runtime_command({
        "command": "spawn_usd", "name": "MyObject", "path": str(usd), "position": [1.0, 0.0, 0.8],
    })
    prim = scene.stage.prims[f"{_OBJECTS}/MyObject"]
    assert (prim.references, prim.translate) == ([str(usd.resolve())], (1.0, 0.0, 0.8))

    scene.launcher.execute_runtime_command({"command": "remove", "name": "MyObject"})
    assert f"{_OBJECTS}/MyObject" not in scene.stage.prims
    # The commander's own objects never enter the object state.
    assert [record[0] for record in _captured(scene_control)] == [object_id]


def test_scene_control_still_removes_its_own_objects(scene, scene_control):
    object_id = _spawn(scene, scene_control)

    result = scene_control._execute(scene.launcher, "remove_object", {"object_id": object_id})

    assert result == {"success": True, "message": f"Removed {object_id}"}
    assert scene.stage.removed == [f"{_OBJECTS}/{object_id}"]
    assert not scene_control.owns(object_id)
    assert _captured(scene_control) == []
