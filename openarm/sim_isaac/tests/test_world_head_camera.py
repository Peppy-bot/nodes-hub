"""A robot's head camera seats under the prim it stands at, from the pack
its model draws: the v2 draws one, the v1 none, and a robot whose head
camera cannot be seated leaves nothing on the stage. The stage is stood in;
what the pack does to the stage is test_head_camera.py's."""

import sys
from types import ModuleType
from unittest.mock import Mock

import pytest

from _world import stand_in_pack, world_module


@pytest.fixture(name="world")
def _world(monkeypatch, tmp_path):
    module = world_module()
    # One recorder for the stage and the attach, so the order they are
    # called in is read off it.
    calls = Mock()
    stage = calls.stage
    context = Mock()
    context.get_stage.return_value = stage
    usd = ModuleType("omni.usd")
    usd.get_context = Mock(return_value=context)
    monkeypatch.setitem(sys.modules, "omni.usd", usd)
    # `import omni.usd` binds the package, so the submodule rides on it.
    monkeypatch.setattr(sys.modules["omni"], "usd", usd, raising=False)
    monkeypatch.setattr(module.World, "_place", staticmethod(lambda prim, position, yaw: None))
    attach = calls.attach
    attach.side_effect = lambda stage, root, pack: f"{root}/openarm_body_link0/openarm_head_camera"
    monkeypatch.setattr(module.head_camera, "attach", attach)
    pack = stand_in_pack(module.head_camera, tmp_path / "head_camera")
    stages = {"openarm_v1": tmp_path / "v1.usd", "openarm_v2": tmp_path / "v2.usd"}
    for path in stages.values():
        path.write_text("stage")
    catalogue = module.Catalogue(stages, head_camera_pack=pack)
    return module.World(catalogue), calls, pack, module


def test_each_v2_draws_its_head_camera_under_its_own_prim_once_referenced(world):
    stood, calls, pack, module = world

    bravo = stood.add("bravo", "openarm_v2", module.Placement.of([0.0, -1.5, 0.0], 0.0))
    charlie = stood.add("charlie", "openarm_v2", module.Placement.of([1.5, 0.0, 0.0], 0.0))

    assert calls.attach.call_args_list == [
        ((calls.stage, bravo.prim(), pack),),
        ((calls.stage, charlie.prim(), pack),),
    ]
    assert [bravo.prim(), charlie.prim()] == ["/World/bravo", "/World/charlie"]
    # The head camera seats on the pedestal the reference brings in.
    names = [name for name, _, _ in calls.mock_calls]
    referenced = names.index("stage.DefinePrim().GetReferences().AddReference")
    assert names.index("attach") > referenced


def test_a_v1_draws_none(world):
    stood, calls, _, module = world

    stood.add("charlie", "openarm_v1", module.Placement.of([1.5, 0.0, 0.0], 0.0))

    calls.attach.assert_not_called()


def test_a_robot_whose_head_camera_cannot_be_seated_leaves_nothing_behind(world):
    stood, calls, _, module = world
    calls.attach.side_effect = RuntimeError("head camera link 'openarm_body_link0' is not under /World/bravo")
    spot = module.Placement.of([0.0, -1.5, 0.0], 0.0)

    with pytest.raises(RuntimeError, match="openarm_body_link0"):
        stood.add("bravo", "openarm_v2", spot)

    calls.stage.RemovePrim.assert_called_once_with("/World/bravo")
    assert stood.robots() == []
    assert not stood.occupied(spot)
