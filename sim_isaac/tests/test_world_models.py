"""What a robot's model asks of the stage as it joins, on a stage that is
stood in: the stage its entry names is referenced under the prim it stands
at, the head camera seats there for a model that draws one and for no other,
and a robot whose head camera cannot be seated leaves nothing on the stage.
What the pack does to the stage is test_head_camera.py's, and what a posture
and gravity compensation author is test_world_joining.py's."""

import sys
from types import ModuleType
from unittest.mock import Mock

import pytest

from _world import known, stand_in_pack, world_module


@pytest.fixture(name="world")
def _world(monkeypatch, tmp_path):
    module = world_module()
    # One recorder for the stage and what is authored on it, so the order
    # they are called in is read off it.
    calls = Mock()
    stage = calls.stage
    context = Mock()
    context.get_stage.return_value = stage
    usd = ModuleType("omni.usd")
    usd.get_context = Mock(return_value=context)
    monkeypatch.setitem(sys.modules, "omni.usd", usd)
    # `import omni.usd` binds the package, so the submodule rides on it.
    monkeypatch.setattr(sys.modules["omni"], "usd", usd, raising=False)
    monkeypatch.setattr(module.World, "place", staticmethod(lambda prim, position, yaw: None))
    monkeypatch.setattr(module.World, "_start_in_posture", staticmethod(calls.start_in_posture))
    calls.compensate_gravity.return_value = 0
    monkeypatch.setattr(module.World, "_compensate_gravity", staticmethod(calls.compensate_gravity))
    attach = calls.attach
    attach.side_effect = lambda stage, root, pack: f"{root}/openarm_body_link0/openarm_head_camera"
    monkeypatch.setattr(module.head_camera, "attach", attach)
    pack = stand_in_pack(module.head_camera, tmp_path / "head_camera")
    # Every model's stage is on disk where its entry names it.
    monkeypatch.setattr(sys.modules["isaac_models"], "ASSETS_DIR", tmp_path)
    for model in ("openarm_v1", "openarm_v2", "so101"):
        path = tmp_path / known(model).stage
        path.parent.mkdir(exist_ok=True)
        path.write_text("stage")
    return module.World(pack), calls, pack, module


def test_each_v2_draws_its_head_camera_under_its_own_prim_once_referenced(world):
    stood, calls, pack, module = world

    bravo = stood.add("bravo", known("openarm_v2"), module.Placement.of([0.0, -1.5, 0.0], 0.0))
    charlie = stood.add("charlie", known("openarm_v2"), module.Placement.of([1.5, 0.0, 0.0], 0.0))

    assert calls.attach.call_args_list == [
        ((calls.stage, bravo.prim(), pack),),
        ((calls.stage, charlie.prim(), pack),),
    ]
    assert [bravo.prim(), charlie.prim()] == ["/World/bravo", "/World/charlie"]
    # The head camera seats on the pedestal the reference brings in.
    names = [name for name, _, _ in calls.mock_calls]
    referenced = names.index("stage.DefinePrim().GetReferences().AddReference")
    assert names.index("attach") > referenced


@pytest.mark.parametrize("model", ["openarm_v1", "so101"])
def test_a_model_whose_entry_asks_for_no_head_camera_draws_none(world, model):
    stood, calls, _, module = world

    stood.add("charlie", known(model), module.Placement.of([1.5, 0.0, 0.0], 0.0))

    calls.attach.assert_not_called()


def test_a_robot_whose_head_camera_cannot_be_seated_leaves_nothing_behind(world):
    stood, calls, _, module = world
    calls.attach.side_effect = RuntimeError("head camera link 'openarm_body_link0' is not under /World/bravo")
    spot = module.Placement.of([0.0, -1.5, 0.0], 0.0)

    with pytest.raises(RuntimeError, match="openarm_body_link0"):
        stood.add("bravo", known("openarm_v2"), spot)

    calls.stage.RemovePrim.assert_called_once_with("/World/bravo")
    assert stood.robots() == []
    assert not stood.occupied(spot)


def test_a_model_that_draws_the_head_camera_is_not_stood_without_its_pack(world):
    """No model of the engine drew the pack at setup, so none was staged."""
    _, calls, _, module = world
    stood = module.World(None)

    with pytest.raises(RuntimeError, match="openarm_v2 draws the head camera, and no pack was staged"):
        stood.add("bravo", known("openarm_v2"), module.Placement.of([0.0, 0.0, 0.0], 0.0))

    calls.stage.RemovePrim.assert_called_once_with("/World/bravo")
    assert stood.robots() == []


def test_a_world_with_no_pack_stands_the_models_that_draw_none(world):
    _, calls, _, module = world
    stood = module.World(None)

    charlo = stood.add("charlo", known("so101"), module.Placement.of([0.0, 0.0, 0.0], 0.0))

    assert (charlo.model, charlo.prim()) == ("so101", "/World/charlo")
    calls.attach.assert_not_called()


def test_each_robot_references_the_stage_its_own_model_names(world, tmp_path):
    stood, calls, _, module = world

    stood.add("alpha", known("openarm_v1"), module.Placement.of([0.0, 0.0, 0.0], 0.0))
    stood.add("charlo", known("so101"), module.Placement.of([1.5, 0.0, 0.0], 0.0))

    references = calls.stage.DefinePrim.return_value.GetReferences.return_value.AddReference
    assert [args for args, _ in references.call_args_list] == [
        (str(tmp_path / "openarm" / "openarm_bimanual.usd"),),
        (str(tmp_path / "so101" / "so101.usd"),),
    ]
    assert [args for args, _ in calls.stage.DefinePrim.call_args_list] == [
        ("/World/alpha", "Xform"),
        ("/World/charlo", "Xform"),
    ]


def test_a_model_whose_stage_is_not_baked_is_not_stood(world, tmp_path):
    stood, calls, _, module = world
    (tmp_path / "so101" / "so101.usd").unlink()

    with pytest.raises(FileNotFoundError, match=r"so101\.usd is missing"):
        stood.add("charlo", known("so101"), module.Placement.of([0.0, 0.0, 0.0], 0.0))

    assert stood.robots() == []


def test_gravity_is_compensated_for_the_models_whose_entry_asks_for_it(world):
    stood, calls, _, module = world
    prim = calls.stage.DefinePrim.return_value

    stood.add("alpha", known("openarm_v2"), module.Placement.of([0.0, 0.0, 0.0], 0.0))
    calls.compensate_gravity.assert_called_once_with(prim)

    # The SO-101's drives hold it against gravity, as its servos do.
    calls.compensate_gravity.reset_mock()
    stood.add("charlo", known("so101"), module.Placement.of([1.5, 0.0, 0.0], 0.0))
    calls.compensate_gravity.assert_not_called()


def test_every_robot_is_put_in_the_posture_its_own_model_starts_in(world):
    stood, calls, _, module = world
    prim = calls.stage.DefinePrim.return_value

    stood.add("alpha", known("openarm_v2"), module.Placement.of([0.0, 0.0, 0.0], 0.0))
    stood.add("charlo", known("so101"), module.Placement.of([1.5, 0.0, 0.0], 0.0))

    assert [args for args, _ in calls.start_in_posture.call_args_list] == [
        (prim, known("openarm_v2")),
        (prim, known("so101")),
    ]
