"""What is authored on a robot as it joins, against real USD: the posture its
model starts in, written to each joint's PhysX state and its drive's target
in the units USD keeps them in, and gravity taken off the robot's own links
and off nothing else on the stage."""

import math
import sys
from pathlib import Path

import pytest

pytest.importorskip("pxr", reason="USD Python wheels are unavailable on this platform")

from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402  pylint: disable=C0413
from sim_robot_core.models import EngineModel, parse_entry  # noqa: E402  pylint: disable=C0413

from _world import known, world_module  # noqa: E402  pylint: disable=C0413

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))

import isaac_models  # noqa: E402  pylint: disable=C0413

_ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def _so101_stage():
    """An SO-101 as its entry expects it on the stage: a prim per link under
    the robot's prim, and a revolute joint named after each joint of the
    shared entry, kept under a scope as an importer leaves them."""
    stage = Usd.Stage.CreateInMemory()
    root = stage.DefinePrim("/World/charlo", "Xform")
    for link in ("base_link", "shoulder_link", "gripper_link"):
        body = UsdGeom.Xform.Define(stage, f"/World/charlo/{link}").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body)
    for joint in (*_ARM_JOINTS, "gripper"):
        UsdPhysics.RevoluteJoint.Define(stage, f"/World/charlo/joints/{joint}")
    return stage, root


def _state(stage, joint: str, kind: str):
    return stage.GetPrimAtPath(f"/World/charlo/joints/{joint}").GetAttribute(
        f"state:{kind}:physics:position"
    )


def _target(stage, joint: str, kind: str):
    prim = stage.GetPrimAtPath(f"/World/charlo/joints/{joint}")
    return UsdPhysics.DriveAPI(prim, kind).GetTargetPositionAttr()


def _model_starting_in(posture: dict, joints=("slide", "hinge")):
    """A model whose one arm moves `joints` and starts in `posture`."""
    entry = parse_entry(
        "rig", "rig.json5", {"arms": [{"name": "arm", "joints": list(joints)}], "start_posture": posture}
    )
    return isaac_models.parse(
        EngineModel(entry=entry, engine={"stage": "rig/rig.usd", "articulation_root": "."})
    )


class TestStartPosture:
    def test_an_so101_starts_folded_and_is_held_there(self):
        world = world_module()
        stage, root = _so101_stage()
        so101 = known("so101")

        world.World._start_in_posture(root, so101)

        posture = so101.entry.start_posture
        assert set(_ARM_JOINTS) <= set(posture)
        for joint, position in posture.items():
            # USD keeps a revolute joint's position in degrees.
            degrees = pytest.approx(math.degrees(position), abs=1e-4)
            assert _state(stage, joint, "angular").Get() == degrees
            assert _target(stage, joint, "angular").Get() == degrees
            prim = stage.GetPrimAtPath(f"/World/charlo/joints/{joint}")
            # usd-core does not register PhysX's schemas, so the applied
            # schema is read as authored.
            applied = prim.GetMetadata("apiSchemas").ApplyOperations([])
            assert f"{world.JOINT_STATE_API}:angular" in applied
            assert "PhysicsDriveAPI:angular" in applied

    def test_a_joint_the_posture_does_not_name_is_left_as_its_stage_has_it(self):
        world = world_module()
        stage, root = _so101_stage()
        so101 = known("so101")

        world.World._start_in_posture(root, so101)

        for joint in set((*_ARM_JOINTS, "gripper")) - set(so101.entry.start_posture):
            assert not _state(stage, joint, "angular").IsValid()

    def test_a_model_with_no_posture_starts_where_its_stage_puts_it(self):
        world = world_module()
        stage = Usd.Stage.CreateInMemory()
        root = stage.DefinePrim("/World/alpha", "Xform")
        UsdPhysics.RevoluteJoint.Define(stage, "/World/alpha/joints/openarm_left_joint1")
        before = stage.GetRootLayer().ExportToString()

        world.World._start_in_posture(root, known("openarm_v2"))

        assert known("openarm_v2").entry.start_posture == {}
        assert stage.GetRootLayer().ExportToString() == before

    def test_a_prismatic_joint_starts_in_metres_and_a_revolute_one_in_degrees(self):
        world = world_module()
        stage = Usd.Stage.CreateInMemory()
        root = stage.DefinePrim("/World/charlo", "Xform")
        UsdPhysics.PrismaticJoint.Define(stage, "/World/charlo/joints/slide")
        UsdPhysics.RevoluteJoint.Define(stage, "/World/charlo/joints/hinge")

        world.World._start_in_posture(
            root, _model_starting_in({"slide": 0.02, "hinge": math.pi / 2})
        )

        assert _state(stage, "slide", "linear").Get() == pytest.approx(0.02)
        assert _target(stage, "slide", "linear").Get() == pytest.approx(0.02)
        assert _state(stage, "hinge", "angular").Get() == pytest.approx(90.0)
        assert _target(stage, "hinge", "angular").Get() == pytest.approx(90.0)

    def test_a_posture_naming_a_joint_the_stage_lacks_is_refused_with_its_name(self):
        world = world_module()
        stage, root = _so101_stage()
        stage.RemovePrim("/World/charlo/joints/wrist_roll")

        with pytest.raises(
            RuntimeError,
            match=r"the so101 entry starts joints not in so101/so101\.usd: \['wrist_roll'\]",
        ):
            world.World._start_in_posture(root, known("so101"))

    def test_a_link_named_like_a_joint_is_not_taken_for_it(self):
        world = world_module()
        stage, root = _so101_stage()
        stage.RemovePrim("/World/charlo/joints/wrist_roll")
        UsdGeom.Xform.Define(stage, "/World/charlo/wrist_roll")

        with pytest.raises(RuntimeError, match=r"\['wrist_roll'\]"):
            world.World._start_in_posture(root, known("so101"))

    def test_a_joint_a_posture_cannot_place_is_refused(self):
        world = world_module()
        stage = Usd.Stage.CreateInMemory()
        root = stage.DefinePrim("/World/charlo", "Xform")
        UsdPhysics.SphericalJoint.Define(stage, "/World/charlo/joints/ball")

        with pytest.raises(RuntimeError, match="a posture places revolute and prismatic joints"):
            world.World._start_in_posture(
                root, _model_starting_in({"ball": 0.1}, joints=("ball",))
            )

    def test_each_robot_is_placed_on_its_own_joints(self):
        """Two robots of one model share every joint name, and each is
        placed under its own prim."""
        world = world_module()
        stage, root = _so101_stage()
        other = stage.DefinePrim("/World/delta", "Xform")
        UsdPhysics.RevoluteJoint.Define(stage, "/World/delta/joints/shoulder_pan")

        world.World._start_in_posture(root, known("so101"))

        assert other.IsValid()
        untouched = stage.GetPrimAtPath("/World/delta/joints/shoulder_pan")
        assert not untouched.GetAttribute("state:angular:physics:position").IsValid()


class TestGravityCompensation:
    def test_gravity_comes_off_every_link_of_the_robot_and_off_nothing_else(self):
        world = world_module()
        stage, root = _so101_stage()
        # A spawned object, and another robot, beside it on the stage.
        crate = UsdGeom.Xform.Define(stage, "/World/RuntimeObjects/crate").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(crate)
        other = UsdGeom.Xform.Define(stage, "/World/delta/base_link").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(other)

        compensated = world.World._compensate_gravity(root)

        assert compensated == 3
        for link in ("base_link", "shoulder_link", "gripper_link"):
            prim = stage.GetPrimAtPath(f"/World/charlo/{link}")
            assert prim.GetAttribute(world.DISABLE_GRAVITY).Get() is True
            applied = prim.GetMetadata("apiSchemas").ApplyOperations([])
            assert world.PHYSX_RIGID_BODY_API in applied
        for prim in (crate, other):
            assert not prim.GetAttribute(world.DISABLE_GRAVITY).IsValid()

    def test_a_prim_that_is_no_rigid_body_is_left_alone(self):
        world = world_module()
        stage, root = _so101_stage()

        world.World._compensate_gravity(root)

        scope = stage.GetPrimAtPath("/World/charlo/joints")
        assert not scope.GetAttribute(world.DISABLE_GRAVITY).IsValid()
        assert not root.GetAttribute(world.DISABLE_GRAVITY).IsValid()

    def test_links_are_reached_by_what_they_are_not_by_what_they_are_called(self):
        """The SO-101's links carry no robot's prefix."""
        world = world_module()
        stage, root = _so101_stage()

        assert world.World._compensate_gravity(root) == 3
