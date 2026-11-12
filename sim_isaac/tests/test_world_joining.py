"""What is authored on a robot as it joins, against real USD: the posture its
model starts in, written to each joint's PhysX state and to the target of
each joint's drive in the units USD keeps them in, each finger that mimics
another bound to it, and gravity taken off the robot's own links and off
nothing else on the stage."""

import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import robot_description_py

pytest.importorskip("pxr", reason="USD Python wheels are unavailable on this platform")

from pxr import Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402  pylint: disable=C0413
from entries import entry  # noqa: E402  pylint: disable=C0413
from sim_robot_core.models import FingerSpan, Gripper, Mimic, ModelEntry  # noqa: E402  pylint: disable=C0413

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
    rig = entry("rig", arms={"arm": list(joints)}, start_posture=posture)
    return isaac_models.stood(
        "rig", rig, isaac_models.parse("rig", {"stage": "rig/rig.usd", "articulation_root": "."}), {}, {}
    )


class TestStartPosture:
    def test_an_so101_starts_at_ready_and_is_held_there(self):
        world = world_module()
        stage, root = _so101_stage()
        so101 = known("so101")

        world.World._start_in_posture(root, so101)

        # The arm at its ready state and the jaw closed.
        posture = so101.entry.start_posture
        assert set(posture) == {*_ARM_JOINTS, "gripper"}
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
        # A joint of the stage that no limb of the description moves.
        UsdPhysics.RevoluteJoint.Define(stage, "/World/charlo/joints/idler")
        so101 = known("so101")

        world.World._start_in_posture(root, so101)

        assert "idler" not in so101.entry.start_posture
        assert not _state(stage, "idler", "angular").IsValid()
        assert not _target(stage, "idler", "angular").IsValid()

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

        with pytest.raises(RuntimeError, match="a robot's limbs move revolute and prismatic joints"):
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


def _openarm_v2_stage():
    """An OpenArm v2 as its entry expects it on the stage: a revolute joint
    along X named after each joint of the shared entry, under a scope."""
    stage = Usd.Stage.CreateInMemory()
    root = stage.DefinePrim("/World/charlo", "Xform")
    for joint in known("openarm_v2").entry.joints():
        UsdPhysics.RevoluteJoint.Define(stage, f"/World/charlo/joints/{joint}")
    return stage, root


def _mimic(stage, joint: str, instance: str):
    """The gearing, the offset and the joint followed of the mimic `joint`
    of /World/charlo carries on its degree of freedom `instance`."""
    prim = stage.GetPrimAtPath(f"/World/charlo/joints/{joint}")
    prefix = f"physxMimicJoint:{instance}"
    return (
        prim.GetAttribute(f"{prefix}:gearing").Get(),
        prim.GetAttribute(f"{prefix}:offset").Get(),
        prim.GetRelationship(f"{prefix}:referenceJoint").GetTargets(),
    )


def _rig_of_followers(*pairs):
    """A model whose one gripper drives the first joint of each of `pairs`,
    (followed, follower, multiplier, offset), and has the second follow it."""
    joints, spans, mimics = [], [], []
    for followed, follower, multiplier, offset in pairs:
        joints += [followed, follower]
        spans += [FingerSpan(closed=0.0, travel=1.0), FingerSpan(closed=offset, travel=multiplier)]
        mimics += [None, Mimic(joint=followed, multiplier=multiplier, offset=offset)]
    rig = ModelEntry(
        model="rig",
        arms=(),
        grippers=(
            Gripper(name="gripper", joints=tuple(joints), spans=tuple(spans), mimics=tuple(mimics)),
        ),
        cameras=(),
        start_posture={joint: 0.0 for joint in joints},
    )
    raw = {"stage": "rig/rig.usd", "articulation_root": "."}
    return isaac_models.stood("rig", rig, isaac_models.parse("rig", raw), {}, {})


class TestFollowers:
    def test_an_openarms_second_finger_follows_its_first_one_to_one_and_has_no_drive(self):
        world = world_module()
        stage, root = _openarm_v2_stage()
        openarm = known("openarm_v2")

        world.World._start_in_posture(root, openarm)
        world.World._couple_followers(root, openarm)

        for side in ("left", "right"):
            followed, follower = (f"openarm_{side}_finger_joint{n}" for n in (1, 2))
            prim = stage.GetPrimAtPath(f"/World/charlo/joints/{follower}")
            applied = prim.GetMetadata("apiSchemas").ApplyOperations([])
            assert f"{world.MIMIC_JOINT_API}:rotX" in applied
            assert _mimic(stage, follower, "rotX") == (
                -1.0,
                0.0,
                [Sdf.Path(f"/World/charlo/joints/{followed}")],
            )
            # Both start closed; the followed finger's drive holds the pair.
            assert _state(stage, follower, "angular").Get() == pytest.approx(0.0)
            assert not _target(stage, follower, "angular").IsValid()
            assert _target(stage, followed, "angular").Get() == pytest.approx(0.0)

    def test_a_follower_stands_at_its_multiplier_and_offset_in_the_stages_unit(self):
        world = world_module()
        stage = Usd.Stage.CreateInMemory()
        root = stage.DefinePrim("/World/charlo", "Xform")
        for name in ("hinge", "hinge_mirror"):
            UsdPhysics.RevoluteJoint.Define(stage, f"/World/charlo/joints/{name}")
        for name in ("slide", "slide_mirror"):
            UsdPhysics.PrismaticJoint.Define(stage, f"/World/charlo/joints/{name}").CreateAxisAttr("Y")
        rig = _rig_of_followers(("hinge", "hinge_mirror", -1.0, 0.1), ("slide", "slide_mirror", 2.0, 0.01))

        world.World._couple_followers(root, rig)

        # position + gearing * followed + offset = 0, the offset in degrees on
        # a revolute joint and in metres on a prismatic one.
        gearing, offset, followed = _mimic(stage, "hinge_mirror", "rotX")
        assert (gearing, followed) == (1.0, [Sdf.Path("/World/charlo/joints/hinge")])
        assert offset == pytest.approx(-math.degrees(0.1))
        gearing, offset, followed = _mimic(stage, "slide_mirror", "rotY")
        assert (gearing, followed) == (-2.0, [Sdf.Path("/World/charlo/joints/slide")])
        assert offset == pytest.approx(-0.01)

    def test_a_follower_of_another_kind_than_the_joint_it_follows_is_refused(self):
        world = world_module()
        stage = Usd.Stage.CreateInMemory()
        root = stage.DefinePrim("/World/charlo", "Xform")
        UsdPhysics.PrismaticJoint.Define(stage, "/World/charlo/joints/slide")
        UsdPhysics.RevoluteJoint.Define(stage, "/World/charlo/joints/hinge")

        with pytest.raises(
            RuntimeError,
            match=r"the rig entry has 'hinge' follow 'slide', and rig/rig\.usd makes one angular and "
            r"the other linear: a finger follows a joint of its own kind",
        ):
            world.World._couple_followers(root, _rig_of_followers(("slide", "hinge", 1.0, 0.0)))

    def test_a_follower_the_stage_lacks_is_refused_with_its_name(self):
        world = world_module()
        stage = Usd.Stage.CreateInMemory()
        root = stage.DefinePrim("/World/charlo", "Xform")
        UsdPhysics.RevoluteJoint.Define(stage, "/World/charlo/joints/hinge")

        with pytest.raises(
            RuntimeError,
            match=r"the rig entry couples finger joints not in rig/rig\.usd: \['hinge_mirror'\]",
        ):
            world.World._couple_followers(
                root, _rig_of_followers(("hinge", "hinge_mirror", 1.0, 0.0))
            )


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


def _described_stage(model: str, slack: float = 0.0):
    """A stage of `model` whose joints are its description's: each joint a
    limb moves a revolute or prismatic joint under a scope, its limits the
    description's in USD's units and single precision, each finger that
    follows another `slack` of its travel wider on each side, as an importer
    leaves a mimic joint."""
    model_known = known(model)
    followers = {joint for gripper in model_known.entry.grippers for joint, _ in gripper.followers()}
    kinds = {
        joint.get("name"): joint.get("type")
        for joint in ET.fromstring(robot_description_py.model(model).urdf).findall("joint")
    }
    stage = Usd.Stage.CreateInMemory()
    root = stage.DefinePrim("/World/charlo", "Xform")
    for name, (lower, upper) in model_known.position_limits.items():
        widen = slack * (upper - lower) if name in followers else 0.0
        lower, upper = lower - widen, upper + widen
        path = f"/World/charlo/joints/{name}"
        if kinds[name] == "prismatic":
            joint = UsdPhysics.PrismaticJoint.Define(stage, path)
        else:
            joint = UsdPhysics.RevoluteJoint.Define(stage, path)
            lower, upper = math.degrees(lower), math.degrees(upper)
        joint.CreateLowerLimitAttr(float(np.float32(lower)))
        joint.CreateUpperLimitAttr(float(np.float32(upper)))
    return stage, root, model_known


class TestStageHeldToTheDescription:
    @pytest.mark.parametrize("model", ["openarm_v1", "openarm_v2", "so101"])
    def test_a_stage_of_the_descriptions_joints_and_limits_is_held(self, model):
        world = world_module()
        _, root, model_known = _described_stage(model)

        world.World._hold_to_description(root, model_known)

    def test_a_limit_off_the_descriptions_is_refused_naming_the_joint(self):
        world = world_module()
        stage, root, model_known = _described_stage("openarm_v2")
        joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath("/World/charlo/joints/openarm_left_joint2"))
        joint.GetUpperLimitAttr().Set(joint.GetUpperLimitAttr().Get() + 0.5)

        with pytest.raises(RuntimeError, match=r"openarm/openarm_bimanual_v2\.usd is not the openarm_v2 its description describes: 'openarm_left_joint2' moves over"):
            world.World._hold_to_description(root, model_known)

    @pytest.mark.parametrize("model", ["openarm_v1", "openarm_v2"])
    def test_a_follower_finger_held_wider_than_its_description_is_held(self, model):
        world = world_module()
        _, root, model_known = _described_stage(model, slack=0.1)

        world.World._hold_to_description(root, model_known)

    def test_a_follower_finger_held_narrower_than_its_description_is_refused(self):
        world = world_module()
        _, root, model_known = _described_stage("openarm_v2", slack=-0.1)

        with pytest.raises(RuntimeError, match=r"'openarm_left_finger_joint2' moves over"):
            world.World._hold_to_description(root, model_known)

    def test_a_joint_the_description_fixes_is_held_while_it_is_locked(self):
        world = world_module()
        stage, root, model_known = _described_stage("openarm_v1")
        hand = UsdPhysics.RevoluteJoint.Define(stage, "/World/charlo/joints/openarm_left_hand")
        hand.CreateLowerLimitAttr(0.0)
        hand.CreateUpperLimitAttr(5.7e-13)

        world.World._hold_to_description(root, model_known)

        hand.GetUpperLimitAttr().Set(10.0)
        with pytest.raises(RuntimeError, match=r"'openarm_left_hand' moves over \[0, 0\.174533\], and the description moves no joint"):
            world.World._hold_to_description(root, model_known)

    def test_a_joint_the_stage_lacks_is_refused(self):
        world = world_module()
        stage, root, model_known = _described_stage("so101")
        stage.RemovePrim("/World/charlo/joints/wrist_roll")

        with pytest.raises(RuntimeError, match=r"'wrist_roll' is no revolute or prismatic joint of it"):
            world.World._hold_to_description(root, model_known)
