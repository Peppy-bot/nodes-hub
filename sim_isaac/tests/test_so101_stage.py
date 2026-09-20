"""The SO-101 stage scripts/build_so101.py writes, held to so101_description:
link and joint names, joint limits, and the tool frame's pose through the
chain.

The stage is built here from the staged upstream model when
SO101_UPSTREAM_MODEL names it, and read as baked under PEPPY_ROBOT_ASSETS_DIR
otherwise (the base image's suite). The description's URDF comes from
SO101_KINEMATICS_URDF: so101_description needs a newer python than Isaac
Sim's, so it is not a dependency of this project. Without the URDF or a
stage the suite is skipped, and needs mujoco, which the builder runs on.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("pxr")
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402  pylint: disable=C0413

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_so101  # noqa: E402  pylint: disable=C0413

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
LINKS = (
    "base_link",
    "shoulder_link",
    "upper_arm_link",
    "lower_arm_link",
    "wrist_link",
    "gripper_link",
    "moving_jaw_so101_v1_link",
)


@pytest.fixture(scope="module")
def urdf() -> Path:
    value = os.environ.get("SO101_KINEMATICS_URDF")
    if not value:
        pytest.skip("SO101_KINEMATICS_URDF names so101_description's URDF (README, the SO-101 stage)")
    return Path(value)


@pytest.fixture(scope="module")
def stage(tmp_path_factory, urdf) -> Usd.Stage:
    upstream = os.environ.get("SO101_UPSTREAM_MODEL")
    if upstream:
        output = tmp_path_factory.mktemp("so101_stage")
        path = build_so101.build_stage(Path(upstream), urdf, output)
    else:
        assets = os.environ.get("PEPPY_ROBOT_ASSETS_DIR")
        path = Path(assets) / "so101" / build_so101.STAGE_FILENAME if assets else None
        if path is None or not path.is_file():
            pytest.skip("SO101_UPSTREAM_MODEL names the staged upstream to build from, or PEPPY_ROBOT_ASSETS_DIR a baked stage")
    return Usd.Stage.Open(str(path))


def _matrix(pos, quat_wxyz) -> np.ndarray:
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, np.array(quat_wxyz, dtype=np.float64))
    matrix = np.eye(4)
    matrix[:3, :3] = rotation.reshape(3, 3)
    matrix[:3, 3] = pos
    return matrix


def _quat(value) -> tuple[float, float, float, float]:
    return (value.GetReal(), *value.GetImaginary())


def _local(prim) -> np.ndarray:
    """A prim's local transform as a column-vector matrix."""
    return np.array(UsdGeom.Xformable(prim).GetLocalTransformation()).T


def _rot_z(angle: float) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:2, :2] = [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    return matrix


def tool_frame_in_world(stage: Usd.Stage, positions: dict[str, float]) -> np.ndarray:
    """The tool frame's world pose with the joints at `positions`, walked
    through the stage's joint frames from the base link."""
    world = {"base_link": _local(stage.GetPrimAtPath("/so101/base_link"))}
    for name in JOINTS:
        joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(f"/so101/joints/{name}"))
        parent = joint.GetBody0Rel().GetTargets()[0].name
        child = joint.GetBody1Rel().GetTargets()[0].name
        in_parent = _matrix(joint.GetLocalPos0Attr().Get(), _quat(joint.GetLocalRot0Attr().Get()))
        in_child = _matrix(joint.GetLocalPos1Attr().Get(), _quat(joint.GetLocalRot1Attr().Get()))
        world[child] = world[parent] @ in_parent @ _rot_z(positions[name]) @ np.linalg.inv(in_child)
    tool = stage.GetPrimAtPath("/so101/gripper_link/gripper_frame_link")
    return world["gripper_link"] @ _local(tool)


def urdf_tool_frame_in_world(urdf: Path, positions: dict[str, float]) -> np.ndarray:
    """The URDF's gripper_frame_link for the same joint positions. MuJoCo
    fuses the fixed link into gripper_link, so its fixed joint is composed
    onto that body's pose."""
    model = mujoco.MjModel.from_xml_path(str(urdf))
    data = mujoco.MjData(model)
    for name, position in positions.items():
        data.qpos[model.joint(name).qposadr[0]] = position
    mujoco.mj_forward(model, data)
    body = model.body("gripper_link").id
    _, tool = build_so101.read_urdf(urdf)
    fixed = _matrix(tool.xyz, build_so101.quat_from_rpy(tool.rpy))
    return _matrix(data.xpos[body], data.xquat[body]) @ fixed


def test_the_links_are_the_urdfs(stage):
    links = [
        prim.GetName()
        for prim in stage.GetPrimAtPath("/so101").GetChildren()
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    assert links == list(LINKS)
    assert stage.GetDefaultPrim().GetPath() == "/so101"
    assert UsdGeom.GetStageUpAxis(stage) == "Z"
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0


def test_every_link_has_a_mass_and_something_to_draw(stage):
    for link in LINKS:
        prim = stage.GetPrimAtPath(f"/so101/{link}")
        assert UsdPhysics.MassAPI(prim).GetMassAttr().Get() > 0.0, link
        assert stage.GetPrimAtPath(f"/so101/{link}/visuals").GetChildren(), link


def test_the_joints_are_the_descriptions_with_its_limits(stage, urdf):
    limits, _ = build_so101.read_urdf(urdf)
    joints = [prim.GetName() for prim in stage.GetPrimAtPath("/so101/joints").GetChildren()]
    assert sorted(joints) == sorted(JOINTS)
    for name in JOINTS:
        joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(f"/so101/joints/{name}"))
        assert joint.GetAxisAttr().Get() == "Z"
        assert math.radians(joint.GetLowerLimitAttr().Get()) == pytest.approx(limits[name].lower, abs=1e-6)
        assert math.radians(joint.GetUpperLimitAttr().Get()) == pytest.approx(limits[name].upper, abs=1e-6)


def test_wrist_roll_reaches_the_urdfs_upper_limit(stage):
    """Upstream stops wrist_roll at 2.7438 rad; the description's 2.84121
    wins."""
    joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath("/so101/joints/wrist_roll"))
    assert math.radians(joint.GetUpperLimitAttr().Get()) == pytest.approx(2.84121, abs=1e-6)


def test_every_joint_is_a_position_drive(stage):
    for name in JOINTS:
        drive = UsdPhysics.DriveAPI(stage.GetPrimAtPath(f"/so101/joints/{name}"), "angular")
        assert drive.GetTypeAttr().Get() == "force"
        assert drive.GetStiffnessAttr().Get() > 0.0
        assert drive.GetMaxForceAttr().Get() == pytest.approx(build_so101.SERVO_MAX_TORQUE)


def test_the_articulation_is_rooted_at_the_base(stage):
    root = stage.GetPrimAtPath("/so101/root_joint")
    assert root.HasAPI(UsdPhysics.ArticulationRootAPI)
    joint = UsdPhysics.FixedJoint(root)
    assert not joint.GetBody0Rel().GetTargets()
    assert [target.name for target in joint.GetBody1Rel().GetTargets()] == ["base_link"]


@pytest.mark.parametrize(
    "positions",
    [
        dict.fromkeys(JOINTS, 0.0),
        dict(zip(JOINTS, (-0.05, -1.72, 1.68, 1.24, 0.32, -0.17))),
        dict(zip(JOINTS, (1.1, 0.4, -0.9, 0.7, 2.8, 1.2))),
        dict(zip(JOINTS, (-1.8, -0.6, 1.2, -1.5, -2.7, 0.3))),
    ],
)
def test_the_tool_frame_is_the_urdfs_through_the_chain(stage, urdf, positions):
    ours = tool_frame_in_world(stage, positions)
    theirs = urdf_tool_frame_in_world(urdf, positions)
    assert ours[:3, 3] == pytest.approx(theirs[:3, 3], abs=1e-5)
    assert ours[:3, :3] == pytest.approx(theirs[:3, :3], abs=1e-4)


def test_collisions_are_guides_with_the_collision_api(stage):
    for link in LINKS[1:]:
        collisions = stage.GetPrimAtPath(f"/so101/{link}/collisions")
        assert UsdGeom.Imageable(collisions).GetPurposeAttr().Get() == "guide"
        shapes = collisions.GetChildren()
        assert shapes, link
        assert all(shape.HasAPI(UsdPhysics.CollisionAPI) for shape in shapes)
