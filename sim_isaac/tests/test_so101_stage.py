"""The SO-101 stage scripts/build_so101.py writes, held to so101_description
(link and joint names, joint limits, and the tool frame's pose through the
chain) and to the bundle the base image bakes.

The stage is read as baked where PEPPY_ROBOT_ASSETS_DIR holds one (the base
image's suite), and built here otherwise, from MuJoCo Menagerie's model at
the commit sim_base_images/so101_model.lock.json pins. SO101_UPSTREAM_MODEL
names the directory that model is staged in; without it the suite stages the
model in a directory of its own. Either way every file of the model is
checked against the lock.

The description's URDF is the file SO101_KINEMATICS_URDF names, or
so101_description's at the public-peppy-libs commit this project pins:
so101_description needs a newer python than Isaac Sim's, so it is not a
dependency of this project. The suite reads that file alone, and none of the
meshes its links name. Staging the model or the URDF needs the network. The
suite needs mujoco, which the builder runs on, and USD.
"""

from __future__ import annotations

import math
import os
import platform
import sys
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional
from urllib.request import Request, urlopen

import numpy as np
import pytest

from _base_images import IMAGES_DIR, shell_value

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("pxr")
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402  pylint: disable=C0413

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(IMAGES_DIR))

import build_so101  # noqa: E402  pylint: disable=C0413
import so101_model  # noqa: E402  pylint: disable=C0413

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

# so101_description's URDF in public-peppy-libs, the repository this project
# pins sim_robot_core in.
_LIBS_URL = "https://raw.githubusercontent.com/Peppy-bot/public-peppy-libs"
_URDF_IN_LIBS = "so101_description/src/so101_description/urdf/so101_kinematics.urdf"
_USER_AGENT = "sim_isaac/tests/test_so101_stage.py"

# The bundle is built on Linux x86_64, the platform of the Isaac image. MuJoCo
# rounds what it compiles differently on another architecture, so only there
# is the stage built here the bundle's byte for byte.
_BUILDS_THE_BUNDLES_BYTES = sys.platform == "linux" and platform.machine() == "x86_64"


def _fetch_pinned_urdf(directory: Path) -> Path:
    """Stages so101_description's URDF at the public-peppy-libs commit this
    project pins."""
    project = tomllib.loads((Path(__file__).parent / "pyproject.toml").read_text())
    commit = project["tool"]["uv"]["sources"]["sim_robot_core"]["rev"]
    request = Request(
        f"{_LIBS_URL}/{commit}/{_URDF_IN_LIBS}", headers={"User-Agent": _USER_AGENT}
    )
    with urlopen(request, timeout=120) as response:
        data = response.read()
    path = directory / Path(_URDF_IN_LIBS).name
    path.write_bytes(data)
    return path


def _baked_stage() -> Optional[Path]:
    """The stage the base image bakes, where the assets directory holds one."""
    assets = os.environ.get("PEPPY_ROBOT_ASSETS_DIR")
    if not assets:
        return None
    path = Path(assets) / "so101" / build_so101.STAGE_FILENAME
    return path if path.is_file() else None


@pytest.fixture(scope="module")
def urdf(tmp_path_factory) -> Path:
    named = os.environ.get("SO101_KINEMATICS_URDF")
    if named:
        return Path(named)
    return _fetch_pinned_urdf(tmp_path_factory.mktemp("so101_description"))


@pytest.fixture(scope="module")
def stage_path(tmp_path_factory, urdf) -> Path:
    baked = _baked_stage()
    if baked is not None:
        return baked
    staged = os.environ.get("SO101_UPSTREAM_MODEL")
    model = Path(staged) if staged else tmp_path_factory.mktemp("so101_upstream") / "so101"
    # Stages the model, or checks the one staged there against the lock.
    so101_model.fetch(model)
    return build_so101.build_stage(model, urdf, tmp_path_factory.mktemp("so101_stage"))


@pytest.fixture(scope="module")
def stage(stage_path) -> Usd.Stage:
    return Usd.Stage.Open(str(stage_path))


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


def urdf_chain(urdf: Path) -> mujoco.MjModel:
    """The URDF's links and joints as MuJoCo compiles them, without the
    shapes of its links: a shape names a mesh file beside the URDF, and this
    suite reads the URDF alone."""
    robot = ET.parse(urdf).getroot()
    for link in robot.findall("link"):
        for shape in link.findall("collision") + link.findall("visual"):
            link.remove(shape)
    return mujoco.MjModel.from_xml_string(ET.tostring(robot, encoding="unicode"))


def urdf_tool_frame_in_world(urdf: Path, positions: dict[str, float]) -> np.ndarray:
    """The URDF's gripper_frame_link for the same joint positions. MuJoCo
    fuses the fixed link into gripper_link, so its fixed joint is composed
    onto that body's pose."""
    model = urdf_chain(urdf)
    data = mujoco.MjData(model)
    for name, position in positions.items():
        data.qpos[model.joint(name).qposadr[0]] = position
    mujoco.mj_forward(model, data)
    body = model.body("gripper_link").id
    _, tool = build_so101.read_urdf(urdf)
    fixed = _matrix(tool.xyz, build_so101.quat_from_rpy(tool.rpy))
    return _matrix(data.xpos[body], data.xquat[body]) @ fixed


@pytest.mark.skipif(
    not _BUILDS_THE_BUNDLES_BYTES,
    reason="the bundle is built on Linux x86_64, and MuJoCo rounds differently elsewhere",
)
def test_the_stage_is_the_one_the_pinned_bundle_holds(stage_path):
    """isaac_assets.env pins the bundle the image bakes and the digest of the
    stage in it, and download_assets.sh refuses a bundle whose stage is
    another. A change of the builder, of the model's lock, of the description
    or of the mujoco pin that moves the stage fails here until the bundle is
    built and pinned again (README, the SO-101 stage)."""
    pins = (IMAGES_DIR / "isaac_assets.env").read_text()
    assert build_so101.sha256(stage_path) == shell_value(pins, "SO101_STAGE_SHA256")


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
