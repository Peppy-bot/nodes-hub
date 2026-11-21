#!/usr/bin/env python3
"""Build the SO-101's Isaac Sim stage from its MuJoCo model.

SO-101's description in robot_description carries a URDF of the arm's joints
and its links' collision meshes: no visual mesh, no inertia. The geometry
comes from the SO-101's MuJoCo model in the release of Waldo's catalogue that
sim_base_images/waldo_catalogue.py pins: MuJoCo Menagerie's robotstudio_so101
(Apache-2.0) with the files Waldo lays over it, the model the MuJoCo node
bakes and Waldo loads, so the three engines stand the same robot.

The description is the SO-101's model in robot_description_py, its URDF
and its SRDF. MuJoCo compiles the upstream MJCF, and the stage is
written from the compiled model: every body that carries a joint becomes a
link prim at its pose in the zero configuration, with the body's mass and
inertia, its visual meshes and its collision shapes, under the name of the
URDF link that the URDF joint of the same name places (the world body, which
the fixed base folds into, is the URDF's root link); every hinge becomes a
revolute joint between two links with a position drive; the base is fixed
to the world by the articulation's root joint. Bodies with no joint of their
own (upstream's base and its camera mount) are fused into the link they are
rigid with. Where upstream and the description differ the description wins:
each joint's limits are the URDF's, and the tool frame is the URDF's fixed
joint that places the grasp link of the SRDF's arm, `gripper_frame_link`.

CPU only: it needs the `mujoco`, `usd-core`, `numpy`, `robot_description_py`
and `control_core_py` packages, no Isaac Sim.

    python3 build_so101.py --release <staged release> --output <directory> \\
        --date <YYYY-MM-DD> [--archive <bundle.tar.gz>]

`--release` is a directory `waldo_catalogue.py fetch <directory> --robot so101`
staged, which holds the SO-101's model under so101/. The builder refuses any
other directory, and gives the command that stages it.

`--archive` packs the output directory into the bundle the Isaac base image
downloads: entries sorted, owners and times zeroed, so the same directory
packs to the same bytes and the bundle is published under its own SHA-256.
The directory holds the stage, upstream's LICENSE, MODIFICATIONS.md, which
says how the stage changes upstream's files, by whom and on `--date`, and a
manifest of what built it: the release and the tree the model came from,
the digest of this script and of mujoco_usd.py, the digest
robot_description_py gives the SO-101's model (its URDF, its SRDF and its
collision meshes), the date, and the versions of the tools. So the bundle's
bytes change with any of them, also when the stage's do not, and a build of
the same release on the same date packs to the same bytes.
"""

from __future__ import annotations

import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import robot_description_py
from control_core_py.geometry import wxyz_of_xyzw
from mujoco_usd import Bundle, bundle_parser, define_geom, finish_bundle, gf_quat, set_pose, write_bundle
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

# The SO-101 of the release, whose model directory the builder reads.
ROBOT_ID = "so101"
STAGE_FILENAME = "so101.usd"
ROBOT_PATH = "/so101"
JOINTS_PATH = f"{ROBOT_PATH}/joints"
LOOKS_PATH = f"{ROBOT_PATH}/Looks"
ROOT_JOINT = "root_joint"
# Upstream's base body, which MuJoCo folds into the world body.
BASE_BODY = "base"
BUNDLE = Bundle(
    robot_id=ROBOT_ID,
    stage=STAGE_FILENAME,
    builder=Path(__file__),
    license="Apache-2.0",
    upstream="",
    changes=(
        "MuJoCo compiles so101.xml, as Waldo's catalogue holds it, and the stage "
        "is written from the compiled model: each body that carries a joint a link prim at its "
        "pose with every joint at zero, with its mass and inertia, its meshes as USD meshes and "
        "its collision shapes, named after the URDF link of robot_description's model so101 the "
        "joint of the same name places; each body with no joint of its own fused into the link it "
        "is rigid with; each hinge a revolute joint with a position drive and the URDF's limits; "
        "the base fixed to the world; and the URDF's tool frame, gripper_frame_link."
    ),
    notices={"LICENSE": "LICENSE"},
)

# Upstream's geom groups: what is drawn, and what collides.
VISUAL_GROUP = 2
# The STS3215 position servo upstream models, per radian; a USD angular drive
# is per degree.
SERVO_KP = 998.22
SERVO_KV = 2.731
SERVO_MAX_TORQUE = 2.94

ARTICULATION_POSITION_ITERATIONS = 32
ARTICULATION_VELOCITY_ITERATIONS = 1


@dataclass(frozen=True)
class UrdfJoint:
    """A joint of the URDF: the link it hangs from and the link it places."""

    parent_link: str
    child_link: str


@dataclass(frozen=True)
class Description:
    """What the builder takes of the SO-101's description: the digest of its
    model, the limits of each joint its arm and its gripper move, every joint
    of the URDF by name, the URDF's root link, and the tool frame: where the
    fixed joint that places the arm's grasp link stands it in its parent."""

    digest: str
    limits: dict[str, robot_description_py.JointLimits]
    joints: dict[str, UrdfJoint]
    root_link: str
    tool: robot_description_py.LinkTransform


def _urdf_joint(joint: ET.Element) -> UrdfJoint:
    return UrdfJoint(
        parent_link=joint.find("parent").get("link"),
        child_link=joint.find("child").get("link"),
    )


def read_description() -> Description:
    """The SO-101's description as robot_description_py bundles it. The
    bindings give the arm, its gripper, the joint limits and the tool frame;
    ElementTree reads what they do not: each joint's parent and child."""
    model = robot_description_py.model(ROBOT_ID)
    described = model.description()
    (arm,) = described.arms
    moved = (*arm.joints, *(finger.joint for finger in arm.gripper.fingers))
    # Direct children only: transmission blocks nest limitless <joint> stubs
    # under the same names.
    elements = ET.fromstring(model.urdf).findall("joint")
    joints = {joint.get("name"): _urdf_joint(joint) for joint in elements}
    (root_link,) = {joint.parent_link for joint in joints.values()} - {
        joint.child_link for joint in joints.values()
    }
    (tool,) = [fixed for fixed in described.fixed_transforms() if fixed.child == arm.grasp_link]
    return Description(
        digest=model.digest,
        limits={joint: described.joint_limits(joint) for joint in moved},
        joints=joints,
        root_link=root_link,
        tool=tool,
    )


def links_of_bodies(model: mujoco.MjModel, description: Description) -> dict[str, str]:
    """The URDF link each body of the compiled model is, by the body's name.
    MuJoCo folds the fixed base into the world body, which is the URDF's
    root link, and the body a hinge moves is the link the URDF's joint of the
    same name places."""
    links = {model.body(0).name: description.root_link}
    for joint_id in range(model.njnt):
        body = model.body(int(model.jnt_bodyid[joint_id])).name
        links[body] = description.joints[model.joint(joint_id).name].child_link
    return links


def compile_model(model_dir: Path, fuse: bool) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(model_dir / "so101.xml"))
    spec.compiler.fusestatic = fuse
    return spec.compile()


def define_link(stage, model, data, body_id: int, link: str, base_inertial, materials: dict) -> str:
    path = f"{ROBOT_PATH}/{link}"
    prim = UsdGeom.Xform.Define(stage, path).GetPrim()
    set_pose(prim, data.xpos[body_id], data.xquat[body_id])
    UsdPhysics.RigidBodyAPI.Apply(prim)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass, ipos, iquat, inertia = (
        base_inertial
        if body_id == 0
        else (
            model.body_mass[body_id],
            model.body_ipos[body_id],
            model.body_iquat[body_id],
            model.body_inertia[body_id],
        )
    )
    mass_api.CreateMassAttr(float(mass))
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*(float(v) for v in ipos)))
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*(float(v) for v in inertia)))
    mass_api.CreatePrincipalAxesAttr(gf_quat(iquat))
    UsdGeom.Xform.Define(stage, f"{path}/visuals")
    collisions = UsdGeom.Xform.Define(stage, f"{path}/collisions")
    UsdGeom.Imageable(collisions).CreatePurposeAttr(UsdGeom.Tokens.guide)
    index = 0
    for geom_id in range(model.ngeom):
        if int(model.geom_bodyid[geom_id]) != body_id:
            continue
        name = model.geom(geom_id).name or f"geom_{index}"
        define_geom(stage, path, name, model, geom_id, VISUAL_GROUP, LOOKS_PATH, materials)
        index += 1
    return path


def define_joint(stage, model, joint_id: int, links: dict[str, str], limits: dict) -> None:
    joint = model.joint(joint_id)
    if int(model.jnt_type[joint_id]) != mujoco.mjtJoint.mjJNT_HINGE:
        raise ValueError(f"joint '{joint.name}' is not a hinge")
    axis = model.jnt_axis[joint_id]
    if not np.allclose(axis, (0.0, 0.0, 1.0), atol=1e-9):
        raise ValueError(f"joint '{joint.name}' turns about {axis}, not its body's z")
    child = int(model.jnt_bodyid[joint_id])
    parent = int(model.body_parentid[child])
    limit = limits[joint.name]
    prim = UsdPhysics.RevoluteJoint.Define(stage, f"{JOINTS_PATH}/{joint.name}")
    prim.CreateBody0Rel().SetTargets([f"{ROBOT_PATH}/{links[model.body(parent).name]}"])
    prim.CreateBody1Rel().SetTargets([f"{ROBOT_PATH}/{links[model.body(child).name]}"])
    prim.CreateAxisAttr(UsdPhysics.Tokens.z)
    # The joint frame is the child body's, moved to the joint's anchor: in the
    # parent it sits where the zero configuration puts the child.
    anchor = model.jnt_pos[joint_id]
    rotated = np.zeros(3)
    mujoco.mju_rotVecQuat(rotated, anchor, model.body_quat[child])
    prim.CreateLocalPos0Attr(Gf.Vec3f(*(float(v) for v in model.body_pos[child] + rotated)))
    prim.CreateLocalRot0Attr(gf_quat(model.body_quat[child]))
    prim.CreateLocalPos1Attr(Gf.Vec3f(*(float(v) for v in anchor)))
    prim.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    prim.CreateLowerLimitAttr(math.degrees(limit.lower))
    prim.CreateUpperLimitAttr(math.degrees(limit.upper))
    drive = UsdPhysics.DriveAPI.Apply(prim.GetPrim(), UsdPhysics.Tokens.angular)
    drive.CreateTypeAttr(UsdPhysics.Tokens.force)
    drive.CreateStiffnessAttr(math.radians(SERVO_KP))
    drive.CreateDampingAttr(math.radians(SERVO_KV))
    drive.CreateMaxForceAttr(SERVO_MAX_TORQUE)
    drive.CreateTargetPositionAttr(0.0)


def define_root_joint(stage, root_link: str) -> None:
    """Fixes the base to the world and roots the articulation there."""
    joint = UsdPhysics.FixedJoint.Define(stage, f"{ROBOT_PATH}/{ROOT_JOINT}")
    joint.CreateBody1Rel().SetTargets([f"{ROBOT_PATH}/{root_link}"])
    prim = joint.GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(prim)
    prim.CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool).Set(
        False
    )
    prim.CreateAttribute(
        "physxArticulation:solverPositionIterationCount", Sdf.ValueTypeNames.Int
    ).Set(ARTICULATION_POSITION_ITERATIONS)
    prim.CreateAttribute(
        "physxArticulation:solverVelocityIterationCount", Sdf.ValueTypeNames.Int
    ).Set(ARTICULATION_VELOCITY_ITERATIONS)


def define_tool_frame(stage, tool: robot_description_py.LinkTransform) -> None:
    prim = UsdGeom.Xform.Define(stage, f"{ROBOT_PATH}/{tool.parent}/{tool.child}")
    set_pose(prim.GetPrim(), tool.position, wxyz_of_xyzw(tool.orientation))


def build_stage(model_dir: Path, description: Description, output: Path) -> Path:
    model = compile_model(model_dir, fuse=True)
    unfused = compile_model(model_dir, fuse=False)
    base = unfused.body(BASE_BODY).id
    base_inertial = (
        unfused.body_mass[base],
        unfused.body_ipos[base],
        unfused.body_iquat[base],
        unfused.body_inertia[base],
    )
    joints = [model.joint(i).name for i in range(model.njnt)]
    if sorted(joints) != sorted(description.limits):
        raise ValueError(
            f"upstream's joints are {joints}, and the description's are {sorted(description.limits)}"
        )
    links = links_of_bodies(model, description)
    bodies = [model.body(i).name for i in range(model.nbody)]
    if sorted(bodies) != sorted(links):
        raise ValueError(
            f"upstream's bodies are {bodies}, and the ones the world and its hinges make links of "
            f"are {sorted(links)}"
        )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    output.mkdir(parents=True, exist_ok=True)
    path = output / STAGE_FILENAME
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    robot = UsdGeom.Xform.Define(stage, ROBOT_PATH)
    stage.SetDefaultPrim(robot.GetPrim())
    UsdGeom.Scope.Define(stage, LOOKS_PATH)
    UsdGeom.Scope.Define(stage, JOINTS_PATH)
    materials: dict = {}
    for body_id in range(model.nbody):
        define_link(stage, model, data, body_id, links[bodies[body_id]], base_inertial, materials)
    for joint_id in range(model.njnt):
        define_joint(stage, model, joint_id, links, description.limits)
    define_root_joint(stage, description.root_link)
    define_tool_frame(stage, description.tool)
    stage.GetRootLayer().Save()
    return path


def main(argv=None) -> int:
    args = bundle_parser(__doc__, ROBOT_ID).parse_args(argv)
    description = read_description()
    stage_path = build_stage(args.release / ROBOT_ID, description, args.output)
    write_bundle(BUNDLE, args.release, args.output, description.digest, args.date)
    finish_bundle(stage_path, args.output, args.archive)
    return 0


if __name__ == "__main__":
    sys.exit(main())
