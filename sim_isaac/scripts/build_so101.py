#!/usr/bin/env python3
"""Build the SO-101's Isaac Sim stage from its upstream MuJoCo model.

so101_description, the hardware's source of truth, carries a kinematics-only
URDF: no meshes, no collision, no inertia. The geometry comes from MuJoCo
Menagerie's robotstudio_so101 (Apache-2.0) at the commit
sim_base_images/so101_model.lock.json pins, the same files the MuJoCo node
bakes and Waldo's catalogue names, so the three engines stand the same robot.

MuJoCo compiles the upstream MJCF, and the stage is written from the
compiled model: every body that carries a joint becomes a link prim under
the URDF's link name at its pose in the zero configuration, with the body's
mass and inertia, its visual meshes and its collision shapes; every hinge
becomes a revolute joint between two links with a position drive; the base
is fixed to the world by the articulation's root joint. Bodies with no joint
of their own (upstream's base and its camera mount) are fused into the link
they are rigid with. Where upstream and the description differ the
description wins: each joint's limits are the URDF's, and the tool frame
`gripper_frame_link` is the URDF's fixed joint.

CPU only: it needs the `mujoco`, `usd-core` and `numpy` wheels, no Isaac Sim.

    python3 build_so101.py --model <staged upstream directory> \\
        --urdf <so101_kinematics.urdf> --lock <so101_model.lock.json> \\
        --output <directory> [--archive <bundle.tar.gz>]

`--archive` packs the output directory into the bundle the Isaac base image
downloads: entries sorted, owners and times zeroed, so the same stage packs
to the same bytes and the bundle is published under its own SHA-256.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import sys
import tarfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

STAGE_FILENAME = "so101.usd"
MANIFEST_FILENAME = "bundle_manifest.json"
ROBOT_PATH = "/so101"
JOINTS_PATH = f"{ROBOT_PATH}/joints"
LOOKS_PATH = f"{ROBOT_PATH}/Looks"
ROOT_JOINT = "root_joint"
TOOL_FRAME_JOINT = "gripper_frame_joint"

# Upstream's bodies under the URDF's link names. MuJoCo folds the fixed base
# into the world body, so the world body's geoms are the base link's.
LINK_OF_BODY = {
    "world": "base_link",
    "shoulder": "shoulder_link",
    "upper_arm": "upper_arm_link",
    "lower_arm": "lower_arm_link",
    "wrist": "wrist_link",
    "gripper": "gripper_link",
    "moving_jaw_so101_v1": "moving_jaw_so101_v1_link",
}
BASE_BODY = "base"

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
    lower: float
    upper: float


@dataclass(frozen=True)
class ToolFrame:
    parent_link: str
    child_link: str
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_urdf(path: Path) -> tuple[dict[str, UrdfJoint], ToolFrame]:
    """The description's joint limits by joint name, and its tool frame."""
    root = ET.parse(path).getroot()
    limits: dict[str, UrdfJoint] = {}
    tool = None
    # Direct children only: transmission blocks nest limitless <joint> stubs
    # under the same names.
    for joint in root.findall("joint"):
        name = joint.get("name")
        if joint.get("type") == "revolute":
            limit = joint.find("limit")
            limits[name] = UrdfJoint(float(limit.get("lower")), float(limit.get("upper")))
        if name == TOOL_FRAME_JOINT:
            origin = joint.find("origin")
            tool = ToolFrame(
                parent_link=joint.find("parent").get("link"),
                child_link=joint.find("child").get("link"),
                xyz=tuple(float(v) for v in origin.get("xyz").split()),
                rpy=tuple(float(v) for v in origin.get("rpy").split()),
            )
    if tool is None:
        raise ValueError(f"{path} has no {TOOL_FRAME_JOINT}")
    return limits, tool


def quat_from_rpy(rpy: tuple[float, float, float]) -> np.ndarray:
    """URDF fixed-axis roll, pitch, yaw as a wxyz quaternion."""
    quat = np.zeros(4)
    mujoco.mju_euler2Quat(quat, np.array(rpy, dtype=np.float64), "XYZ")
    return quat


def compile_model(model_dir: Path, fuse: bool) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(model_dir / "so101.xml"))
    spec.compiler.fusestatic = fuse
    return spec.compile()


def gf_quat(wxyz, kind=Gf.Quatf):
    w, x, y, z = (float(v) for v in wxyz)
    return kind(w, x, y, z)


def set_pose(prim, pos, quat_wxyz, scale=None) -> None:
    xform = UsdGeom.Xformable(prim)
    xform.AddTranslateOp().Set(Gf.Vec3d(*(float(v) for v in pos)))
    xform.AddOrientOp().Set(gf_quat(quat_wxyz))
    xform.AddScaleOp().Set(Gf.Vec3f(*(scale if scale is not None else (1.0, 1.0, 1.0))))


def material_for(stage, model, geom_id: int, cache: dict) -> UsdShade.Material:
    """One preview-surface material per colour the model draws."""
    mat_id = int(model.geom_matid[geom_id])
    rgba = model.mat_rgba[mat_id] if mat_id >= 0 else model.geom_rgba[geom_id]
    name = model.material(mat_id).name if mat_id >= 0 else "geom_colour"
    key = (name, tuple(round(float(v), 4) for v in rgba))
    if key in cache:
        return cache[key]
    material = UsdShade.Material.Define(stage, f"{LOOKS_PATH}/{name}")
    shader = UsdShade.Shader.Define(stage, f"{LOOKS_PATH}/{name}/surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*(float(v) for v in rgba[:3]))
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    cache[key] = material
    return material


def define_mesh(stage, path: str, model, mesh_id: int) -> UsdGeom.Mesh:
    """The compiled mesh, in the frame MuJoCo placed its geom in."""
    vert_adr, vert_num = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
    face_adr, face_num = int(model.mesh_faceadr[mesh_id]), int(model.mesh_facenum[mesh_id])
    points = model.mesh_vert[vert_adr : vert_adr + vert_num]
    faces = model.mesh_face[face_adr : face_adr + face_num]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * face_num))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.astype(np.int32).reshape(-1)))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    low, high = points.min(axis=0), points.max(axis=0)
    mesh.CreateExtentAttr([Gf.Vec3f(*map(float, low)), Gf.Vec3f(*map(float, high))])
    return mesh


def define_geom(stage, parent: str, name: str, model, geom_id: int, materials: dict):
    """One geom of a link, drawn or collided with."""
    visual = int(model.geom_group[geom_id]) == VISUAL_GROUP
    kind = int(model.geom_type[geom_id])
    size = model.geom_size[geom_id]
    path = f"{parent}/{'visuals' if visual else 'collisions'}/{name}"
    pos, quat = model.geom_pos[geom_id], model.geom_quat[geom_id]
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        prim = define_mesh(stage, path, model, int(model.geom_dataid[geom_id])).GetPrim()
        set_pose(prim, pos, quat)
        approximation = UsdPhysics.Tokens.convexHull
    elif kind == mujoco.mjtGeom.mjGEOM_BOX:
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(2.0)
        cube.CreateExtentAttr([Gf.Vec3f(-1, -1, -1), Gf.Vec3f(1, 1, 1)])
        prim = cube.GetPrim()
        set_pose(prim, pos, quat, scale=tuple(float(v) for v in size))
        approximation = None
    elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        sphere = UsdGeom.Sphere.Define(stage, path)
        sphere.CreateRadiusAttr(float(size[0]))
        prim = sphere.GetPrim()
        set_pose(prim, pos, quat)
        approximation = None
    elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        capsule = UsdGeom.Capsule.Define(stage, path)
        capsule.CreateRadiusAttr(float(size[0]))
        capsule.CreateHeightAttr(2.0 * float(size[1]))
        capsule.CreateAxisAttr(UsdGeom.Tokens.z)
        prim = capsule.GetPrim()
        set_pose(prim, pos, quat)
        approximation = None
    else:
        raise ValueError(f"geom {geom_id} has a type this builder does not write: {kind}")
    if visual:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material_for(stage, model, geom_id, materials))
        return
    UsdPhysics.CollisionAPI.Apply(prim)
    if approximation is not None:
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(approximation)


def define_link(stage, model, data, body_id: int, base_inertial, materials: dict) -> str:
    body = model.body(body_id)
    link = LINK_OF_BODY[body.name]
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
        define_geom(stage, path, name, model, geom_id, materials)
        index += 1
    return path


def define_joint(stage, model, joint_id: int, limits: dict[str, UrdfJoint]) -> None:
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
    prim.CreateBody0Rel().SetTargets([f"{ROBOT_PATH}/{LINK_OF_BODY[model.body(parent).name]}"])
    prim.CreateBody1Rel().SetTargets([f"{ROBOT_PATH}/{LINK_OF_BODY[model.body(child).name]}"])
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


def define_root_joint(stage) -> None:
    """Fixes the base to the world and roots the articulation there."""
    joint = UsdPhysics.FixedJoint.Define(stage, f"{ROBOT_PATH}/{ROOT_JOINT}")
    joint.CreateBody1Rel().SetTargets([f"{ROBOT_PATH}/{LINK_OF_BODY['world']}"])
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


def define_tool_frame(stage, tool: ToolFrame) -> None:
    prim = UsdGeom.Xform.Define(stage, f"{ROBOT_PATH}/{tool.parent_link}/{tool.child_link}")
    set_pose(prim.GetPrim(), tool.xyz, quat_from_rpy(tool.rpy))


def build_stage(model_dir: Path, urdf: Path, output: Path) -> Path:
    limits, tool = read_urdf(urdf)
    model = compile_model(model_dir, fuse=True)
    unfused = compile_model(model_dir, fuse=False)
    base = unfused.body(BASE_BODY).id
    base_inertial = (
        unfused.body_mass[base],
        unfused.body_ipos[base],
        unfused.body_iquat[base],
        unfused.body_inertia[base],
    )
    bodies = [model.body(i).name for i in range(model.nbody)]
    if bodies != list(LINK_OF_BODY):
        raise ValueError(f"upstream's bodies are {bodies}, and this builder knows {list(LINK_OF_BODY)}")
    joints = [model.joint(i).name for i in range(model.njnt)]
    if sorted(joints) != sorted(limits):
        raise ValueError(f"upstream's joints are {joints}, and the URDF's are {sorted(limits)}")
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
        define_link(stage, model, data, body_id, base_inertial, materials)
    for joint_id in range(model.njnt):
        define_joint(stage, model, joint_id, limits)
    define_root_joint(stage)
    define_tool_frame(stage, tool)
    stage.GetRootLayer().Save()
    return path


def write_manifest(output: Path, stage_path: Path, model_dir: Path, urdf: Path, lock: Path) -> None:
    pinned = json.loads(lock.read_text())
    manifest = {
        "version": 1,
        "stage": STAGE_FILENAME,
        "source": {
            "repository": pinned["repository"],
            "commit": pinned["commit"],
            "directory": pinned["directory"],
            "lock_sha256": sha256(lock),
            "license": "Apache-2.0",
        },
        "description": {"urdf": urdf.name, "sha256": sha256(urdf)},
        "builder": {"script": Path(__file__).name, "sha256": sha256(Path(__file__))},
        "tools": {
            "mujoco": mujoco.__version__,
            "usd": ".".join(str(v) for v in Usd.GetVersion()),
            "numpy": np.__version__,
            "python": sys.version.split()[0],
        },
        "files": {
            STAGE_FILENAME: sha256(stage_path),
            "LICENSE": sha256(model_dir / "LICENSE"),
        },
    }
    (output / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2) + "\n")


def pack(directory: Path, archive: Path) -> str:
    """Packs `directory` into `archive` reproducibly and returns the archive's
    SHA-256, the key it is published under."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for path in sorted(directory.iterdir()):
            info = tar.gettarinfo(str(path), arcname=path.name)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            info.mode = 0o644
            with path.open("rb") as source:
                tar.addfile(info, source)
    with archive.open("wb") as target, gzip.GzipFile(
        fileobj=target, mode="wb", mtime=0, filename=""
    ) as compressed:
        compressed.write(buffer.getvalue())
    return sha256(archive)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", type=Path, required=True, help="the staged upstream directory")
    parser.add_argument("--urdf", type=Path, required=True, help="so101_kinematics.urdf")
    parser.add_argument("--lock", type=Path, required=True, help="so101_model.lock.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path, help="pack the output into this .tar.gz")
    args = parser.parse_args(argv)
    stage_path = build_stage(args.model, args.urdf, args.output)
    (args.output / "LICENSE").write_bytes((args.model / "LICENSE").read_bytes())
    write_manifest(args.output, stage_path, args.model, args.urdf, args.lock)
    print(f"{stage_path}: {stage_path.stat().st_size} bytes")
    if args.archive is not None:
        print(f"{args.archive}: sha256 {pack(args.output, args.archive)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
