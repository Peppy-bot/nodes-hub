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

CPU only: it needs the `mujoco`, `usd-core`, `numpy` and
`robot_description_py` packages, no Isaac Sim.

    python3 build_so101.py --release <staged release> --output <directory> \\
        [--archive <bundle.tar.gz>]

`--release` is a directory `waldo_catalogue.py fetch <directory> --robot so101`
staged, which holds the SO-101's model under so101/.

`--archive` packs the output directory into the bundle the Isaac base image
downloads: entries sorted, owners and times zeroed, so the same directory
packs to the same bytes and the bundle is published under its own SHA-256.
The directory holds the stage and a manifest of what built it: the release
and the tree the model came from, the digest of this script, the digest
robot_description_py gives the SO-101's model (its URDF, its SRDF and its
collision meshes), and the versions of the tools. So the bundle's bytes
change with any of them, also when the stage's do not.
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
import robot_description_py
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

# The SO-101 of the release, whose model directory the builder reads.
ROBOT_ID = "so101"
STAGE_FILENAME = "so101.usd"
MANIFEST_FILENAME = "bundle_manifest.json"
ROBOT_PATH = "/so101"
JOINTS_PATH = f"{ROBOT_PATH}/joints"
LOOKS_PATH = f"{ROBOT_PATH}/Looks"
ROOT_JOINT = "root_joint"
# Upstream's base body, which MuJoCo folds into the world body.
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
    """A joint of the URDF: the link it hangs from, the link it places, and
    where it places it in the parent's frame."""

    parent_link: str
    child_link: str
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]


@dataclass(frozen=True)
class Description:
    """What the builder takes of the SO-101's description: the digest of its
    model, the limits of each joint its arm and its gripper move, every joint
    of the URDF by name, the URDF's root link, and the tool frame, the fixed
    joint that places the arm's grasp link."""

    digest: str
    limits: dict[str, robot_description_py.JointLimits]
    joints: dict[str, UrdfJoint]
    root_link: str
    tool: UrdfJoint


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _urdf_joint(joint: ET.Element) -> UrdfJoint:
    origin = joint.find("origin")
    return UrdfJoint(
        parent_link=joint.find("parent").get("link"),
        child_link=joint.find("child").get("link"),
        xyz=tuple(float(v) for v in origin.get("xyz").split()),
        rpy=tuple(float(v) for v in origin.get("rpy").split()),
    )


def read_description() -> Description:
    """The SO-101's description as robot_description_py bundles it. The
    bindings give the arm, its gripper and the joint limits; ElementTree
    reads what they do not: each joint's parent, child and origin."""
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
    (tool,) = [joint for joint in joints.values() if joint.child_link == arm.grasp_link]
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
    elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        cylinder = UsdGeom.Cylinder.Define(stage, path)
        cylinder.CreateRadiusAttr(float(size[0]))
        cylinder.CreateHeightAttr(2.0 * float(size[1]))
        cylinder.CreateAxisAttr(UsdGeom.Tokens.z)
        prim = cylinder.GetPrim()
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
        define_geom(stage, path, name, model, geom_id, materials)
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


def define_tool_frame(stage, tool: UrdfJoint) -> None:
    prim = UsdGeom.Xform.Define(stage, f"{ROBOT_PATH}/{tool.parent_link}/{tool.child_link}")
    set_pose(prim.GetPrim(), tool.xyz, quat_from_rpy(tool.rpy))


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


def model_source(release: Path) -> dict:
    """The release a staged SO-101 model came from and the tree of it."""
    index = json.loads((release / "index.json").read_text())
    robots = json.loads((release / "robots.json").read_text())
    (tree,) = [robot["tree"] for robot in robots if robot["id"] == ROBOT_ID]
    return {"release": index["prefix"], "tree": tree, "license": "Apache-2.0"}


def write_manifest(output: Path, stage_path: Path, release: Path, description: Description) -> None:
    manifest = {
        "version": 1,
        "stage": STAGE_FILENAME,
        "source": model_source(release),
        "description": {"model": ROBOT_ID, "digest": description.digest},
        "builder": {"script": Path(__file__).name, "sha256": sha256(Path(__file__).read_bytes())},
        "tools": {
            "mujoco": mujoco.__version__,
            "usd": ".".join(str(v) for v in Usd.GetVersion()),
            "numpy": np.__version__,
            "python": sys.version.split()[0],
        },
        "files": {
            STAGE_FILENAME: sha256(stage_path.read_bytes()),
            "LICENSE": sha256((release / ROBOT_ID / "LICENSE").read_bytes()),
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
    return sha256(archive.read_bytes())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--release", type=Path, required=True, help="the staged release of Waldo's catalogue")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path, help="pack the output into this .tar.gz")
    args = parser.parse_args(argv)
    model = args.release / ROBOT_ID
    description = read_description()
    stage_path = build_stage(model, description, args.output)
    (args.output / "LICENSE").write_bytes((model / "LICENSE").read_bytes())
    write_manifest(args.output, stage_path, args.release, description)
    print(f"{stage_path}: {stage_path.stat().st_size} bytes")
    if args.archive is not None:
        print(f"{args.archive}: sha256 {pack(args.output, args.archive)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
