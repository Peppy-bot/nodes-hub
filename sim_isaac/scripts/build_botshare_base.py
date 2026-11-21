#!/usr/bin/env python3
"""Build the botshare base's Isaac Sim stage from its description.

The botshare base is botshare.ai's OpenAMRobot 2.0 base, robot_description's
model botshare_base: a chassis on two driven hub-motor wheels and four
casters. Its URDF, as robot_description_py bundles it, is the one source of
the stage: MuJoCo compiles it, with the visual meshes it names, and the stage
is written from the compiled model, as build_so101.py writes the SO-101's.
MuJoCo fuses every link that fixed joints join into one body, the first link
of the group, with the sum of their inertials: the root link base_footprint
and every link fixed to it is the chassis, and each wheel, on a continuous
joint, is a body of its own. Waldo's MJCF of the base fuses the URDF the same
way, so both engines roll the same masses.

Each body is a link prim at its pose with every joint at zero, with its
mass, its visual meshes and its collision shapes; each link fixed to it is a
prim under it, nested as the URDF nests them and placed by the URDF's fixed
joints, so a camera stands under the link it hangs from. Each wheel joint is
a revolute joint without limits, as its continuous joint turns, with a
velocity drive: no stiffness, its damping and its torque cap set by the
engine from its entry of the model. The articulation's root carries no joint
to the world: the base rolls on its wheels.

What the URDF does not hold:

- each wheel's armature, WHEEL_ARMATURE, the provisional rotor inertia Waldo's
  MJCF gives the same wheel;
- the wheels' grip, WHEEL_FRICTION, the sliding friction MuJoCo gives a
  contact by default, which Waldo's wheels run on;
- the casters' spheres take no friction, the least of any two materials in
  contact, so the wheels alone grip the floor.

The visual meshes are botshare's, from openAMRobot/openamr-platform-sw at the
commit the URDF was written from (MESH_SOURCE), each held to its SHA-256
(MESHES). `fetch_meshes` stages them, with the repository's licenses.

CPU only: it needs the `mujoco`, `usd-core`, `numpy` and
`robot_description_py` packages, no Isaac Sim.

    python3 build_botshare_base.py --meshes <directory> --output <directory> \\
        [--archive <bundle.tar.gz>]

`--meshes` is a directory `fetch_meshes` staged, or that `--fetch` stages
first. `--archive` packs the output directory into the bundle the Isaac base
image downloads, as build_so101.py packs the SO-101's: the stage, the
licenses of the meshes and a manifest of what built it (the meshes' source,
the digest of this script, the digest robot_description_py gives the base's
model, and the versions of the tools).
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import mujoco
import numpy as np
import robot_description_py
from control_core_py.geometry import wxyz_of_xyzw
from mujoco_usd import define_geom, pack, set_pose, sha256
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

ROBOT_ID = "botshare_base"
STAGE_FILENAME = "botshare_base.usd"
MANIFEST_FILENAME = "bundle_manifest.json"
ROBOT_PATH = f"/{ROBOT_ID}"
JOINTS_PATH = f"{ROBOT_PATH}/joints"
LOOKS_PATH = f"{ROBOT_PATH}/Looks"
MATERIALS_PATH = f"{ROBOT_PATH}/PhysicsMaterials"

# What MuJoCo makes of a URDF's visual elements: drawn geoms of group 1.
VISUAL_GROUP = 1

# Where the URDF's visual meshes live, and where they are staged from: the
# directory of openamrobot_description's visual meshes, and its repository,
# at the commit the URDF was written from (its header names it).
MESH_PACKAGE = "package://openamrobot_description/meshes/visual/"
MESH_SOURCE = "https://raw.githubusercontent.com/openAMRobot/openamr-platform-sw/69847ab7c9c6cd49b3022dc1a62f86c044ba10ff/"
MESH_DIRECTORY = "ros2/src/openamrobot_description/meshes/visual/"
MESHES = {
    "base_camera.stl": "26dd5d0899140ef7db660a41544e2d21b763d03f0dddf07e07524d06e43c6391",
    "base_chassis.stl": "693c71f0e6aac3af27e024fdb8bcf32287240d5177195ad002135002c38c8089",
    "base_covers.stl": "26250674f5a6492cbd66a97aaf44593c840484366ceb5eda9352b278e04c6232",
    "base_equipment.stl": "2cfe6d6b973e68cda68e36901a607703153335a653ab66e58d4cd5d7e80c267d",
    "caster.stl": "83739568c39cf68bc4dd2fe0c06680352ae69dab145a5463fcffaa6f7bdf9deb",
    "hub_motor_wheel.stl": "638b308b212dd992b3bbd5331f49032119bea12ddd2656c8881680a5dd3af820",
    "imu.stl": "8a8d2b79d03fee0a5a71624ae77c34304ba84e76dcd8776fa1d642f218779a65",
    "lidar.stl": "421b2dfa45a743527c53c61b0fb5cf206a4b7f7a259f5457a483f4489f18611b",
    "tof.stl": "2a64dcb271f41c4f7aeaa510c0b5762b78d498a3eca0e327dbc0014d9e473fb6",
    "ultrasonic.stl": "bf7b746729a7fbe30a56ae172ac09ee9fba2a1f6a93c349150e4ae5fa12de09f",
}
# The repository's licenses, which cover the meshes (CERN-OHL-P-2.0 for those
# derived from the hardware's CAD, MIT for the rest, as LICENSING.md maps them).
LICENSES = {
    "LICENSE": "3b255148b447030828e24e478cd1ee2cc711155abc1c50cf6985fbe3611e7737",
    "LICENSING.md": "19d39bdbd398243fccfe2a80148ca067ce5fc3bcfd783c11c79c0a4dedaef6f1",
    "NOTICE.md": "4903cce2ebdde634ac790fc1c6fc778ba56dac6cdc743072733b815008748475",
}

# The wheels' rotor inertia (kg m^2): provisional, as Waldo's MJCF gives it,
# since no ZLTech document gives the hub motor's.
WHEEL_ARMATURE = 0.05
# The tires' sliding friction: MuJoCo's default, which Waldo's wheels grip with.
WHEEL_FRICTION = 1.0

ARTICULATION_POSITION_ITERATIONS = 32
ARTICULATION_VELOCITY_ITERATIONS = 1


@dataclass(frozen=True)
class Description:
    """What the builder takes of the base's description: its URDF, the
    digest of its model, its root link, its wheel joints, left then right,
    and every fixed joint of the URDF by the link it places."""

    urdf: str
    digest: str
    root_link: str
    wheels: tuple[str, str]
    fixed: dict[str, robot_description_py.LinkTransform]


def read_description() -> Description:
    """The base's description as robot_description_py bundles it."""
    model = robot_description_py.model(ROBOT_ID)
    described = model.description()
    base = described.differential_base()
    return Description(
        urdf=model.urdf,
        digest=model.digest,
        root_link=described.root_link,
        wheels=(base.left.joint, base.right.joint),
        fixed={fixed.child: fixed for fixed in described.fixed_transforms()},
    )


def fetch_meshes(directory: Path) -> None:
    """Stages the visual meshes under `directory` and the licenses under
    `directory`/license, each from MESH_SOURCE and held to its SHA-256. A
    file staged before and still held to it is kept."""
    wanted = [
        *((directory / name, MESH_DIRECTORY + name, digest) for name, digest in MESHES.items()),
        *((directory / "license" / name, name, digest) for name, digest in LICENSES.items()),
    ]
    for path, source, digest in wanted:
        if path.is_file() and sha256(path.read_bytes()) == digest:
            continue
        with urlopen(MESH_SOURCE + source, timeout=60) as response:
            data = response.read()
        if sha256(data) != digest:
            raise ValueError(f"{MESH_SOURCE + source} has SHA-256 {sha256(data)}, and the pin is {digest}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def check_meshes(directory: Path) -> None:
    """Refuses a mesh directory whose files are not the pinned ones."""
    for name, digest in {**MESHES, **{f"license/{n}": d for n, d in LICENSES.items()}}.items():
        path = directory / name
        if not path.is_file() or sha256(path.read_bytes()) != digest:
            raise ValueError(
                f"{path} is not the pinned file: stage the meshes with `--fetch` or fetch_meshes"
            )


def compiled_urdf(urdf: str, meshes: Path, root_link: str) -> mujoco.MjModel:
    """The URDF as MuJoCo compiles it, each visual mesh read from `meshes`
    and drawn in the URDF's material of its name, every group of links that
    fixed joints join fused into one body, and the root link free, so its
    body carries its own inertial."""
    robot = ET.fromstring(urdf)
    drawn_in: dict[str, str] = {}
    for link in robot.findall("link"):
        for index, visual in enumerate(link.findall("visual")):
            mesh = visual.find("geometry/mesh")
            filename = mesh.get("filename")
            if not filename.startswith(MESH_PACKAGE):
                raise ValueError(f"link '{link.get('name')}' draws {filename}, which is not under {MESH_PACKAGE}")
            mesh.set("filename", filename[len(MESH_PACKAGE) :])
            name = f"{link.get('name')}_visual_{index}"
            visual.set("name", name)
            drawn_in[name] = visual.find("material").get("name")
    for gazebo in robot.findall("gazebo"):
        robot.remove(gazebo)
    extension = ET.SubElement(robot, "mujoco")
    ET.SubElement(extension, "compiler", meshdir=str(meshes), discardvisual="false", fusestatic="true")
    spec = mujoco.MjSpec.from_string(ET.tostring(robot, encoding="unicode"))
    for material in robot.findall("material"):
        rgba = [float(value) for value in material.find("color").get("rgba").split()]
        spec.add_material(name=material.get("name"), rgba=rgba)
    for geom, material in drawn_in.items():
        spec.geom(geom).material = material
    spec.body(root_link).add_freejoint()
    return spec.compile()


def link_parents(description: Description) -> dict[str, str]:
    """The link each fixed link hangs from, by the fixed link."""
    return {child: fixed.parent for child, fixed in description.fixed.items()}


def body_of(link: str, parents: dict[str, str]) -> str:
    """The body a link is fused into: the first link up its fixed joints."""
    while link in parents:
        link = parents[link]
    return link


def define_fixed_links(stage, body_path: str, body: str, description: Description) -> None:
    """Each link fixed to `body`, a prim under the prim of the link it hangs
    from, placed by its fixed joint."""
    parents = link_parents(description)
    paths = {body: body_path}

    def path_of(link: str) -> str:
        if link not in paths:
            paths[link] = f"{path_of(parents[link])}/{link}"
            fixed = description.fixed[link]
            prim = UsdGeom.Xform.Define(stage, paths[link]).GetPrim()
            set_pose(prim, fixed.position, wxyz_of_xyzw(fixed.orientation))
        return paths[link]

    for link in sorted(parents):
        if body_of(link, parents) == body:
            path_of(link)


def physics_material(stage, name: str, friction: float, combine: str) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, f"{MATERIALS_PATH}/{name}")
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(friction)
    api.CreateDynamicFrictionAttr(friction)
    api.CreateRestitutionAttr(0.0)
    prim = material.GetPrim()
    prim.AddAppliedSchema("PhysxMaterialAPI")
    prim.CreateAttribute(
        "physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token, False, Sdf.VariabilityUniform
    ).Set(combine)
    return material


def bind_physics(prim, material: UsdShade.Material) -> None:
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        material, UsdShade.Tokens.weakerThanDescendants, "physics"
    )


def define_body(
    stage, model, data, body_id: int, description: Description, looks: dict, physics: dict
) -> str:
    """One body: its prim at its pose with every joint at zero, its inertial,
    its geoms and the links fixed to it. `looks` holds the drawn materials
    written so far, `physics` the tire's and the casters' contact."""
    link = model.body(body_id).name
    path = f"{ROBOT_PATH}/{link}"
    prim = UsdGeom.Xform.Define(stage, path).GetPrim()
    set_pose(prim, data.xpos[body_id], data.xquat[body_id])
    UsdPhysics.RigidBodyAPI.Apply(prim)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(float(model.body_mass[body_id]))
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*(float(v) for v in model.body_ipos[body_id])))
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*(float(v) for v in model.body_inertia[body_id])))
    mass_api.CreatePrincipalAxesAttr(Gf.Quatf(*(float(v) for v in model.body_iquat[body_id])))
    UsdGeom.Xform.Define(stage, f"{path}/visuals")
    collisions = UsdGeom.Xform.Define(stage, f"{path}/collisions")
    UsdGeom.Imageable(collisions).CreatePurposeAttr(UsdGeom.Tokens.guide)
    is_wheel = link in {wheel_link(model, joint) for joint in description.wheels}
    geoms = [geom_id for geom_id in range(model.ngeom) if int(model.geom_bodyid[geom_id]) == body_id]
    for index, geom_id in enumerate(geoms):
        geom = define_geom(stage, path, f"geom_{index}", model, geom_id, VISUAL_GROUP, LOOKS_PATH, looks)
        if int(model.geom_group[geom_id]) == VISUAL_GROUP:
            continue
        if is_wheel:
            bind_physics(geom, physics["tire"])
        elif int(model.geom_type[geom_id]) == mujoco.mjtGeom.mjGEOM_SPHERE:
            bind_physics(geom, physics["caster"])
    define_fixed_links(stage, path, link, description)
    return path


def wheel_link(model, joint: str) -> str:
    return model.body(int(model.joint(joint).bodyid[0])).name


def define_wheel_joint(stage, model, joint: str) -> None:
    """A wheel's continuous joint: a revolute joint without limits between
    the chassis and the wheel, turning about the joint's axis, with a
    velocity drive and the wheel's armature."""
    joint_id = model.joint(joint).id
    child = int(model.jnt_bodyid[joint_id])
    parent = int(model.body_parentid[child])
    axis = model.jnt_axis[joint_id]
    if not np.allclose(axis, (0.0, 1.0, 0.0), atol=1e-9):
        raise ValueError(f"wheel joint '{joint}' turns about {axis}, not its body's y")
    prim = UsdPhysics.RevoluteJoint.Define(stage, f"{JOINTS_PATH}/{joint}")
    prim.CreateBody0Rel().SetTargets([f"{ROBOT_PATH}/{model.body(parent).name}"])
    prim.CreateBody1Rel().SetTargets([f"{ROBOT_PATH}/{model.body(child).name}"])
    prim.CreateAxisAttr(UsdPhysics.Tokens.y)
    anchor = model.jnt_pos[joint_id]
    rotated = np.zeros(3)
    mujoco.mju_rotVecQuat(rotated, anchor, model.body_quat[child])
    prim.CreateLocalPos0Attr(Gf.Vec3f(*(float(v) for v in model.body_pos[child] + rotated)))
    prim.CreateLocalRot0Attr(Gf.Quatf(*(float(v) for v in model.body_quat[child])))
    prim.CreateLocalPos1Attr(Gf.Vec3f(*(float(v) for v in anchor)))
    prim.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    drive = UsdPhysics.DriveAPI.Apply(prim.GetPrim(), UsdPhysics.Tokens.angular)
    drive.CreateTypeAttr(UsdPhysics.Tokens.force)
    drive.CreateStiffnessAttr(0.0)
    drive.CreateTargetVelocityAttr(0.0)
    joint_prim = prim.GetPrim()
    joint_prim.AddAppliedSchema("PhysxJointAPI")
    joint_prim.CreateAttribute("physxJoint:armature", Sdf.ValueTypeNames.Float, False).Set(WHEEL_ARMATURE)


def define_articulation_root(stage, robot) -> None:
    """Roots the articulation at the robot's prim, with no joint to the
    world: the base stands free on its wheels."""
    prim = robot.GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(prim)
    prim.AddAppliedSchema("PhysxArticulationAPI")
    prim.CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool).Set(False)
    prim.CreateAttribute("physxArticulation:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(
        ARTICULATION_POSITION_ITERATIONS
    )
    prim.CreateAttribute("physxArticulation:solverVelocityIterationCount", Sdf.ValueTypeNames.Int).Set(
        ARTICULATION_VELOCITY_ITERATIONS
    )


def build_stage(meshes: Path, description: Description, output: Path) -> Path:
    check_meshes(meshes)
    model = compiled_urdf(description.urdf, meshes, description.root_link)
    hinges = sorted(
        model.joint(j).name for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
    )
    if hinges != sorted(description.wheels):
        raise ValueError(f"the URDF turns {hinges}, and its differential base's wheels are {description.wheels}")
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
    UsdGeom.Scope.Define(stage, MATERIALS_PATH)
    UsdGeom.Scope.Define(stage, JOINTS_PATH)
    # A contact takes the larger friction of a tire's and the smaller of a
    # caster's, as MuJoCo takes the larger of two geoms' and the caster's
    # alone at the higher priority Waldo gives it.
    physics = {
        "tire": physics_material(stage, "tire", WHEEL_FRICTION, "max"),
        "caster": physics_material(stage, "caster", 0.0, "min"),
    }
    looks: dict = {}
    # Body 0 is MuJoCo's world, which holds nothing of the base.
    for body_id in range(1, model.nbody):
        define_body(stage, model, data, body_id, description, looks, physics)
    for joint in description.wheels:
        define_wheel_joint(stage, model, joint)
    define_articulation_root(stage, robot)
    stage.GetRootLayer().Save()
    return path


def write_manifest(output: Path, stage_path: Path, description: Description) -> None:
    manifest = {
        "version": 1,
        "stage": STAGE_FILENAME,
        "source": {"meshes": MESH_SOURCE + MESH_DIRECTORY, "license": "CERN-OHL-P-2.0 and MIT"},
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
            **{name: sha256((output / name).read_bytes()) for name in LICENSES},
        },
    }
    (output / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2) + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--meshes", type=Path, required=True, help="the staged visual meshes and licenses")
    parser.add_argument("--fetch", action="store_true", help="stage the meshes at --meshes first")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path, help="pack the output into this .tar.gz")
    args = parser.parse_args(argv)
    if args.fetch:
        fetch_meshes(args.meshes)
    description = read_description()
    stage_path = build_stage(args.meshes, description, args.output)
    for name in LICENSES:
        (args.output / name).write_bytes((args.meshes / "license" / name).read_bytes())
    write_manifest(args.output, stage_path, description)
    print(f"{stage_path}: {stage_path.stat().st_size} bytes")
    if args.archive is not None:
        print(f"{args.archive}: sha256 {pack(args.output, args.archive)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
