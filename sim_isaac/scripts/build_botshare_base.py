#!/usr/bin/env python3
"""Build the botshare base's Isaac Sim stage from its MuJoCo model.

The botshare base is botshare.ai's OpenAMRobot 2.0 base, robot_description's
model botshare_base. Its model is the one in the release of Waldo's catalogue:
the MJCF Waldo's tools/botshare_base writes from the base's URDF, with the
meshes its tree stages from botshare's openamr-platform-sw. The MuJoCo node
bakes the same model and Waldo loads it, so the three engines stand the same
robot, and the URDF is the one source of its geometry.

MuJoCo compiles the MJCF with its mount body hanging from the world on a free
joint, as Waldo hangs a rolling robot, and the stage is written from the
compiled model:

- the mount body is the URDF's root link, base_footprint, with the sum of the
  inertials of every link fixed to it, as the MJCF gives it; each body a hinge
  moves is the link the URDF joint of the same name places, a wheel;
- each link the URDF fixes to a body is a prim under the prim of the link it
  hangs from, placed by the URDF's fixed joint, so a camera stands under the
  link it hangs from;
- each drawn geom is a mesh in a material named after the mesh, of its colour,
  and each collision geom a hidden collider: a geom that takes no friction
  (condim 1, a caster's sphere) in a frictionless material that wins over any
  other, and a wheel's collider in a material of the friction the MJCF gives
  it, named by that friction, the larger of two winning, as MuJoCo combines
  two geoms' friction;
- each hinge is a revolute joint without limits about its axis, with the
  wheel's armature and its rate loop. The MJCF runs the loop as a velocity
  actuator, its kv the loop's proportional gain, and an integrated-velocity
  actuator, its kp the loop's integral gain, the two capped together by the
  joint's actuatorfrcrange. The joint's drive is a velocity drive: no
  stiffness, the proportional gain as its damping and the cap as its maximum
  force; the integral gain is the joint's attribute of its own,
  INTEGRAL_GAIN_ATTRIBUTE, per radian. The engine reads the three back as the
  loop it runs (engine/wheel_loop.py);
- the robot's prim roots the articulation with no joint to the world: the
  base rolls on its wheels.

The hinges are the wheels of the description's differential base, and the
stage refuses any other.

CPU only: it needs the `mujoco`, `usd-core`, `numpy`, `robot_description_py`,
`control_core_py` and `sim_robot_core` packages, no Isaac Sim.

    python3 build_botshare_base.py --release <staged release> --output <directory> \\
        --date <YYYY-MM-DD> [--archive <bundle.tar.gz>]

`--release` is a directory `waldo_catalogue.py fetch <directory> --robot
botshare_base` staged, which holds the base's model under botshare_base/. The
builder refuses any other directory, and gives the command that stages it.

The output directory holds the bundle: the stage, the license, the licensing
policy and the notice of botshare's repository, the provenance of its
meshes, which gives their licences, MODIFICATIONS.md, which says how the
stage changes the meshes, by whom and on `--date`, and a manifest of what
built it (the release and the tree the model came from, the digest of this
script and of mujoco_usd.py, the digest robot_description_py gives the
base's model, the date, and the versions of the tools). `--archive` packs it
into the bundle the Isaac base image downloads, as build_so101.py packs the
SO-101's, so a build of the same release on the same date packs to the same
bytes.
"""

from __future__ import annotations

import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import mujoco
import numpy as np
import robot_description_py
from control_core_py.geometry import wxyz_of_xyzw
from mujoco_usd import (
    Bundle,
    bundle_parser,
    catalogue_entry,
    define_geom,
    finish_bundle,
    set_pose,
    write_bundle,
)
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
from sim_robot_core.rate_loop import wheel_rate_loop

# The engine's wheel loop, pure Python, names the attribute the stage carries.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
from wheel_loop import INTEGRAL_GAIN_ATTRIBUTE  # noqa: E402  pylint: disable=C0413

ROBOT_ID = "botshare_base"
STAGE_FILENAME = "botshare_base.usd"
ROBOT_PATH = f"/{ROBOT_ID}"
JOINTS_PATH = f"{ROBOT_PATH}/joints"
LOOKS_PATH = f"{ROBOT_PATH}/Looks"
MATERIALS_PATH = f"{ROBOT_PATH}/PhysicsMaterials"
# The files of botshare's repository the bundle carries beside the stage,
# each by its path in the model's tree: the repository's license, licensing
# policy and notice, and the provenance of its meshes, whose licence block
# gives CERN-OHL-P-2.0 for the chassis meshes derived from the hardware's CAD
# and MIT for the generated envelopes.
NOTICES = {
    "LICENSE": "license/LICENSE",
    "LICENSING.md": "license/LICENSING.md",
    "NOTICE.md": "license/NOTICE.md",
    "PROVENANCE.yaml": "meshes/PROVENANCE.yaml",
}
BUNDLE = Bundle(
    robot_id=ROBOT_ID,
    stage=STAGE_FILENAME,
    builder=Path(__file__),
    license="CERN-OHL-P-2.0 and MIT",
    upstream="meshes/",
    changes=(
        "The stage draws the meshes under meshes/visual/ as the base's MJCF in Waldo's catalogue draws them: "
        "the triangles of each a USD mesh under the prim of the body it belongs to, at the body's "
        "pose with every joint at zero, in a material named after the mesh, of its MJCF colour. "
        "The bodies, their inertials and colliders, and the wheels' joints and rate loops come "
        "from that MJCF."
    ),
    notices=NOTICES,
)

# The MJCF's geom groups: what is drawn.
VISUAL_GROUP = 2
# The contact dimension of a geom that takes no friction.
FRICTIONLESS_CONDIM = 1

ARTICULATION_POSITION_ITERATIONS = 32
ARTICULATION_VELOCITY_ITERATIONS = 1


@dataclass(frozen=True)
class Description:
    """What the builder takes of the base's description: the digest of its
    model, its root link, its wheel joints, left then right, the link each
    joint of the URDF places, and every fixed joint by the link it places."""

    digest: str
    root_link: str
    wheels: tuple[str, str]
    child_links: dict[str, str]
    fixed: dict[str, robot_description_py.LinkTransform]


@dataclass(frozen=True)
class RateLoop:
    """A wheel's rate loop as the MJCF runs it: the proportional gain, its
    velocity actuator's kv (N*m*s/rad), the integral gain, its
    integrated-velocity actuator's kp (N*m/rad), and the cap of the two
    (N*m)."""

    proportional_gain: float
    integral_gain: float
    cap: float


def read_description() -> Description:
    """The base's description as robot_description_py bundles it. The
    bindings give the root link, the wheels and the fixed joints;
    ElementTree reads the link each joint places."""
    model = robot_description_py.model(ROBOT_ID)
    described = model.description()
    base = described.differential_base()
    joints = ET.fromstring(model.urdf).findall("joint")
    return Description(
        digest=model.digest,
        root_link=described.root_link,
        wheels=(base.left.joint, base.right.joint),
        child_links={joint.get("name"): joint.find("child").get("link") for joint in joints},
        fixed={fixed.child: fixed for fixed in described.fixed_transforms()},
    )


def compile_model(release: Path) -> mujoco.MjModel:
    """The base's MJCF, compiled with its mount body free, as Waldo hangs a
    rolling robot from the world, and each drawn geom in a material named
    after its mesh. A mesh drawn in two colours is refused."""
    entry = catalogue_entry(release, ROBOT_ID)
    spec = mujoco.MjSpec.from_file(str(release / ROBOT_ID / entry["wrapper"]))
    colours: dict[str, tuple[float, ...]] = {}
    for geom in spec.geoms:
        if geom.group != VISUAL_GROUP:
            continue
        colour = tuple(float(value) for value in geom.rgba)
        if colours.setdefault(geom.meshname, colour) != colour:
            raise ValueError(f"the MJCF draws the mesh '{geom.meshname}' in two colours")
        geom.material = geom.meshname
    for mesh, colour in colours.items():
        spec.add_material(name=mesh, rgba=list(colour))
    spec.body(entry["mount"]["body"]).add_freejoint()
    return spec.compile()


def links_of_bodies(model: mujoco.MjModel, mount: str, description: Description) -> dict[str, str]:
    """The URDF link each body of the compiled model is, by the body's name:
    the mount body is the root link, and the body a hinge moves is the link
    the URDF's joint of the same name places."""
    links = {mount: description.root_link}
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE:
            body = model.body(int(model.jnt_bodyid[joint_id])).name
            links[body] = description.child_links[model.joint(joint_id).name]
    return links


def rate_loop_of(model: mujoco.MjModel, joint: str) -> RateLoop:
    """The rate loop the MJCF runs the wheel `joint` with, as
    sim_robot_core.rate_loop's rule holds its actuators, the rule the MuJoCo
    node holds the same MJCF to, capped together by its actuatorfrcrange,
    the same either way, as a PhysX drive's one maximum force caps it.
    Anything else is refused naming what the wheel needs."""
    joint_id = model.joint(joint).id
    loop = wheel_rate_loop(model, joint, joint_id)
    low, high = (float(value) for value in model.jnt_actfrcrange[joint_id])
    # MuJoCo compiles a limited range only with its low below its high, so an
    # even range caps at a torque above zero.
    if not model.jnt_actfrclimited[joint_id] or low != -high:
        raise ValueError(
            f"wheel '{joint}': its actuators are capped at [{low:g}, {high:g}]; give the joint an "
            "actuatorfrcrange of -cap cap"
        )
    return RateLoop(
        proportional_gain=loop.proportional_gain, integral_gain=loop.integral_gain, cap=high
    )


def body_of(link: str, description: Description) -> str:
    """The body a link is fused into: the first link up its fixed joints."""
    while link in description.fixed:
        link = description.fixed[link].parent
    return link


def define_fixed_links(stage, body_path: str, body: str, description: Description) -> None:
    """Each link fixed to `body`, a prim under the prim of the link it hangs
    from, placed by its fixed joint."""
    paths = {body: body_path}

    def path_of(link: str) -> str:
        if link not in paths:
            fixed = description.fixed[link]
            paths[link] = f"{path_of(fixed.parent)}/{link}"
            prim = UsdGeom.Xform.Define(stage, paths[link]).GetPrim()
            set_pose(prim, fixed.position, wxyz_of_xyzw(fixed.orientation))
        return paths[link]

    for link in sorted(description.fixed):
        if body_of(link, description) == body:
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


def tire_name(friction: float) -> str:
    """The name of the material of a tire of friction `friction`: `tire_`
    and the friction's shortest exact decimal, its point written `p`, so
    each friction names a material of its own."""
    return "tire_" + format(Decimal(repr(friction)), "f").replace(".", "p")


def contact_of(stage, model, geom_id: int, wheel: bool, contacts: dict):
    """The physics material of a collision geom, each defined once under its
    name, the key of `contacts`: `caster`, frictionless, for a geom that
    takes no friction, the one `tire_name` names for a wheel's collider, of
    its MJCF friction, and none for any other geom."""
    if int(model.geom_condim[geom_id]) == FRICTIONLESS_CONDIM:
        name, friction, combine = "caster", 0.0, "min"
    elif wheel:
        friction = float(model.geom_friction[geom_id][0])
        name, combine = tire_name(friction), "max"
    else:
        return None
    if name not in contacts:
        contacts[name] = physics_material(stage, name, friction, combine)
    return contacts[name]


def define_body(
    stage, model, data, body_id: int, link: str, description: Description, looks: dict, contacts: dict
) -> str:
    """One body: its prim at its pose with every joint at zero, its
    inertial, its geoms and the links fixed to it. `looks` and `contacts`
    hold the drawn and the physics materials written so far."""
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
    wheel = link in {description.child_links[joint] for joint in description.wheels}
    geoms = [geom_id for geom_id in range(model.ngeom) if int(model.geom_bodyid[geom_id]) == body_id]
    for index, geom_id in enumerate(geoms):
        geom = define_geom(stage, path, f"geom_{index}", model, geom_id, VISUAL_GROUP, LOOKS_PATH, looks)
        if int(model.geom_group[geom_id]) == VISUAL_GROUP:
            continue
        contact = contact_of(stage, model, geom_id, wheel, contacts)
        if contact is not None:
            UsdShade.MaterialBindingAPI.Apply(geom).Bind(contact, UsdShade.Tokens.weakerThanDescendants, "physics")
    define_fixed_links(stage, path, link, description)
    return path


def define_wheel_joint(stage, model, joint: str, links: dict[str, str]) -> None:
    """A wheel's hinge: a revolute joint without limits between its parent
    body and the wheel, with the wheel's armature and its rate loop: a
    velocity drive of its proportional gain capped at its torque, and its
    integral gain."""
    joint_id = model.joint(joint).id
    child = int(model.jnt_bodyid[joint_id])
    parent = int(model.body_parentid[child])
    axis = model.jnt_axis[joint_id]
    if not np.allclose(axis, (0.0, 1.0, 0.0), atol=1e-9):
        raise ValueError(f"wheel joint '{joint}' turns about {axis}, not its body's y")
    loop = rate_loop_of(model, joint)
    prim = UsdPhysics.RevoluteJoint.Define(stage, f"{JOINTS_PATH}/{joint}")
    prim.CreateBody0Rel().SetTargets([f"{ROBOT_PATH}/{links[model.body(parent).name]}"])
    prim.CreateBody1Rel().SetTargets([f"{ROBOT_PATH}/{links[model.body(child).name]}"])
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
    # USD writes an angular drive's gains per degree.
    drive.CreateDampingAttr(math.radians(loop.proportional_gain))
    drive.CreateMaxForceAttr(loop.cap)
    joint_prim = prim.GetPrim()
    joint_prim.CreateAttribute(INTEGRAL_GAIN_ATTRIBUTE, Sdf.ValueTypeNames.Float, True).Set(
        loop.integral_gain
    )
    joint_prim.AddAppliedSchema("PhysxJointAPI")
    joint_prim.CreateAttribute("physxJoint:armature", Sdf.ValueTypeNames.Float, False).Set(
        float(model.dof_armature[int(model.jnt_dofadr[joint_id])])
    )


def define_articulation_root(robot) -> None:
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


def build_stage(release: Path, description: Description, output: Path) -> Path:
    model = compile_model(release)
    hinges = sorted(
        model.joint(j).name for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
    )
    if hinges != sorted(description.wheels):
        raise ValueError(f"the MJCF turns {hinges}, and the description's differential base {description.wheels}")
    links = links_of_bodies(model, catalogue_entry(release, ROBOT_ID)["mount"]["body"], description)
    bodies = [model.body(i).name for i in range(1, model.nbody)]
    if sorted(bodies) != sorted(links):
        raise ValueError(
            f"the MJCF's bodies are {bodies}, and the ones its mount and its hinges make links of are {sorted(links)}"
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
    UsdGeom.Scope.Define(stage, MATERIALS_PATH)
    UsdGeom.Scope.Define(stage, JOINTS_PATH)
    looks: dict = {}
    contacts: dict = {}
    # Body 0 is MuJoCo's world, which holds nothing of the base.
    for body_id in range(1, model.nbody):
        define_body(stage, model, data, body_id, links[model.body(body_id).name], description, looks, contacts)
    for joint in description.wheels:
        define_wheel_joint(stage, model, joint, links)
    define_articulation_root(robot)
    stage.GetRootLayer().Save()
    return path


def main(argv=None) -> int:
    args = bundle_parser(__doc__, ROBOT_ID).parse_args(argv)
    description = read_description()
    stage_path = build_stage(args.release, description, args.output)
    write_bundle(BUNDLE, args.release, args.output, description.digest, args.date)
    finish_bundle(stage_path, args.output, args.archive)
    return 0


if __name__ == "__main__":
    sys.exit(main())
