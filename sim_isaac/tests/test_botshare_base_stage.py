"""The botshare base's stage scripts/build_botshare_base.py writes from its
MuJoCo model, held to its description, the URDF robot_description bundles
(the bodies its fixed joints fuse with the sum of their masses, every link
at its place, the wheels' continuous joints), to the model (each wheel's
rate loop as its drive, the contact of its tires and casters, and the
colour of each mesh), and to the bundle the base image bakes.

The stage is read as baked where PEPPY_ROBOT_ASSETS_DIR holds one (the base
image's suite), and built here otherwise, from the base's model in the
release of Waldo's catalogue that sim_base_images/waldo_catalogue.py pins.
BOTSHARE_RELEASE names a directory a release holding the base is staged in;
without it the suite stages the pinned release in a directory of its own,
which needs the network. The suite needs mujoco, which the builder runs on,
and USD.
"""

from __future__ import annotations

import math
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import robot_description_py
from control_core_py.geometry import rotation_of

from _base_images import IMAGES_DIR, baked_stage, only_on_linux_x86_64, shell_value
from _world import known, world_module

pytest.importorskip("mujoco")
pytest.importorskip("pxr")
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402  pylint: disable=C0413

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(IMAGES_DIR))

import build_botshare_base as builder  # noqa: E402  pylint: disable=C0413
import waldo_catalogue  # noqa: E402  pylint: disable=C0413

ROBOT = builder.ROBOT_PATH
BODIES = ("base_footprint", "left_wheel_link", "right_wheel_link")
WHEELS = ("left_wheel_joint", "right_wheel_joint")


@pytest.fixture(scope="module")
def description() -> builder.Description:
    return builder.read_description()


@pytest.fixture(scope="module")
def release(tmp_path_factory) -> Path:
    staged = os.environ.get("BOTSHARE_RELEASE")
    if staged:
        return Path(staged)
    release = tmp_path_factory.mktemp("botshare_release") / "release"
    waldo_catalogue.fetch(release, robots=[builder.ROBOT_ID])
    return release


@pytest.fixture(scope="module")
def model(release):
    """The base's MuJoCo model, as the builder compiles it."""
    return builder.compile_model(release)


@pytest.fixture(scope="module")
def stage_path(tmp_path_factory, description, release) -> Path:
    baked = baked_stage(builder.ROBOT_ID, builder.STAGE_FILENAME)
    if baked is not None:
        return baked
    return builder.build_stage(release, description, tmp_path_factory.mktemp("botshare_stage"))


@pytest.fixture(scope="module")
def stage(stage_path) -> Usd.Stage:
    return Usd.Stage.Open(str(stage_path))


@pytest.fixture(scope="module")
def urdf() -> ET.Element:
    return ET.fromstring(robot_description_py.model(builder.ROBOT_ID).urdf)


def _matrix(position, xyzw) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = rotation_of(tuple(xyzw))
    matrix[:3, 3] = position
    return matrix


def _world(prim) -> np.ndarray:
    """A prim's world transform as a column-vector matrix."""
    return np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T


def _prim_named(stage, name: str):
    (prim,) = [prim for prim in Usd.PrimRange(stage.GetPrimAtPath(ROBOT)) if prim.GetName() == name]
    return prim


def _urdf_link_in_root(description: builder.Description, link: str) -> np.ndarray:
    """Where the URDF's fixed joints stand `link` in the root link's frame."""
    if link == description.root_link:
        return np.eye(4)
    fixed = description.fixed[link]
    return _urdf_link_in_root(description, fixed.parent) @ _matrix(fixed.position, fixed.orientation)


def _mass_of(urdf: ET.Element, link: str) -> float:
    mass = urdf.find(f"link[@name='{link}']/inertial/mass")
    return 0.0 if mass is None else float(mass.get("value"))


@only_on_linux_x86_64
def test_the_stage_is_the_one_the_pinned_bundle_holds(stage_path):
    """isaac_assets.env pins the bundle the image bakes and the digest of the
    stage in it, and download_assets.sh refuses a bundle whose stage is
    another. A change of the builder, of the release, of the description
    or of the mujoco pin that moves the stage fails here until the bundle is
    built and pinned again (README, the botshare base's stage)."""
    pins = (IMAGES_DIR / "isaac_assets.env").read_text()
    assert builder.sha256(stage_path.read_bytes()) == shell_value(pins, "BOTSHARE_BASE_STAGE_SHA256")


def test_the_stage_is_z_up_in_metres_with_the_robot_as_its_default_prim(stage):
    assert stage.GetDefaultPrim().GetPath() == ROBOT
    assert UsdGeom.GetStageUpAxis(stage) == "Z"
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0


def test_the_bodies_are_the_root_link_and_the_wheels(stage, description):
    bodies = [
        prim.GetName() for prim in stage.GetPrimAtPath(ROBOT).GetChildren() if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    assert bodies == list(BODIES)
    assert bodies[0] == description.root_link


def test_each_body_weighs_the_links_its_fixed_joints_join(stage, description, urdf):
    """The chassis carries every link fixed to the root; a wheel its own."""
    links = [link.get("name") for link in urdf.findall("link")]
    expected = {
        "base_footprint": sum(
            _mass_of(urdf, link) for link in links if builder.body_of(link, description) == "base_footprint"
        ),
        "left_wheel_link": _mass_of(urdf, "left_wheel_link"),
        "right_wheel_link": _mass_of(urdf, "right_wheel_link"),
    }
    for body, mass in expected.items():
        stood = UsdPhysics.MassAPI(stage.GetPrimAtPath(f"{ROBOT}/{body}")).GetMassAttr().Get()
        assert stood == pytest.approx(mass, rel=1e-6), body
    assert expected["base_footprint"] == pytest.approx(51.263, abs=1e-9)


def test_every_link_of_the_urdf_stands_where_its_fixed_joints_place_it(stage, description, urdf):
    """Each link is one prim under the robot, which a camera hanging from it
    is defined under, at the URDF's place with every joint at zero."""
    for link in (element.get("name") for element in urdf.findall("link")):
        if link in ("left_wheel_link", "right_wheel_link"):
            continue
        ours = _world(_prim_named(stage, link))
        theirs = _urdf_link_in_root(description, link)
        assert ours[:3, 3] == pytest.approx(theirs[:3, 3], abs=1e-6), link
        assert ours[:3, :3] == pytest.approx(theirs[:3, :3], abs=1e-6), link


def test_the_wheels_turn_without_limits_about_their_axle(stage, description):
    for joint, link in zip(description.wheels, ("left_wheel_link", "right_wheel_link"), strict=True):
        prim = stage.GetPrimAtPath(f"{builder.JOINTS_PATH}/{joint}")
        revolute = UsdPhysics.RevoluteJoint(prim)
        assert revolute.GetAxisAttr().Get() == "Y"
        assert revolute.GetLowerLimitAttr().Get() == -math.inf
        assert revolute.GetUpperLimitAttr().Get() == math.inf
        assert [t.name for t in revolute.GetBody0Rel().GetTargets()] == ["base_footprint"]
        assert [t.name for t in revolute.GetBody1Rel().GetTargets()] == [link]
        # The joint's frame on the chassis is the wheel's place in the URDF.
        wheel = _world(stage.GetPrimAtPath(f"{ROBOT}/{link}"))
        assert tuple(revolute.GetLocalPos0Attr().Get()) == pytest.approx(tuple(wheel[:3, 3]), abs=1e-6)
    assert description.wheels == WHEELS


def test_the_robot_roots_its_articulation_and_stands_free(stage):
    root = stage.GetPrimAtPath(ROBOT)
    assert root.HasAPI(UsdPhysics.ArticulationRootAPI)
    joints = [prim for prim in Usd.PrimRange(root) if prim.IsA(UsdPhysics.Joint)]
    assert sorted(prim.GetName() for prim in joints) == list(WHEELS)


def test_each_wheel_carries_the_rate_loop_its_model_runs_it_with(stage, model, description):
    """A velocity drive, no stiffness, of the proportional gain per degree
    as USD writes an angular drive's, capped at the loop's torque, the
    integral gain in the joint's attribute of its own, and the armature of
    the model's wheel."""
    for joint in description.wheels:
        loop = builder.rate_loop_of(model, joint)
        assert loop.proportional_gain > 0.0 and loop.integral_gain > 0.0 and loop.cap > 0.0
        prim = stage.GetPrimAtPath(f"{builder.JOINTS_PATH}/{joint}")
        drive = UsdPhysics.DriveAPI(prim, "angular")
        assert drive.GetTypeAttr().Get() == "force"
        assert drive.GetStiffnessAttr().Get() == 0.0
        assert drive.GetDampingAttr().Get() == pytest.approx(math.radians(loop.proportional_gain))
        assert drive.GetMaxForceAttr().Get() == pytest.approx(loop.cap)
        assert prim.GetAttribute(builder.INTEGRAL_GAIN_ATTRIBUTE).Get() == pytest.approx(loop.integral_gain)
        armature = model.dof_armature[int(model.jnt_dofadr[model.joint(joint).id])]
        assert prim.GetAttribute("physxJoint:armature").Get() == pytest.approx(armature)


# A wheel on a hinge whose actuator force its range caps, driven by the
# actuators a test gives it.
_WHEEL_MJCF = """<mujoco><worldbody><body><joint name="wheel" type="hinge" actuatorfrclimited="true"
actuatorfrcrange="-16 16"/><geom type="sphere" size="0.1"/></body></worldbody>
<actuator>{actuators}</actuator></mujoco>"""


def _wheel_driven_by(actuators: str):
    import mujoco  # pylint: disable=C0415

    return mujoco.MjModel.from_xml_string(_WHEEL_MJCF.format(actuators=actuators))


def test_a_wheel_driven_as_the_engines_run_it_gives_its_loop():
    model = _wheel_driven_by(
        '<velocity name="p" joint="wheel" kv="64"/><intvelocity name="i" joint="wheel" kp="128"/>'
    )

    assert builder.rate_loop_of(model, "wheel") == builder.RateLoop(
        proportional_gain=64.0, integral_gain=128.0, cap=16.0
    )


@pytest.mark.parametrize(
    ("actuators", "refusal"),
    [
        (
            '<position name="p" joint="wheel" kp="64"/>'
            '<intvelocity name="i" joint="wheel" kp="128"/>',
            "wheel 'wheel': 2 actuators drive it; drive the wheel with one <velocity> actuator",
        ),
        (
            '<velocity name="p" joint="wheel" kv="64"/>'
            '<intvelocity name="i" joint="wheel" kp="128" actrange="-1 1"/>',
            "wheel 'wheel': its actuator 'i' clamps its activation; a wheel turns without end",
        ),
    ],
    ids=["position_servo", "clamped_integral"],
)
def test_a_wheel_the_mujoco_node_refuses_is_refused(actuators, refusal):
    """The builder holds the MJCF to the rule the MuJoCo node holds it to
    (sim_robot_core.rate_loop), so a bundle never bakes a wheel that engine
    refuses."""
    with pytest.raises(ValueError, match=f"^{re.escape(refusal)}"):
        builder.rate_loop_of(_wheel_driven_by(actuators), "wheel")


def test_the_engine_runs_the_loop_the_stage_carries(stage, model, description):
    """What the engine reads back off the stage as each wheel stands is the
    loop the model runs it with."""
    world = world_module()
    loops = world.World._wheel_loops(stage.GetPrimAtPath(ROBOT), known(builder.ROBOT_ID))
    expected = [builder.rate_loop_of(model, joint) for joint in description.wheels]
    assert [(loop.proportional_gain, loop.integral_gain, loop.max_torque) for loop in loops] == [
        (
            pytest.approx(rate.proportional_gain, rel=1e-6),
            pytest.approx(rate.integral_gain, rel=1e-6),
            pytest.approx(rate.cap),
        )
        for rate in expected
    ]


def _bound_physics(prim):
    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
    return material.GetPrim().GetName() if material else None


def test_the_tires_grip_as_the_model_gives_and_the_casters_slide(stage, model):
    for body in ("left_wheel_link", "right_wheel_link"):
        (tire,) = stage.GetPrimAtPath(f"{ROBOT}/{body}/collisions").GetChildren()
        assert tire.IsA(UsdGeom.Cylinder)
        assert _bound_physics(tire) == "tire"
    chassis = stage.GetPrimAtPath(f"{ROBOT}/base_footprint/collisions").GetChildren()
    casters = [shape for shape in chassis if shape.IsA(UsdGeom.Sphere)]
    assert len(casters) == 4
    assert all(_bound_physics(shape) == "caster" for shape in casters)
    assert all(_bound_physics(shape) is None for shape in chassis if not shape.IsA(UsdGeom.Sphere))
    caster = stage.GetPrimAtPath(f"{builder.MATERIALS_PATH}/caster")
    assert UsdPhysics.MaterialAPI(caster).GetDynamicFrictionAttr().Get() == 0.0
    assert caster.GetAttribute("physxMaterial:frictionCombineMode").Get() == "min"
    wheel = model.body("left_wheel_link").id
    (tire_geom,) = [
        geom for geom in range(model.ngeom)
        if model.geom_bodyid[geom] == wheel and model.geom_group[geom] != builder.VISUAL_GROUP
    ]
    tire_material = stage.GetPrimAtPath(f"{builder.MATERIALS_PATH}/tire")
    assert UsdPhysics.MaterialAPI(tire_material).GetDynamicFrictionAttr().Get() == pytest.approx(
        model.geom_friction[tire_geom][0]
    )
    assert tire_material.GetAttribute("physxMaterial:frictionCombineMode").Get() == "max"


def test_every_body_draws_its_meshes_in_their_model_colors_and_hides_its_collisions(stage, model):
    colors = {model.material(i).name: tuple(float(v) for v in model.mat_rgba[i][:3]) for i in range(model.nmat)}
    for body in BODIES:
        drawn = stage.GetPrimAtPath(f"{ROBOT}/{body}/visuals").GetChildren()
        assert drawn, body
        for mesh in drawn:
            material, _ = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()
            name = material.GetPrim().GetName()
            surface = UsdShade.Shader(stage.GetPrimAtPath(f"{material.GetPath()}/surface"))
            assert tuple(surface.GetInput("diffuseColor").Get()) == pytest.approx(colors[name], abs=1e-6)
        collisions = stage.GetPrimAtPath(f"{ROBOT}/{body}/collisions")
        assert UsdGeom.Imageable(collisions).GetPurposeAttr().Get() == "guide"
        assert all(shape.HasAPI(UsdPhysics.CollisionAPI) for shape in collisions.GetChildren())
