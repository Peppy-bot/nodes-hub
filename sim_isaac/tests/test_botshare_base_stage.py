"""The botshare base's stage scripts/build_botshare_base.py writes, held to its
description, the URDF robot_description bundles: the bodies its fixed joints
fuse with the sum of their masses, every link at its place, the wheels'
continuous joints, and the contact of its tires and casters.

The stage is read as baked where PEPPY_ROBOT_ASSETS_DIR holds one (the base
image's suite), and built here otherwise, from the visual meshes
BOTSHARE_MESHES names, or that the suite stages in a directory of its own
over the network, each held to its pin. The suite needs mujoco, which the
builder runs on, and USD.
"""

from __future__ import annotations

import math
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import robot_description_py
from control_core_py.geometry import rotation_of

pytest.importorskip("mujoco")
pytest.importorskip("pxr")
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402  pylint: disable=C0413

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_botshare_base as builder  # noqa: E402  pylint: disable=C0413

ROBOT = builder.ROBOT_PATH
BODIES = ("base_footprint", "left_wheel_link", "right_wheel_link")
WHEELS = ("left_wheel_joint", "right_wheel_joint")


def _baked_stage() -> Optional[Path]:
    """The stage the base image bakes, where the assets directory holds one."""
    assets = os.environ.get("PEPPY_ROBOT_ASSETS_DIR")
    if not assets:
        return None
    path = Path(assets) / builder.ROBOT_ID / builder.STAGE_FILENAME
    return path if path.is_file() else None


@pytest.fixture(scope="module")
def description() -> builder.Description:
    return builder.read_description()


@pytest.fixture(scope="module")
def stage(tmp_path_factory, description) -> Usd.Stage:
    baked = _baked_stage()
    if baked is not None:
        return Usd.Stage.Open(str(baked))
    staged = os.environ.get("BOTSHARE_MESHES")
    meshes = Path(staged) if staged else tmp_path_factory.mktemp("botshare_meshes")
    builder.fetch_meshes(meshes)
    path = builder.build_stage(meshes, description, tmp_path_factory.mktemp("botshare_stage"))
    return Usd.Stage.Open(str(path))


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
    parents = builder.link_parents(description)
    links = [link.get("name") for link in urdf.findall("link")]
    expected = {
        "base_footprint": sum(
            _mass_of(urdf, link) for link in links if builder.body_of(link, parents) == "base_footprint"
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


def test_the_wheels_turn_without_limits_on_velocity_drives_about_their_axle(stage, description):
    for joint, link in zip(description.wheels, ("left_wheel_link", "right_wheel_link"), strict=True):
        prim = stage.GetPrimAtPath(f"{builder.JOINTS_PATH}/{joint}")
        revolute = UsdPhysics.RevoluteJoint(prim)
        assert revolute.GetAxisAttr().Get() == "Y"
        assert revolute.GetLowerLimitAttr().Get() == -math.inf
        assert revolute.GetUpperLimitAttr().Get() == math.inf
        assert [t.name for t in revolute.GetBody0Rel().GetTargets()] == ["base_footprint"]
        assert [t.name for t in revolute.GetBody1Rel().GetTargets()] == [link]
        drive = UsdPhysics.DriveAPI(prim, "angular")
        assert drive.GetTypeAttr().Get() == "force"
        assert drive.GetStiffnessAttr().Get() == 0.0
        assert prim.GetAttribute("physxJoint:armature").Get() == pytest.approx(builder.WHEEL_ARMATURE)
        # The joint's frame on the chassis is the wheel's place in the URDF.
        wheel = _world(stage.GetPrimAtPath(f"{ROBOT}/{link}"))
        assert tuple(revolute.GetLocalPos0Attr().Get()) == pytest.approx(tuple(wheel[:3, 3]), abs=1e-6)
    assert description.wheels == WHEELS


def test_the_robot_roots_its_articulation_and_stands_free(stage):
    root = stage.GetPrimAtPath(ROBOT)
    assert root.HasAPI(UsdPhysics.ArticulationRootAPI)
    joints = [prim for prim in Usd.PrimRange(root) if prim.IsA(UsdPhysics.Joint)]
    assert sorted(prim.GetName() for prim in joints) == list(WHEELS)


def _bound_physics(prim):
    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
    return material.GetPrim().GetName() if material else None


def test_the_tires_grip_and_the_casters_slide(stage):
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
    tire_material = stage.GetPrimAtPath(f"{builder.MATERIALS_PATH}/tire")
    assert UsdPhysics.MaterialAPI(tire_material).GetDynamicFrictionAttr().Get() == builder.WHEEL_FRICTION
    assert tire_material.GetAttribute("physxMaterial:frictionCombineMode").Get() == "max"


def test_every_body_draws_its_meshes_in_the_urdfs_materials_and_hides_its_collisions(stage, urdf):
    colors = {
        material.get("name"): tuple(float(v) for v in material.find("color").get("rgba").split()[:3])
        for material in urdf.findall("material")
    }
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
