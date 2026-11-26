"""The botshare base's stage scripts/build_botshare_base.py writes from its
MuJoCo model, held to its description, the URDF robot_description bundles
(the bodies its fixed joints fuse with the sum of their masses, every link
at its place on the body that carries it, the wheels' continuous joints),
to the model (the mount's mass properties, its own and those of the bodies
no joint moves on it combined, every geom at its place, each wheel's rate
loop as its drive, the contact of its tires and casters, and the colour of
each mesh), to what the engine reads off it (each wheel's loop and the
lidar's mount), and to the bundle the base image bakes.

The stage is read as baked where PEPPY_ROBOT_ASSETS_DIR holds one (the base
image's suite), and built here otherwise, from the base's model in the
release of Waldo's catalogue that sim_base_images/waldo_catalogue.py pins.
BOTSHARE_RELEASE names the directory that release is staged in; without it
the suite stages the release in a directory of its own, which needs the
network. Either way every file of the model is checked against the release.
The suite needs mujoco, which the builder runs on, and USD.
"""

from __future__ import annotations

import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import robot_description_py
from control_core_py.geometry import rotation_of

from _base_images import IMAGES_DIR, baked_stage, only_on_linux_x86_64, shell_value
from _bundles import check_bundle, check_the_date_is_required, check_the_release_is_the_pinned_one, staged_release
from _world import known, world_module

pytest.importorskip("mujoco")
pytest.importorskip("pxr")
import mujoco  # noqa: E402  pylint: disable=C0413
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402  pylint: disable=C0413

_SIM_ISAAC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SIM_ISAAC_DIR / "scripts"))
sys.path.insert(0, str(_SIM_ISAAC_DIR / "engine"))
sys.path.insert(0, str(_SIM_ISAAC_DIR / "engine" / "exts"))

import build_botshare_base as builder  # noqa: E402  pylint: disable=C0413
import lidar_sensor  # noqa: E402  pylint: disable=C0413
import mujoco_usd  # noqa: E402  pylint: disable=C0413

ROBOT = builder.ROBOT_PATH
BODIES = ("base_footprint", "left_wheel_link", "right_wheel_link")
WHEELS = ("left_wheel_joint", "right_wheel_joint")


@pytest.fixture(scope="module")
def description() -> builder.Description:
    return builder.read_description()


@pytest.fixture(scope="module")
def release(tmp_path_factory) -> Path:
    return staged_release(tmp_path_factory, "BOTSHARE_RELEASE", builder.ROBOT_ID)


@pytest.fixture(scope="module")
def model(release):
    """The base's MuJoCo model, as the builder compiles it."""
    return builder.compile_model(release)


@pytest.fixture(scope="module")
def entry(release) -> dict:
    """The base's entry in the release's catalogue."""
    return mujoco_usd.catalogue_entry(release, builder.ROBOT_ID)


@pytest.fixture(scope="module")
def mount(entry) -> str:
    """The name of the MJCF's mount body."""
    return entry["mount"]["body"]


@pytest.fixture(scope="module")
def released(release, entry):
    """The base's MJCF as the release gives it, compiled with each of its
    bodies a body of its own, and the model's data with every joint at zero."""
    released = mujoco.MjModel.from_xml_path(str(release / builder.ROBOT_ID / entry["wrapper"]))
    data = mujoco.MjData(released)
    mujoco.mj_forward(released, data)
    return released, data


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


def _inertia_about(mass: float, centre: np.ndarray, axes: np.ndarray, principal, point: np.ndarray) -> np.ndarray:
    """The inertia tensor about `point` of a body of `mass` at `centre`
    whose principal moments `principal` lie along the columns of `axes`."""
    offset = centre - point
    return axes @ np.diag(principal) @ axes.T + mass * (offset @ offset * np.eye(3) - np.outer(offset, offset))


def _combined_in(model, data, frame: int, bodies: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """The centre of mass of `bodies` of `model` and their inertia tensor
    about it, in the frame of the body `frame`."""
    to_frame = data.xmat[frame].reshape(3, 3).T
    masses = [float(model.body_mass[body]) for body in bodies]
    centres = [to_frame @ (data.xipos[body] - data.xpos[frame]) for body in bodies]
    axes = [to_frame @ data.ximat[body].reshape(3, 3) for body in bodies]
    centre = sum(m * c for m, c in zip(masses, centres, strict=True)) / sum(masses)
    inertia = sum(
        _inertia_about(m, c, a, model.body_inertia[body], centre)
        for body, m, c, a in zip(bodies, masses, centres, axes, strict=True)
    )
    return centre, inertia


def _stood_mass_properties(prim) -> tuple[float, np.ndarray, np.ndarray]:
    """The mass, the centre of mass and the inertia tensor about it that a
    body's MassAPI gives, in the body's frame."""
    api = UsdPhysics.MassAPI(prim)
    axes = api.GetPrincipalAxesAttr().Get()
    rotation = np.array(rotation_of((*axes.GetImaginary(), axes.GetReal())))
    inertia = rotation @ np.diag(api.GetDiagonalInertiaAttr().Get()) @ rotation.T
    return api.GetMassAttr().Get(), np.array(api.GetCenterOfMassAttr().Get()), inertia


def _carrier(model, mount: int, body: int) -> int:
    """The body of `model` that `body` is fused into: the first body up from
    it that is the mount or that a joint moves."""
    while body != mount and model.body_jntnum[body] == 0:
        body = int(model.body_parentid[body])
    return body


def _placed(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The position and the rotation of a column-vector transform, its scale
    taken out of the rotation."""
    rotation = matrix[:3, :3]
    return matrix[:3, 3], rotation / np.linalg.norm(rotation, axis=0)


def _same_place(geom, other) -> bool:
    """Whether two geoms, each its parent's path, its position and its
    rotation, stand under one prim at one place."""
    (path, position, rotation), (other_path, other_position, other_rotation) = geom, other
    return (
        path == other_path
        and np.allclose(position, other_position, rtol=0.0, atol=1e-6)
        and np.allclose(rotation, other_rotation, rtol=0.0, atol=1e-6)
    )


@only_on_linux_x86_64
def test_the_stage_is_the_one_the_pinned_bundle_holds(stage_path):
    """isaac_assets.env pins the bundle the image bakes and the digest of the
    stage in it, and download_assets.sh refuses a bundle whose stage is
    another. A change of the builder, of the release, of the description
    or of the mujoco pin that moves the stage fails here until the bundle is
    built and pinned again (README, the botshare base's stage)."""
    pins = (IMAGES_DIR / "isaac_assets.env").read_text()
    assert mujoco_usd.sha256(stage_path.read_bytes()) == shell_value(pins, "BOTSHARE_BASE_STAGE_SHA256")


def test_the_bundle_carries_botshares_notices_and_how_the_stage_changes_its_meshes(release, tmp_path):
    """Beside the stage, the bundle carries the license, licensing policy and
    notice of botshare's repository and the provenance of its meshes, whose
    licence block is the source of the manifest's license, and
    MODIFICATIONS.md names the meshes the stage changes."""
    output, manifest, modifications = check_bundle(builder, release, tmp_path)
    assert re.search(
        r"files at https://raw\.githubusercontent\.com/openAMRobot/openamr-platform-sw/[0-9a-f]{40}/"
        r"ros2/src/openamrobot_description/meshes/, under CERN-OHL-P-2\.0 and MIT\. The stage draws the "
        r"meshes under meshes/visual/",
        modifications,
    )
    provenance = (output / "PROVENANCE.yaml").read_text()
    assert "chassis_meshes: CERN-OHL-P-2.0" in provenance and "envelopes: MIT" in provenance
    assert manifest["source"]["license"] == "CERN-OHL-P-2.0 and MIT"


def test_a_build_with_no_date_is_refused(release, tmp_path, capsys):
    check_the_date_is_required(builder, release, tmp_path, capsys)


def test_a_build_from_another_release_or_one_without_the_robot_is_refused(release, tmp_path, capsys):
    check_the_release_is_the_pinned_one(builder, release, tmp_path, capsys)


def _release_of(tmp_path: Path, *robots: dict) -> Path:
    """A staged release that lists `robots` in its robots.json."""
    (tmp_path / "robots.json").write_text(json.dumps(list(robots)))
    return tmp_path


_SO101 = {"id": "so101", "source": {"base": "https://example.org/so101"}}
_MESHES = {"prefix": "meshes/", "base": "https://example.org/meshes"}


def test_a_staged_release_that_lists_no_robot_of_the_bundle_is_refused(tmp_path):
    release = _release_of(tmp_path, _SO101)
    refusal = (
        f"the release staged at {release} lists no robot 'botshare_base' (it lists ['so101']): set the builder's "
        "ROBOT_ID to the id it lists for the robot"
    )

    with pytest.raises(ValueError, match=f"^{re.escape(refusal)}$"):
        mujoco_usd.catalogue_entry(release, "botshare_base")


def test_a_staged_release_that_lists_the_bundles_robot_twice_is_refused(tmp_path):
    base = {"id": "botshare_base", "source": {"base": "https://example.org/base"}}
    release = _release_of(tmp_path, base, _SO101, base)
    refusal = (
        f"the release staged at {release} lists the robot 'botshare_base' 2 times: pin a release that lists it "
        "once, RELEASE_DIGEST in sim_base_images/waldo_catalogue.py"
    )

    with pytest.raises(ValueError, match=f"^{re.escape(refusal)}$"):
        mujoco_usd.catalogue_entry(release, "botshare_base")


@pytest.mark.parametrize(
    ("overlays", "laid"), [(None, []), ([{"prefix": "urdf/", "base": "https://example.org/urdf"}], ["urdf/"])],
    ids=["no-overlays", "another-prefix"],
)
def test_an_upstream_no_overlay_of_the_robot_lays_is_refused(tmp_path, overlays, laid):
    source = {"base": "https://example.org/base"} | ({} if overlays is None else {"overlays": overlays})
    release = _release_of(tmp_path, {"id": "botshare_base", "source": source})
    refusal = (
        f"the robot 'botshare_base' of the release staged at {release} lays no overlay at 'meshes/' (its overlays "
        f"lay at {laid}): set `upstream` in the builder's BUNDLE to '' for the tree's base or to a prefix an "
        "overlay lays at"
    )

    with pytest.raises(ValueError, match=f"^{re.escape(refusal)}$"):
        mujoco_usd.upstream_of(release, "botshare_base", "meshes/")


def test_an_upstream_two_overlays_of_the_robot_lay_is_refused(tmp_path):
    source = {"base": "https://example.org/base", "overlays": [_MESHES, _MESHES]}
    release = _release_of(tmp_path, {"id": "botshare_base", "source": source})
    refusal = (
        f"the robot 'botshare_base' of the release staged at {release} lays 2 overlays at 'meshes/' (its overlays "
        "lay at ['meshes/', 'meshes/']): set `upstream` in the builder's BUNDLE to '' for the tree's base or to a "
        "prefix one overlay lays at"
    )

    with pytest.raises(ValueError, match=f"^{re.escape(refusal)}$"):
        mujoco_usd.upstream_of(release, "botshare_base", "meshes/")


def test_an_upstream_is_the_base_of_the_one_overlay_laid_at_its_prefix_or_the_trees_base(tmp_path):
    visual = {"prefix": "meshes/visual/", "base": "https://example.org/visual"}
    source = {"base": "https://example.org/base", "overlays": [_MESHES, visual]}
    release = _release_of(tmp_path, {"id": "botshare_base", "source": source})

    assert mujoco_usd.upstream_of(release, "botshare_base", "meshes/") == "https://example.org/meshes"
    assert mujoco_usd.upstream_of(release, "botshare_base", "meshes/visual/") == "https://example.org/visual"
    assert mujoco_usd.upstream_of(release, "botshare_base", "") == "https://example.org/base"


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


def test_the_mount_carries_the_mjcfs_mount_and_lidar_link_combined_about_its_frame(stage, mount, released):
    """lidar_link, the body of its own the MJCF gives the lidar's housing,
    welded to the mount, is part of the mount on the stage: their masses
    added, at their joint centre of mass, their inertias moved to it."""
    model, data = released
    mount_id, lidar_id = model.body(mount).id, model.body("lidar_link").id
    assert (model.body_parentid[lidar_id], model.body_jntnum[lidar_id]) == (mount_id, 0)
    centre, inertia = _combined_in(model, data, mount_id, [mount_id, lidar_id])

    stood_mass, stood_centre, stood_inertia = _stood_mass_properties(stage.GetPrimAtPath(f"{ROBOT}/base_footprint"))

    assert stood_mass == pytest.approx(model.body_mass[mount_id] + model.body_mass[lidar_id], rel=1e-6)
    assert stood_centre == pytest.approx(centre, abs=1e-6)
    assert stood_inertia == pytest.approx(inertia, abs=1e-6)


def test_every_geom_of_the_model_stands_on_the_body_it_is_fused_into_where_the_model_places_it(
    stage, model, mount, released, description
):
    """Each geom of the MJCF is drawn or collides under the prim of the body
    that carries it, a body no joint moves fused into the body it hangs from,
    at the place the MJCF gives it with every joint at zero: the lidar's
    housing on the mount. Each geom stands at one place on the stage, and
    each geom the stage holds at one of the MJCF's."""
    unfused, data = released
    links = builder.links_of_bodies(model, mount, description)
    mount_id = unfused.body(mount).id

    def carrier_path(geom: int) -> str:
        carrier = unfused.body(_carrier(unfused, mount_id, int(unfused.geom_bodyid[geom]))).name
        folder = "visuals" if unfused.geom_group[geom] == builder.VISUAL_GROUP else "collisions"
        return f"{ROBOT}/{links[carrier]}/{folder}"

    placed = [
        (carrier_path(geom), data.geom_xpos[geom], data.geom_xmat[geom].reshape(3, 3)) for geom in range(unfused.ngeom)
    ]
    stood = [
        (str(prim.GetParent().GetPath()), *_placed(_world(prim)))
        for body in BODIES
        for folder in ("visuals", "collisions")
        for prim in stage.GetPrimAtPath(f"{ROBOT}/{body}/{folder}").GetChildren()
    ]
    assert [sum(_same_place(geom, other) for other in stood) for geom in placed] == [1] * len(placed)
    assert [sum(_same_place(geom, other) for other in placed) for geom in stood] == [1] * len(stood)


def test_every_link_of_the_urdf_stands_where_its_fixed_joints_place_it(stage, description, urdf):
    """Each link is one prim under the prim of the body that carries it,
    which a camera hanging from it is defined under and a lidar's frame
    placed by, at the URDF's place with every joint at zero: lidar_link and
    lidar_frame on the mount."""
    links = [element.get("name") for element in urdf.findall("link")]
    assert {builder.body_of(link, description) for link in ("lidar_link", "lidar_frame")} == {description.root_link}
    for link in links:
        if link in ("left_wheel_link", "right_wheel_link"):
            continue
        prim = _prim_named(stage, link)
        assert prim.GetPath().HasPrefix(f"{ROBOT}/{builder.body_of(link, description)}"), link
        ours = _world(prim)
        theirs = _urdf_link_in_root(description, link)
        assert ours[:3, 3] == pytest.approx(theirs[:3, 3], abs=1e-6), link
        assert ours[:3, :3] == pytest.approx(theirs[:3, :3], abs=1e-6), link


def test_the_engine_sweeps_the_lidar_from_lidar_frame_on_the_mount(stage, description):
    """The engine mounts the lidar in the rigid body that carries the prim of
    its parent link, lidar_link: the mount, whose colliders its rays pass,
    at the place the description gives lidar_frame."""
    botshare = known(builder.ROBOT_ID)
    (lidar,) = botshare.entry.lidars

    lidar_mount = lidar_sensor.mount_of(stage.GetPrimAtPath(ROBOT), botshare, lidar)

    frame = _urdf_link_in_root(description, lidar.frame)
    assert lidar_mount.body_path == f"{ROBOT}/{description.root_link}"
    assert lidar_mount.frame_in_body.position == pytest.approx(tuple(frame[:3, 3]), abs=1e-6)
    rotation = np.array(rotation_of(lidar_mount.frame_in_body.orientation.xyzw))
    assert rotation == pytest.approx(frame[:3, :3], abs=1e-6)


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


# A wheel on a hinge of the attributes `joint` a test gives it, driven by the
# actuators the test gives it.
_WHEEL_MJCF = """<mujoco><worldbody><body><joint name="wheel" type="hinge" {joint}/>
<geom type="sphere" size="0.1"/></body></worldbody><actuator>{actuators}</actuator></mujoco>"""
# A hinge whose actuator force its range caps at 16 N m either way.
_CAPPED = 'actuatorfrclimited="true" actuatorfrcrange="-16 16"'
# The two actuators a wheel's rate loop runs on.
_RATE_LOOP = '<velocity name="p" joint="wheel" kv="64"/><intvelocity name="i" joint="wheel" kp="128"/>'


def _wheel_driven_by(actuators: str, joint: str = _CAPPED):
    return mujoco.MjModel.from_xml_string(_WHEEL_MJCF.format(joint=joint, actuators=actuators))


def test_a_wheel_driven_as_the_engines_run_it_gives_its_loop():
    model = _wheel_driven_by(_RATE_LOOP)

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


@pytest.mark.parametrize(
    ("joint", "cap"),
    [
        ('actuatorfrclimited="false" actuatorfrcrange="-16 16"', "[-16, 16]"),
        ('actuatorfrclimited="true" actuatorfrcrange="-8 16"', "[-8, 16]"),
    ],
    ids=["no_limit", "uneven"],
)
def test_a_wheel_capped_at_no_one_torque_either_way_is_refused(joint, cap):
    refusal = f"wheel 'wheel': its actuators are capped at {cap}; give the joint an actuatorfrcrange of -cap cap"
    with pytest.raises(ValueError, match=f"^{re.escape(refusal)}$"):
        builder.rate_loop_of(_wheel_driven_by(_RATE_LOOP, joint), "wheel")


def test_a_wheel_that_turns_about_an_axis_other_than_its_bodys_y_is_refused():
    model = _wheel_driven_by(_RATE_LOOP, f'{_CAPPED} axis="1 0 0"')
    refusal = f"wheel joint 'wheel' turns about {model.jnt_axis[0]}, not its body's y"
    with pytest.raises(ValueError, match=f"^{re.escape(refusal)}$"):
        builder.define_wheel_joint(Usd.Stage.CreateInMemory(), model, "wheel", {})


# The description of a base on the wheels left_wheel_joint and right_wheel_joint.
_TWO_WHEELS = builder.Description(
    digest="digest",
    root_link="base_footprint",
    wheels=("left_wheel_joint", "right_wheel_joint"),
    child_links={"left_wheel_joint": "left_wheel_link", "right_wheel_joint": "right_wheel_link"},
    fixed={},
)


def _release(directory: Path, inside: str, assets: str = "") -> Path:
    """A release of the base in `directory`: its MJCF's mount, the body
    `base`, holds `inside`, and the MJCF's assets are `assets`."""
    (directory / builder.ROBOT_ID).mkdir()
    (directory / builder.ROBOT_ID / "base.xml").write_text(
        f'<mujoco><asset>{assets}</asset><worldbody><body name="base">'
        f'<geom type="sphere" size="0.2"/>{inside}</body></worldbody></mujoco>'
    )
    (directory / "robots.json").write_text(
        json.dumps([{"id": builder.ROBOT_ID, "wrapper": "base.xml", "mount": {"body": "base"}}])
    )
    return directory


def _wheel_body(joint: str) -> str:
    """A body `<joint>_body` that the hinge `joint` turns about its y."""
    return (
        f'<body name="{joint}_body"><joint name="{joint}" type="hinge" axis="0 1 0"/>'
        '<geom type="sphere" size="0.05"/></body>'
    )


def test_a_mesh_drawn_in_two_colours_is_refused(tmp_path):
    drawn = "".join(
        f'<geom type="mesh" mesh="shell" group="{builder.VISUAL_GROUP}" rgba="{rgba}"/>'
        for rgba in ("1 0 0 1", "0 1 0 1")
    )
    release = _release(tmp_path, drawn, '<mesh name="shell" vertex="0 0 0 1 0 0 0 1 0 0 0 1"/>')
    with pytest.raises(ValueError, match="^the MJCF draws the mesh 'shell' in two colours$"):
        builder.compile_model(release)


def test_hinges_other_than_the_descriptions_wheels_are_refused(tmp_path):
    release = _release(tmp_path, _wheel_body("left_wheel_joint"))
    refusal = (
        "the MJCF turns ['left_wheel_joint'], and the description's differential base "
        "('left_wheel_joint', 'right_wheel_joint')"
    )
    with pytest.raises(ValueError, match=f"^{re.escape(refusal)}$"):
        builder.build_stage(release, _TWO_WHEELS, tmp_path / "stage")


def test_a_body_a_joint_other_than_a_wheels_moves_is_refused(tmp_path):
    bumper = '<body name="bumper"><joint name="bumper" type="slide"/><geom type="sphere" size="0.05"/></body>'
    release = _release(tmp_path, _wheel_body("left_wheel_joint") + _wheel_body("right_wheel_joint") + bumper)
    refusal = (
        "the MJCF's bodies, each body no joint moves fused into the one it hangs from, are "
        "['base', 'left_wheel_joint_body', 'right_wheel_joint_body', 'bumper'], and the ones its mount and its "
        "hinges make links of are ['base', 'left_wheel_joint_body', 'right_wheel_joint_body']"
    )
    with pytest.raises(ValueError, match=f"^{re.escape(refusal)}$"):
        builder.build_stage(release, _TWO_WHEELS, tmp_path / "stage")


def test_bodies_no_joint_moves_fuse_into_the_body_they_hang_from(tmp_path):
    """A body no joint moves, on the mount or on another such body, is part
    of the mount: its mass adds to the mount's and its geoms stand on it."""
    bracket = (
        '<body name="bracket" pos="0.1 0 0"><geom type="sphere" size="0.05"/>'
        '<body name="bumper" pos="0.1 0 0"><geom type="sphere" size="0.05"/></body></body>'
    )
    release = _release(tmp_path, bracket)
    unfused = mujoco.MjModel.from_xml_path(str(release / builder.ROBOT_ID / "base.xml"))

    model = builder.compile_model(release)

    assert [model.body(body).name for body in range(model.nbody)] == ["world", "base"]
    assert model.body_mass[1] == pytest.approx(sum(unfused.body_mass))
    assert sorted(float(x) for x in model.geom_pos[:, 0]) == pytest.approx([0.0, 0.1, 0.2])


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
    wheel = model.body("left_wheel_link").id
    (tire_geom,) = [
        geom for geom in range(model.ngeom)
        if model.geom_bodyid[geom] == wheel and model.geom_group[geom] != builder.VISUAL_GROUP
    ]
    tire_name = builder.tire_name(float(model.geom_friction[tire_geom][0]))
    for body in ("left_wheel_link", "right_wheel_link"):
        (tire,) = stage.GetPrimAtPath(f"{ROBOT}/{body}/collisions").GetChildren()
        assert tire.IsA(UsdGeom.Cylinder)
        assert _bound_physics(tire) == tire_name
    chassis = stage.GetPrimAtPath(f"{ROBOT}/base_footprint/collisions").GetChildren()
    casters = [shape for shape in chassis if shape.IsA(UsdGeom.Sphere)]
    assert len(casters) == 4
    assert all(_bound_physics(shape) == "caster" for shape in casters)
    assert all(_bound_physics(shape) is None for shape in chassis if not shape.IsA(UsdGeom.Sphere))
    caster = stage.GetPrimAtPath(f"{builder.MATERIALS_PATH}/caster")
    assert UsdPhysics.MaterialAPI(caster).GetDynamicFrictionAttr().Get() == 0.0
    assert caster.GetAttribute("physxMaterial:frictionCombineMode").Get() == "min"
    tire_material = stage.GetPrimAtPath(f"{builder.MATERIALS_PATH}/{tire_name}")
    assert UsdPhysics.MaterialAPI(tire_material).GetDynamicFrictionAttr().Get() == pytest.approx(
        model.geom_friction[tire_geom][0]
    )
    assert tire_material.GetAttribute("physxMaterial:frictionCombineMode").Get() == "max"


def test_each_tire_friction_names_a_material_of_its_own():
    """Two tires of one friction share its material, and a tire of another
    friction takes one of its own."""
    frictions = (1.0, 0.8, 1.0)
    model = SimpleNamespace(
        geom_condim=np.full(len(frictions), 3),
        geom_friction=np.array([(friction, 0.005, 0.0001) for friction in frictions]),
    )
    stage = Usd.Stage.CreateInMemory()
    contacts: dict = {}
    tires = [builder.contact_of(stage, model, geom, True, contacts) for geom in range(len(frictions))]
    assert [tire.GetPath().name for tire in tires] == ["tire_1p0", "tire_0p8", "tire_1p0"]
    assert [
        UsdPhysics.MaterialAPI(tire.GetPrim()).GetDynamicFrictionAttr().Get() for tire in tires
    ] == pytest.approx(frictions)


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
