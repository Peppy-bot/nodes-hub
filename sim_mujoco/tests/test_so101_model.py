"""The SO-101 this engine stands, held to so101_description: the hardware's
source of truth for the joints, their limits, the tool frame, the start
posture and the front camera.

The model is MuJoCo Menagerie's robotstudio_so101 at the commit
sim_base_images/so101_model.lock.json pins, read from the baked assets
(PEPPY_ROBOT_ASSETS_DIR, as <dir>/so101/so101.xml). Where that directory
holds no SO-101 the suite stages the pinned model itself, which needs the
network. Either way every file is checked against the lock, so what is
compared with the description is the model the image carries.
"""

from __future__ import annotations

import importlib.util
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest
from so101_description import limits, simulation
from so101_description.model import END_EFFECTOR_FRAME, KINEMATICS_URDF_PATH
from so101_description.units import GRIPPER_NAME, JOINT_NAMES

import runtime_fakes  # noqa: F401  pylint: disable=W0611  (installs the runtime the bridge imports)

_NODES_HUB = Path(__file__).resolve().parents[2]
_ENGINE_DIR = _NODES_HUB / "sim_mujoco" / "engine"
sys.path.insert(0, str(_ENGINE_DIR))

import mujoco_models  # noqa: E402  pylint: disable=C0413
from bridge_extension import MujocoBridgeExtension  # noqa: E402  pylint: disable=C0413
from exts.camera_sensor import add_cameras  # noqa: E402  pylint: disable=C0413
from mujoco_models import MujocoModels  # noqa: E402  pylint: disable=C0413
from world import World, compile_spec  # noqa: E402  pylint: disable=C0413

# The name this robot stands under, which every name it answers to carries.
ROBOT = "charlo"

MOTOR_JOINTS = (*JOINT_NAMES, GRIPPER_NAME)
# The URDF rounds its limits to five decimals and upstream writes seven.
LIMIT_TOLERANCE_RAD = 1e-4
TOOL_POSITION_TOLERANCE_M = 1e-5
TOOL_ORIENTATION_TOLERANCE = 1e-4
# The link the URDF hangs its tool frame from, and the fixed joint that
# places it there. MuJoCo fuses a fixed link into its parent, so the frame is
# composed from the two.
TOOL_PARENT_LINK = "gripper_link"
TOOL_JOINT = "gripper_frame_joint"
# The site that is the tool frame in the MJCF, and the body the URDF's base
# link is there.
TOOL_SITE = "gripperframe"
BASE_BODY = "base"

# Where each motor joint sits in its range, per configuration: mid-range,
# two spread poses, and the limits, with wrist_roll at the upper limit only
# the corrected range reaches.
_CONFIGURATIONS = [
    (0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
    (0.1, 0.8, 0.3, 0.6, 0.9, 0.2),
    (0.9, 0.2, 0.7, 0.1, 0.05, 0.8),
    (0.0, 1.0, 0.0, 1.0, 1.0, 1.0),
    (1.0, 0.0, 1.0, 0.0, 0.0, 0.0),
]


def _load_script(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


so101_model = _load_script(_NODES_HUB / "sim_base_images" / "so101_model.py")


@pytest.fixture(name="staged", scope="module", autouse=True)
def staged_fixture(tmp_path_factory):
    """The pinned model under the assets the engine reads: the baked one, or
    one this suite stages where none is baked."""
    assets = mujoco_models.ASSETS_DIR
    if not (assets / "so101").is_dir():
        assets = tmp_path_factory.mktemp("robot_assets")
    # Stages the model, or checks the one staged there against the lock.
    so101_model.fetch(assets / "so101")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(mujoco_models, "ASSETS_DIR", assets)
        yield


@pytest.fixture(name="known", scope="module")
def known_fixture():
    return MujocoModels.read().of("so101")


@pytest.fixture(name="model", scope="module")
def model_fixture(known):
    """The scene as a stand compiles it, with the entry's corrections."""
    return compile_spec(known).compile()


@pytest.fixture(name="upstream", scope="module")
def upstream_fixture(known):
    """The scene as upstream ships it."""
    return mujoco.MjModel.from_xml_path(str(known.scene_path()))


@pytest.fixture(name="rendered", scope="module")
def rendered_fixture(known):
    """The scene as a rendering engine composes it, with this robot standing
    in it."""
    world = World(head_camera_pack=None, renders=True)
    world.add(ROBOT, known, world.free_spot())
    return world.compose().compile()


@pytest.fixture(name="urdf", scope="module")
def urdf_fixture():
    """The description's kinematics URDF, as MuJoCo reads it."""
    return mujoco.MjModel.from_xml_path(KINEMATICS_URDF_PATH)


def _urdf_joints() -> dict:
    # Direct children only: transmission blocks nest <joint> stubs under the
    # same names.
    return {joint.get("name"): joint for joint in ET.parse(KINEMATICS_URDF_PATH).getroot().findall("joint")}


def _urdf_limits() -> dict[str, tuple[float, float]]:
    """Every motor joint's limits: the arm's as the description parses them,
    the gripper's read from the same file."""
    arm = limits.from_urdf(KINEMATICS_URDF_PATH)
    described = dict(zip(JOINT_NAMES, zip(arm.lower, arm.upper)))
    gripper = _urdf_joints()[GRIPPER_NAME].find("limit")
    described[GRIPPER_NAME] = (float(gripper.get("lower")), float(gripper.get("upper")))
    return described


def _off_the_urdf(model) -> dict[str, tuple[float, float]]:
    """The joints whose range is not the URDF's, with the range they have."""
    return {
        name: tuple(model.joint(name).range)
        for name, described in _urdf_limits().items()
        if np.abs(model.joint(name).range - described).max() > LIMIT_TOLERANCE_RAD
    }


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """A URDF rpy as a rotation: fixed-axis roll, pitch, yaw."""
    cr, sr, cp, sp, cy, sy = (
        np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw),
    )
    rot_x = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    rot_y = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rot_z = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rot_z @ rot_y @ rot_x


def _pose(model, configuration) -> mujoco.MjData:
    """The model's kinematics with every motor joint at its share of the
    URDF's range."""
    data = mujoco.MjData(model)
    for (name, (lower, upper)), share in zip(_urdf_limits().items(), configuration):
        data.qpos[model.joint(name).qposadr[0]] = lower + share * (upper - lower)
    mujoco.mj_kinematics(model, data)
    return data


def _in_frame(frame_pos, frame_mat, pos, mat):
    return frame_mat.T @ (pos - frame_pos), frame_mat.T @ mat


class TestJoints:
    def test_the_motor_joints_are_the_descriptions_in_wire_order(self, model):
        assert [model.joint(index).name for index in range(model.njnt)] == list(MOTOR_JOINTS)

    def test_each_joint_is_a_hinge_about_its_own_z_like_the_urdfs(self, model):
        described = _urdf_joints()
        for name in MOTOR_JOINTS:
            axis = [float(v) for v in described[name].find("axis").get("xyz").split()]
            assert described[name].get("type") == "revolute"
            assert model.joint(name).type[0] == mujoco.mjtJoint.mjJNT_HINGE
            assert axis == [0.0, 0.0, 1.0]
            assert model.joint(name).axis.tolist() == axis

    def test_every_joint_range_is_the_urdfs_limit(self, model):
        for name, described in _urdf_limits().items():
            assert model.joint(name).limited[0]
            assert model.joint(name).range == pytest.approx(described, abs=LIMIT_TOLERANCE_RAD)

    def test_upstream_alone_stops_wrist_roll_short_of_the_urdf(self, upstream):
        """Why the entry corrects a joint range: upstream's wrist_roll ends at
        2.7438 rad where the URDF, and so the hardware's backbone, reaches
        2.84121. Every other joint is the URDF's as shipped."""
        off = _off_the_urdf(upstream)

        assert list(off) == ["wrist_roll"]
        assert off["wrist_roll"][1] == pytest.approx(2.7438473)
        assert _urdf_limits()["wrist_roll"][1] == pytest.approx(2.84121)

    def test_the_entry_corrects_exactly_what_upstream_has_off(self, known, model, upstream):
        assert set(known.joint_ranges) == set(_off_the_urdf(upstream))
        assert _off_the_urdf(model) == {}


class TestTheSolverItStandsUnder:
    """A scene steps one set of solver settings, so every model standing in
    one asks for the same. The OpenArms ask for MuJoCo's defaults with an
    implicit-fast integrator, and this entry stands the SO-101 under them."""

    def test_upstream_tunes_the_solver_for_a_scene_it_has_to_itself(self, upstream):
        assert upstream.opt.timestep == pytest.approx(0.005)
        assert upstream.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
        assert upstream.opt.iterations == 10
        assert upstream.opt.ls_iterations == 20
        assert upstream.opt.impratio == pytest.approx(10.0)

    def test_the_entry_stands_it_under_what_the_openarms_stand_under(self, model):
        openarm = mujoco.MjModel.from_xml_string(
            '<mujoco><option integrator="implicitfast"/><worldbody/></mujoco>'
        )

        assert model.opt.integrator == openarm.opt.integrator
        assert model.opt.timestep == pytest.approx(openarm.opt.timestep)
        assert model.opt.cone == openarm.opt.cone
        assert model.opt.iterations == openarm.opt.iterations
        assert model.opt.ls_iterations == openarm.opt.ls_iterations
        assert model.opt.impratio == pytest.approx(openarm.opt.impratio)


class TestActuators:
    def test_a_position_servo_drives_every_motor_joint(self, model):
        for name in MOTOR_JOINTS:
            actuator = model.actuator(name)
            assert actuator.trntype[0] == mujoco.mjtTrn.mjTRN_JOINT
            assert actuator.trnid[0] == model.joint(name).id
            # A position servo: force = kp * (ctrl - q) - kv * dq.
            assert actuator.gaintype[0] == mujoco.mjtGain.mjGAIN_FIXED
            assert actuator.biastype[0] == mujoco.mjtBias.mjBIAS_AFFINE
            assert actuator.gainprm[0] > 0.0
            assert actuator.biasprm[1] == -actuator.gainprm[0]
        assert model.nu == len(MOTOR_JOINTS)

    def test_every_servo_is_clamped_to_its_joints_corrected_range(self, model):
        for name in MOTOR_JOINTS:
            assert model.actuator(name).ctrllimited[0]
            assert model.actuator(name).ctrlrange == pytest.approx(
                model.joint(name).range, abs=LIMIT_TOLERANCE_RAD
            )

    def test_the_jaws_servo_can_be_force_capped(self, model):
        assert model.actuator(GRIPPER_NAME).forcelimited[0]


class TestToolPoint:
    @staticmethod
    def _described(urdf, configuration):
        """The URDF's tool frame in its base link's frame, which MuJoCo
        fuses into the world."""
        joint = _urdf_joints()[TOOL_JOINT]
        assert (joint.find("parent").get("link"), joint.find("child").get("link")) == (
            TOOL_PARENT_LINK,
            END_EFFECTOR_FRAME,
        )
        origin = joint.find("origin")
        offset = np.array([float(v) for v in origin.get("xyz").split()])
        turn = _rpy_matrix(*(float(v) for v in origin.get("rpy").split()))
        parent = _pose(urdf, configuration).body(TOOL_PARENT_LINK)
        parent_mat = parent.xmat.reshape(3, 3)
        return parent.xpos + parent_mat @ offset, parent_mat @ turn

    @staticmethod
    def _site(model, configuration):
        """The MJCF's tool site in its base body's frame."""
        data = _pose(model, configuration)
        base, site = data.body(BASE_BODY), data.site(TOOL_SITE)
        return _in_frame(base.xpos, base.xmat.reshape(3, 3), site.xpos, site.xmat.reshape(3, 3))

    @pytest.mark.parametrize("configuration", _CONFIGURATIONS)
    def test_the_tool_site_is_the_urdfs_tool_frame(self, model, urdf, configuration):
        position, orientation = self._site(model, configuration)
        described_position, described_orientation = self._described(urdf, configuration)

        assert position == pytest.approx(described_position, abs=TOOL_POSITION_TOLERANCE_M)
        assert orientation == pytest.approx(described_orientation, abs=TOOL_ORIENTATION_TOLERANCE)

    def test_upstream_alone_puts_its_tool_site_off_the_urdfs_frame(self, upstream, urdf):
        """Why the entry corrects a site pose: upstream's site sits 19.9 mm
        from the URDF's tool frame and is turned a quarter turn from it."""
        position, orientation = self._site(upstream, _CONFIGURATIONS[0])
        described_position, described_orientation = self._described(urdf, _CONFIGURATIONS[0])

        assert np.linalg.norm(position - described_position) == pytest.approx(0.0199, abs=1e-4)
        turned = np.arccos((np.trace(described_orientation.T @ orientation) - 1.0) / 2.0)
        assert turned == pytest.approx(np.pi / 2, abs=1e-3)


class TestStartPosture:
    @pytest.fixture(name="standing")
    def standing_fixture(self, known):
        """The scene composed and started the way a stand starts it."""
        world = World(head_camera_pack=None, renders=False)
        robot = world.add(ROBOT, known, world.free_spot())
        model = world.compose().compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        MujocoBridgeExtension(
            model, data, None, [robot], 50, renders=False, time_base_s=0.0
        ).startup(posture_for=[robot])
        mujoco.mj_forward(model, data)
        return model, data

    def test_every_joint_starts_where_the_description_says(self, standing):
        """The arm in its start posture and the jaw closed."""
        model, data = standing
        described = simulation.start_positions_rad()
        started = {
            name: float(data.qpos[model.joint(f"{ROBOT}/{name}").qposadr[0]]) for name in described
        }

        assert list(described) == [*JOINT_NAMES, GRIPPER_NAME]
        assert started == pytest.approx(described)

    def test_the_start_posture_is_inside_the_urdfs_limits(self, standing):
        model, data = standing
        started = tuple(
            float(data.qpos[model.joint(f"{ROBOT}/{name}").qposadr[0]]) for name in JOINT_NAMES
        )

        assert limits.from_urdf(KINEMATICS_URDF_PATH).contains(started)

    def test_the_servos_hold_every_joint_where_it_starts(self, standing):
        model, data = standing
        described = simulation.start_positions_rad()
        targets = {
            name: float(data.ctrl[model.actuator(f"{ROBOT}/{name}").id]) for name in described
        }

        assert targets == pytest.approx(described)


class TestFrontCamera:
    def test_it_hangs_from_the_base_at_the_descriptions_pose(self, rendered):
        described = simulation.front_camera()
        camera = rendered.camera(f"{ROBOT}/{described.name}")
        orientation = np.array(described.quat_wxyz) / np.linalg.norm(described.quat_wxyz)

        assert rendered.body(int(camera.bodyid[0])).name == f"{ROBOT}/{BASE_BODY}"
        assert camera.pos == pytest.approx(described.pos)
        assert camera.quat == pytest.approx(orientation)
        assert camera.fovy[0] == pytest.approx(described.fovy_deg)

    def test_the_base_body_is_the_urdfs_base_link(self, known, rendered):
        """The camera's pose is written in the URDF's base link frame, so the
        body it hangs from has to be that frame."""
        base = rendered.body(f"{ROBOT}/{BASE_BODY}")

        assert known.body_of(simulation.front_camera().parent_link) == BASE_BODY
        assert base.pos.tolist() == [0.0, 0.0, 0.0]
        assert base.quat.tolist() == [1.0, 0.0, 0.0, 0.0]

    def test_it_streams_the_descriptions_resolution(self, known, rendered):
        described = simulation.front_camera()
        (camera,) = known.entry.cameras

        assert (camera.name, camera.width, camera.height, camera.fps) == (
            described.name,
            described.width,
            described.height,
            described.fps,
        )
        assert camera.depth is None
        # The offscreen buffer the renderer draws into covers the stream.
        assert rendered.vis.global_.offwidth >= described.width
        assert rendered.vis.global_.offheight >= described.height

    def test_the_scene_lights_the_floor_it_lays_for_the_arm(self, known, rendered, upstream):
        """The arm brings no light of its own, and the scene lights the floor
        it works against, once however many arms stand on it."""
        assert not known.camera_lights
        assert known.floor
        assert upstream.nlight == 0
        assert rendered.nlight == 1
