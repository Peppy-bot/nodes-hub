"""The MuJoCo model of each robot this engine stands, held to its description,
the URDF and SRDF robot_description bundles: the joints and their limits, the
start posture, and where the cameras hang. The models are the release of
Waldo's catalogue that sim_base_images/waldo_catalogue.py pins, read as baked
where PEPPY_ROBOT_ASSETS_DIR holds one, and staged here otherwise, which
needs the network. Either way every file is checked against the pin, so what
is compared with the description is the model the image carries."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
import robot_description_py
from entries import staged_release

import runtime_fakes  # noqa: F401  pylint: disable=W0611  (installs the runtime the bridge imports)

import mujoco_models  # noqa: E402  pylint: disable=C0413
from bridge_extension import MujocoBridgeExtension  # noqa: E402  pylint: disable=C0413
from mujoco_models import MujocoModels  # noqa: E402  pylint: disable=C0413
from world import World, compile_spec  # noqa: E402  pylint: disable=C0413

# The name each robot stands under, which every name it answers to carries.
ROBOT = "charlo"
MODELS = ("openarm_v1", "openarm_v2", "so101")
# The URDFs round their limits to five decimals and the MJCFs write seven.
LIMIT_TOLERANCE_RAD = 1e-4
POSITION_TOLERANCE_M = 1e-4
ORIENTATION_TOLERANCE = 1e-5


@pytest.fixture(name="release", scope="module", autouse=True)
def release_fixture():
    release = staged_release(mujoco_models.ASSETS_DIR)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(mujoco_models, "ASSETS_DIR", release)
        yield release


def _described(model: str):
    return robot_description_py.model(model).description()


def _known(model: str):
    return MujocoModels.read().of(model, _described(model))


def _standing(model: str, renders: bool = False):
    """The scene composed and started the way a stand starts it."""
    known = _known(model)
    world = World(renders=renders)
    robot = world.add(ROBOT, known, world.free_spot())
    compiled = world.compose().compile()
    data = mujoco.MjData(compiled)
    mujoco.mj_forward(compiled, data)
    MujocoBridgeExtension(compiled, data, None, [robot], 50, renders=False, time_base_s=0.0).startup(
        posture_for=[robot]
    )
    mujoco.mj_forward(compiled, data)
    return known, compiled, data


@pytest.mark.parametrize("model", MODELS)
class TestJoints:
    def test_every_joint_the_description_moves_is_a_joint_of_the_model(self, model):
        known, compiled, _ = _standing(model)
        for joint in known.entry.joints():
            assert mujoco.mj_name2id(compiled, mujoco.mjtObj.mjOBJ_JOINT, f"{ROBOT}/{joint}") >= 0

    def test_every_joint_range_is_the_urdfs_limit(self, model):
        known, compiled, _ = _standing(model)
        description = _described(model)
        for joint in known.entry.joints():
            limits = description.joint_limits(joint)
            assert compiled.joint(f"{ROBOT}/{joint}").range == pytest.approx(
                (limits.lower, limits.upper), abs=LIMIT_TOLERANCE_RAD
            ), joint


@pytest.mark.parametrize("model", MODELS)
class TestStartPosture:
    def test_every_joint_starts_where_the_description_says(self, model):
        known, compiled, data = _standing(model)
        started = {
            joint: float(data.qpos[compiled.joint(f"{ROBOT}/{joint}").qposadr[0]])
            for joint in known.entry.start_posture
        }
        assert started == pytest.approx(known.entry.start_posture)

    def test_the_servos_hold_every_joint_where_it_starts(self, model):
        known, compiled, data = _standing(model)
        for _ in range(200):
            mujoco.mj_step(compiled, data)
        held = {
            joint: float(data.qpos[compiled.joint(f"{ROBOT}/{joint}").qposadr[0]])
            for joint in known.entry.start_posture
        }
        assert held == pytest.approx(known.entry.start_posture, abs=1e-3)


# The SO-101's tool frame: the fixed joint of the URDF that places it on the
# gripper link, and the site that is it in the MJCF.
_SO101_TOOL_JOINT = "gripper_frame_joint"
_SO101_TOOL_SITE = "gripperframe"
_SO101_BASE_BODY = "base"
# Where each motor joint sits in its range, per configuration.
_CONFIGURATIONS = [
    (0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
    (0.1, 0.8, 0.3, 0.6, 0.9, 0.2),
    (0.9, 0.2, 0.7, 0.1, 0.05, 0.8),
    (0.0, 1.0, 0.0, 1.0, 1.0, 1.0),
    (1.0, 0.0, 1.0, 0.0, 0.0, 0.0),
]


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """A URDF rpy as a rotation: fixed-axis roll, pitch, yaw."""
    cr, sr, cp, sp, cy, sy = (
        np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw),
    )
    rot_x = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    rot_y = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rot_z = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rot_z @ rot_y @ rot_x


@pytest.fixture(name="urdf", scope="module")
def urdf_fixture(tmp_path_factory):
    """The SO-101 description's URDF, as MuJoCo reads it beside its meshes."""
    directory = tmp_path_factory.mktemp("so101")
    robot_description_py.model("so101").write_to(directory)
    return mujoco.MjModel.from_xml_path(str(directory / "so101.urdf"))


@pytest.fixture(name="model", scope="module")
def model_fixture():
    """The SO-101's MJCF as a stand compiles it."""
    return compile_spec(_known("so101")).compile()


class TestTheSo101:
    @staticmethod
    def _pose(model, configuration):
        known = _known("so101")
        data = mujoco.MjData(model)
        description = _described("so101")
        for joint, share in zip(known.entry.joints(), configuration):
            limits = description.joint_limits(joint)
            data.qpos[model.joint(joint).qposadr[0]] = limits.lower + share * (limits.upper - limits.lower)
        mujoco.mj_kinematics(model, data)
        return data

    @staticmethod
    def _in_base(data, body: str):
        base = data.body(_SO101_BASE_BODY)
        frame = base.xmat.reshape(3, 3)
        return frame.T @ (data.body(body).xpos - base.xpos), frame.T @ data.body(body).xmat.reshape(3, 3)

    @pytest.mark.parametrize("configuration", _CONFIGURATIONS)
    def test_the_tool_site_is_the_urdfs_tool_frame(self, model, urdf, configuration):
        data = self._pose(model, configuration)
        base = data.body(_SO101_BASE_BODY)
        frame = base.xmat.reshape(3, 3)
        site = data.site(_SO101_TOOL_SITE)
        position = frame.T @ (site.xpos - base.xpos)
        orientation = frame.T @ site.xmat.reshape(3, 3)

        joint = {j.get("name"): j for j in ET.fromstring(robot_description_py.model("so101").urdf).findall("joint")}[_SO101_TOOL_JOINT]
        origin = joint.find("origin")
        offset = np.array([float(v) for v in origin.get("xyz").split()])
        turn = _rpy_matrix(*(float(v) for v in origin.get("rpy").split()))
        parent = self._pose(urdf, configuration).body(joint.find("parent").get("link"))
        parent_mat = parent.xmat.reshape(3, 3)

        assert position == pytest.approx(parent.xpos + parent_mat @ offset, abs=1e-5)
        assert orientation == pytest.approx(parent_mat @ turn, abs=1e-4)

    @pytest.mark.parametrize("configuration", _CONFIGURATIONS)
    def test_the_body_the_wrist_camera_hangs_from_is_the_urdfs_link(self, model, urdf, configuration):
        known = _known("so101")
        (camera,) = known.entry.cameras
        position, orientation = self._in_base(self._pose(model, configuration), known.body_of(camera.parent_link))
        described = self._pose(urdf, configuration).body(camera.parent_link)

        assert position == pytest.approx(described.xpos, abs=1e-5)
        assert orientation == pytest.approx(described.xmat.reshape(3, 3), abs=1e-4)

    def test_the_scene_lights_the_floor_it_lays_for_the_arm(self):
        known, compiled, _ = _standing("so101", renders=True)
        assert (known.camera_lights, known.floor) == (False, True)
        assert compiled.nlight == 1


class TestTheOpenArmV2HeadCamera:
    def test_the_chest_camera_is_the_left_eye_of_the_head_camera_the_model_draws(self):
        """The description puts the chest camera where the CAD seats the ZED
        Mini's left lens front, and the model draws that ZED Mini with its two
        eyes where the same CAD puts them: the stream looks out of the drawn
        lens."""
        known = _known("openarm_v2")
        chest = next(camera for camera in known.entry.cameras if camera.name == "chest")
        model = mujoco.MjModel.from_xml_path(str(known.scene_path()))
        data = mujoco.MjData(model)
        mujoco.mj_kinematics(model, data)
        mujoco.mj_camlight(model, data)
        carrier = data.body(known.body_of(chest.parent_link))
        frame = carrier.xmat.reshape(3, 3)
        eye = data.cam("camera_head_left")
        position = frame.T @ (eye.xpos - carrier.xpos)
        rotation = frame.T @ eye.xmat.reshape(3, 3)
        described = np.zeros(9)
        mujoco.mju_quat2Mat(described, np.array(chest.quat_wxyz) / np.linalg.norm(chest.quat_wxyz))

        assert position == pytest.approx(chest.pos, abs=POSITION_TOLERANCE_M)
        assert rotation == pytest.approx(described.reshape(3, 3), abs=ORIENTATION_TOLERANCE)
