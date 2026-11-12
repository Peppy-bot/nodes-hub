"""The models this engine stands: its entry of each, parsed strictly, the
MuJoCo model of each from the release of Waldo's catalogue, and both held to
a robot's description through sim_robot_core's entry of the same name.

The release is read as baked where PEPPY_ROBOT_ASSETS_DIR holds one, and
staged here otherwise, which needs the network."""

from __future__ import annotations

from pathlib import Path

import pyjson5
import pytest
from entries import entry, known, staged_release
from robot_description_py import model as bundled
from sim_robot_core.errors import ModelEntryError
from sim_robot_core.models import shipped_entry

import mujoco_models
from catalogue_robots import ServoGains
from mujoco_models import MujocoModels, Servo, parse

# The node whose defaults are the real OpenArm arm's MIT gains.
_ARM_NODE = Path(__file__).resolve().parents[2] / "openarm" / "arm" / "peppy.json5"


@pytest.fixture(name="release", scope="module", autouse=True)
def release_fixture():
    release = staged_release(mujoco_models.ASSETS_DIR)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(mujoco_models, "ASSETS_DIR", release)
        yield release


def _entry(model: str):
    """The entry of a robot of the bundled model `model`."""
    return shipped_entry(model, bundled(model).description())


def _read(model: str):
    """The model of the shipped entries and the release, as a robot of it stands."""
    return MujocoModels.read().of(model, bundled(model).description())


def _driver_gains() -> tuple[ServoGains, ...]:
    """The MIT gains the real arm runs: the defaults of the arm node's kp1..kp7
    and kd1..kd7."""
    parameters = pyjson5.loads(_ARM_NODE.read_text())["execution"]["parameters"]
    return tuple(
        ServoGains(kp=parameters[f"kp{j}"]["$default"], kd=parameters[f"kd{j}"]["$default"])
        for j in range(1, 8)
    )


class TestTheShippedModels:
    def test_the_engine_stands_the_three_models_it_has_an_entry_for(self):
        assert MujocoModels.read().names() == ["openarm_v1", "openarm_v2", "so101"]

    @pytest.mark.parametrize("model", ["openarm_v1", "openarm_v2"])
    def test_an_openarm_runs_the_real_arms_gains_on_every_arm(self, model):
        """The release holds the gains both engines run, and they are the real
        arm's, so a sim moves the way the hardware does under the same
        commands."""
        known = _read(model)
        for arm in known.entry.arms:
            assert known.robot.arm_gains[arm.name] == _driver_gains()
        assert known.robot.gravity_compensation
        assert known.camera_lights
        assert not known.floor

    def test_an_so101_runs_its_files_servos_against_a_floor(self):
        known = _read("so101")
        assert known.robot.arm_gains == {}
        assert not known.robot.gravity_compensation
        assert (known.camera_lights, known.floor) == (False, True)

    def test_every_model_stands_from_its_wrapper_by_the_body_that_carries_it(self, release):
        for name in MujocoModels.read().names():
            known = _read(name)
            assert known.scene_path().is_relative_to(release / name)
            assert known.robot.mount_body

    def test_every_entry_is_the_shared_entry_of_the_same_model(self):
        models = MujocoModels.read()
        for name in models.names():
            assert models.of(name, bundled(name).description()).entry == _entry(name)

    def test_an_unknown_model_says_what_the_engine_stands(self):
        with pytest.raises(
            ValueError,
            match="unknown model 'openarm_v3': this engine stands openarm_v1, openarm_v2, so101",
        ):
            MujocoModels.read().of("openarm_v3", bundled("so101").description())


class TestReadingADirectory:
    def test_a_directory_stands_the_models_it_holds_an_entry_for(self, tmp_path):
        (tmp_path / "so101.json5").write_text("{ floor: true }")
        models = MujocoModels.read(tmp_path)
        assert models.names() == ["so101"]
        with pytest.raises(ValueError, match="unknown model 'openarm_v2': this engine stands so101"):
            models.of("openarm_v2", bundled("openarm_v2").description())

    def test_a_bad_entry_fails_the_read_naming_its_file(self, tmp_path):
        (tmp_path / "so101.json5").write_text('{ scene: "so101/scene.xml" }')
        with pytest.raises(ModelEntryError, match=r"models/so101\.json5: the entry has unknown key\(s\) \['scene'\]"):
            MujocoModels.read(tmp_path)

    def test_a_model_the_release_does_not_stand_fails_the_read(self, tmp_path):
        entries = tmp_path / "models"
        entries.mkdir()
        (entries / "openarm_v1.json5").write_text("{}")
        (tmp_path / "robots.json").write_text("[]")
        with pytest.raises(ModelEntryError, match=r"stands \['openarm_v1'\], which the release of the assets does not"):
            MujocoModels.read(entries, tmp_path)


class TestStrictParsing:
    def test_an_unknown_key_is_refused_with_the_ones_allowed(self):
        with pytest.raises(ModelEntryError) as refused:
            parse("so101", {"gravity_compensation": True})
        message = str(refused.value)
        assert "models/so101.json5: the entry has unknown key(s) ['gravity_compensation']" in message
        assert "['camera_lights', 'floor']" in message

    @pytest.mark.parametrize("key", ["camera_lights", "floor"])
    def test_a_flag_is_true_or_false(self, key):
        with pytest.raises(ModelEntryError, match=f"{key} must be true or false, got 1"):
            parse("so101", {key: 1})


_ARM = entry("arm", arms={"left": ["a", "b"], "right": ["c", "d"]}, grippers={"left_jaw": {"f": (0.0, 1.0)}, "right_jaw": {"g": (0.0, 1.0)}})


def _known(**release):
    return known("arm", _ARM, Path("arm.xml"), "base", **release)


class TestTheReleasesGains:
    def test_each_arm_runs_its_own_gains_and_each_gripper_its_own(self):
        known = _known(
            arm_gains={"left": [(1.0, 0.1), (2.0, 0.2)], "right": [(3.0, 0.3), (4.0, 0.4)]},
            gripper_gains={"right": (5.0, 0.5)},
        )
        assert known.servos() == {
            "a": Servo(ServoGains(1.0, 0.1), critical=True),
            "b": Servo(ServoGains(2.0, 0.2), critical=True),
            "c": Servo(ServoGains(3.0, 0.3), critical=True),
            "d": Servo(ServoGains(4.0, 0.4), critical=True),
            "g": Servo(ServoGains(5.0, 0.5), critical=False),
        }
        # The controller raises the arms' damping to critical, and the
        # fingers keep theirs.
        assert known.actuator_params() == {"joint_names": ["a", "b", "c", "d"], "kp": [1.0, 2.0, 3.0, 4.0], "kd": [0.1, 0.2, 0.3, 0.4]}

    def test_a_model_given_no_gains_runs_its_files_actuators(self):
        assert _known().servos() == {}
        assert _known().actuator_params() == {"joint_names": [], "kp": [], "kd": []}

    def test_gains_for_an_arm_the_description_lacks_are_refused(self):
        with pytest.raises(ModelEntryError, match=r"robot 'arm': gains name the arm 'middle', and the description's arms are \['left', 'right'\]"):
            _known(arm_gains={"middle": [(1.0, 0.1)]})
        with pytest.raises(ModelEntryError, match="gains name the arm 'middle'"):
            _known(gripper_gains={"middle": (1.0, 0.1)})

    def test_gains_for_another_count_of_joints_are_refused(self):
        with pytest.raises(ModelEntryError, match="arm 'left' has gains for 1 joints, and its group moves 2"):
            _known(arm_gains={"left": [(1.0, 0.1)]})


class TestBodies:
    def test_a_mapped_link_is_the_body_the_release_names(self):
        assert _known(link_bodies={"base_link": "base"}).body_of("base_link") == "base"

    def test_any_other_link_is_the_body_of_its_own_name(self):
        assert _known(link_bodies={"base_link": "base"}).body_of("gripper_link") == "gripper_link"
