"""The models this engine stands, one entry per model under models/: what
MuJoCo alone knows about each, parsed strictly beside sim_robot_core's entry
of the same name."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from sim_robot_core.models import EngineModel, parse_entry, shipped_entry

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))

import mujoco_models  # noqa: E402  pylint: disable=C0413
from mujoco_models import ArmGains, MujocoModels, SitePose, parse  # noqa: E402  pylint: disable=C0413

_OPENARM_KP = (240.0, 240.0, 240.0, 240.0, 24.0, 31.0, 25.0)
_OPENARM_KD = (3.0, 3.0, 3.0, 3.0, 0.2, 0.2, 0.2)


def _parse(model: str, **engine):
    """This engine's entry of a shipped model, written as `engine` on top of
    the one key every entry needs."""
    return parse(EngineModel(entry=shipped_entry(model), engine={"scene": "scene.xml", **engine}))


class TestTheShippedEntries:
    def test_the_engine_stands_the_three_models_it_has_an_entry_for(self):
        assert MujocoModels.read().names() == ["openarm_v1", "openarm_v2", "so101"]

    def test_an_openarm_v1_is_servoed_and_compensated_and_draws_no_head_camera(self):
        known = MujocoModels.read().of("openarm_v1")
        assert known.model == "openarm_v1"
        assert known.scene == "openarm/openarm_bimanual_v1.xml"
        assert known.arm_gains == ArmGains(kp=_OPENARM_KP, kd=_OPENARM_KD)
        assert (known.gravity_compensation, known.camera_lights, known.head_camera) == (
            True,
            True,
            False,
        )
        assert known.world_links == frozenset()
        assert (known.link_bodies, known.joint_ranges, known.site_poses) == ({}, {}, {})

    def test_an_openarm_v2_draws_the_head_camera_on_its_folded_pedestal(self):
        known = MujocoModels.read().of("openarm_v2")
        assert known.scene == "openarm/openarm_bimanual_v2.xml"
        assert known.arm_gains == ArmGains(kp=_OPENARM_KP, kd=_OPENARM_KD)
        assert (known.gravity_compensation, known.camera_lights, known.head_camera) == (
            True,
            True,
            True,
        )
        assert known.world_links == frozenset({"openarm_body_link0"})

    def test_an_so101_runs_its_files_servos_corrected_to_its_description(self):
        known = MujocoModels.read().of("so101")
        assert known.scene == "so101/so101.xml"
        assert known.arm_gains is None
        assert (known.gravity_compensation, known.camera_lights, known.head_camera) == (
            False,
            False,
            False,
        )
        assert known.link_bodies == {"base_link": "base"}
        assert known.joint_ranges == {"wrist_roll": (-2.74385, 2.84121)}
        assert known.site_poses == {
            "gripperframe": SitePose(
                pos=(-0.0079, -0.000218121, -0.0981274), quat_wxyz=(0.0, 0.0, 1.0, 0.0)
            )
        }

    def test_every_entry_is_the_shared_entry_of_the_same_model(self):
        models = MujocoModels.read()
        for name in models.names():
            assert models.of(name).entry == shipped_entry(name)

    def test_an_unknown_model_says_what_the_engine_stands(self):
        with pytest.raises(
            ValueError,
            match="unknown model 'openarm_v3': this engine stands openarm_v1, openarm_v2, so101",
        ):
            MujocoModels.read().of("openarm_v3")


class TestReadingADirectory:
    def test_a_directory_stands_the_models_it_holds_an_entry_for(self, tmp_path):
        (tmp_path / "so101.json5").write_text('{ scene: "so101/scene.xml" }')
        models = MujocoModels.read(tmp_path)
        assert models.names() == ["so101"]
        with pytest.raises(ValueError, match="unknown model 'openarm_v2': this engine stands so101"):
            models.of("openarm_v2")

    def test_a_bad_entry_fails_the_read_naming_its_file(self, tmp_path):
        (tmp_path / "so101.json5").write_text('{ scene: "so101/scene.xml", gains: {} }')
        with pytest.raises(RuntimeError, match=r"models/so101\.json5: the entry has unknown key"):
            MujocoModels.read(tmp_path)


class TestStrictParsing:
    def test_an_unknown_key_is_refused_with_the_ones_allowed(self):
        with pytest.raises(RuntimeError) as refused:
            _parse("so101", gravity_compensated=True)
        message = str(refused.value)
        assert "models/so101.json5: the entry has unknown key(s) ['gravity_compensated']" in message
        assert "'gravity_compensation'" in message

    @pytest.mark.parametrize("scene", [None, "", 7])
    def test_an_entry_names_its_scene(self, scene):
        with pytest.raises(RuntimeError, match="scene must name the model's MJCF"):
            parse(EngineModel(entry=shipped_entry("so101"), engine={"scene": scene}))

    @pytest.mark.parametrize(
        "solver",
        [
            7,
            ["<option/>"],
            "<optionally this is prose>",
            "a note about the <option we want",
            "<option garbage",
            '<option integrator="nonsense"/>',
            "",
            "<!-- nothing -->",
            # An element that is not the one the key names.
            '<compiler angle="degree"/>',
            '<statistic meansize="0.05"/>',
            '<option timestep="0.004"/><size memory="10M"/>',
            '<option timestep="0.004"/><option timestep="0.009"/>',
        ],
    )
    def test_a_solver_mujoco_will_not_take_is_refused_at_the_entry(self, solver):
        """Every entry is read at node setup, so an element MuJoCo will not
        take is answered for there and not by the first robot to attach as
        this model."""
        with pytest.raises(
            RuntimeError,
            match=r"models/so101\.json5: solver (must be an|is no|is one) MJCF",
        ):
            _parse("so101", solver=solver)

    @pytest.mark.parametrize("timestep", ["0", "-1"])
    def test_a_solver_that_steps_nowhere_is_refused_at_the_entry(self, timestep):
        """A scene paces on its timestep: zero divides by it and a negative
        one never comes due, and both take until the first step to show."""
        with pytest.raises(RuntimeError, match=r"solver steps a timestep above zero"):
            _parse("so101", solver=f'<option timestep="{timestep}"/>')

    @pytest.mark.parametrize(
        "setting",
        [
            'timestep="nan"',
            'timestep="inf"',
            'gravity="0 0 nan"',
            'wind="nan 0 0"',
            'magnetic="nan 0 0"',
            'density="nan"',
            'impratio="nan"',
            'tolerance="nan"',
        ],
    )
    def test_a_solver_that_is_not_finite_is_refused_at_the_entry(self, setting):
        """A scene steps every setting of its `<option>`, so a value that is
        no number steps every robot standing in it to NaN and publishes
        that, from the first step on."""
        with pytest.raises(RuntimeError, match=r"solver is finite in every setting"):
            _parse("so101", solver=f"<option {setting}/>")

    def test_the_solver_an_entry_names_is_what_the_model_stands_under(self):
        known = _parse("so101", solver='<option integrator="implicitfast" timestep="0.007"/>')

        assert known.solver.timestep == pytest.approx(0.007)

    def test_an_entry_naming_no_solver_stands_under_its_own_file(self):
        assert _parse("so101").solver is None

    @pytest.mark.parametrize("key", ["gravity_compensation", "camera_lights", "head_camera"])
    def test_a_flag_is_true_or_false(self, key):
        with pytest.raises(RuntimeError, match=f"{key} must be true or false, got 1"):
            _parse("so101", **{key: 1})

    @pytest.mark.parametrize("world_links", ["base_link", ["base_link", ""], [3]])
    def test_world_links_is_a_list_of_names(self, world_links):
        with pytest.raises(RuntimeError, match="world_links must be a list of names"):
            _parse("so101", world_links=world_links)

    @pytest.mark.parametrize("link_bodies", [["base"], {"base_link": ""}, {"base_link": 3}])
    def test_link_bodies_maps_links_to_bodies(self, link_bodies):
        with pytest.raises(RuntimeError, match="link_bodies must map link names to body names"):
            _parse("so101", link_bodies=link_bodies)


class TestArmGains:
    @pytest.mark.parametrize(
        ("gains", "field"),
        [
            ({"kp": list(_OPENARM_KP[:6]), "kd": list(_OPENARM_KD)}, "kp"),
            ({"kp": list(_OPENARM_KP), "kd": [*_OPENARM_KD, 0.2]}, "kd"),
            ({"kp": list(_OPENARM_KP)}, "kd"),
        ],
    )
    def test_gains_come_one_per_joint_of_an_arm(self, gains, field):
        with pytest.raises(RuntimeError, match=rf"arm_gains\.{field} must be 7 finite numbers"):
            _parse("openarm_v1", arm_gains=gains)

    def test_a_gain_is_a_finite_number(self):
        gains = {"kp": [*_OPENARM_KP[:6], float("nan")], "kd": list(_OPENARM_KD)}
        with pytest.raises(RuntimeError, match=r"arm_gains\.kp must be 7 finite numbers"):
            _parse("openarm_v1", arm_gains=gains)

    def test_an_unknown_gain_key_is_refused(self):
        gains = {"kp": list(_OPENARM_KP), "kd": list(_OPENARM_KD), "ki": [0.0] * 7}
        with pytest.raises(RuntimeError, match=r"arm_gains has unknown key\(s\) \['ki'\]"):
            _parse("openarm_v1", arm_gains=gains)

    def test_gains_are_an_object(self):
        with pytest.raises(RuntimeError, match="arm_gains must be an object"):
            _parse("openarm_v1", arm_gains=[240.0])

    def test_one_set_of_gains_cannot_serve_arms_of_different_joint_counts(self):
        entry = parse_entry(
            "lopsided",
            "lopsided.json5",
            {"arms": [{"name": "long", "joints": ["a", "b"]}, {"name": "short", "joints": ["c"]}]},
        )
        engine = {"scene": "scene.xml", "arm_gains": {"kp": [1.0, 1.0], "kd": [0.1, 0.1]}}
        with pytest.raises(RuntimeError, match=r"arms of one joint count, and these have \[1, 2\]"):
            parse(EngineModel(entry=entry, engine=engine))


class TestCorrections:
    def test_a_joint_range_names_a_joint_a_limb_moves(self):
        with pytest.raises(
            RuntimeError, match=r"joint_ranges names joints no limb moves: \['wrist_yaw'\]"
        ):
            _parse("so101", joint_ranges={"wrist_yaw": [-1.0, 1.0], "wrist_roll": [-1.0, 1.0]})

    def test_a_joint_range_may_correct_a_finger_joint(self):
        known = _parse("so101", joint_ranges={"gripper": [-0.17, 1.74]})
        assert known.joint_ranges == {"gripper": (-0.17, 1.74)}

    @pytest.mark.parametrize("limits", [[1.0, -1.0], [0.5, 0.5]])
    def test_a_joint_range_runs_from_its_lower_to_its_upper_limit(self, limits):
        with pytest.raises(RuntimeError, match=r"joint_ranges\.wrist_roll .* is not a range"):
            _parse("so101", joint_ranges={"wrist_roll": limits})

    @pytest.mark.parametrize("limits", [[-1.0], [-1.0, 0.0, 1.0], [-1.0, "1.0"], [-1.0, float("inf")]])
    def test_a_joint_range_is_two_finite_numbers(self, limits):
        with pytest.raises(
            RuntimeError, match=r"joint_ranges\.wrist_roll must be 2 finite numbers"
        ):
            _parse("so101", joint_ranges={"wrist_roll": limits})

    def test_a_site_pose_carries_a_unit_quaternion(self):
        pose = {"pos": [0.0, 0.0, 0.0], "quat_wxyz": [1.0, 1.0, 0.0, 0.0]}
        with pytest.raises(
            RuntimeError, match=r"site_poses\.gripperframe\.quat_wxyz norm 1\.41\d* is not 1"
        ):
            _parse("so101", site_poses={"gripperframe": pose})

    def test_a_site_pose_spells_out_its_position_and_orientation(self):
        with pytest.raises(
            RuntimeError, match=r"site_poses\.gripperframe\.pos must be 3 finite numbers"
        ):
            _parse("so101", site_poses={"gripperframe": {"quat_wxyz": [1.0, 0.0, 0.0, 0.0]}})
        with pytest.raises(
            RuntimeError, match=r"site_poses\.gripperframe\.quat_wxyz must be 4 finite numbers"
        ):
            _parse("so101", site_poses={"gripperframe": {"pos": [0.0, 0.0, 0.0]}})

    def test_an_unknown_site_pose_key_is_refused(self):
        pose = {"pos": [0.0, 0.0, 0.0], "quat_wxyz": [1.0, 0.0, 0.0, 0.0], "euler": [0, 0, 0]}
        with pytest.raises(
            RuntimeError, match=r"site_poses\.gripperframe has unknown key\(s\) \['euler'\]"
        ):
            _parse("so101", site_poses={"gripperframe": pose})


class TestBodies:
    def test_a_mapped_link_is_the_body_its_entry_names(self):
        known = _parse("so101", link_bodies={"base_link": "base"})
        assert known.body_of("base_link") == "base"

    def test_any_other_link_is_the_body_of_its_own_name(self):
        known = _parse("so101", link_bodies={"base_link": "base"})
        assert known.body_of("gripper_link") == "gripper_link"


class TestActuatorParams:
    def test_the_gains_of_one_arm_repeat_on_every_arm(self):
        known = MujocoModels.read().of("openarm_v2")
        params = known.actuator_params()

        assert params["joint_names"] == known.entry.arm_joints()
        assert len(params["joint_names"]) == 14
        assert params["kp"] == [*_OPENARM_KP, *_OPENARM_KP]
        assert params["kd"] == [*_OPENARM_KD, *_OPENARM_KD]
        # The fingers keep their file's actuators.
        assert not set(params["joint_names"]) & set(known.entry.finger_joints())

    def test_a_model_driven_by_its_files_actuators_carries_none(self):
        assert MujocoModels.read().of("so101").actuator_params() == {
            "joint_names": [],
            "kp": [],
            "kd": [],
        }

    def test_a_one_arm_model_given_gains_carries_them_once(self):
        params = _parse("so101", arm_gains={"kp": [5.0] * 5, "kd": [0.5] * 5}).actuator_params()
        assert params == {
            "joint_names": list(shipped_entry("so101").arms[0].joints),
            "kp": [5.0] * 5,
            "kd": [0.5] * 5,
        }

    def test_gravity_compensation_is_the_scenes_and_not_the_actuators(self):
        """The robot's weight is compensated on the spec its scene compiles
        from (test_launcher.py), so asking for it changes no actuator."""
        assert _parse("so101", gravity_compensation=True).actuator_params() == (
            _parse("so101").actuator_params()
        )


class TestScenePath:
    def test_a_scene_is_read_from_the_baked_assets(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mujoco_models, "ASSETS_DIR", tmp_path)
        scene = tmp_path / "so101" / "scene.xml"
        scene.parent.mkdir()
        scene.write_text("<mujoco/>")
        assert _parse("so101", scene="so101/scene.xml").scene_path() == scene

    def test_a_scene_missing_from_the_image_is_reported(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mujoco_models, "ASSETS_DIR", tmp_path)
        with pytest.raises(FileNotFoundError, match="baked into the container image"):
            _parse("so101", scene="so101/scene.xml").scene_path()
