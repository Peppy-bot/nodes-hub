"""The models this engine stands, one entry per model under models/: what
Isaac Sim alone knows about each, parsed strictly beside sim_robot_core's
entry of the same name."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from sim_robot_core.models import EngineModel, parse_entry, shipped_entry

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))

import isaac_models  # noqa: E402  pylint: disable=C0413
from isaac_models import DriveGains, IsaacModels, parse  # noqa: E402  pylint: disable=C0413

_OPENARM_ARM_GAINS = DriveGains(
    kp=(240.0, 240.0, 240.0, 240.0, 24.0, 31.0, 25.0),
    kd=(3.0, 3.0, 3.0, 3.0, 0.2, 0.2, 0.2),
    max_efforts=(40.0, 40.0, 27.0, 27.0, 7.0, 7.0, 7.0),
)
_OPENARM_GRIPPER_GAINS = DriveGains(kp=(80.0, 80.0), kd=(2.0, 2.0), max_efforts=(5.0, 5.0))


def _parse(model: str, **engine):
    """This engine's entry of a shipped model, written as `engine` on top of
    the keys every entry needs."""
    raw = {"stage": f"{model}/{model}.usd", "articulation_root": ".", **engine}
    return parse(EngineModel(entry=shipped_entry(model), engine=raw))


def _gains(joints: int, **replaced) -> dict:
    return {"kp": [1.0] * joints, "kd": [0.1] * joints, "max_efforts": [2.0] * joints, **replaced}


class TestTheShippedEntries:
    def test_the_engine_stands_the_three_models_it_has_an_entry_for(self):
        assert IsaacModels.read().names() == ["openarm_v1", "openarm_v2", "so101"]

    def test_an_openarm_v1_is_servoed_and_compensated_and_draws_no_head_camera(self):
        known = IsaacModels.read().of("openarm_v1")
        assert known.model == "openarm_v1"
        assert known.stage == "openarm/openarm_bimanual.usd"
        assert (known.articulation_root, known.link_prims) == (".", {})
        assert known.arm_gains == _OPENARM_ARM_GAINS
        assert known.gripper_gains == _OPENARM_GRIPPER_GAINS
        assert (known.gravity_compensation, known.head_camera) == (True, False)

    def test_an_openarm_v2_is_the_same_servo_on_its_own_stage_with_the_head_camera(self):
        known = IsaacModels.read().of("openarm_v2")
        assert known.stage == "openarm/openarm_bimanual_v2.usd"
        assert (known.articulation_root, known.link_prims) == (".", {})
        assert known.arm_gains == _OPENARM_ARM_GAINS
        assert known.gripper_gains == _OPENARM_GRIPPER_GAINS
        assert (known.gravity_compensation, known.head_camera) == (True, True)

    def test_an_so101_is_held_by_its_own_position_drives_against_gravity(self):
        known = IsaacModels.read().of("so101")
        assert known.stage == "so101/so101.usd"
        assert (known.articulation_root, known.link_prims) == (".", {})
        assert known.arm_gains == DriveGains(
            kp=(998.22,) * 5, kd=(2.731,) * 5, max_efforts=(2.94,) * 5
        )
        assert known.gripper_gains == DriveGains(kp=(998.22,), kd=(2.731,), max_efforts=(2.94,))
        assert (known.gravity_compensation, known.head_camera) == (False, False)

    def test_every_entry_is_the_shared_entry_of_the_same_model(self):
        models = IsaacModels.read()
        for name in models.names():
            assert models.of(name).entry == shipped_entry(name)

    def test_an_unknown_model_says_what_the_engine_stands(self):
        with pytest.raises(
            ValueError,
            match="unknown model 'openarm_v3': this engine stands openarm_v1, openarm_v2, so101",
        ):
            IsaacModels.read().of("openarm_v3")


class TestReadingADirectory:
    def test_a_directory_stands_the_models_it_holds_an_entry_for(self, tmp_path):
        (tmp_path / "so101.json5").write_text('{ stage: "so101/so101.usd", articulation_root: "." }')
        models = IsaacModels.read(tmp_path)
        assert models.names() == ["so101"]
        with pytest.raises(ValueError, match="unknown model 'openarm_v2': this engine stands so101"):
            models.of("openarm_v2")

    def test_a_model_with_no_entry_falls_back_to_no_others(self, tmp_path):
        """An engine with an OpenArm's entry alone stands no SO-101 on it."""
        (tmp_path / "openarm_v2.json5").write_text(
            '{ stage: "openarm/openarm_bimanual_v2.usd", articulation_root: "." }'
        )
        with pytest.raises(ValueError, match="unknown model 'so101': this engine stands openarm_v2"):
            IsaacModels.read(tmp_path).of("so101")

    def test_a_bad_entry_fails_the_read_naming_its_file(self, tmp_path):
        (tmp_path / "so101.json5").write_text(
            '{ stage: "so101/so101.usd", articulation_root: ".", gains: {} }'
        )
        with pytest.raises(RuntimeError, match=r"models/so101\.json5: the entry has unknown key"):
            IsaacModels.read(tmp_path)

    def test_an_entry_for_a_model_nothing_describes_is_refused(self, tmp_path):
        (tmp_path / "openarm_v3.json5").write_text('{ stage: "x.usd", articulation_root: "." }')
        with pytest.raises(ValueError, match="no entry describes a 'openarm_v3'"):
            IsaacModels.read(tmp_path)


class TestStrictParsing:
    def test_an_unknown_key_is_refused_with_the_ones_allowed(self):
        with pytest.raises(RuntimeError) as refused:
            _parse("so101", gravity_compensated=True)
        message = str(refused.value)
        assert "models/so101.json5: the entry has unknown key(s) ['gravity_compensated']" in message
        assert "'gravity_compensation'" in message

    @pytest.mark.parametrize("stage", [None, "", 7])
    def test_an_entry_names_its_stage(self, stage):
        with pytest.raises(RuntimeError, match="stage must name the model's USD"):
            _parse("so101", stage=stage)

    @pytest.mark.parametrize("stage", ["/opt/so101.usd", "../so101.usd", "so101/../../so101.usd"])
    def test_a_stage_stays_under_the_baked_assets(self, stage):
        with pytest.raises(RuntimeError, match="stage must stay under the baked assets"):
            _parse("so101", stage=stage)

    @pytest.mark.parametrize("key", ["gravity_compensation", "head_camera"])
    def test_a_flag_is_true_or_false(self, key):
        with pytest.raises(RuntimeError, match=f"{key} must be true or false, got 1"):
            _parse("so101", **{key: 1})

    @pytest.mark.parametrize("root", [None, "", 3, "/World/so101", "../base_link", "base_link/", "a//b"])
    def test_an_articulation_root_is_the_robots_prim_or_a_path_under_it(self, root):
        with pytest.raises(
            RuntimeError, match="articulation_root must be a prim path under the robot's prim"
        ):
            _parse("so101", articulation_root=root)

    @pytest.mark.parametrize(
        "link_prims", [["base"], {"base_link": ""}, {"base_link": 3}, {"base_link": "so101/base"}]
    )
    def test_link_prims_maps_links_to_prim_names(self, link_prims):
        with pytest.raises(RuntimeError, match="link_prims must map link names to prim names"):
            _parse("so101", link_prims=link_prims)


class TestGains:
    @pytest.mark.parametrize("field", ["kp", "kd", "max_efforts"])
    def test_arm_gains_come_one_per_joint_of_an_arm(self, field):
        with pytest.raises(RuntimeError, match=rf"arm_gains\.{field} must be 7 finite numbers"):
            _parse("openarm_v1", arm_gains=_gains(7, **{field: [1.0] * 6}))

    @pytest.mark.parametrize("field", ["kp", "kd", "max_efforts"])
    def test_gripper_gains_come_one_per_finger_of_a_gripper(self, field):
        with pytest.raises(RuntimeError, match=rf"gripper_gains\.{field} must be 2 finite numbers"):
            _parse("openarm_v1", gripper_gains=_gains(2, **{field: [1.0]}))
        # The SO-101's gripper is one jaw.
        with pytest.raises(RuntimeError, match=rf"gripper_gains\.{field} must be 1 finite numbers"):
            _parse("so101", gripper_gains=_gains(1, **{field: [1.0, 1.0]}))

    @pytest.mark.parametrize("field", ["kp", "kd", "max_efforts"])
    def test_gains_spell_out_their_effort_ceilings_beside_kp_and_kd(self, field):
        gains = _gains(5)
        del gains[field]
        with pytest.raises(RuntimeError, match=rf"arm_gains\.{field} must be 5 finite numbers"):
            _parse("so101", arm_gains=gains)

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0, "240", True])
    def test_a_gain_is_a_finite_number_and_never_negative(self, value):
        with pytest.raises(RuntimeError, match=r"arm_gains\.kp must be 5 finite numbers, none negative"):
            _parse("so101", arm_gains=_gains(5, kp=[1.0, 1.0, 1.0, 1.0, value]))

    def test_an_unknown_gain_key_is_refused(self):
        with pytest.raises(RuntimeError, match=r"gripper_gains has unknown key\(s\) \['ki'\]"):
            _parse("so101", gripper_gains=_gains(1, ki=[0.0]))

    @pytest.mark.parametrize("key", ["arm_gains", "gripper_gains"])
    def test_gains_are_an_object(self, key):
        with pytest.raises(RuntimeError, match=f"{key} must be an object"):
            _parse("so101", **{key: [240.0]})

    def test_one_set_of_gains_cannot_serve_arms_of_different_joint_counts(self):
        entry = parse_entry(
            "lopsided",
            "lopsided.json5",
            {"arms": [{"name": "long", "joints": ["a", "b"]}, {"name": "short", "joints": ["c"]}]},
        )
        engine = {"stage": "lopsided.usd", "articulation_root": ".", "arm_gains": _gains(2)}
        with pytest.raises(RuntimeError, match=r"arms of one joint count, and these have \[1, 2\]"):
            parse(EngineModel(entry=entry, engine=engine))

    def test_a_model_with_no_gripper_carries_no_gripper_gains(self):
        entry = parse_entry(
            "bare", "bare.json5", {"arms": [{"name": "arm", "joints": ["a", "b"]}]}
        )
        engine = {"stage": "bare.usd", "articulation_root": ".", "gripper_gains": _gains(1)}
        with pytest.raises(RuntimeError, match=r"grippers of one joint count, and these have \[\]"):
            parse(EngineModel(entry=entry, engine=engine))


class TestDriveParams:
    def test_the_gains_of_one_arm_are_applied_to_every_arm(self):
        known = IsaacModels.read().of("openarm_v2")
        left, right = known.entry.arms

        for arm in (left, right):
            assert known.arm_params(arm) == {
                "joint_names": list(arm.joints),
                "kp": list(_OPENARM_ARM_GAINS.kp),
                "kd": list(_OPENARM_ARM_GAINS.kd),
                "max_efforts": list(_OPENARM_ARM_GAINS.max_efforts),
            }
        assert known.arm_params(left)["joint_names"] != known.arm_params(right)["joint_names"]

    def test_the_gains_of_one_gripper_are_applied_to_every_gripper(self):
        known = IsaacModels.read().of("openarm_v1")

        for gripper in known.entry.grippers:
            assert known.gripper_params(gripper) == {
                "joint_names": list(gripper.joints),
                "kp": [80.0, 80.0],
                "kd": [2.0, 2.0],
                "max_efforts": [5.0, 5.0],
            }

    def test_each_model_drives_its_limbs_with_its_own_gains(self):
        models = IsaacModels.read()
        so101, openarm = models.of("so101"), models.of("openarm_v2")

        (arm,) = so101.entry.arms
        (jaw,) = so101.entry.grippers
        assert so101.arm_params(arm) == {
            "joint_names": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            "kp": [998.22] * 5,
            "kd": [2.731] * 5,
            "max_efforts": [2.94] * 5,
        }
        assert so101.gripper_params(jaw) == {
            "joint_names": ["gripper"],
            "kp": [998.22],
            "kd": [2.731],
            "max_efforts": [2.94],
        }
        assert so101.arm_params(arm)["kp"] != openarm.arm_params(openarm.entry.arms[0])["kp"]

    def test_a_limb_whose_model_carries_no_gains_keeps_its_stages_drives(self):
        known = _parse("so101")
        (arm,) = known.entry.arms
        (jaw,) = known.entry.grippers

        assert known.arm_params(arm) == {
            "joint_names": list(arm.joints), "kp": [], "kd": [], "max_efforts": []
        }
        assert known.gripper_params(jaw) == {
            "joint_names": ["gripper"], "kp": [], "kd": [], "max_efforts": []
        }


class TestPrims:
    def test_a_stage_whose_default_prim_is_the_articulation_is_read_at_the_robots_prim(self):
        assert _parse("so101").articulation_path("/World/charlo") == "/World/charlo"

    def test_an_articulation_root_under_the_robots_prim_is_read_there(self):
        known = _parse("so101", articulation_root="base_link")
        assert known.articulation_path("/World/charlo") == "/World/charlo/base_link"

    def test_a_mapped_link_is_the_prim_its_entry_names(self):
        known = _parse("so101", link_prims={"base_link": "base"})
        assert known.prim_of("base_link") == "base"

    def test_any_other_link_is_the_prim_of_its_own_name(self):
        known = _parse("so101", link_prims={"base_link": "base"})
        assert known.prim_of("gripper_link") == "gripper_link"


class TestStagePath:
    def test_a_stage_is_read_from_the_baked_assets_under_its_robots_directory(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(isaac_models, "ASSETS_DIR", tmp_path)
        for model in ("openarm_v1", "openarm_v2", "so101"):
            known = IsaacModels.read().of(model)
            path = tmp_path / known.stage
            path.parent.mkdir(exist_ok=True)
            path.write_text("#usda 1.0")
            assert known.stage_path() == path
        assert sorted(path.name for path in tmp_path.iterdir()) == ["openarm", "so101"]

    def test_a_stage_missing_from_the_image_is_reported(self, tmp_path, monkeypatch):
        monkeypatch.setattr(isaac_models, "ASSETS_DIR", tmp_path)
        with pytest.raises(FileNotFoundError, match="baked into the container image"):
            IsaacModels.read().of("so101").stage_path()


class TestAssetsDir:
    @staticmethod
    def _loaded(monkeypatch):
        """isaac_models.py as a process starting now reads it, beside the one
        every other suite shares."""
        path = Path(isaac_models.__file__)
        spec = importlib.util.spec_from_file_location("_isaac_models_under_test", path)
        module = importlib.util.module_from_spec(spec)
        # Its dataclasses resolve postponed annotations through sys.modules.
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        return module

    def test_the_baked_assets_are_read_from_where_the_deployment_points(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PEPPY_ROBOT_ASSETS_DIR", str(tmp_path))
        module = self._loaded(monkeypatch)
        stage = tmp_path / "so101" / "so101.usd"
        stage.parent.mkdir()
        stage.write_text("#usda 1.0")

        assert module.ASSETS_DIR == tmp_path
        assert module.IsaacModels.read().of("so101").stage_path() == stage

    def test_a_native_run_reads_them_beside_the_engine(self, monkeypatch):
        monkeypatch.delenv("PEPPY_ROBOT_ASSETS_DIR", raising=False)
        module = self._loaded(monkeypatch)

        assert module.ASSETS_DIR == Path(isaac_models.__file__).parent / "assets" / "robots"
