"""Tests for the camera plumbing: config parsing, slot validation, z16
conversion, and pacing. Pure python against camera_common; no engine import.

Run inside a base image carrying numpy and pyjson5, e.g.

    docker run --rm -v <openarm-nodes>:/repo peppybot/openarm-mujoco-sim:3.10.0-20 \\
      bash -c "pip install -q pytest && cd /repo/<engine-node> && python3 -m pytest tests/"
"""

import struct
import sys
from pathlib import Path

import numpy as np
import pytest

_ROBOT_DIR = Path(__file__).resolve().parents[1] / "robots" / "openarm"
sys.path.insert(0, str(_ROBOT_DIR))

from camera_common import (
    CameraConfig,
    DepthSpec,
    FrameIdCounter,
    FramePacer,
    depth_to_z16,
    load_camera_configs,
    validate_camera_slots,
)

_CAMERAS_CONFIG = _ROBOT_DIR / "config" / "cameras.json5"


class TestLoadCameraConfigs:
    def test_checked_in_config_parses(self):
        configs = load_camera_configs(_CAMERAS_CONFIG)
        assert [c.name for c in configs] == [
            "wrist_left",
            "wrist_right",
            "chest",
        ]
        by_name = {c.name: c for c in configs}
        for wrist in ("wrist_left", "wrist_right"):
            assert (by_name[wrist].width, by_name[wrist].height) == (960, 600)
            assert by_name[wrist].depth is None
            assert by_name[wrist].fps == 15
        chest = by_name["chest"]
        assert (chest.width, chest.height) == (1280, 720)
        assert chest.parent_link == "openarm_body_link0"
        assert chest.depth is not None
        assert (chest.depth.width, chest.depth.height) == (640, 360)

    def _write_config(self, tmp_path, body: str) -> Path:
        path = tmp_path / "cameras.json5"
        path.write_text(body)
        return path

    def _entry(self, **overrides) -> str:
        fields = {
            "name": '"cam"',
            "parent_link": '"link"',
            "pos": "[0, 0, 0]",
            "quat_wxyz": "[1, 0, 0, 0]",
            "fovy_deg": "60",
            "color": "{ width: 8, height: 4 }",
            "fps": "15",
        }
        fields.update(overrides)
        return "{" + ", ".join(f"{k}: {v}" for k, v in fields.items()) + "}"

    def test_unknown_camera_key_rejected(self, tmp_path):
        """A misspelled optional key must fail, not silently change the stream:
        `dpeth` would otherwise yield a colour-only camera where the file
        plainly describes an rgbd one."""
        path = self._write_config(
            tmp_path,
            "{ cameras: ["
            + self._entry(dpeth="{ width: 4, height: 2, min_depth_m: 0.1, max_range_m: 10 }")
            + "] }",
        )
        with pytest.raises(RuntimeError, match="unknown key"):
            load_camera_configs(path)

    def test_unknown_color_key_rejected(self, tmp_path):
        path = self._write_config(
            tmp_path,
            "{ cameras: [" + self._entry(color="{ width: 8, height: 4, fps: 15 }") + "] }",
        )
        with pytest.raises(RuntimeError, match="unknown key"):
            load_camera_configs(path)

    def test_unknown_depth_key_rejected(self, tmp_path):
        path = self._write_config(
            tmp_path,
            "{ cameras: ["
            + self._entry(
                depth="{ width: 4, height: 2, min_depth_m: 0.1, max_range_m: 10, units: 0.001 }"
            )
            + "] }",
        )
        with pytest.raises(RuntimeError, match="unknown key"):
            load_camera_configs(path)

    def test_duplicate_names_rejected(self, tmp_path):
        path = self._write_config(
            tmp_path, f"{{ cameras: [{self._entry()}, {self._entry()}] }}"
        )
        with pytest.raises(RuntimeError, match="duplicate"):
            load_camera_configs(path)

    def test_empty_list_rejected(self, tmp_path):
        path = self._write_config(tmp_path, "{ cameras: [] }")
        with pytest.raises(RuntimeError, match="non-empty"):
            load_camera_configs(path)

    def test_unnormalized_quat_rejected(self, tmp_path):
        path = self._write_config(
            tmp_path,
            f"{{ cameras: [{self._entry(quat_wxyz='[1, 1, 0, 0]')}] }}",
        )
        with pytest.raises(RuntimeError, match="norm"):
            load_camera_configs(path)

    def test_nonfinite_pos_rejected(self, tmp_path):
        path = self._write_config(
            tmp_path, f"{{ cameras: [{self._entry(pos='[0, 0, NaN]')}] }}"
        )
        with pytest.raises(RuntimeError, match="finite"):
            load_camera_configs(path)

    @pytest.mark.parametrize("fovy", ["0", "180", "-10"])
    def test_fovy_out_of_range_rejected(self, tmp_path, fovy):
        path = self._write_config(
            tmp_path, f"{{ cameras: [{self._entry(fovy_deg=fovy)}] }}"
        )
        with pytest.raises(RuntimeError, match="fovy"):
            load_camera_configs(path)

    @pytest.mark.parametrize("fps", ["0", "-1", "1.5", "256", "true"])
    def test_bad_fps_rejected(self, tmp_path, fps):
        path = self._write_config(tmp_path, f"{{ cameras: [{self._entry(fps=fps)}] }}")
        with pytest.raises(RuntimeError, match="fps"):
            load_camera_configs(path)

    def test_bool_fovy_rejected(self, tmp_path):
        path = self._write_config(
            tmp_path, f"{{ cameras: [{self._entry(fovy_deg='true')}] }}"
        )
        with pytest.raises(RuntimeError, match="fovy"):
            load_camera_configs(path)

    def test_bool_pos_component_rejected(self, tmp_path):
        path = self._write_config(
            tmp_path, f"{{ cameras: [{self._entry(pos='[0, true, 0]')}] }}"
        )
        with pytest.raises(RuntimeError, match="finite"):
            load_camera_configs(path)

    def test_non_object_entry_rejected(self, tmp_path):
        path = self._write_config(tmp_path, "{ cameras: [7] }")
        with pytest.raises(RuntimeError, match="object"):
            load_camera_configs(path)

    def test_non_object_depth_rejected(self, tmp_path):
        path = self._write_config(
            tmp_path, f"{{ cameras: [{self._entry(depth='7')}] }}"
        )
        with pytest.raises(RuntimeError, match="depth must be an object"):
            load_camera_configs(path)

    def test_bool_depth_range_rejected(self, tmp_path):
        entry = self._entry(
            depth="{ width: 4, height: 2, min_depth_m: true, max_range_m: 1.0 }"
        )
        path = self._write_config(tmp_path, f"{{ cameras: [{entry}] }}")
        with pytest.raises(RuntimeError, match="depth range"):
            load_camera_configs(path)

    def test_inverted_depth_range_rejected(self, tmp_path):
        entry = self._entry(
            depth="{ width: 4, height: 2, min_depth_m: 2.0, max_range_m: 1.0 }"
        )
        path = self._write_config(tmp_path, f"{{ cameras: [{entry}] }}")
        with pytest.raises(RuntimeError, match="depth range"):
            load_camera_configs(path)

    def test_fps_upper_bound_accepted(self, tmp_path):
        path = self._write_config(tmp_path, f"{{ cameras: [{self._entry(fps='255')}] }}")
        (config,) = load_camera_configs(path)
        assert config.fps == 255

    @pytest.mark.parametrize(
        "depth_range",
        [
            "min_depth_m: 0.1, max_range_m: 100.0",
            "min_depth_m: NaN, max_range_m: 1.0",
            "min_depth_m: 0.1, max_range_m: Infinity",
        ],
    )
    def test_depth_range_outside_wire_ceiling_rejected(self, tmp_path, depth_range):
        path = self._write_config(
            tmp_path,
            f"{{ cameras: [{self._entry(depth=f'{{ width: 4, height: 2, {depth_range} }}')}] }}",
        )
        with pytest.raises(RuntimeError, match="wire ceiling"):
            load_camera_configs(path)

    @pytest.mark.parametrize("dims", ["width: 3, height: 2", "width: 4, height: 1"])
    def test_depth_grid_must_subsample_color_grid(self, tmp_path, dims):
        path = self._write_config(
            tmp_path,
            f"{{ cameras: [{self._entry(depth=f'{{ {dims}, min_depth_m: 0.1, max_range_m: 1.0 }}')}] }}",
        )
        with pytest.raises(RuntimeError, match="subsample"):
            load_camera_configs(path)


class TestValidateCameraSlots:
    _COLOR = frozenset({"wrist_left", "wrist_right"})
    _RGBD = frozenset({"chest"})

    def _camera(self, name: str, depth: DepthSpec | None = None) -> CameraConfig:
        return CameraConfig(
            name=name,
            parent_link="link",
            pos=(0.0, 0.0, 0.0),
            quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            fovy_deg=60.0,
            width=8,
            height=4,
            fps=15,
            depth=depth,
        )

    def _depth(self) -> DepthSpec:
        return DepthSpec(width=4, height=2, min_depth_m=0.1, max_range_m=1.0)

    def test_exact_partition_passes(self):
        cameras = [
            self._camera("wrist_left"),
            self._camera("wrist_right"),
            self._camera("chest", self._depth()),
        ]
        validate_camera_slots(cameras, self._COLOR, self._RGBD)

    def test_missing_camera_rejected(self):
        cameras = [self._camera("wrist_left"), self._camera("chest", self._depth())]
        with pytest.raises(RuntimeError, match="does not match"):
            validate_camera_slots(cameras, self._COLOR, self._RGBD)

    def test_unknown_camera_rejected(self):
        cameras = [
            self._camera("wrist_left"),
            self._camera("wrist_right"),
            self._camera("head"),
            self._camera("chest", self._depth()),
        ]
        with pytest.raises(RuntimeError, match="does not match"):
            validate_camera_slots(cameras, self._COLOR, self._RGBD)

    def test_kind_mismatch_rejected(self):
        cameras = [
            self._camera("wrist_left", self._depth()),
            self._camera("wrist_right"),
            self._camera("chest"),
        ]
        with pytest.raises(RuntimeError, match="does not match"):
            validate_camera_slots(cameras, self._COLOR, self._RGBD)


class TestDepthSpec:
    def test_range_beyond_wire_ceiling_rejected(self):
        with pytest.raises(ValueError, match="wire ceiling"):
            DepthSpec(width=4, height=2, min_depth_m=0.1, max_range_m=65.536)

    def test_nonpositive_min_rejected(self):
        with pytest.raises(ValueError, match="wire ceiling"):
            DepthSpec(width=4, height=2, min_depth_m=0.0, max_range_m=1.0)

    def test_inverted_range_rejected(self):
        with pytest.raises(ValueError, match="wire ceiling"):
            DepthSpec(width=4, height=2, min_depth_m=2.0, max_range_m=1.0)


class TestDepthToZ16:
    _SPEC = DepthSpec(width=3, height=1, min_depth_m=0.4, max_range_m=65.535)

    def _decode(self, payload: bytes) -> list[int]:
        return list(struct.unpack(f"<{len(payload) // 2}H", payload))

    def test_in_range_converts_to_little_endian_mm(self):
        payload = depth_to_z16(np.array([[0.5, 1.0, 2.5]], dtype=np.float32), self._SPEC)
        assert self._decode(payload) == [500, 1000, 2500]

    def test_out_of_range_and_nonfinite_become_zero(self):
        depth = np.array([[0.1, np.nan, np.inf]], dtype=np.float32)
        assert self._decode(depth_to_z16(depth, self._SPEC)) == [0, 0, 0]

    def test_rounds_to_nearest_millimeter(self):
        depth = np.array([[1.2349, 1.2351, 0.4004]], dtype=np.float64)
        assert self._decode(depth_to_z16(depth, self._SPEC)) == [1235, 1235, 400]

    def test_exact_bounds_encode(self):
        depth = np.array([[0.4, 65.535, 65.0]], dtype=np.float64)
        assert self._decode(depth_to_z16(depth, self._SPEC)) == [400, 65535, 65000]

    def test_sub_millimeter_clips_to_one(self):
        spec = DepthSpec(width=1, height=1, min_depth_m=0.0002, max_range_m=1.0)
        depth = np.array([[0.0004]], dtype=np.float64)
        assert self._decode(depth_to_z16(depth, spec)) == [1]

    def test_shape_mismatch_rejected(self):
        with pytest.raises(ValueError, match="shape"):
            depth_to_z16(np.zeros((2, 3), dtype=np.float32), self._SPEC)


class TestFrameIdCounter:
    def test_wraps_at_u32(self):
        counter = FrameIdCounter()
        counter._next = 0xFFFFFFFF
        assert counter.next() == 0xFFFFFFFF
        assert counter.next() == 0


class TestFramePacer:
    def test_first_call_is_due(self):
        pacer = FramePacer(10)
        assert pacer.take_if_due(100.0)
        assert not pacer.take_if_due(100.05)
        assert pacer.take_if_due(100.1)

    def test_stall_resyncs_instead_of_bursting(self):
        pacer = FramePacer(10)
        assert pacer.take_if_due(100.0)
        assert pacer.take_if_due(105.0)
        # After the resync the next frame is one period out, not a burst of
        # 50 missed frames.
        assert not pacer.take_if_due(105.05)
        assert pacer.take_if_due(105.1)
