"""Tests for the capture throttle state machine: warm, promote, read, disarm,
and the consecutive-failure bound. Fakes stand in for annotators and render
products; the renderer's own behavior is exercised only in a live engine."""

import sys
from pathlib import Path

import numpy as np
import pytest

_ROBOT_DIR = Path(__file__).resolve().parents[1] / "robots" / "openarm"
sys.path.insert(0, str(_ROBOT_DIR))
sys.path.insert(0, str(_ROBOT_DIR / "exts"))

import camera_sensor
from camera_common import CameraConfig, DepthSpec
from camera_sensor import _MAX_CAPTURE_FAILURES, IsaacCameraSensor

_WIDTH, _HEIGHT = 4, 2
_FPS = 10


class FakeAnnotator:
    def __init__(self):
        self.data = np.zeros((0,), dtype=np.uint8)

    def get_data(self):
        return self.data

    def make_valid(self):
        self.data = np.ones((_HEIGHT, _WIDTH, 4), dtype=np.uint8)


class FakeHydraTexture:
    def __init__(self, log):
        self._log = log

    def set_updates_enabled(self, enabled):
        self._log.append(enabled)


class FakeRenderProduct:
    def __init__(self):
        self.updates = []
        self.hydra_texture = FakeHydraTexture(self.updates)


class FakeIO:
    def __init__(self, delivers=True):
        self.frames = []
        self.infos = []
        self.depth_payloads = []
        self.delivers = delivers

    def camera_timestamp_s(self):
        return 123.0

    def publish_color_frame(self, name, timestamp_s, frame_id, *rest):
        self.frames.append((name, frame_id))
        return self.delivers

    def publish_rgbd_frames(self, name, timestamp_s, frame_id, align_mode, color, depth):
        self.frames.append((name, frame_id))
        self.depth_payloads.append(depth)
        return self.delivers

    def publish_color_stream_info(self, *args):
        self.infos.append(args)

    def publish_rgbd_stream_info(self, *args):
        self.infos.append(args)


@pytest.fixture
def clock(monkeypatch):
    state = {"t": 0.0}
    monkeypatch.setattr(camera_sensor.time, "monotonic", lambda: state["t"])
    return state


def _sensor(throttled=True):
    camera = CameraConfig(
        name="cam",
        parent_link="link",
        pos=(0.0, 0.0, 0.0),
        quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        fovy_deg=60.0,
        width=_WIDTH,
        height=_HEIGHT,
        fps=_FPS,
        depth=None,
    )
    io = FakeIO()
    sensor = IsaacCameraSensor("/openarm", [camera], io)
    annotator = FakeAnnotator()
    render_product = FakeRenderProduct()
    sensor._annotators[("cam", "color")] = annotator
    sensor._render_products[("cam", "color")] = render_product
    sensor._ready = True
    sensor._throttled = throttled
    return sensor, annotator, render_product, io


class TestThrottledCycle:
    def test_reads_only_on_the_second_step_after_arming(self, clock):
        sensor, annotator, render_product, io = _sensor()
        annotator.make_valid()

        sensor.step()
        assert render_product.updates == [True]
        assert io.frames == []

        sensor.step()
        assert io.frames == []

        sensor.step()
        assert io.frames == [("cam", 0)]
        assert render_product.updates == [True, False]

    def test_empty_annotator_keeps_camera_armed(self, clock):
        sensor, annotator, render_product, io = _sensor()

        for _ in range(5):
            sensor.step()
        assert io.frames == []
        assert render_product.updates == [True]

        annotator.make_valid()
        sensor.step()
        assert io.frames == [("cam", 0)]
        assert render_product.updates == [True, False]

    def test_next_cycle_waits_for_the_pacer(self, clock):
        sensor, annotator, render_product, io = _sensor()
        annotator.make_valid()
        for _ in range(3):
            sensor.step()
        assert io.frames == [("cam", 0)]

        sensor.step()
        assert render_product.updates == [True, False]

        clock["t"] += 1.0 / _FPS
        for _ in range(3):
            sensor.step()
        assert io.frames == [("cam", 0), ("cam", 1)]
        assert render_product.updates == [True, False, True, False]

    def test_deadline_during_capture_does_not_double_arm(self, clock):
        sensor, _annotator, render_product, _io = _sensor()

        sensor.step()
        clock["t"] += 1.0 / _FPS
        sensor.step()
        clock["t"] += 1.0 / _FPS
        sensor.step()
        assert render_product.updates == [True]
        assert sensor._warming | sensor._armed == {"cam"}

    def test_persistent_failure_raises_at_bound(self, clock):
        sensor, _annotator, _render_product, _io = _sensor()

        sensor.step()
        sensor.step()
        with pytest.raises(RuntimeError, match="consecutive reads"):
            for _ in range(_MAX_CAPTURE_FAILURES):
                sensor.step()

    def test_transport_drops_are_not_renderer_failures(self, clock):
        """A publish the transport drops is a consumer falling behind, not a
        renderer that disagrees with the config. Counting drops toward the bound
        would raise "renderer output and camera config disagree" at a camera
        whose renderer and config are both correct, and would hold it armed so
        it renders every update while the machine is already late."""
        sensor, annotator, render_product, io = _sensor()
        annotator.make_valid()
        io.delivers = False

        # Well past the bound: if a drop counted, this would raise.
        for _ in range(_MAX_CAPTURE_FAILURES + 10):
            clock["t"] += 1.0 / _FPS
            sensor.step()

        assert sensor._capture_failures["cam"] == 0
        assert io.frames, "frames were rendered and offered to the transport"
        # The camera arms and disarms once per paced cycle. Treating a drop as a
        # failed read skips the disarm, which leaves the render product enabled
        # forever; the repeated False entries are that disarm still happening.
        assert render_product.updates.count(False) > 1

    def test_success_resets_the_failure_count(self, clock):
        sensor, annotator, _render_product, _io = _sensor()
        for _ in range(4):
            sensor.step()
        assert sensor._capture_failures["cam"] > 0

        annotator.make_valid()
        sensor.step()
        assert sensor._capture_failures["cam"] == 0


class TestUnthrottled:
    def test_due_captures_immediately_without_arming(self, clock):
        sensor, annotator, render_product, io = _sensor(throttled=False)
        annotator.make_valid()

        sensor.step()
        assert io.frames == [("cam", 0)]
        assert render_product.updates == []
        assert sensor._armed == set() and sensor._warming == set()


class TestRgbdCapture:
    def test_throttled_cycle_publishes_subsampled_depth(self, clock):
        camera = CameraConfig(
            name="rgbd",
            parent_link="link",
            pos=(0.0, 0.0, 0.0),
            quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            fovy_deg=60.0,
            width=_WIDTH,
            height=_HEIGHT,
            fps=_FPS,
            depth=DepthSpec(
                width=_WIDTH // 2, height=_HEIGHT // 2, min_depth_m=0.1, max_range_m=1.0
            ),
        )
        io = FakeIO()
        sensor = IsaacCameraSensor("/openarm", [camera], io)
        color = FakeAnnotator()
        color.make_valid()
        depth = FakeAnnotator()
        depth.data = np.full((_HEIGHT, _WIDTH), 0.5, dtype=np.float32)
        render_product = FakeRenderProduct()
        sensor._annotators[("rgbd", "color")] = color
        sensor._annotators[("rgbd", "depth")] = depth
        sensor._render_products[("rgbd", "color")] = render_product
        sensor._ready = True
        sensor._throttled = True

        for _ in range(3):
            sensor.step()
        assert io.frames == [("rgbd", 0)]
        assert render_product.updates == [True, False]
        (_, _, _, payload) = io.depth_payloads[0]
        assert len(payload) == (_WIDTH // 2) * (_HEIGHT // 2) * 2
