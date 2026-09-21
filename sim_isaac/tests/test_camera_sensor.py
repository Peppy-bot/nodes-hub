"""Tests for the capture throttle state machine: warm, promote, read, disarm,
and the consecutive-failure bound, and for where a model's cameras are hung
on its robot. Fakes stand in for annotators and render products; the
renderer's own behavior is exercised only in a live engine."""

import sys
from pathlib import Path

import numpy as np
import pytest

_ENGINE_DIR = Path(__file__).resolve().parents[1] / "engine"
sys.path.insert(0, str(_ENGINE_DIR))
sys.path.insert(0, str(_ENGINE_DIR / "exts"))

import camera_sensor
import isaac_models
from camera_geometry import pinhole
from camera_sensor import _MAX_CAPTURE_FAILURES, IsaacCameraSensor
from sim_robot_core.cameras import CameraConfig, DepthSpec
from sim_robot_core.models import Arm, EngineModel, ModelEntry, shipped_entry

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
        self.geometries = []
        self.depth_payloads = []
        self.delivers = delivers

    def timestamp_s(self):
        return 123.0

    def publish_color_frame(self, robot, name, timestamp_s, frame_id, *rest):
        self.frames.append((robot, name, frame_id))
        return self.delivers

    def publish_rgbd_frames(self, robot, name, timestamp_s, frame_id, align_mode, color, depth):
        self.frames.append((robot, name, frame_id))
        self.depth_payloads.append(depth)
        return self.delivers

    def publish_color_stream_info(self, *args):
        self.infos.append(args)

    def publish_rgbd_stream_info(self, *args):
        self.infos.append(args)

    def publish_color_geometry(self, *args):
        self.geometries.append(args)

    def publish_rgbd_geometry(self, *args):
        self.geometries.append(args)


def _known(*cameras, **engine):
    """A model carrying `cameras`, as this engine knows it."""
    entry = ModelEntry(
        model="rig",
        arms=(Arm(name="arm", joints=("joint",)),),
        grippers=(),
        cameras=tuple(cameras),
        start_posture={},
    )
    raw = {"stage": "rig/rig.usd", "articulation_root": ".", **engine}
    return isaac_models.parse(EngineModel(entry=entry, engine=raw))


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
    sensor = IsaacCameraSensor("alpha", "/World/alpha", _known(camera), io)
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
        assert io.frames == [("alpha", "cam", 0)]
        assert render_product.updates == [True, False]

    def test_empty_annotator_keeps_camera_armed(self, clock):
        sensor, annotator, render_product, io = _sensor()

        for _ in range(5):
            sensor.step()
        assert io.frames == []
        assert render_product.updates == [True]

        annotator.make_valid()
        sensor.step()
        assert io.frames == [("alpha", "cam", 0)]
        assert render_product.updates == [True, False]

    def test_next_cycle_waits_for_the_pacer(self, clock):
        sensor, annotator, render_product, io = _sensor()
        annotator.make_valid()
        for _ in range(3):
            sensor.step()
        assert io.frames == [("alpha", "cam", 0)]

        sensor.step()
        assert render_product.updates == [True, False]

        clock["t"] += 1.0 / _FPS
        for _ in range(3):
            sensor.step()
        assert io.frames == [("alpha", "cam", 0), ("alpha", "cam", 1)]
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
        assert io.frames == [("alpha", "cam", 0)]
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
        sensor = IsaacCameraSensor("alpha", "/World/alpha", _known(camera), io)
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
        assert io.frames == [("alpha", "rgbd", 0)]
        assert render_product.updates == [True, False]
        (_, _, _, payload) = io.depth_payloads[0]
        assert len(payload) == (_WIDTH // 2) * (_HEIGHT // 2) * 2


class TestGeometry:
    def test_geometry_goes_out_with_the_stream_info(self, clock):
        sensor, _, _, io = _sensor()
        sensor.step()
        assert len(io.infos) == 1
        assert io.geometries == [("alpha", "cam", pinhole(60.0, _WIDTH, _HEIGHT))]

    def test_an_rgbd_cameras_depth_grid_is_a_centred_pinhole(self, clock):
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
        sensor = IsaacCameraSensor("alpha", "/World/alpha", _known(camera), io)

        sensor._publish_geometry(camera)

        color = pinhole(60.0, _WIDTH, _HEIGHT)
        depth = pinhole(60.0, _WIDTH // 2, _HEIGHT // 2)
        assert io.geometries == [
            ("alpha", "rgbd", color, depth, "z", 0.1, 1.0, "depth_to_color",
             (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        ]
        # The same field of view at the depth stream's size, centred.
        assert depth.fx == pytest.approx(color.fx / 2)
        assert (depth.cx, depth.cy) == ((_WIDTH // 2 - 1) / 2, (_HEIGHT // 2 - 1) / 2)


class TestWhereAModelsCamerasHang:
    """Against real USD: a camera hangs from the prim its parent link is in
    its own model's stage, under its own robot's prim."""

    @pytest.fixture(autouse=True)
    def _usd(self):
        pytest.importorskip("pxr", reason="USD Python wheels are unavailable on this platform")

    @staticmethod
    def _stage(*link_paths):
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.CreateInMemory()
        for path in link_paths:
            UsdGeom.Xform.Define(stage, path)
        return stage

    @staticmethod
    def _hang(stage, root: str, known):
        sensor = IsaacCameraSensor("charlo", root, known, FakeIO())
        return [
            sensor._define_camera_prim(stage, stage.GetPrimAtPath(root), camera)
            for camera in known.entry.cameras
        ]

    def test_the_rig_is_the_cameras_of_the_robots_own_model(self):
        so101 = isaac_models.IsaacModels.read().of("so101")
        sensor = IsaacCameraSensor("charlo", "/World/charlo", so101, FakeIO())
        assert list(sensor._cameras) == ["front"]

        openarm = isaac_models.IsaacModels.read().of("openarm_v2")
        sensor = IsaacCameraSensor("alpha", "/World/alpha", openarm, FakeIO())
        assert list(sensor._cameras) == ["wrist_left", "wrist_right", "chest"]

    def test_an_so101s_front_camera_hangs_from_its_own_base_link(self):
        from pxr import UsdGeom

        # An OpenArm stands beside it, and another SO-101 with the same links.
        stage = self._stage(
            "/World/alpha/openarm_body_link0",
            "/World/bravo/base_link",
            "/World/charlo/base_link",
        )
        so101 = isaac_models.IsaacModels.read().of("so101")

        assert self._hang(stage, "/World/charlo", so101) == ["/World/charlo/base_link/front"]

        camera = stage.GetPrimAtPath("/World/charlo/base_link/front")
        assert camera.IsA(UsdGeom.Camera)
        (front,) = shipped_entry("so101").cameras
        translate, _orient = UsdGeom.Xformable(camera).GetOrderedXformOps()
        assert tuple(translate.Get()) == pytest.approx(front.pos)
        assert not stage.GetPrimAtPath("/World/bravo/base_link/front").IsValid()

    def test_an_openarm_v2s_cameras_hang_from_its_wrists_and_its_pedestal(self):
        stage = self._stage(
            "/World/alpha/openarm_body_link0",
            "/World/alpha/openarm_left_ee_base_link",
            "/World/alpha/openarm_right_ee_base_link",
        )
        openarm = isaac_models.IsaacModels.read().of("openarm_v2")

        assert self._hang(stage, "/World/alpha", openarm) == [
            "/World/alpha/openarm_left_ee_base_link/wrist_left",
            "/World/alpha/openarm_right_ee_base_link/wrist_right",
            "/World/alpha/openarm_body_link0/chest",
        ]

    def test_a_link_hangs_from_the_prim_its_models_entry_maps_it_to(self):
        stage = self._stage("/World/charlo/base")
        (front,) = shipped_entry("so101").cameras
        known = _known(front, link_prims={"base_link": "base"})

        assert self._hang(stage, "/World/charlo", known) == ["/World/charlo/base/front"]

    def test_a_parent_link_the_robots_stage_lacks_is_refused_with_what_it_has(self):
        stage = self._stage("/World/charlo/shoulder_link")
        so101 = isaac_models.IsaacModels.read().of("so101")

        with pytest.raises(RuntimeError) as refused:
            self._hang(stage, "/World/charlo", so101)

        message = str(refused.value)
        assert "camera 'front' parent link 'base_link' is not under /World/charlo as 'base_link'" in message
        assert "shoulder_link" in message
