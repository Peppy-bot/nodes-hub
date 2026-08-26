"""Tests for the MuJoCo camera path: attaching cameras to a compiled scene,
and the render thread's pacing, framing and failure handling. The scene is a
real (tiny) MJCF compiled by MuJoCo; only the Renderers are fakes, because
rendering needs a GL context the test host may not have.
"""

import dataclasses
import logging
import struct
import sys
import threading
import time
from pathlib import Path

import mujoco
import numpy as np
import pytest

_ROBOT_DIR = Path(__file__).resolve().parents[1] / "robots" / "openarm"
sys.path.insert(0, str(_ROBOT_DIR))
sys.path.insert(0, str(_ROBOT_DIR / "exts"))

import camera_sensor
from camera_common import CameraConfig, DepthSpec
from camera_sensor import (
    _LATE_REPORT_PERIOD_S,
    MujocoCameraSensor,
    _PoseSnapshot,
    compile_model_with_cameras,
)

_SCENE_OFFSCREEN = (320, 240)
_WELDED_LINK = "openarm_body_link0"
_COLOR = (8, 4)  # width, height
_DEPTH = (4, 2)
_FPS = 10
# Heartbeat bound this suite patches in, small enough that waiting it out costs
# a fraction of a second rather than the module's real five.
_PATCHED_HEARTBEAT_S = 0.2


@pytest.fixture(name="scene")
def scene_fixture(tmp_path):
    """A scene with one movable body, one camera of its own, and an offscreen
    buffer smaller than the streams under test."""
    path = tmp_path / "scene.xml"
    path.write_text(
        f"""<mujoco model="test">
          <visual>
            <global offwidth="{_SCENE_OFFSCREEN[0]}" offheight="{_SCENE_OFFSCREEN[1]}"/>
          </visual>
          <worldbody>
            <camera name="scene_cam" pos="0 0 1"/>
            <body name="link_a" pos="0 0 0.5">
              <joint name="hinge_a" type="hinge" axis="0 1 0"/>
              <geom type="box" size="0.1 0.1 0.1"/>
            </body>
          </worldbody>
        </mujoco>"""
    )
    return path


def color_camera(name="wrist", parent="link_a", fps=_FPS):
    return CameraConfig(
        name=name,
        parent_link=parent,
        pos=(0.0, 0.0, 0.1),
        quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        fovy_deg=60.0,
        width=_COLOR[0],
        height=_COLOR[1],
        fps=fps,
        depth=None,
    )


def rgbd_camera(name="chest", parent=_WELDED_LINK, fps=_FPS):
    return CameraConfig(
        name=name,
        parent_link=parent,
        pos=(0.0, 0.0, 0.2),
        quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        fovy_deg=50.0,
        width=_COLOR[0],
        height=_COLOR[1],
        fps=fps,
        depth=DepthSpec(
            width=_DEPTH[0], height=_DEPTH[1], min_depth_m=0.1, max_range_m=10.0
        ),
    )


def camera_body(model, name):
    return model.cam_bodyid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)]


class FakeRenderer:
    def __init__(self, height, width, depth_m=None):
        self._height = height
        self._width = width
        self._depth_m = depth_m
        self.rendered = []
        self.poses = []
        self.camera_positions = []
        self.closed = False

    def update_scene(self, data, camera):
        self.rendered.append(camera)
        self.poses.append(np.array(data.qpos, copy=True))
        self.camera_positions.append(np.array(data.cam_xpos, copy=True))

    def render(self):
        if self._depth_m is not None:
            return np.full((self._height, self._width), self._depth_m, dtype=np.float32)
        return np.full((self._height, self._width, 3), 7, dtype=np.uint8)

    @property
    def renders_depth(self):
        return self._depth_m is not None

    def enable_depth_rendering(self):
        self._depth_m = 0.0

    def close(self):
        self.closed = True


class FakeIO:
    def __init__(self, delivers=True):
        self.frames = []
        self.rgbd = []
        self.infos = []
        self.delivers = delivers
        self.now_s = 123.0

    def camera_timestamp_s(self):
        return self.now_s

    def publish_color_frame(
        self, name, timestamp_s, frame_id, encoding, width, height, frame
    ):
        self.frames.append((name, frame_id, encoding, width, height, frame, timestamp_s))
        return self.delivers

    def publish_rgbd_frames(self, name, timestamp_s, frame_id, align_mode, color, depth):
        self.rgbd.append((name, frame_id, align_mode, color, depth, timestamp_s))
        return self.delivers

    def publish_color_stream_info(self, name, width, height, fps, encoding):
        self.infos.append((name, width, height, fps, encoding))

    def publish_rgbd_stream_info(
        self, name, width, height, fps, encoding,
        depth_width, depth_height, depth_encoding, depth_unit,
    ):
        self.infos.append((name, width, height, fps, encoding,
                           depth_width, depth_height, depth_encoding, depth_unit))


@pytest.fixture(name="clock")
def clock_fixture(monkeypatch):
    """A monotonic clock the test sets to an absolute time, so a deadline
    lands on the tick that should carry it rather than a float epsilon short
    of it."""

    class Clock:
        def __init__(self):
            self.now = 0.0

        def set(self, seconds):
            self.now = seconds

    clock = Clock()
    monkeypatch.setattr(camera_sensor.time, "monotonic", lambda: clock.now)
    return clock


def sensor_with_fakes(model, cameras, io, depth_m=1.5):
    """A sensor plus the renderer dicts its render loop would have built."""
    sensor = MujocoCameraSensor(model, cameras, io)
    color = {(c.height, c.width): FakeRenderer(c.height, c.width) for c in cameras}
    depth = {
        (c.depth.height, c.depth.width): FakeRenderer(
            c.depth.height, c.depth.width, depth_m=depth_m
        )
        for c in cameras
        if c.depth is not None
    }
    return sensor, color, depth


def render_due(sensor, model, color, depth, pose=None):
    data = mujoco.MjData(model)
    qpos = np.zeros(model.nq)
    if pose is not None:
        sensor.snapshot(pose)
    sensor._render_due(data, qpos, color, depth)  # pylint: disable=W0212


def test_compile_attaches_a_camera_to_its_parent_body(scene):
    model = compile_model_with_cameras(scene, [color_camera()])
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link_a")
    assert camera_body(model, "wrist") == body


def test_compile_welds_the_base_link_camera_onto_the_world_body(scene):
    model = compile_model_with_cameras(scene, [rgbd_camera()])
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, _WELDED_LINK) == -1
    assert camera_body(model, "chest") == 0


def test_compile_rejects_a_parent_link_the_scene_does_not_carry(scene):
    with pytest.raises(RuntimeError) as excinfo:
        compile_model_with_cameras(scene, [color_camera(parent="no_such_link")])
    message = str(excinfo.value)
    assert "no_such_link" in message
    assert "link_a" in message


def test_compile_rejects_a_camera_name_the_scene_already_uses(scene):
    with pytest.raises(RuntimeError, match="scene_cam"):
        compile_model_with_cameras(scene, [color_camera(name="scene_cam")])


def test_compile_grows_the_offscreen_buffer_to_cover_every_stream(scene):
    wide = dataclasses.replace(color_camera(), width=640, height=480)
    model = compile_model_with_cameras(scene, [wide])
    assert (model.vis.global_.offwidth, model.vis.global_.offheight) == (640, 480)


def test_compile_keeps_an_offscreen_buffer_already_large_enough(scene):
    model = compile_model_with_cameras(scene, [color_camera(), rgbd_camera()])
    assert (model.vis.global_.offwidth, model.vis.global_.offheight) == _SCENE_OFFSCREEN


def test_pose_snapshot_reports_nothing_until_the_physics_thread_writes():
    snapshot = _PoseSnapshot(2)
    out = np.array([9.0, 9.0])
    assert snapshot.read_into(out) is None
    assert out.tolist() == [9.0, 9.0]
    snapshot.write(np.array([1.0, 2.0]), 17.5)
    assert snapshot.read_into(out) == 17.5
    assert out.tolist() == [1.0, 2.0]


def test_no_frames_before_the_physics_thread_publishes_a_pose(scene, clock):
    model = compile_model_with_cameras(scene, [color_camera()])
    io = FakeIO()
    sensor, color, depth = sensor_with_fakes(model, [color_camera()], io)

    render_due(sensor, model, color, depth)
    assert io.frames == []
    # Stream info describes the config, not the scene, so it does not wait.
    assert [info[0] for info in io.infos] == ["wrist"]

    clock.set(1.0 / _FPS)
    render_due(sensor, model, color, depth, pose=np.zeros(model.nq))
    assert [name for name, *_ in io.frames] == ["wrist"]


def test_captures_only_the_cameras_whose_deadline_is_due(scene, clock):
    cameras = [color_camera(name="fast", fps=20), color_camera(name="slow", fps=5)]
    model = compile_model_with_cameras(scene, cameras)
    io = FakeIO()
    sensor, color, depth = sensor_with_fakes(model, cameras, io)
    pose = np.zeros(model.nq)

    render_due(sensor, model, color, depth, pose=pose)
    assert sorted(name for name, *_ in io.frames) == ["fast", "slow"]

    io.frames.clear()
    clock.set(1.0 / 20)
    render_due(sensor, model, color, depth, pose=pose)
    assert [name for name, *_ in io.frames] == ["fast"]

    io.frames.clear()
    clock.set(1.0 / 5)
    render_due(sensor, model, color, depth, pose=pose)
    assert sorted(name for name, *_ in io.frames) == ["fast", "slow"]


def test_color_capture_publishes_a_full_rgb8_frame(scene, clock):
    model = compile_model_with_cameras(scene, [color_camera()])
    io = FakeIO()
    sensor, color, depth = sensor_with_fakes(model, [color_camera()], io)

    render_due(sensor, model, color, depth, pose=np.zeros(model.nq))

    name, frame_id, encoding, width, height, frame, _ = io.frames[0]
    assert (name, frame_id, encoding) == ("wrist", 0, "rgb8")
    assert (width, height) == _COLOR
    assert len(frame) == _COLOR[0] * _COLOR[1] * 3
    assert set(frame) == {7}


def test_rgbd_capture_pairs_color_and_depth_under_one_frame_id(scene, clock):
    camera = rgbd_camera()
    model = compile_model_with_cameras(scene, [camera])
    io = FakeIO()
    sensor, color, depth = sensor_with_fakes(model, [camera], io, depth_m=1.5)

    render_due(sensor, model, color, depth, pose=np.zeros(model.nq))

    name, frame_id, align_mode, color_frame, depth_frame, _ = io.rgbd[0]
    assert (name, frame_id, align_mode) == ("chest", 0, "depth_to_color")
    assert color_frame[:3] == ("rgb8", _COLOR[0], _COLOR[1])
    assert depth_frame[:3] == ("z16", _DEPTH[0], _DEPTH[1])
    # Depth renders at the depth stream's own size, so no resampling happens.
    assert len(depth_frame[3]) == _DEPTH[0] * _DEPTH[1] * 2
    millimeters = struct.unpack(f"<{_DEPTH[0] * _DEPTH[1]}H", depth_frame[3])
    assert set(millimeters) == {1500}
    assert color[(camera.height, camera.width)].rendered == ["chest"]
    assert depth[(camera.depth.height, camera.depth.width)].rendered == ["chest"]


def test_frame_ids_count_captures_per_camera(scene, clock):
    cameras = [color_camera(name="fast", fps=20), color_camera(name="slow", fps=5)]
    model = compile_model_with_cameras(scene, cameras)
    io = FakeIO()
    sensor, color, depth = sensor_with_fakes(model, cameras, io)
    pose = np.zeros(model.nq)

    for tick in range(3):
        clock.set(tick / 20)
        render_due(sensor, model, color, depth, pose=pose)

    per_camera = {}
    for name, frame_id, *_ in io.frames:
        per_camera.setdefault(name, []).append(frame_id)
    assert per_camera["fast"] == [0, 1, 2]
    assert per_camera["slow"] == [0]


def test_stream_info_repeats_on_its_own_slow_cadence(scene, clock):
    model = compile_model_with_cameras(scene, [color_camera()])
    io = FakeIO()
    sensor, color, depth = sensor_with_fakes(model, [color_camera()], io)
    pose = np.zeros(model.nq)

    # One second of ticks carries the stream's description twice (the one it
    # opens with, and the 1 Hz repeat) against a full second of frames.
    for tick in range(_FPS + 1):
        clock.set(tick / _FPS)
        render_due(sensor, model, color, depth, pose=pose)
    assert [info[0] for info in io.infos] == ["wrist", "wrist"]
    assert len(io.frames) == _FPS


def test_captures_at_rate_are_never_reported_as_late(scene, clock, caplog):
    model = compile_model_with_cameras(scene, [color_camera()])
    sensor, color, depth = sensor_with_fakes(model, [color_camera()], FakeIO())
    pose = np.zeros(model.nq)

    with caplog.at_level(logging.WARNING):
        for tick in range(2 * _FPS):
            clock.set(tick / _FPS)
            render_due(sensor, model, color, depth, pose=pose)
    assert caplog.messages == []


def test_a_stalled_camera_is_reported_while_it_is_still_stalled(scene, clock, caplog):
    """The report must not wait for recovery: a camera that never delivers
    again is the case worth hearing about."""
    model = compile_model_with_cameras(scene, [color_camera()])
    io = FakeIO(delivers=False)
    sensor, color, depth = sensor_with_fakes(model, [color_camera()], io)
    pose = np.zeros(model.nq)

    io.delivers = True
    render_due(sensor, model, color, depth, pose=pose)
    io.delivers = False
    with caplog.at_level(logging.WARNING):
        for period in (1, 2):
            clock.set(period * _LATE_REPORT_PERIOD_S)
            render_due(sensor, model, color, depth, pose=pose)
    assert caplog.messages == [
        "1 late frame interval(s): worst 1000 ms on wrist",
        "1 late frame interval(s): worst 2000 ms on wrist",
    ]


def test_a_dropped_publish_is_not_a_delivery(scene, clock):
    model = compile_model_with_cameras(scene, [color_camera()])
    io = FakeIO(delivers=False)
    sensor, color, depth = sensor_with_fakes(model, [color_camera()], io)

    render_due(sensor, model, color, depth, pose=np.zeros(model.nq))

    # The frame was rendered and handed to the transport, which dropped it.
    assert len(io.frames) == 1
    assert sensor._streams["wrist"].last_delivery_s is None  # pylint: disable=W0212


def test_every_camera_in_a_cycle_is_stamped_with_the_pose_it_shows(scene, clock):
    cameras = [color_camera(name="wrist"), rgbd_camera()]
    model = compile_model_with_cameras(scene, cameras)
    io = FakeIO()
    sensor, color, depth = sensor_with_fakes(model, cameras, io)

    io.now_s = 100.0
    sensor.snapshot(np.zeros(model.nq))
    # Time moves on while the cycle renders; the frames still name the pose.
    io.now_s = 100.5
    render_due(sensor, model, color, depth)

    assert io.frames[0][-1] == 100.0
    assert io.rgbd[0][-1] == 100.0


def test_the_rendered_scene_carries_the_published_pose(scene, clock):
    """The renderer is handed a private MjData; only driving it from a real
    pose proves the snapshot, the kinematics and the camera frames all run."""
    camera = color_camera()
    model = compile_model_with_cameras(scene, [camera])
    io = FakeIO()
    sensor, color, depth = sensor_with_fakes(model, [camera], io)
    hinge = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hinge_a")]

    poses = []
    for angle in (0.0, 0.9):
        pose = np.zeros(model.nq)
        pose[hinge] = angle
        render_due(sensor, model, color, depth, pose=pose)
        clock.set(1.0 / _FPS)
        poses.append(angle)

    renderer = color[(camera.height, camera.width)]
    assert [p[hinge] for p in renderer.poses] == poses
    # The camera hangs off the body the hinge moves, so its world frame moved.
    first, second = renderer.camera_positions
    assert not np.allclose(first, second)


def test_the_snapshot_copies_the_pose_it_is_given(scene):
    """The physics thread hands over its live qpos view, so the snapshot has
    to copy: aliasing would let the solver mutate what the renderer reads."""
    snapshot = _PoseSnapshot(2)
    live = np.array([1.0, 2.0])
    snapshot.write(live, 1.0)
    live[0] = 99.0

    out = np.zeros(2)
    snapshot.read_into(out)
    assert out.tolist() == [1.0, 2.0]


def test_camera_placement_reaches_the_compiled_model(scene):
    """A quaternion read in the wrong order still compiles and still points a
    camera somewhere, so the pose has to be asserted, not just the parent."""
    camera = dataclasses.replace(
        color_camera(), quat_wxyz=(0.6761, -0.2067, -0.0493, -0.7054), fovy_deg=66.0
    )
    model = compile_model_with_cameras(scene, [camera])
    index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera.name)

    # MuJoCo normalizes what it is given, which reorders nothing: a wxyz read
    # as xyzw still fails here.
    expected = np.array(camera.quat_wxyz) / np.linalg.norm(camera.quat_wxyz)
    assert model.cam_pos[index].tolist() == pytest.approx(list(camera.pos))
    assert model.cam_quat[index].tolist() == pytest.approx(expected.tolist())
    assert model.cam_fovy[index] == pytest.approx(camera.fovy_deg)


def test_the_camera_light_rig_reaches_the_compiled_model(scene):
    model = compile_model_with_cameras(scene, [color_camera()])

    assert model.nlight == len(camera_sensor._LIGHT_DIRECTIONS)
    assert not model.light_castshadow.any()
    assert model.vis.headlight.ambient[0] == pytest.approx(
        camera_sensor._HEADLIGHT_AMBIENT
    )


def test_an_rgbd_camera_advertises_its_depth_stream(scene, clock):
    camera = rgbd_camera()
    model = compile_model_with_cameras(scene, [camera])
    io = FakeIO()
    sensor, color, depth = sensor_with_fakes(model, [camera], io)

    render_due(sensor, model, color, depth, pose=np.zeros(model.nq))

    assert io.infos == [
        ("chest", _COLOR[0], _COLOR[1], _FPS, "rgb8",
         _DEPTH[0], _DEPTH[1], "z16", 0.001)
    ]


def test_renderers_open_at_each_streams_own_size(scene, monkeypatch):
    """Renderer takes (height, width); a transposed pair renders every
    non-square stream sideways and nothing downstream would notice."""
    cameras = [color_camera(), rgbd_camera()]
    model = compile_model_with_cameras(scene, cameras)
    sensor = MujocoCameraSensor(model, cameras, FakeIO())
    opened = []

    def fake_renderer(_model, height, width):
        made = FakeRenderer(height, width)
        opened.append(made)
        return made

    monkeypatch.setattr(mujoco, "Renderer", fake_renderer)
    color, depth = sensor._create_renderers([])  # pylint: disable=W0212

    assert sorted((r._height, r._width) for r in opened) == [  # pylint: disable=W0212
        (_DEPTH[1], _DEPTH[0]),
        (_COLOR[1], _COLOR[0]),
    ]
    # Only the depth renderer is switched into depth mode.
    assert depth[(_DEPTH[1], _DEPTH[0])].renders_depth
    assert not color[(_COLOR[1], _COLOR[0])].renders_depth


def test_a_renderer_opened_before_a_failure_is_still_closed(scene, monkeypatch):
    cameras = [color_camera(), rgbd_camera()]
    model = compile_model_with_cameras(scene, cameras)
    sensor = MujocoCameraSensor(model, cameras, FakeIO())
    opened = []

    def fail_on_the_second(_model, height, width):
        if len(opened) == 1:
            raise RuntimeError("no GL context")
        made = FakeRenderer(height, width)
        opened.append(made)
        return made

    monkeypatch.setattr(mujoco, "Renderer", fail_on_the_second)
    sensor.start()
    sensor.stop()

    assert len(opened) == 1 and opened[0].closed
    with pytest.raises(RuntimeError, match="camera render thread failed"):
        sensor.raise_if_failed()


def test_background_depth_publishes_as_invalid(scene, clock):
    """MuJoCo returns the far clipping distance for empty space, which is
    past any configured range and must reach the wire as the invalid marker."""
    camera = rgbd_camera()
    model = compile_model_with_cameras(scene, [camera])
    io = FakeIO()
    far_plane = camera.depth.max_range_m * 5.0
    sensor, color, depth = sensor_with_fakes(model, [camera], io, depth_m=far_plane)

    render_due(sensor, model, color, depth, pose=np.zeros(model.nq))

    payload = io.rgbd[0][4][3]
    assert set(struct.unpack(f"<{_DEPTH[0] * _DEPTH[1]}H", payload)) == {0}


def test_a_wedged_renderer_surfaces_on_the_physics_thread(scene, monkeypatch):
    """A renderer that blocks raises nothing, so only the heartbeat catches it."""
    model = compile_model_with_cameras(scene, [color_camera()])
    sensor = MujocoCameraSensor(model, [color_camera()], FakeIO())
    wedged = threading.Event()
    release = threading.Event()

    class WedgedRenderer(FakeRenderer):
        def render(self):
            wedged.set()
            release.wait(timeout=10.0)
            return super().render()

    monkeypatch.setattr(mujoco, "Renderer", lambda _model, h, w: WedgedRenderer(h, w))
    # Patch the bound the sensor reads and sleep that same value: sleeping the
    # module's real 5s bound would pass while making the patch pointless.
    monkeypatch.setattr(camera_sensor, "_HEARTBEAT_TIMEOUT_S", _PATCHED_HEARTBEAT_S)
    try:
        sensor.snapshot(np.zeros(model.nq))
        sensor.start()
        assert wedged.wait(timeout=5.0)
        time.sleep(_PATCHED_HEARTBEAT_S * 2)
        with pytest.raises(RuntimeError, match="stopped responding"):
            sensor.raise_if_failed()
    finally:
        release.set()
        sensor.stop()


def test_a_dead_renderer_surfaces_on_the_physics_thread(scene, monkeypatch):
    model = compile_model_with_cameras(scene, [color_camera()])
    sensor = MujocoCameraSensor(model, [color_camera()], FakeIO())

    def explode(*_args, **_kwargs):
        raise RuntimeError("no GL context")

    monkeypatch.setattr(mujoco, "Renderer", explode)
    sensor.raise_if_failed()
    sensor.start()
    sensor.stop()
    with pytest.raises(RuntimeError, match="camera render thread failed"):
        sensor.raise_if_failed()


def test_stop_closes_every_renderer_it_opened(scene, monkeypatch):
    cameras = [color_camera(), rgbd_camera()]
    model = compile_model_with_cameras(scene, cameras)
    sensor = MujocoCameraSensor(model, cameras, FakeIO())
    opened = []
    running = threading.Event()

    def fake_renderer(_model, height, width):
        renderer = FakeRenderer(height, width)
        opened.append(renderer)
        if len(opened) == 2:
            running.set()
        return renderer

    monkeypatch.setattr(mujoco, "Renderer", fake_renderer)
    sensor.start()
    assert running.wait(timeout=5.0)
    sensor.stop()

    assert len(opened) == 2
    assert all(renderer.closed for renderer in opened)
    sensor.raise_if_failed()


def test_stop_clears_the_heartbeat_so_shutdown_is_not_a_wedged_renderer(
    scene, monkeypatch
):
    """A stopped thread stops beating on purpose. If stop() left the last beat
    behind, any raise_if_failed() more than the bound after a deliberate stop
    would report a wedged renderer, and the physics loop calls it on every
    step, so an orderly shutdown would surface as a renderer fault."""
    model = compile_model_with_cameras(scene, [color_camera()])
    monkeypatch.setattr(mujoco, "Renderer", lambda _model, h, w: FakeRenderer(h, w))
    monkeypatch.setattr(camera_sensor, "_HEARTBEAT_TIMEOUT_S", _PATCHED_HEARTBEAT_S)
    sensor = MujocoCameraSensor(model, [color_camera()], FakeIO())
    sensor.snapshot(np.zeros(model.nq))
    sensor.start()
    sensor.stop()

    # Well past the bound a beating thread would have to meet.
    time.sleep(_PATCHED_HEARTBEAT_S * 2)
    sensor.raise_if_failed()
