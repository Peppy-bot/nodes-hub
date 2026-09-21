from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from camera_geometry import (
    DEPTH_MODEL,
    DEPTH_TO_COLOR_ORIENTATION,
    DEPTH_TO_COLOR_POSITION,
    pinhole,
)
from sim_robot_core.cameras import (
    ALIGN_MODE,
    COLOR_ENCODING,
    DEPTH_ENCODING,
    DEPTH_UNIT_M_PER_LSB,
    CameraConfig,
    FrameIdCounter,
    FramePacer,
    depth_to_z16,
)

logger = logging.getLogger(__name__)

_STREAM_INFO_HZ = 1

# A scene that ships no lights, around a robot whose meshes are matte black,
# renders nearly unlit under the default camera-mounted headlight alone. For a
# model whose entry asks for it, enabling cameras adds three equal directional
# lights and dims the headlight to a fill, so every camera sees one lit scene
# rather than its own travelling highlight.
# The rig is on the model, so the operator's viewer shows the same lighting the
# cameras record. None of the lights casts a shadow: the scene is the robot
# alone with no floor to catch one, and shadow maps measured a third of the
# render budget on this scene for no visible difference, which is the
# difference between keeping and missing the frame rate on a machine whose
# EGL lands on a software rasterizer.
_HEADLIGHT_AMBIENT = 0.35
_HEADLIGHT_DIFFUSE = 0.3
_HEADLIGHT_SPECULAR = 0.1
_LIGHT_POS = (0.0, 0.0, 3.0)
_LIGHT_DIRECTIONS = ((0.0, 0.0, -1.0), (1.0, -0.4, -0.7), (-1.0, 0.4, -0.7))
_LIGHT_DIFFUSE = 0.5
_LIGHT_SPECULAR = 0.1

# Idle period of the render thread between due checks: short enough that a 15
# fps deadline is met within a frame's tolerance, long enough to cost nothing.
_POLL_PERIOD_S = 0.002
# Grace for the render thread to finish its current frame and close its GL
# context on shutdown; one full capture cycle is milliseconds.
_JOIN_TIMEOUT_S = 5.0
# A capture this many periods after the previous one is late: the machine could
# not render at the configured rate. Consumers age frames against their own
# staleness bounds (the recorder ends an episode over its), so the engine says
# when it is the one falling behind instead of leaving a camera to look silent.
_LATE_CAPTURE_PERIODS = 3.0
_LATE_REPORT_PERIOD_S = 1.0
# The render thread touches its heartbeat every poll, so silence this long is a
# thread wedged inside a GL call rather than a slow one: no exception is ever
# raised there, and without this the engine would keep publishing joint states
# beside a video track that stopped.
_HEARTBEAT_TIMEOUT_S = 5.0


def add_cameras(spec, known, xml_path: Path) -> None:
    """Add to one model's spec a camera per camera of its entry, on the body
    its entry hangs it from. The robot is attached into the scene under its
    own prefix, so these are the cameras of that robot alone.

    The baked MJCF is never modified: the cameras go on an mjSpec parsed from
    it, so a camera-less launch compiles the scene without them.
    """
    cameras = list(known.entry.cameras)
    taken = {camera.name for camera in spec.cameras}
    for camera in cameras:
        if camera.name in taken:
            raise RuntimeError(
                f"camera '{camera.name}' collides with a camera the scene "
                f"{xml_path.name} already defines"
            )
        mjcf_camera = _parent_body(spec, known, camera, xml_path).add_camera()
        mjcf_camera.name = camera.name
        mjcf_camera.pos = list(camera.pos)
        mjcf_camera.quat = list(camera.quat_wxyz)
        mjcf_camera.fovy = camera.fovy_deg

    logger.info(
        f"{known.model} carries cameras: "
        + ", ".join(f"{c.name}@{c.parent_link}" for c in cameras)
    )


def widen_offscreen(spec, cameras) -> None:
    """Widen the scene's offscreen framebuffer to cover every render size it
    holds, colour and depth. The scene's own values stand where they are
    already larger, and a scene rendering nothing keeps them."""
    sizes = list(cameras)
    if not sizes:
        return
    global_ = spec.visual.global_
    # Depth never exceeds color: the config parser rejects a depth grid that is
    # not an integer decimation of it.
    global_.offwidth = max([global_.offwidth, *(camera.width for camera in sizes)])
    global_.offheight = max([global_.offheight, *(camera.height for camera in sizes)])
    logger.info(f"offscreen buffer {global_.offwidth}x{global_.offheight}")


def add_light_rig(spec) -> None:
    import mujoco  # pylint: disable=C0415

    headlight = spec.visual.headlight
    headlight.ambient = [_HEADLIGHT_AMBIENT] * 3
    headlight.diffuse = [_HEADLIGHT_DIFFUSE] * 3
    headlight.specular = [_HEADLIGHT_SPECULAR] * 3
    for index, direction in enumerate(_LIGHT_DIRECTIONS):
        light = spec.worldbody.add_light()
        light.name = f"camera_light_{index}"
        light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
        light.pos = list(_LIGHT_POS)
        light.dir = list(direction)
        light.diffuse = [_LIGHT_DIFFUSE] * 3
        light.specular = [_LIGHT_SPECULAR] * 3
        light.castshadow = False


def _parent_body(spec, known, camera: CameraConfig, xml_path: Path):
    """The body a camera hangs from: the one its parent link is in this
    model's MJCF, or the world body for a link the compiler folds into it.
    Any other missing parent is an error of the model's entry."""
    name = known.body_of(camera.parent_link)
    body = spec.body(name)
    if body is not None:
        return body
    if camera.parent_link in known.world_links:
        return spec.worldbody
    raise RuntimeError(
        f"camera '{camera.name}' parent link '{camera.parent_link}' is not a "
        f"body of {xml_path.name}; bodies: {sorted(b.name for b in spec.bodies)}"
    )


class _PoseSnapshot:
    """Latest scene pose, written by the physics thread and read by the render
    thread: the joint positions and the time they were stepped at. Copying
    under the lock is microseconds at this model size, and it keeps the
    renderer off the live MjData the solver is stepping."""

    def __init__(self, size: int) -> None:
        self._lock = threading.Lock()
        self._qpos = np.zeros(size)
        self._timestamp_s: Optional[float] = None

    def write(self, qpos: np.ndarray, timestamp_s: float) -> None:
        with self._lock:
            self._qpos[:] = qpos
            self._timestamp_s = timestamp_s

    def read_into(self, out: np.ndarray) -> Optional[float]:
        """The time of the pose copied into out, or None before the physics
        thread has published one, so the first frame shows the running scene
        rather than the model's zero pose."""
        with self._lock:
            if self._timestamp_s is not None:
                out[:] = self._qpos
            return self._timestamp_s


@dataclass(frozen=True)
class Rig:
    """The cameras one standing robot renders: the robot they publish for,
    and the prefix its names carry in the composed scene."""

    robot: str
    prefix: str
    cameras: tuple[CameraConfig, ...]


class _CameraStream:
    """One camera's schedules and identity: the robot it renders for and the
    camera it is in the scene, when its next frame and next description are
    due, its capture counter, and when it last put a frame on the wire."""

    def __init__(
        self, config: CameraConfig, robot: str, scene_name: str, frame_ids=None
    ) -> None:
        self.config = config
        # The robot whose camera pair carries these frames.
        self.robot = robot
        # What this camera is called in the composed scene, which is the
        # robot's prefix and the camera's own name.
        self.scene_name = scene_name
        self.frames = FramePacer(config.fps)
        self.info = FramePacer(_STREAM_INFO_HZ)
        self.frame_ids = frame_ids or FrameIdCounter()
        self.last_delivery_s: Optional[float] = None

    def gap_s(self, now: float) -> Optional[float]:
        """How long this camera has gone without delivering, if that is longer
        than its own frame rate allows."""
        if self.last_delivery_s is None:
            return None
        gap = now - self.last_delivery_s
        return gap if gap > _LATE_CAPTURE_PERIODS / self.config.fps else None


class MujocoCameraSensor:
    """Renders the configured robot-mounted cameras off the physics thread and
    publishes their frames.

    MuJoCo renders on demand rather than on a pipeline clock, so a private
    thread owns the GL context and every Renderer (EGL contexts are bound to
    the thread that creates them) and paces itself against each camera's fps.
    It renders a private MjData carrying the latest pose the physics thread
    published, which keeps rendering off the data the solver is stepping and
    lets a slow frame fall behind instead of stalling physics.

    Frames carry the time of the pose they show, not the time their render
    finished, so a whole capture cycle shares one timestamp and a consumer
    aligning video against joint states reads both off the same instant.

    An rgbd camera renders depth from the same camera at the depth stream's own
    resolution: one pinhole, one vertical FOV, so color and depth are aligned by
    construction with no resampling.
    """

    def __init__(self, model, rigs: "list[Rig]", io, counters=None) -> None:
        self._model = model
        # Every camera of every robot standing, keyed by the robot it belongs
        # to and the camera it is: two robots of one model carry the same
        # camera names, and each publishes on its own pair. A camera that was
        # already streaming keeps counting its frames from where it was, so
        # what a consumer pairs a frame by runs on across a scene composed
        # again around it.
        kept = counters or {}
        self._streams = {
            (rig.robot, camera.name): _CameraStream(
                camera,
                rig.robot,
                f"{rig.prefix}{camera.name}",
                kept.get((rig.robot, camera.name)),
            )
            for rig in rigs
            for camera in rig.cameras
        }
        self._io = io
        self._pose = _PoseSnapshot(model.nq)
        self._late_deliveries = 0
        self._worst_late: Optional[tuple[float, str]] = None
        self._next_late_report_s: Optional[float] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # Whether a render thread is still running after its stop gave up
        # waiting for it, which is what the frame counters may not outlive.
        self._outlived_its_stop = False
        self._failure: Optional[Exception] = None
        self._heartbeat_s: Optional[float] = None

    def start(self) -> None:
        if self._thread is not None or self._stop.is_set():
            raise RuntimeError("MujocoCameraSensor cannot be started twice")
        self._heartbeat_s = time.monotonic()
        self._thread = threading.Thread(
            target=self._render_loop, name="camera-render", daemon=True
        )
        self._thread.start()
        logger.info(
            "MujocoCameraSensor started: "
            + ", ".join(
                f"{s.robot}/{s.config.name} {s.config.width}x{s.config.height}@{s.config.fps}"
                for s in self._streams.values()
            )
        )

    def counters(self) -> dict:
        """Each camera's frame counter, for the scene composed next. A render
        thread that outlived its stop is still counting on these, so a scene
        it may still be rendering hands none of them on and the cameras of
        the next one count from zero."""
        if self._outlived_its_stop:
            return {}
        return {key: stream.frame_ids for key, stream in self._streams.items()}

    def snapshot(self, qpos: np.ndarray) -> None:
        """Publish the just-stepped pose, and the time it holds for, to the
        render thread. Called every physics tick; rendering picks up whichever
        pose is current when a camera's frame is due."""
        self._pose.write(qpos, self._io.timestamp_s())

    def raise_if_failed(self) -> None:
        """Re-raise a render-thread failure on the physics thread, whose
        exceptions reach the node. A dead renderer must take the engine down:
        the alternative is a recording session whose video silently stops. A
        wedged renderer raises nothing at all, so silence past the heartbeat
        bound counts as death too. A sensor that was never started has no
        thread to be wedged."""
        if self._failure is not None:
            raise RuntimeError("camera render thread failed") from self._failure
        if self._heartbeat_s is None:
            return
        silence_s = time.monotonic() - self._heartbeat_s
        if silence_s > _HEARTBEAT_TIMEOUT_S:
            raise RuntimeError(
                f"camera render thread stopped responding {silence_s:.1f}s ago"
            )

    def stop(self) -> None:
        self._stop.set()
        if self._thread is None:
            return
        self._thread.join(timeout=_JOIN_TIMEOUT_S)
        if self._thread.is_alive():
            self._outlived_its_stop = True
            logger.warning("camera render thread did not exit within the join timeout")
        self._thread = None
        # A stopped thread stops beating on purpose. Leaving the last beat in
        # place would let a slow shutdown age past the bound and surface a
        # deliberate stop as a wedged renderer; raise_if_failed reads None as
        # "no thread to be wedged".
        self._heartbeat_s = None

    def _render_loop(self) -> None:
        import mujoco  # pylint: disable=C0415

        # Every renderer opened, in open order, so the close path is total
        # even when creation itself fails partway through.
        opened: list = []
        try:
            color_renderers, depth_renderers = self._create_renderers(opened)
            data = mujoco.MjData(self._model)
            qpos = np.zeros(self._model.nq)
            while not self._stop.is_set():
                self._heartbeat_s = time.monotonic()
                self._render_due(data, qpos, color_renderers, depth_renderers)
                self._stop.wait(_POLL_PERIOD_S)
        except Exception as exc:  # pylint: disable=W0718
            logger.exception("camera render thread failed")
            self._failure = exc
        finally:
            self._close(opened)

    @staticmethod
    def _close(renderers: list) -> None:
        """Closing a GL context can itself raise, so one bad context must not
        strand the others or replace the failure that got us here."""
        for renderer in renderers:
            try:
                renderer.close()
            except Exception as exc:  # pylint: disable=W0718
                logger.warning(f"closing a renderer failed: {exc}")

    def _create_renderers(self, opened: list) -> tuple[dict, dict]:
        """One Renderer per distinct render size, created here because an EGL
        context belongs to the thread that made it, and appended to opened as
        it is made so the caller can close what exists."""
        import mujoco  # pylint: disable=C0415

        def renderer(height: int, width: int):
            made = mujoco.Renderer(self._model, height, width)
            opened.append(made)
            return made

        color: dict[tuple[int, int], object] = {}
        depth: dict[tuple[int, int], object] = {}
        for camera in (stream.config for stream in self._streams.values()):
            size = (camera.height, camera.width)
            if size not in color:
                color[size] = renderer(*size)
            if camera.depth is None:
                continue
            size = (camera.depth.height, camera.depth.width)
            if size not in depth:
                depth[size] = renderer(*size)
                depth[size].enable_depth_rendering()
        return color, depth

    def _render_due(self, data, qpos, color_renderers, depth_renderers) -> None:
        import mujoco  # pylint: disable=C0415

        now = time.monotonic()
        self._report_late_deliveries(now)
        for stream in self._streams.values():
            if stream.info.take_if_due(now):
                self._publish_stream_info(stream)
                self._publish_geometry(stream)

        due = [s for s in self._streams.values() if s.frames.take_if_due(now)]
        if not due:
            return
        timestamp_s = self._pose.read_into(qpos)
        if timestamp_s is None:
            return
        data.qpos[:] = qpos
        # Everything a render reads off the pose: body and geom frames, then
        # the camera and light frames that hang off them. The solver's work is
        # the physics thread's and none of it reaches the pixels.
        mujoco.mj_kinematics(self._model, data)
        mujoco.mj_camlight(self._model, data)
        for stream in due:
            delivered = self._capture(
                stream, timestamp_s, data, color_renderers, depth_renderers
            )
            if not delivered:
                continue
            self._note_gap(stream, now)
            stream.last_delivery_s = now

    def _note_gap(self, stream: _CameraStream, now: float) -> None:
        gap = stream.gap_s(now)
        if gap is None:
            return
        self._late_deliveries += 1
        if self._worst_late is None or gap > self._worst_late[0]:
            self._worst_late = (gap, f"{stream.robot}/{stream.config.name}")

    def _report_late_deliveries(self, now: float) -> None:
        """Aggregate per second: a machine that cannot keep up would otherwise
        log at frame rate. A camera still in its gap is counted here rather
        than on recovery, so an ongoing stall reports every second instead of
        going quiet until it ends."""
        if self._next_late_report_s is None:
            self._next_late_report_s = now + _LATE_REPORT_PERIOD_S
            return
        if now < self._next_late_report_s:
            return
        self._next_late_report_s = now + _LATE_REPORT_PERIOD_S
        for stream in self._streams.values():
            self._note_gap(stream, now)
        if self._worst_late is None:
            return
        gap, name = self._worst_late
        logger.warning(
            f"{self._late_deliveries} late frame interval(s): "
            f"worst {gap * 1000:.0f} ms on {name}"
        )
        self._late_deliveries = 0
        self._worst_late = None

    def _capture(
        self,
        stream: _CameraStream,
        timestamp_s: float,
        data,
        color_renderers,
        depth_renderers,
    ) -> bool:
        """Render one camera and put it on the wire. False when the publish was
        dropped because the previous one is still in flight: the consumer never
        sees that frame, so it counts against the camera's delivery rate."""
        camera = stream.config
        renderer = color_renderers[(camera.height, camera.width)]
        renderer.update_scene(data, camera=stream.scene_name)
        rgb = renderer.render()
        frame_id = stream.frame_ids.next()
        if camera.depth is None:
            return self._io.publish_color_frame(
                stream.robot,
                camera.name,
                timestamp_s,
                frame_id,
                COLOR_ENCODING,
                camera.width,
                camera.height,
                rgb.tobytes(),
            )
        depth_renderer = depth_renderers[(camera.depth.height, camera.depth.width)]
        depth_renderer.update_scene(data, camera=stream.scene_name)
        depth_m = depth_renderer.render()
        return self._io.publish_rgbd_frames(
            stream.robot,
            camera.name,
            timestamp_s,
            frame_id,
            ALIGN_MODE,
            (COLOR_ENCODING, camera.width, camera.height, rgb.tobytes()),
            (
                DEPTH_ENCODING,
                camera.depth.width,
                camera.depth.height,
                depth_to_z16(depth_m, camera.depth),
            ),
        )

    def _publish_geometry(self, stream: _CameraStream) -> None:
        """Where this camera's pixels point, in the camera_geometry contract's
        terms. Depth renders from the
        same camera at the depth stream's own size, so its grid is a pinhole of
        the same field of view, centred like the colour one."""
        camera = stream.config
        color = pinhole(camera.fovy_deg, camera.width, camera.height)
        if camera.depth is None:
            self._io.publish_color_geometry(stream.robot, camera.name, color)
            return
        self._io.publish_rgbd_geometry(
            stream.robot,
            camera.name,
            color,
            pinhole(camera.fovy_deg, camera.depth.width, camera.depth.height),
            DEPTH_MODEL,
            camera.depth.min_depth_m,
            camera.depth.max_range_m,
            ALIGN_MODE,
            DEPTH_TO_COLOR_POSITION,
            DEPTH_TO_COLOR_ORIENTATION,
        )

    def _publish_stream_info(self, stream: _CameraStream) -> None:
        camera = stream.config
        if camera.depth is None:
            self._io.publish_color_stream_info(
                stream.robot, camera.name, camera.width, camera.height, camera.fps, COLOR_ENCODING
            )
            return
        self._io.publish_rgbd_stream_info(
            stream.robot,
            camera.name,
            camera.width,
            camera.height,
            camera.fps,
            COLOR_ENCODING,
            camera.depth.width,
            camera.depth.height,
            DEPTH_ENCODING,
            DEPTH_UNIT_M_PER_LSB,
        )
