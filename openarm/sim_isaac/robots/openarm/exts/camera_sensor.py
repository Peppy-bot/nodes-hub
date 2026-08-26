from __future__ import annotations

import enum
import logging
import math
import time

import numpy as np

from camera_common import (
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


class CaptureOutcome(enum.Enum):
    """What one capture attempt produced. A frame the renderer never made and a
    frame the transport dropped are different faults: the first means the
    renderer and the camera config disagree and no retry fixes it, the second
    means a consumer is behind and the next frame will very likely land."""

    NO_FRAME = enum.auto()
    DROPPED = enum.auto()
    DELIVERED = enum.auto()


def _delivery(published: bool) -> CaptureOutcome:
    """A publish that returns False rendered fine and was dropped in flight."""
    return CaptureOutcome.DELIVERED if published else CaptureOutcome.DROPPED


_STREAM_INFO_HZ = 1
# Any consistent film-back size works; FOV is the invariant and focal length is
# derived from it.
_VERTICAL_APERTURE_MM = 15.0
_CLIPPING_RANGE_M = (0.02, 100.0)
# Consecutive failed reads (empty or mis-shaped annotator data) tolerated per
# camera before the renderer and config are declared incompatible; first
# frames legitimately take a few updates to appear.
_MAX_CAPTURE_FAILURES = 300


class IsaacCameraSensor:
    """Renders the configured robot-mounted cameras through Replicator render
    products and publishes their frames.

    Setup is deferred like every other ext: camera prims and render products
    can only be created once the USD stage is live. Between captures each
    color-only camera's hydra texture updates are paused so the RTX renderer
    only pays for camera frames at camera rate, not at physics rate; a
    capture enables updates, lets one update render, and reads the annotator
    on the update after that, the first on which it exposes the new frame.
    Each capture therefore spends two updates, so the effective frame rate is
    bounded by half the app update rate; deadlines that fire mid-capture are
    absorbed by the pacer's resync. An rgbd camera renders one product at
    color resolution with both annotators attached, and its depth is
    subsampled to the depth wire format at capture, like the real rig's
    half-resolution depth.
    """

    def __init__(self, root_prim: str, cameras: list[CameraConfig], io) -> None:
        self._root = root_prim
        self._cameras = {camera.name: camera for camera in cameras}
        self._io = io
        self._ready = False
        self._render_products: dict[tuple[str, str], object] = {}
        self._annotators: dict[tuple[str, str], object] = {}
        self._throttled = False
        self._armed: set[str] = set()
        self._warming: set[str] = set()
        self._frame_pacers = {c.name: FramePacer(c.fps) for c in cameras}
        self._info_pacers = {c.name: FramePacer(_STREAM_INFO_HZ) for c in cameras}
        self._frame_ids = {c.name: FrameIdCounter() for c in cameras}
        self._capture_failures = {c.name: 0 for c in cameras}
        self._dropping = False

    def setup(self) -> bool:
        """Create camera prims, render products and annotators against the live
        stage. False while the stage or the robot is not loaded yet; a
        parent link missing from a loaded robot raises, because that is a
        config error no retry can fix."""
        if self._ready:
            return True
        try:
            import omni.usd  # pylint: disable=E0401
        except ImportError as exc:
            logger.error(f"omni.usd unavailable: {exc}")
            return False
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return False
        root = stage.GetPrimAtPath(self._root)
        if not root or not root.IsValid():
            return False

        import omni.replicator.core as rep  # pylint: disable=E0401

        for camera in self._cameras.values():
            camera_path = self._define_camera_prim(stage, root, camera)
            render_product = rep.create.render_product(
                camera_path, (camera.width, camera.height)
            )
            color_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            color_annotator.attach(render_product)
            self._render_products[(camera.name, "color")] = render_product
            self._annotators[(camera.name, "color")] = color_annotator
            if camera.depth is not None:
                # Depth rides the same render product at color resolution and
                # is subsampled to the depth wire format at capture; a second
                # render product on the same prim freezes when its updates
                # are cycled.
                depth_annotator = rep.AnnotatorRegistry.get_annotator(
                    "distance_to_image_plane"
                )
                depth_annotator.attach(render_product)
                self._annotators[(camera.name, "depth")] = depth_annotator

        # Evaluate every product (no short-circuit) so none is left paused if
        # support were ever mixed across products.
        paused = [
            self._set_updates_enabled(rp, False)
            for rp in self._render_products.values()
        ]
        self._throttled = all(paused)
        if not self._throttled:
            for rp in self._render_products.values():
                self._set_updates_enabled(rp, True)
            logger.warning(
                "render product throttling unavailable; cameras render every update"
            )
        self._ready = True
        logger.info(
            "IsaacCameraSensor ready: "
            + ", ".join(
                f"{c.name} {c.width}x{c.height}@{c.fps}" for c in self._cameras.values()
            )
            + (" (throttled)" if self._throttled else "")
        )
        return True

    def _define_camera_prim(self, stage, root, camera: CameraConfig) -> str:
        from pxr import Gf, Usd, UsdGeom  # pylint: disable=E0401

        parent = next(
            (prim for prim in Usd.PrimRange(root) if prim.GetName() == camera.parent_link),
            None,
        )
        if parent is None:
            names = sorted(prim.GetName() for prim in Usd.PrimRange(root))
            raise RuntimeError(
                f"camera '{camera.name}' parent link '{camera.parent_link}' is not"
                f" under {self._root}; prims: {names}"
            )

        camera_path = parent.GetPath().AppendChild(camera.name)
        usd_camera = UsdGeom.Camera.Define(stage, camera_path)
        xform = UsdGeom.Xformable(usd_camera.GetPrim())
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(*camera.pos))
        w, x, y, z = camera.quat_wxyz
        xform.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
        focal_mm = (_VERTICAL_APERTURE_MM / 2.0) / math.tan(
            math.radians(camera.fovy_deg) / 2.0
        )
        usd_camera.CreateFocalLengthAttr(focal_mm)
        usd_camera.CreateVerticalApertureAttr(_VERTICAL_APERTURE_MM)
        usd_camera.CreateHorizontalApertureAttr(
            _VERTICAL_APERTURE_MM * camera.width / camera.height
        )
        usd_camera.CreateClippingRangeAttr(Gf.Vec2f(*_CLIPPING_RANGE_M))
        return str(camera_path)

    @staticmethod
    def _set_updates_enabled(render_product, enabled: bool) -> bool:
        hydra_texture = getattr(render_product, "hydra_texture", None)
        setter = getattr(hydra_texture, "set_updates_enabled", None)
        if setter is None:
            return False
        setter(enabled)
        return True

    def teardown(self) -> None:
        for annotator in self._annotators.values():
            try:
                annotator.detach()
            except Exception as exc:
                logger.warning(f"annotator detach failed: {exc}")
        for render_product in self._render_products.values():
            try:
                render_product.destroy()
            except Exception as exc:
                logger.warning(f"render product destroy failed: {exc}")
        self._annotators.clear()
        self._render_products.clear()
        self._armed.clear()
        self._warming.clear()
        self._ready = False

    def step(self) -> None:
        """Runs on the main thread right after sim_app.update(): read the
        cameras armed earlier, promote warming ones, then arm those due."""
        now = time.monotonic()
        for name, pacer in self._info_pacers.items():
            if pacer.take_if_due(now):
                self._publish_stream_info(self._cameras[name])

        # An armed camera stays armed (render product updates on) until a
        # capture actually lands: the annotator yields empty data until the
        # pipeline's first frame, and can keep yielding it transiently.
        for name in sorted(self._armed):
            if not self._tracked_capture(name):
                continue
            self._set_camera_updates(name, False)
            self._armed.discard(name)

        # A camera enabled last step rendered during this step's update; the
        # annotator exposes that frame on the next update, so warming cameras
        # become readable one step after arming, never on the arming step
        # (reading earlier returns the previous cycle's pixels).
        self._armed |= self._warming
        self._warming.clear()

        for name, pacer in self._frame_pacers.items():
            if not pacer.take_if_due(now):
                continue
            if not self._throttled:
                # Unthrottled render products render every update, so the
                # frame is already current.
                self._tracked_capture(name)
                continue
            if name in self._armed or name in self._warming:
                continue
            self._set_camera_updates(name, True)
            self._warming.add(name)

    def _set_camera_updates(self, name: str, enabled: bool) -> None:
        for kind in ("color", "depth"):
            render_product = self._render_products.get((name, kind))
            if render_product is not None:
                self._set_updates_enabled(render_product, enabled)

    def _tracked_capture(self, name: str) -> bool:
        """Capture, and report whether the renderer produced a frame. The
        consecutive-failure bound counts only missing renderer output, because
        that is the fault no retry fixes; a transport drop means the consumer is
        behind, which is the MuJoCo twin's reading of the same signal. Returns
        True for a rendered frame whether or not the publish landed, so a run of
        drops cannot hold the camera armed and lose its rate throttling."""
        outcome = self._capture(self._cameras[name])
        if outcome is CaptureOutcome.NO_FRAME:
            self._capture_failures[name] += 1
            if self._capture_failures[name] >= _MAX_CAPTURE_FAILURES:
                raise RuntimeError(
                    f"{name}: no usable annotator frame in {_MAX_CAPTURE_FAILURES} "
                    "consecutive reads; renderer output and camera config disagree"
                )
            return False
        self._capture_failures[name] = 0
        if outcome is CaptureOutcome.DROPPED and not self._dropping:
            self._dropping = True
            logger.warning(
                "a rendered frame was dropped before the wire because the "
                "previous publish is still in flight, suppressing repeats"
            )
        elif outcome is CaptureOutcome.DELIVERED:
            self._dropping = False
        return True

    def _capture(self, camera: CameraConfig) -> CaptureOutcome:
        """Publish the camera's current annotator data."""
        rgba = self._annotators[(camera.name, "color")].get_data()
        rgb = self._as_rgb(rgba, camera)
        if rgb is None:
            return CaptureOutcome.NO_FRAME
        timestamp_s = self._io.camera_timestamp_s()
        frame_id = self._frame_ids[camera.name].next()
        if camera.depth is None:
            return _delivery(
                self._io.publish_color_frame(
                    camera.name,
                    timestamp_s,
                    frame_id,
                    COLOR_ENCODING,
                    camera.width,
                    camera.height,
                    rgb.tobytes(),
                )
            )
        depth_m = self._annotators[(camera.name, "depth")].get_data()
        expected = (camera.height, camera.width)
        if not isinstance(depth_m, np.ndarray) or depth_m.shape != expected:
            shape = getattr(depth_m, "shape", None)
            logger.warning(f"{camera.name}: depth annotator shape {shape} != {expected}")
            return CaptureOutcome.NO_FRAME
        depth_m = depth_m[:: camera.height // camera.depth.height, :: camera.width // camera.depth.width]
        return _delivery(
            self._io.publish_rgbd_frames(
                camera.name,
                timestamp_s,
                frame_id,
                ALIGN_MODE,
                (COLOR_ENCODING, camera.width, camera.height, rgb.tobytes()),
                (
                    DEPTH_ENCODING,
                    camera.depth.width,
                    camera.depth.height,
                    depth_to_z16(depth_m.astype(np.float32, copy=False), camera.depth),
                ),
            )
        )

    @staticmethod
    def _as_rgb(rgba, camera: CameraConfig):
        """The rgb annotator yields HxWx4 uint8; strip alpha to a contiguous
        rgb8 buffer. None for an empty frame, which the annotator yields until
        a render lands."""
        if not isinstance(rgba, np.ndarray) or rgba.size == 0:
            return None
        if rgba.shape != (camera.height, camera.width, 4) or rgba.dtype != np.uint8:
            logger.warning(
                f"{camera.name}: rgb annotator shape {rgba.shape} dtype {rgba.dtype}"
                f" != ({camera.height}, {camera.width}, 4) uint8"
            )
            return None
        return np.ascontiguousarray(rgba[..., :3])

    def _publish_stream_info(self, camera: CameraConfig) -> None:
        if camera.depth is None:
            self._io.publish_color_stream_info(
                camera.name, camera.width, camera.height, camera.fps, COLOR_ENCODING
            )
        else:
            self._io.publish_rgbd_stream_info(
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

    @property
    def is_ready(self) -> bool:
        return self._ready
