"""Thin wrapper over the Stereolabs ZED SDK (``pyzed.sl``).

Every SDK call lives here so the rest of the package stays importable without
CUDA or the SDK present; ``pyzed`` is imported lazily, on the first camera open.
The camera handle is shared between the grab loop and the set_color_* services
behind a lock, so a settings call waits at most one frame.
"""

from __future__ import annotations

import threading

from .conversions import bgra_to_bgr_bytes, depth_mm_to_z16_bytes


def _load_sl():
    import pyzed.sl as sl

    return sl


class ZedFrame:
    """One grabbed capture: rectified left BGR and z16 depth, shared geometry."""

    __slots__ = ("bgr", "depth_z16", "width", "height")

    def __init__(self, bgr: bytes, depth_z16: bytes, width: int, height: int):
        self.bgr = bgr
        self.depth_z16 = depth_z16
        self.width = width
        self.height = height


class ZedCamera:
    """An opened ZED camera producing rectified left color and neural depth."""

    def __init__(self, sl, camera):
        self._sl = sl
        self._camera = camera
        self._lock = threading.Lock()
        self._runtime = sl.RuntimeParameters()
        self._left = sl.Mat()
        self._depth = sl.Mat()

    @classmethod
    def open(
        cls,
        resolution_member: str,
        frame_rate: int,
        depth_mode_member: str,
        min_depth_mm: float,
        serial_number: int,
    ) -> "ZedCamera":
        sl = _load_sl()
        init = sl.InitParameters()
        init.camera_resolution = getattr(sl.RESOLUTION, resolution_member)
        init.camera_fps = frame_rate
        init.depth_mode = getattr(sl.DEPTH_MODE, depth_mode_member)
        init.coordinate_units = sl.UNIT.MILLIMETER
        init.depth_minimum_distance = min_depth_mm
        if serial_number:
            init.set_from_serial_number(serial_number)
        camera = sl.Camera()
        err = camera.open(init)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED open failed: {err}")
        return cls(sl, camera)

    def grab(self) -> ZedFrame | None:
        """The next capture, or None when the SDK reports no fresh frame."""
        sl = self._sl
        with self._lock:
            if self._camera.grab(self._runtime) != sl.ERROR_CODE.SUCCESS:
                return None
            self._camera.retrieve_image(self._left, sl.VIEW.LEFT)
            self._camera.retrieve_measure(self._depth, sl.MEASURE.DEPTH)
            left = self._left.get_data()
            bgr = bgra_to_bgr_bytes(left)
            depth_z16 = depth_mm_to_z16_bytes(self._depth.get_data())
            height, width = left.shape[0], left.shape[1]
        return ZedFrame(bgr, depth_z16, width, height)

    def close(self) -> None:
        with self._lock:
            self._camera.close()

    def _set(self, setting_name: str, value: int) -> tuple[bool, str]:
        sl = self._sl
        setting = getattr(sl.VIDEO_SETTINGS, setting_name)
        with self._lock:
            err = self._camera.set_camera_settings(setting, value)
        if err == sl.ERROR_CODE.SUCCESS:
            return True, ""
        return False, str(err)

    def _get(self, setting_name: str) -> int:
        sl = self._sl
        setting = getattr(sl.VIDEO_SETTINGS, setting_name)
        with self._lock:
            err, value = self._camera.get_camera_settings(setting)
        return value if err == sl.ERROR_CODE.SUCCESS else -1

    def set_exposure(self, mode: str, value_us: int) -> tuple[bool, str, int]:
        """Manual mode drives EXPOSURE_TIME (microseconds); auto hands exposure
        and gain back to AEC_AGC."""
        if mode == "auto":
            ok, message = self._set("AEC_AGC", 1)
            return ok, message, self._get("EXPOSURE_TIME")
        auto_off, message = self._set("AEC_AGC", 0)
        if not auto_off:
            return False, message, value_us
        ok, message = self._set("EXPOSURE_TIME", value_us)
        return ok, message, self._get("EXPOSURE_TIME")

    def set_white_balance(self, mode: str, temperature: int) -> tuple[bool, str, int]:
        if mode == "auto":
            ok, message = self._set("WHITEBALANCE_AUTO", 1)
            return ok, message, self._get("WHITEBALANCE_TEMPERATURE")
        auto_off, message = self._set("WHITEBALANCE_AUTO", 0)
        if not auto_off:
            return False, message, temperature
        ok, message = self._set("WHITEBALANCE_TEMPERATURE", temperature)
        return ok, message, self._get("WHITEBALANCE_TEMPERATURE")

    def set_gain(self, value: int) -> tuple[bool, str, int]:
        ok, message = self._set("GAIN", value)
        return ok, message, self._get("GAIN")

    def set_brightness(self, value: int) -> tuple[bool, str, int]:
        ok, message = self._set("BRIGHTNESS", value)
        return ok, message, self._get("BRIGHTNESS")

    def set_contrast(self, value: int) -> tuple[bool, str, int]:
        ok, message = self._set("CONTRAST", value)
        return ok, message, self._get("CONTRAST")
