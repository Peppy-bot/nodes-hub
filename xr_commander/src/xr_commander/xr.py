"""The headset session: everything that knows teleop_xr exists.

Owns the server lifecycle only; what an XR frame is lives in `devices`, so the
clutch and publishers never import a web stack.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path

import uvicorn
from teleop_xr import Teleop
from teleop_xr.config import TeleopSettings
from teleop_xr.video_stream import VideoSource

from xr_commander.bus import log
from xr_commander.devices import XrFrame, parse_hands

_SERVER_START_TIMEOUT_S = 5.0
_SERVER_STOP_TIMEOUT_S = 2.0


def wait_until_started(
    started: Callable[[], bool],
    alive: Callable[[], bool],
    timeout_s: float,
    tick_s: float = 0.05,
) -> bool:
    """Poll `started()` until true, the thread dies, or the timeout passes."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if started():
            return True
        if not alive():
            return False
        time.sleep(tick_s)
    return False


class XrSession:
    """Owns the teleop_xr server and the latest frame it produced.

    The server runs its own loop on a thread; the lock covers the frame
    hand-off, and the frames themselves are immutable.
    """

    def __init__(
        self,
        *,
        host: str,
        port: int,
        tls_cert_path: Path,
        tls_key_path: Path,
        video_sources: dict[str, VideoSource],
        camera_views: dict[str, dict],
    ) -> None:
        self._lock = threading.Lock()
        self._latest: XrFrame | None = None
        self._thread: threading.Thread | None = None

        settings = TeleopSettings(
            host=host,
            port=port,
            input_mode="controller",
            # A tracked controller reports its true attitude; zero the
            # phone-tilt offset.
            natural_phone_orientation_euler=[0.0, 0.0, 0.0],
            # The frontend builds its camera panel from these keys at connect
            # and matches incoming WebRTC tracks to them by stream id.
            camera_views=camera_views,
        )
        self._teleop = Teleop(settings, video_sources=video_sources)
        self._teleop.subscribe(self._on_xr_update)

        self._server = uvicorn.Server(
            uvicorn.Config(
                self._teleop.app,
                host=host,
                port=port,
                ssl_certfile=str(tls_cert_path),
                ssl_keyfile=str(tls_key_path),
                # Connection lines are the only visibility into browser-side
                # failures; safe now that the runscript drains stderr.
                log_level="info",
            )
        )

    def _on_xr_update(self, _pose, message) -> None:
        devices = (message or {}).get("devices") or []
        frame = XrFrame(
            received_monotonic_s=time.monotonic(),
            hands=parse_hands(devices),
        )
        with self._lock:
            self._latest = frame

    def latest(self) -> XrFrame | None:
        with self._lock:
            return self._latest

    def start(self) -> None:
        """Serve on a thread, returning once the server is actually up.

        Raises when it dies or never binds: a node that looks healthy while
        nothing listens is undiagnosable from the headset. Blocks briefly, so
        call it off the event loop.
        """
        thread = threading.Thread(
            target=self._server.run, name="xr-commander-http", daemon=True
        )
        self._thread = thread
        thread.start()
        if wait_until_started(
            lambda: self._server.started, thread.is_alive, _SERVER_START_TIMEOUT_S
        ):
            return
        # A hung bind must not leave a daemon thread holding the port.
        self.stop()
        raise RuntimeError(
            "WebXR server failed to start on "
            f"{self._server.config.host}:{self._server.config.port} "
            "(port already in use, or unreadable TLS material?)"
        )

    def stop(self) -> None:
        self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=_SERVER_STOP_TIMEOUT_S)
            if self._thread.is_alive():
                log("WebXR server thread did not stop inside the join window")
