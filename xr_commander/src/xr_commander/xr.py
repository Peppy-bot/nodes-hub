"""The headset session: everything that knows teleop_xr exists.

Owns the server lifecycle only; what an XR frame is lives in `devices`, so the
clutch and publishers never import a web stack. teleop_xr serves the page at
'/' itself; the routes behind the page's Robot card are `robot_choice`'s,
and the task page beside it is `task_page`'s.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path

import teleop_xr
import uvicorn
from fastapi import APIRouter, FastAPI
from teleop_xr.config import TeleopSettings, ViewConfig
from teleop_xr.video_stream import VideoSource

from xr_commander.bus import REAL_CLOCK, log
from xr_commander.devices import XrFrame, parse_hands

_SERVER_START_TIMEOUT_S = 5.0
_SERVER_STOP_TIMEOUT_S = 2.0


def include_router_first(app: FastAPI, router: APIRouter) -> None:
    """Add `router` ahead of every route the app already has.

    teleop_xr mounts its frontend at '/', which matches every path, so routes
    appended after it are unreachable.
    """
    appended_from = len(app.router.routes)
    app.include_router(router)
    appended = app.router.routes[appended_from:]
    del app.router.routes[appended_from:]
    app.router.routes[:0] = appended


def _log_failure(shown: concurrent.futures.Future) -> None:
    if not shown.cancelled() and shown.exception() is not None:
        log(f"failed to send the camera panels to the pages: {shown.exception()!r}")


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
    hand-off, and the frames themselves are immutable. The views are the
    headset's panels, one per named video source: the session's own, floating
    above the row of the ones `show_views` names, on every page.
    """

    def __init__(
        self,
        *,
        host: str,
        port: int,
        tls_cert_path: Path,
        tls_key_path: Path,
        video_sources: dict[str, VideoSource],
    ) -> None:
        self._lock = threading.Lock()
        self._latest: XrFrame | None = None
        self._thread: threading.Thread | None = None
        self._own_sources = dict(video_sources)

        self._settings = TeleopSettings(
            host=host,
            port=port,
            input_mode="controller",
            # A tracked controller reports its true attitude; zero the
            # phone-tilt offset.
            natural_phone_orientation_euler=[0.0, 0.0, 0.0],
            # The page builds its camera panels from these keys, listed from
            # its first load, and matches incoming WebRTC tracks to them by
            # stream id.
            camera_views=self._views_of(video_sources),
        )
        self._teleop = teleop_xr.Teleop(self._settings, video_sources=video_sources)
        self._teleop.subscribe(self._on_xr_update)

        self._server = uvicorn.Server(
            uvicorn.Config(
                self._teleop.app,
                host=host,
                port=port,
                ssl_certfile=str(tls_cert_path),
                ssl_keyfile=str(tls_key_path),
                # Connection lines are the only visibility into browser-side
                # failures; the runscript drains stderr.
                log_level="info",
            )
        )

    def _views_of(self, sources: Mapping[str, VideoSource]) -> dict[str, ViewConfig]:
        """One panel per named source; the session's own float above the
        cameras' row."""
        return {name: ViewConfig(raised=name in self._own_sources) for name in sources}

    def _on_xr_update(self, _pose, message) -> None:
        devices = (message or {}).get("devices") or []
        frame = XrFrame(
            received_monotonic_s=REAL_CLOCK.monotonic(),
            hands=parse_hands(devices),
        )
        with self._lock:
            self._latest = frame

    def latest(self) -> XrFrame | None:
        with self._lock:
            return self._latest

    def serve(self, router: APIRouter) -> None:
        """Serve `router` ahead of every route teleop_xr mounts, from the
        next start on."""
        include_router_first(self._teleop.app, router)

    def show_views(self, views: Mapping[str, VideoSource]) -> None:
        """Show the session's own sources and `views` alone, on every page:
        each page in session is told its panels again and its video restarts
        on their tracks, and a page that connects later is told them at
        connect. Applied on the server's loop, between the page messages
        teleop_xr handles there; a server that is not serving has no page to
        show them on."""
        loop = self._server_loop()
        if loop is None:
            return
        sources = {**self._own_sources, **views}
        shown = asyncio.run_coroutine_threadsafe(
            self._teleop.replace_video_sources(sources, self._views_of(sources)), loop
        )
        shown.add_done_callback(_log_failure)

    def _server_loop(self) -> asyncio.AbstractEventLoop | None:
        """The loop the server runs on, None before it serves or after it
        stopped."""
        servers = self._server.servers if self._server.started else []
        loop = servers[0].get_loop() if servers else None
        return loop if loop is not None and not loop.is_closed() else None

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

    def bound_address(self) -> tuple[str, int]:
        """The address the server's listener bound, as the operating system
        reports it: what the node announces as its page endpoint. Valid once
        `start` returned."""
        host, port = self._server.servers[0].sockets[0].getsockname()[:2]
        return host, port

    def stop(self) -> None:
        self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=_SERVER_STOP_TIMEOUT_S)
            if self._thread.is_alive():
                log("WebXR server thread did not stop inside the join window")
