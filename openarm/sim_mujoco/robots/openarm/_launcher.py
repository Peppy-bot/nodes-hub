#!/usr/bin/env python3
"""MuJoCo SimLauncher for the openarm sim engine node."""

# pylint: disable=R0903
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from bridge_extension import MujocoBridgeExtension
from camera_common import CameraConfig
from exts.camera_sensor import compile_model_with_cameras

logger = logging.getLogger(__name__)


class SimLauncher:
    def __init__(
        self,
        xml_path: Path,
        ready: threading.Event,
        stop: threading.Event,
        io,
        state_rate_hz: int,
        headless: bool,
        viewer_host: str,
        viewer_port: int,
        cameras: list[CameraConfig],
    ) -> None:
        self._xml_path = xml_path
        self._ready = ready
        # Set by the asyncio caller on cancel (SIGTERM, peppy node stop). The
        # sim loop runs in run_in_executor and cannot observe asyncio
        # cancellation directly — this Event is the only stop path.
        self._stop = stop
        self._io = io
        self._state_rate_hz = state_rate_hz
        self._headless = headless
        self._viewer_host = viewer_host
        self._viewer_port = viewer_port
        self._cameras = cameras

    def run(self) -> None:
        import mujoco

        if not self._xml_path.exists():
            logger.error(
                "MJCF not found at %s — assets should be baked into the container image",
                self._xml_path,
            )
            raise FileNotFoundError(self._xml_path)

        logger.info(f"Loading model: {self._xml_path}")
        model = self._load_model()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        extension = MujocoBridgeExtension(
            model, data, self._io, self._state_rate_hz, self._cameras
        )
        try:
            extension.startup()
            self._ready.set()
            logger.info("Scene loaded; states will flow")

            if self._headless:
                self._run_streamed(model, data, extension)
            else:
                self._run_windowed(model, data, extension)
        except Exception:
            # Otherwise asyncio.run_in_executor captures the traceback in a
            # Future that may never be awaited — the process exits silently.
            logger.exception("SimLauncher.run failed")
            raise
        finally:
            extension.shutdown()
            self._ready.clear()

    def _load_model(self):
        """The scene as baked, or the scene plus the configured cameras."""
        import mujoco

        if not self._cameras:
            return mujoco.MjModel.from_xml_path(str(self._xml_path))
        return compile_model_with_cameras(self._xml_path, self._cameras)

    def _run_streamed(self, model, data, extension: MujocoBridgeExtension) -> None:
        import mujoco as _mujoco
        import viser
        import mjviser

        # Hand mjviser the bridge extension's step() — it owns mj_step plus the
        # plugin loop, so this single callback is the entire per-tick work.
        def _step_fn(_m, _d) -> None:
            extension.step()

        server = None
        try:
            # Bind address comes from the viewer_host param (all interfaces by
            # default so the viewer is reachable from other machines).
            host = self._viewer_host
            port = self._viewer_port
            server = viser.ViserServer(host=host, port=port)
            viewer = mjviser.Viewer(model, data, server=server, step_fn=_step_fn)

            # Free joints are the scene props (every robot joint is driven);
            # snapshot their spawn pose so the viewer button can restage the
            # scene without touching the arms.
            free_slices = [
                (model.jnt_qposadr[j], model.jnt_dofadr[j])
                for j in range(model.njnt)
                if model.jnt_type[j] == _mujoco.mjtJoint.mjJNT_FREE
            ]
            spawn_qpos = [data.qpos[q : q + 7].copy() for q, _ in free_slices]
            reset_requested = threading.Event()
            if free_slices:
                reset_button = server.gui.add_button("Reset scene objects")

                @reset_button.on_click
                def _(_event) -> None:
                    # Viser callbacks run on server threads; the sim loop owns
                    # the mj state, so only flag the request here.
                    reset_requested.set()

            # viser sends batched position updates as delta messages only —
            # new/refreshing clients receive initial zero positions unless we
            # explicitly push current state on each connection.
            @server.on_client_connect
            def _(client) -> None:
                viewer._refresh_scene_from_gui()  # pylint: disable=W0212

            # viewer.run() installs a SIGINT handler which only works on the
            # main thread; we run inside run_in_executor, so drive the internals
            # by hand instead.
            viewer._setup_gui()  # pylint: disable=W0212
            _mujoco.mj_forward(model, data)
            viewer._render()  # pylint: disable=W0212

            logger.info(
                f"MuJoCo viewer available — open http://{host}:{port} in a browser"
            )
            _render_period = 1.0 / 60.0
            _dt = model.opt.timestep
            _last_render = 0.0
            _last_phys_wall = time.monotonic()
            while not self._stop.is_set():
                if reset_requested.is_set():
                    reset_requested.clear()
                    for (q, d), pose in zip(free_slices, spawn_qpos):
                        data.qpos[q : q + 7] = pose
                        data.qvel[d : d + 6] = 0.0
                    _mujoco.mj_forward(model, data)
                    logger.info("Scene objects reset to spawn poses")
                now = time.monotonic()
                # Step physics at real time, decoupled from render rate.
                n = int((now - _last_phys_wall) / _dt)
                if n > 0:
                    # Cap prevents spiral-of-death after stalls. Trade-off: on
                    # stall recovery the sim falls behind real time
                    # permanently rather than catching up.
                    n = min(n, 200)
                    for _ in range(n):
                        viewer._tick()  # pylint: disable=W0212
                    _last_phys_wall += n * _dt
                if now - _last_render >= _render_period:
                    viewer._render()  # pylint: disable=W0212
                    _last_render = now
                time.sleep(0.001)
        except KeyboardInterrupt:
            logger.info("Shutting down.")
        finally:
            # ViserServer owns non-daemon HTTP/WebSocket threads; without an
            # explicit stop the process can't exit even after the sim loop ends.
            if server is not None:
                server.stop()

    def _run_windowed(self, model, data, extension: MujocoBridgeExtension) -> None:
        import mujoco
        import mujoco.viewer

        dt = model.opt.timestep
        try:
            with mujoco.viewer.launch_passive(model, data) as viewer:
                while viewer.is_running() and not self._stop.is_set():
                    step_start = time.monotonic()
                    extension.step()
                    viewer.sync()
                    elapsed = time.monotonic() - step_start
                    remaining = dt - elapsed
                    if remaining > 0:
                        time.sleep(remaining)
        except KeyboardInterrupt:
            logger.info("Shutting down.")
