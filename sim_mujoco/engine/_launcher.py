#!/usr/bin/env python3
"""MuJoCo SimLauncher for the simulation node: the thread that steps the
scene.

The scene opens empty. A robot's stand loads its model's scene and steps it
until the robot leaves or the engine stops, and the thread waits for the
next stand between two. Each scene is served to the browser by a viewer of
its own, on the same address.
"""

# pylint: disable=R0903
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import head_camera
from bridge_extension import MujocoBridgeExtension
from exts.camera_sensor import add_cameras
from mujoco_models import MujocoModel
from stands import Stands

logger = logging.getLogger(__name__)

# How often an idle thread looks for a stop between stands.
_IDLE_POLL_S = 0.1


def compile_spec(known: MujocoModel):
    """The spec of a model's MJCF, with the joint ranges and site poses its
    entry corrects to the robot's description, and the robot's weight
    compensated where its entry asks for it. A correction naming a joint or a
    site the file lacks is refused, so an upstream rename is caught at the
    first stand."""
    import mujoco

    spec = mujoco.MjSpec.from_file(str(known.scene_path()))
    for name, (lower, upper) in known.joint_ranges.items():
        joint = spec.joint(name)
        if joint is None:
            raise RuntimeError(f"{known.model}: joint_ranges names '{name}', not in {known.scene}")
        joint.range = [lower, upper]
        for actuator in spec.actuators:
            if actuator.target == name and _ctrl_limited(actuator):
                actuator.ctrlrange = [lower, upper]
    for name, pose in known.site_poses.items():
        site = spec.site(name)
        if site is None:
            raise RuntimeError(f"{known.model}: site_poses names '{name}', not in {known.scene}")
        site.pos = list(pose.pos)
        site.quat = list(pose.quat_wxyz)
    if known.gravity_compensation:
        _compensate_gravity(spec, known)
    return spec


def _ctrl_limited(actuator) -> bool:
    """Whether an actuator's control is clamped to its ctrlrange: its file
    says so, or leaves it to the compiler, which clamps the ones that carry a
    range."""
    import mujoco

    if actuator.ctrllimited == mujoco.mjtLimited.mjLIMITED_AUTO:
        return actuator.ctrlrange[0] < actuator.ctrlrange[1]
    return actuator.ctrllimited == mujoco.mjtLimited.mjLIMITED_TRUE


def _compensate_gravity(spec, known: MujocoModel) -> None:
    """Mirror the real driver's in-process gravity feedforward: MuJoCo's body
    gravcomp applies an exact counter-gravity force per body, every step,
    inside the engine. Set on every body the robot's joints move (the real
    arm and gripper drivers both feedforward): the bodies that carry one of
    the model's joints, and everything below them. It goes on the spec
    because MuJoCo compensates only a model compiled with a compensated body,
    so a gravcomp written to the compiled model is never applied. Coriolis is
    intentionally not compensated, negligible at teleop speeds."""
    import mujoco

    compensated = set()
    for name in known.entry.joints():
        joint = spec.joint(name)
        if joint is None:
            raise RuntimeError(
                f"{known.model}: gravity compensation names joint '{name}', not in {known.scene}"
            )
        moved = joint.parent
        for body in (moved, *moved.find_all(mujoco.mjtObj.mjOBJ_BODY)):
            body.gravcomp = 1.0
            compensated.add(body.name)
    logger.info(f"gravity compensation enabled on {len(compensated)} robot bodies")


class SimLauncher:
    def __init__(
        self,
        stands: Stands,
        stop: threading.Event,
        io,
        state_rate_hz: int,
        headless: bool,
        viewer_host: str,
        viewer_port: int,
        renders: bool,
        head_camera_pack: Optional[head_camera.Pack],
    ) -> None:
        self._stands = stands
        # Set by the asyncio caller on cancel (SIGTERM, peppy node stop). The
        # sim loop runs in run_in_executor and cannot observe asyncio
        # cancellation directly, so this Event is the only stop path.
        self._stop = stop
        self._io = io
        self._state_rate_hz = state_rate_hz
        self._headless = headless
        self._viewer_host = viewer_host
        self._viewer_port = viewer_port
        # Whether the robot's cameras are rendered: the rig is its model's.
        self._renders = renders
        # The pack a model that draws the head camera seats on its pedestal.
        self._head_camera_pack = head_camera_pack

    def run(self) -> None:
        import mujoco

        # The engine clock carried from one scene to the next.
        time_base_s = 0.0
        try:
            while not self._stop.is_set():
                # Nothing stands, so an unstand has nothing to let go of.
                idle = self._stands.take_unstand()
                if idle is not None:
                    idle.set_result(None)
                request = self._stands.next_stand(_IDLE_POLL_S)
                if request is None:
                    continue
                try:
                    logger.info(
                        f"Loading the {request.model.model} scene for '{request.robot}'"
                    )
                    model = self._load_model(request.model)
                    data = mujoco.MjData(model)
                    mujoco.mj_forward(model, data)
                    extension = MujocoBridgeExtension(
                        model,
                        data,
                        self._io,
                        request.robot,
                        request.model,
                        self._state_rate_hz,
                        self._renders,
                        time_base_s,
                    )
                    extension.startup()
                except Exception as error:  # pylint: disable=W0718
                    logger.exception(f"standing '{request.robot}' failed")
                    request.future.set_exception(error)
                    continue
                request.future.set_result(None)
                logger.info(f"Scene loaded for '{request.robot}'; states will flow")
                try:
                    if self._headless:
                        self._run_streamed(model, data, extension)
                    else:
                        self._run_windowed(model, data, extension)
                finally:
                    time_base_s = extension.engine_time_s()
                    extension.shutdown()
                    left = self._stands.take_unstand()
                    if left is not None:
                        left.set_result(None)
        except Exception:
            # Otherwise asyncio.run_in_executor captures the traceback in a
            # Future that may never be awaited and the process exits silently.
            logger.exception("SimLauncher.run failed")
            raise
        finally:
            self._stands.cancel_all("the engine stopped")

    def _load_model(self, known: MujocoModel):
        """The model's scene as baked, corrected to the robot's description
        where its entry says so, with the head camera its robot draws and the
        cameras of its own rig when the engine renders."""
        spec = compile_spec(known)
        if known.head_camera:
            if self._head_camera_pack is None:
                raise RuntimeError(
                    f"{known.model} draws the head camera, and no pack was staged"
                )
            head_camera.attach(spec, self._head_camera_pack)
            logger.info("Head camera attached from %s", self._head_camera_pack.directory)
        if self._renders and known.entry.cameras:
            add_cameras(spec, known, known.scene_path())
        return spec.compile()

    def _scene_over(self) -> bool:
        """Whether this scene's loop is done: the engine stops, or the robot
        that stands here is leaving."""
        return self._stop.is_set() or self._stands.unstand_pending()

    def _tick_while_standing(self, tick, ticks: int) -> None:
        """Ticks the viewer up to `ticks` times, and no further once the
        scene is over. A tick paces itself against wall time: on a box that
        cannot step the scene in real time, every tick spends a whole frame
        stepping, so a burst of them lasts seconds, and a robot leaving is
        answered by the tick after the ask, not the burst after it."""
        for _ in range(ticks):
            if self._scene_over():
                return
            tick()

    def _run_streamed(self, model, data, extension: MujocoBridgeExtension) -> None:
        import mujoco as _mujoco
        import viser
        import mjviser

        # Hand mjviser the bridge extension's step(): it owns mj_step plus the
        # plugin loop, so this single callback is the entire per-tick work.
        def _step_fn(_m, _d) -> None:
            extension.step()

        server = None
        try:
            # Bind address comes from the viewer_host param (all interfaces by
            # default so the viewer is reachable from other machines). One
            # server per scene: a browser open on the previous scene
            # reconnects to this one.
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

            # viser sends batched position updates as delta messages only:
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

            logger.info(f"MuJoCo viewer available: open http://{host}:{port} in a browser")
            _render_period = 1.0 / 60.0
            _dt = model.opt.timestep
            _last_render = 0.0
            _last_phys_wall = time.monotonic()
            while not self._scene_over():
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
                    self._tick_while_standing(viewer._tick, n)  # pylint: disable=W0212
                    _last_phys_wall += n * _dt
                if now - _last_render >= _render_period:
                    viewer._render()  # pylint: disable=W0212
                    _last_render = now
                time.sleep(0.001)
        except KeyboardInterrupt:
            logger.info("Shutting down.")
        finally:
            # ViserServer owns non-daemon HTTP/WebSocket threads; without an
            # explicit stop the process can't exit after the sim loop ends,
            # and the next scene's server takes the address back.
            if server is not None:
                server.stop()

    def _run_windowed(self, model, data, extension: MujocoBridgeExtension) -> None:
        import mujoco
        import mujoco.viewer

        dt = model.opt.timestep
        try:
            with mujoco.viewer.launch_passive(model, data) as viewer:
                while viewer.is_running() and not self._scene_over():
                    step_start = time.monotonic()
                    extension.step()
                    viewer.sync()
                    elapsed = time.monotonic() - step_start
                    remaining = dt - elapsed
                    if remaining > 0:
                        time.sleep(remaining)
        except KeyboardInterrupt:
            logger.info("Shutting down.")
