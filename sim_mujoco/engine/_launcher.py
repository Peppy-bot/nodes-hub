#!/usr/bin/env python3
"""MuJoCo SimLauncher for the simulation node: the thread that steps the
scene.

The scene opens empty and stays up. A robot standing and a robot leaving
both compose it again on this thread, which is the only one that may compile
a model while physics reads it, and the robots that stay carry their state
onto the new one: their joint positions, their velocities and the targets
their actuators hold. The engine's clock carries too, so the fleet's time
never goes back.

The scene is watched through a view: a browser one while the engine runs
headless, a window where it does not, and none while no robot stands.
"""

# pylint: disable=R0903
from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

from bridge_extension import MujocoBridgeExtension
from edits import Edits
from world import PREFIX_SEPARATOR, World

logger = logging.getLogger(__name__)

# How often a scene nobody watches looks for a change or a stop.
_IDLE_POLL_S = 0.001
# What the browser view redraws at.
_RENDER_PERIOD_S = 1.0 / 60.0
# How many steps a scene may take in one turn of the loop to catch up with
# the wall clock. What it owes beyond that it keeps as lost time.
_MAX_CATCHUP_STEPS = 200


class SimLauncher:
    """Steps the scene, composes it again as robots join and leave, and
    shows it."""

    def __init__(
        self,
        world: World,
        edits: Edits,
        stop: threading.Event,
        io,
        state_rate_hz: int,
        headless: bool,
        viewer_host: str,
        viewer_port: int,
    ) -> None:
        self._world = world
        # The stands and unstands waiting for this thread.
        self._edits = edits
        # Set by the asyncio caller on cancel (SIGTERM, peppy node stop). The
        # sim loop runs in run_in_executor and cannot observe asyncio
        # cancellation directly, so this Event is the only stop path.
        self._stop = stop
        self._io = io
        self._state_rate_hz = state_rate_hz
        self._headless = headless
        self._viewer_host = viewer_host
        self._viewer_port = viewer_port
        self._model = None
        self._data = None
        self._extension: Optional[MujocoBridgeExtension] = None
        self._view = _NoView()
        # The engine clock, carried from one scene to the next.
        self._time_base_s = 0.0
        # Each standing camera's frame counter, carried the same way.
        self._camera_counters: dict = {}

    def run(self) -> None:
        try:
            self.rebind()
            while not self._stop.is_set():
                # Stand the robots that joined and take out the ones that
                # left, on this thread, which is the only one that may
                # compile the scene.
                self._edits.drain()
                if self._extension is None:
                    raise RuntimeError(
                        "the change just made leaves a scene that does not compose, and "
                        "an engine holding no scene steps nothing: this node stops, and "
                        "the robots standing in it join the engine that comes back"
                    )
                self._view.step(self._scene_is_changing)
        except Exception:
            # Otherwise asyncio.run_in_executor captures the traceback in a
            # Future that may never be awaited and the process exits silently.
            logger.exception("SimLauncher.run failed")
            raise
        finally:
            self._edits.cancel_all("the engine stopped")
            self.unbind()

    def _scene_is_changing(self) -> bool:
        """True once the engine stops or a change to the scene (a robot
        standing, a robot leaving) waits for this thread. Either one ends the
        steps the view is taking, so the change is made before the scene
        takes another step."""
        return self._stop.is_set() or self._edits.pending()

    def unbind(self) -> None:
        """Lets go of the scene so it can be composed again: the view closes
        and every camera, actuator and sensor view of the robots standing
        stops reading the model that is about to be replaced. The engine's
        clock is kept, so the scene that follows carries it on.

        The engine holds nothing of the scene once this returns, however
        badly the scene let go, so whoever composes the next one is the only
        thing standing between this engine and a scene it cannot step."""
        try:
            self._view.close()
        except Exception:  # pylint: disable=W0718
            logger.exception("the view of this scene did not close")
        self._view = _NoView()
        extension, self._extension = self._extension, None
        if extension is None:
            return
        # A scene that does not shut down hands its counters to nobody: what
        # its cameras counted goes with it, and the scene composed next
        # counts its own frames.
        self._camera_counters = {}
        try:
            self._time_base_s = extension.engine_time_s()
            self._camera_counters = extension.shutdown()
        except Exception:  # pylint: disable=W0718
            logger.exception("the scene this engine held did not shut down")

    def rebind(self) -> None:
        """Composes the scene as it stands, carries onto it what the robots
        that stayed were doing, and shows it. A scene that cannot be composed
        (two models asking for different settings, a camera whose body is
        missing) raises out of the stand that asked for it, leaving the
        engine to put the previous scene back."""
        import mujoco  # pylint: disable=C0415

        model = self._world.compose().compile()
        data = mujoco.MjData(model)
        carried = self._carry_state(model, data)
        mujoco.mj_forward(model, data)
        robots = self._world.robots()
        extension = MujocoBridgeExtension(
            model,
            data,
            self._io,
            robots,
            self._state_rate_hz,
            renders=self._world.renders,
            time_base_s=self._time_base_s,
            camera_counters=self._camera_counters,
        )
        extension.startup(posture_for=[r for r in robots if r.instance not in carried])
        # The scene is taken up whole: nothing of it is installed until every
        # part of it is built, so a scene that cannot be built leaves nothing
        # running behind the one the engine keeps stepping.
        view = self._open_view(model, data, extension)
        self._model, self._data, self._extension, self._view = model, data, extension, view
        logger.info(
            "the scene stands %d robot(s): %s",
            len(robots),
            ", ".join(f"{robot.instance} ({robot.model})" for robot in robots) or "none",
        )

    def _carry_state(self, model, data) -> set[str]:
        """Writes what the robots that stayed were doing onto the scene just
        compiled, and answers whose state came across. A joint of the new
        model that the previous one did not carry belongs to a robot that
        just joined, and starts where its model's posture puts it."""
        import mujoco  # pylint: disable=C0415

        if self._model is None or self._data is None:
            return set()
        carried: set[str] = set()
        for joint in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            # A model may leave a joint unnamed, and MuJoCo has no name to
            # look such a joint up by in the scene before this one: it starts
            # where the model it came with puts it.
            if name is None:
                continue
            previous = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if previous < 0:
                continue
            _carry(data.qpos, model, joint, self._data.qpos, self._model, previous, "qpos")
            _carry(data.qvel, model, joint, self._data.qvel, self._model, previous, "dof")
            carried.add(_robot_of(name))
        # The targets the actuators hold, so an arm that was commanded holds
        # its setpoint in the scene it is carried into.
        for actuator in range(model.nu):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator)
            if name is None:
                continue
            previous = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if previous >= 0:
                data.ctrl[actuator] = self._data.ctrl[previous]
        return carried

    def _open_view(self, model, data, extension: MujocoBridgeExtension):
        """How this scene is watched: a scene with no robot in it is stepped
        and shown to nobody, so no viewer serves an empty stage and none is
        left behind by the last robot to leave.

        A view that cannot open leaves the scene stepped and unwatched: the
        robots standing keep their physics, their state and the clock this
        engine publishes, and the operator is told why they have no picture."""
        if not self._world.robots():
            return _NoView(model, extension)
        try:
            if self._headless:
                return _BrowserView(model, data, extension, self._viewer_host, self._viewer_port)
            return _WindowView(model, data, extension)
        except Exception as error:  # pylint: disable=W0718
            logger.error(
                "the scene is stepped unwatched: its view could not open: %s", error
            )
            return _NoView(model, extension)


class _NoView:
    """A scene nobody watches: it steps against the wall clock and draws
    nothing. The engine runs one while no robot stands, so its clock keeps
    ticking for whoever reads it."""

    def __init__(self, model=None, extension=None) -> None:
        self._extension = extension
        self._pacer = _StepPacer(model.opt.timestep) if model is not None else None

    def step(self, scene_is_changing: Callable[[], bool]) -> None:
        if self._extension is None or self._pacer is None:
            time.sleep(_IDLE_POLL_S)
            return
        _take_due_steps(
            self._pacer.due(time.monotonic()), self._extension.step, scene_is_changing
        )
        time.sleep(_IDLE_POLL_S)

    def close(self) -> None:
        return


class _StepPacer:
    """How many steps the scene owes the wall clock, at most
    `_MAX_CATCHUP_STEPS` in one turn; the rest of the debt is lost time."""

    def __init__(self, timestep_s: float) -> None:
        self._timestep_s = timestep_s
        self._stepped_to_s = time.monotonic()

    def due(self, now: float) -> int:
        steps = int((now - self._stepped_to_s) / self._timestep_s)
        if steps <= 0:
            return 0
        steps = min(steps, _MAX_CATCHUP_STEPS)
        self._stepped_to_s += steps * self._timestep_s
        return steps


class _WindowView:
    """The scene in a native MuJoCo window, for an engine that runs with a
    display."""

    def __init__(self, model, data, extension: MujocoBridgeExtension) -> None:
        import mujoco.viewer  # pylint: disable=C0415

        self._model = model
        self._data = data
        self._extension = extension
        self._viewer = mujoco.viewer.launch_passive(model, data)
        self._pacer = _StepPacer(model.opt.timestep)

    def step(self, scene_is_changing: Callable[[], bool]) -> None:
        if not self._viewer.is_running():
            time.sleep(_IDLE_POLL_S)
            return
        _take_due_steps(
            self._pacer.due(time.monotonic()), self._extension.step, scene_is_changing
        )
        self._viewer.sync()
        time.sleep(_IDLE_POLL_S)

    def close(self) -> None:
        self._viewer.close()


class _BrowserView:
    """The scene in a browser, served by viser on the engine's viewer
    address. One server per scene: a browser open on the scene before this
    one reconnects to it."""

    def __init__(
        self,
        model,
        data,
        extension: MujocoBridgeExtension,
        host: str,
        port: int,
    ) -> None:
        import viser  # pylint: disable=C0415

        self._model = model
        self._data = data
        self._extension = extension
        self._pacer = _StepPacer(model.opt.timestep)
        self._next_render_s = 0.0
        self._server = viser.ViserServer(host=host, port=port)
        try:
            self._show(host, port)
        except Exception:
            # The server holds the address and its own non-daemon threads from
            # the moment it is built, so a view that cannot finish hands them
            # back for whatever view comes after it. What went wrong with the
            # view is what the operator is told, so a server that will not
            # stop is reported under it.
            try:
                self._server.stop()
            except Exception:  # pylint: disable=W0718
                logger.exception("the viewer's server did not stop")
            raise

    def _show(self, host: str, port: int) -> None:
        """Draws the scene for the browser: the viewer that renders it, the
        button that restages its props, and the first frame."""
        import mjviser  # pylint: disable=C0415
        import mujoco  # pylint: disable=C0415

        model, data, extension = self._model, self._data, self._extension

        # Hand mjviser the bridge extension's step(): it owns mj_step plus the
        # plugin loop, so this single callback is the entire per-tick work.
        self._viewer = mjviser.Viewer(
            model, data, server=self._server, step_fn=lambda _m, _d: extension.step()
        )
        # Free joints are the scene props (every robot joint is driven);
        # snapshot their spawn pose so the viewer button can restage the
        # scene without touching the arms.
        self._free_joints = [
            (model.jnt_qposadr[joint], model.jnt_dofadr[joint])
            for joint in range(model.njnt)
            if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE
        ]
        self._spawn_qpos = [data.qpos[q : q + 7].copy() for q, _ in self._free_joints]
        self._reset_requested = threading.Event()
        if self._free_joints:
            reset_button = self._server.gui.add_button("Reset scene objects")

            @reset_button.on_click
            def _(_event) -> None:
                # Viser callbacks run on server threads; the sim loop owns
                # the mj state, so only flag the request here.
                self._reset_requested.set()

        # viser sends batched position updates as delta messages only:
        # new/refreshing clients receive initial zero positions unless we
        # explicitly push current state on each connection.
        @self._server.on_client_connect
        def _(_client) -> None:
            self._viewer._refresh_scene_from_gui()  # pylint: disable=W0212

        # viewer.run() installs a SIGINT handler which only works on the
        # main thread; we run inside run_in_executor, so drive the internals
        # by hand instead.
        self._viewer._setup_gui()  # pylint: disable=W0212
        mujoco.mj_forward(model, data)
        self._viewer._render()  # pylint: disable=W0212
        logger.info(f"MuJoCo viewer available: open http://{host}:{port} in a browser")

    def step(self, scene_is_changing: Callable[[], bool]) -> None:
        import mujoco  # pylint: disable=C0415

        if self._reset_requested.is_set():
            self._reset_requested.clear()
            for (qpos, dof), pose in zip(self._free_joints, self._spawn_qpos):
                self._data.qpos[qpos : qpos + 7] = pose
                self._data.qvel[dof : dof + 6] = 0.0
            mujoco.mj_forward(self._model, self._data)
            logger.info("Scene objects reset to spawn poses")
        now = time.monotonic()
        _take_due_steps(
            self._pacer.due(now), self._viewer._tick, scene_is_changing  # pylint: disable=W0212
        )
        if now >= self._next_render_s:
            self._viewer._render()  # pylint: disable=W0212
            self._next_render_s = now + _RENDER_PERIOD_S
        time.sleep(_IDLE_POLL_S)

    def close(self) -> None:
        # ViserServer owns non-daemon HTTP/WebSocket threads; without an
        # explicit stop the process can't exit after the sim loop ends, and
        # the next scene's server takes the address back.
        self._server.stop()


def _take_due_steps(
    steps: int, step: Callable[[], None], scene_is_changing: Callable[[], bool]
) -> None:
    """Takes the steps the scene owes the wall clock, and no further once the
    scene is changing: the steps not taken are lost time, like the ones past
    `_MAX_CATCHUP_STEPS`. A scene stepped slower than real time spends
    seconds on the whole of them (a browser view's tick steps for up to a
    frame of wall time before it returns), so a robot leaving is answered
    after the step during which it asked, not after the last one owed."""
    for _ in range(steps):
        if scene_is_changing():
            return
        step()


def _carry(into, model, joint: int, out_of, previous_model, previous: int, kind: str) -> None:
    """Copies one joint's values from the scene before this one. How many
    values a joint carries is the model's own business (a free joint holds
    seven positions and six velocities), so each model says how wide its
    own joint is."""
    here = _span(model, joint, kind)
    there = _span(previous_model, previous, kind)
    if here[1] - here[0] != there[1] - there[0]:
        raise RuntimeError(
            f"the joint at {here} carried {there[1] - there[0]} value(s) in the scene before "
            f"this one and {here[1] - here[0]} in this one"
        )
    into[here[0] : here[1]] = out_of[there[0] : there[1]]


def _robot_of(scene_name: str) -> str:
    """The robot a name in the composed scene belongs to: every name a robot
    answers to carries its own prefix, and a robot's name carries no
    separator of its own (`world.name_in_the_scene` refuses one)."""
    return scene_name.split(PREFIX_SEPARATOR, 1)[0]


def _span(model, joint: int, kind: str) -> tuple[int, int]:
    """Where one joint's positions or velocities sit, from its own address to
    the next joint's, or to the end of the array for the last joint."""
    addresses, total = (
        (model.jnt_qposadr, model.nq) if kind == "qpos" else (model.jnt_dofadr, model.nv)
    )
    start = int(addresses[joint])
    end = int(addresses[joint + 1]) if joint + 1 < model.njnt else int(total)
    return start, end
