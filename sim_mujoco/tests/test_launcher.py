"""What a robot joining or leaving does to the scene already running: it is
composed again, and the robots that stay carry onto it what they were doing,
a wheel's integral and where a robot that rolls stands among it, while the
one that arrived starts in its model's posture. The scenes are
real (tiny) MJCF composed and compiled by MuJoCo.
"""

from __future__ import annotations

import logging
import math
import sys
import threading
import types
from dataclasses import replace
from pathlib import Path

import mujoco
import pytest
import viewer_fakes
from entries import camera, entry, known as known_model, wheeled

import runtime_fakes  # noqa: F401  pylint: disable=W0611  (installs the runtime the launcher imports)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))

import mujoco_models  # noqa: E402  pylint: disable=C0413
import _launcher as launcher_module  # noqa: E402  pylint: disable=C0413
from _launcher import SimLauncher, _NoView  # noqa: E402  pylint: disable=C0413
from edits import Edits  # noqa: E402  pylint: disable=C0413
from exts import camera_sensor  # noqa: E402  pylint: disable=C0413
from viewer_server import ViewerServer  # noqa: E402  pylint: disable=C0413
from world import World  # noqa: E402  pylint: disable=C0413

_SCENE = "arm/arm.xml"
_STATE_RATE_HZ = 50
_POSTURE = {"lift": 0.25, "flex": -0.5}

# A two-joint arm with a bob on a ball joint, so a rebuild has a joint to
# carry whose positions and speeds differ in width. The base carries the
# robot.
_ARM = """<mujoco model="arm">
  <compiler angle="radian"/>
  <option timestep="0.002" integrator="implicitfast"/>
  <worldbody>
    <body name="base" pos="0 0 0.05">
      <geom type="box" size="0.05 0.05 0.05" mass="1"/>
      <body name="upper" pos="0 0 0.08">
        <joint name="lift" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.02" mass="0.2"/>
        <body name="fore" pos="0 0 0.2">
          <joint name="flex" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.15" size="0.015" mass="0.1"/>
        </body>
      </body>
      <body name="bob" pos="0.1 0 0">
        <joint name="bob" type="ball"/>
        <geom type="sphere" size="0.02" mass="0.05"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="lift" joint="lift" kp="20" kv="1" ctrlrange="-1.5 1.5"/>
    <position name="flex" joint="flex" kp="20" kv="1" ctrlrange="-1.5 1.5"/>
  </actuator>
</mujoco>"""
_ARM_ENTRY = entry("arm", arms={"arm": ["lift", "flex"]}, start_posture=_POSTURE)
_CAMERA_ARM_ENTRY = replace(
    _ARM_ENTRY,
    cameras=(camera("wrist", "fore", (0.0, 0.0, 0.1), (1.0, 0.0, 0.0, 0.0), 60.0, 8, 4, 10),),
)


class _NoSetpoints:
    """The setpoints of a scene no leader drives."""

    def arm(self, robot, arm):
        return None

    def gripper(self, robot, gripper):
        return None

    def base(self, robot, now_s):
        return None

    def take_base_release(self, robot):
        return False


class FakeIO:
    """The transport the bridge publishes on, which these tests only need to
    swallow."""

    def __init__(self) -> None:
        self.setpoints = _NoSetpoints()
        self.engine_times = []

    def record_engine_time(self, seconds):
        self.engine_times.append(seconds)

    def publish_clock_tick(self):
        return None

    def publish_arm_states(self, robot, arm, positions, velocities):
        return None

    def publish_gripper_states(self, robot, gripper, opening):
        return None

    def publish_object_states(self, robots):
        return None

    def publish_base_states(self, robot, positions, velocities):
        return None


class _Unwatched(SimLauncher):
    """A launcher whose scene nobody watches, for the tests that hold the
    scene itself and not how it is shown."""

    def _open_view(self, model, data, extension, robots):
        return _NoView(model, extension)


def _viewer_server() -> ViewerServer:
    """The viewer's server over a viser that serves nobody."""
    return ViewerServer(viewer_fakes.Viser())


@pytest.fixture(name="mjviser")
def mjviser_fixture(monkeypatch):
    """mjviser as the browser view imports it, drawing nothing."""
    monkeypatch.setitem(sys.modules, "mjviser", types.SimpleNamespace(Viewer=viewer_fakes.Viewer))


@pytest.fixture(name="assets", autouse=True)
def assets_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(mujoco_models, "ASSETS_DIR", tmp_path)
    scene = tmp_path / _SCENE
    scene.parent.mkdir()
    scene.write_text(_ARM)
    return tmp_path


def _known(robot=None, **release):
    return known_model("arm", robot or _ARM_ENTRY, mujoco_models.ASSETS_DIR / _SCENE, "base", **release)


def _wheeled():
    return wheeled(mujoco_models.ASSETS_DIR)


def _launcher(world: World) -> SimLauncher:
    return _Unwatched(world, Edits(), threading.Event(), FakeIO(), _STATE_RATE_HZ, _viewer_server())


def _watched(world: World, viewer_server: ViewerServer) -> SimLauncher:
    """A launcher that shows its scene on `viewer_server`."""
    return SimLauncher(world, Edits(), threading.Event(), FakeIO(), _STATE_RATE_HZ, viewer_server)


def _capture_indices(launcher: SimLauncher) -> dict:
    """Each standing camera's capture index counter in the scene the engine holds."""
    return launcher._extension._camera_sensor.counters()  # pylint: disable=W0212


def _stand(launcher: SimLauncher, world: World, name: str, known=None) -> None:
    """A robot joining, the way an edit from the contract server does it."""
    world.add(name, known or _known(), world.free_spot())
    launcher.unbind()
    launcher.rebind()


def _qpos(launcher: SimLauncher, joint: str) -> float:
    model, data = launcher._model, launcher._data  # pylint: disable=W0212
    return float(data.qpos[model.joint(joint).qposadr[0]])


def _ctrl(launcher: SimLauncher, actuator: str) -> float:
    model, data = launcher._model, launcher._data  # pylint: disable=W0212
    return float(data.ctrl[model.actuator(actuator).id])


class TestTheSceneAsRobotsComeAndGo:
    def test_an_empty_scene_steps_and_stands_nobody(self):
        world = World(renders=False)
        launcher = _launcher(world)

        launcher.rebind()

        assert launcher._model.njnt == 0  # pylint: disable=W0212
        launcher._view.step(lambda: False)  # pylint: disable=W0212

    def test_a_robot_that_joins_starts_in_its_models_posture(self):
        world = World(renders=False)
        launcher = _launcher(world)
        launcher.rebind()

        _stand(launcher, world, "alpha")

        assert _qpos(launcher, "alpha/lift") == pytest.approx(_POSTURE["lift"])
        assert _ctrl(launcher, "alpha/lift") == pytest.approx(_POSTURE["lift"])

    def test_a_robot_standing_keeps_its_own_state_when_another_joins(self):
        """Its joints, their speeds and the targets its actuators hold are
        carried onto the scene composed around the robot that arrived."""
        world = World(renders=False)
        launcher = _launcher(world)
        launcher.rebind()
        _stand(launcher, world, "alpha")
        model, data = launcher._model, launcher._data  # pylint: disable=W0212
        data.qpos[model.joint("alpha/lift").qposadr[0]] = 0.9
        data.qvel[model.joint("alpha/flex").dofadr[0]] = 0.4
        data.ctrl[model.actuator("alpha/flex").id] = -0.75

        _stand(launcher, world, "bravo")

        assert _qpos(launcher, "alpha/lift") == pytest.approx(0.9)
        moved, moved_data = launcher._model, launcher._data  # pylint: disable=W0212
        assert float(moved_data.qvel[moved.joint("alpha/flex").dofadr[0]]) == pytest.approx(0.4)
        assert _ctrl(launcher, "alpha/flex") == pytest.approx(-0.75)
        # The robot that just joined starts where its model's posture puts it.
        assert _qpos(launcher, "bravo/lift") == pytest.approx(_POSTURE["lift"])

    def test_a_robot_that_rolls_keeps_where_it_rolled_and_its_wheels_integrals(self):
        """Its free joint is carried as any joint is, and the activation of
        each wheel's integral actuator, the angle its target rates carried
        it to, is carried too, so its rate loop does not snap back."""
        world = World(renders=False)
        launcher = _launcher(world)
        launcher.rebind()
        _stand(launcher, world, "india", _wheeled())
        model, data = launcher._model, launcher._data  # pylint: disable=W0212
        free = model.joint("india").qposadr[0]
        rolled = [0.4, 0.2, 0.0, math.cos(0.2), 0.0, 0.0, math.sin(0.2)]
        data.qpos[free : free + 7] = rolled
        integral = model.actuator_actadr[model.actuator("india/left_wheel_integral").id]
        data.act[integral] = 0.15

        _stand(launcher, world, "juliet", _wheeled())

        moved, moved_data = launcher._model, launcher._data  # pylint: disable=W0212
        moved_free = moved.joint("india").qposadr[0]
        assert list(moved_data.qpos[moved_free : moved_free + 7]) == pytest.approx(rolled)
        moved_integral = moved.actuator_actadr[moved.actuator("india/left_wheel_integral").id]
        assert moved_data.act[moved_integral] == pytest.approx(0.15)
        # The base that just joined starts at rest at its own placement.
        assert moved_data.act[moved.actuator_actadr[moved.actuator("juliet/left_wheel_integral").id]] == 0.0

    def test_a_ball_joint_a_robot_carries_keeps_its_pose_across_a_rebuild(self):
        """A ball joint carries four positions and three speeds, which is a
        width the model states and the carry reads."""
        world = World(renders=False)
        launcher = _launcher(world)
        launcher.rebind()
        _stand(launcher, world, "alpha")
        model, data = launcher._model, launcher._data  # pylint: disable=W0212
        address = model.joint("alpha/bob").qposadr[0]
        data.qpos[address : address + 4] = [0.6, 0.8, 0.0, 0.0]

        _stand(launcher, world, "bravo")

        moved, moved_data = launcher._model, launcher._data  # pylint: disable=W0212
        carried = moved.joint("alpha/bob").qposadr[0]
        assert moved_data.qpos[carried : carried + 4].tolist() == pytest.approx([0.6, 0.8, 0.0, 0.0])

    def test_a_robot_that_leaves_takes_only_itself_out(self):
        world = World(renders=False)
        launcher = _launcher(world)
        launcher.rebind()
        _stand(launcher, world, "alpha")
        _stand(launcher, world, "bravo")
        model, data = launcher._model, launcher._data  # pylint: disable=W0212
        data.qpos[model.joint("bravo/lift").qposadr[0]] = -0.3

        world.remove("alpha")
        launcher.unbind()
        launcher.rebind()

        assert _qpos(launcher, "bravo/lift") == pytest.approx(-0.3)
        with pytest.raises(KeyError):
            launcher._model.joint("alpha/lift")  # pylint: disable=W0212

    def test_the_engine_clock_runs_on_across_a_rebuild(self):
        """The fleet reads one clock, so it never goes back when a robot
        joins."""
        world = World(renders=False)
        launcher = _launcher(world)
        launcher.rebind()
        _stand(launcher, world, "alpha")
        for _ in range(10):
            launcher._extension.step()  # pylint: disable=W0212
        before = launcher._extension.engine_time_s()  # pylint: disable=W0212

        _stand(launcher, world, "bravo")

        assert launcher._extension.engine_time_s() == pytest.approx(before)  # pylint: disable=W0212
        launcher._extension.step()  # pylint: disable=W0212
        assert launcher._extension.engine_time_s() > before  # pylint: disable=W0212


class TestTheThreadThatStepsTheScene:
    def test_it_makes_the_changes_waiting_and_stops_when_it_is_told_to(self):
        """Standing a robot runs on this thread, and the loop ends on the
        stop the node sets when it shuts down."""
        world = World(renders=False)
        edits = Edits()
        stop = threading.Event()
        launcher = _Unwatched(world, edits, stop, FakeIO(), _STATE_RATE_HZ, _viewer_server())
        made = []

        def stand():
            world.add("alpha", _known(), world.free_spot())
            launcher.unbind()
            launcher.rebind()
            made.append("alpha")
            stop.set()

        standing = edits.submit(stand)

        launcher.run()

        assert standing.result(timeout=0) is None
        assert made == ["alpha"]
        assert [robot.instance for robot in world.robots()] == ["alpha"]

    def test_a_change_still_waiting_when_the_engine_stops_is_told_so(self):
        world = World(renders=False)
        edits = Edits()
        stop = threading.Event()
        stop.set()
        launcher = _Unwatched(world, edits, stop, FakeIO(), _STATE_RATE_HZ, _viewer_server())
        waiting = edits.submit(lambda: None)

        launcher.run()

        with pytest.raises(RuntimeError, match="the engine stopped"):
            waiting.result(timeout=0)

    def test_a_camera_counts_its_frames_on_across_a_scene_composed_again(self, monkeypatch):
        """A consumer pairs a frame with its depth by the id it carries, so
        the count runs on into the scene its robot is carried onto."""
        monkeypatch.setattr(camera_sensor.MujocoCameraSensor, "start", lambda self: None)
        world = World(renders=True)
        launcher = _launcher(world)
        world.add("alpha", _known(_CAMERA_ARM_ENTRY), world.free_spot())
        launcher.rebind()
        counted = [_capture_indices(launcher)[("alpha", "wrist")].next() for _ in range(3)]

        world.add("delta", _known(_CAMERA_ARM_ENTRY), world.free_spot())
        launcher.unbind()
        launcher.rebind()

        assert counted == [0, 1, 2]
        assert _capture_indices(launcher)[("alpha", "wrist")].next() == 3
        # The robot that just joined starts its own count.
        assert _capture_indices(launcher)[("delta", "wrist")].next() == 0

    def test_a_scene_that_does_not_compose_takes_the_node_down(self, monkeypatch):
        """An engine holding no scene steps nothing and publishes no clock.
        The node stops saying so, and the robots standing in it join the
        engine that comes back."""
        world = World(renders=False)
        launcher = _launcher(world)
        world.add("alpha", _known(), world.free_spot())

        def leave_no_scene():
            launcher.unbind()
            raise RuntimeError("this scene compiles nothing")

        launcher._edits.submit(leave_no_scene)  # pylint: disable=W0212
        # A run that keeps stepping ends this test at its first step, so the
        # suite answers for it in the time one step takes.
        monkeypatch.setattr(
            launcher_module._NoView,  # pylint: disable=W0212
            "step",
            lambda _self, _scene_is_changing: launcher._stop.set(),  # pylint: disable=W0212
        )

        with pytest.raises(RuntimeError, match="steps nothing"):
            launcher.run()

    @pytest.mark.parametrize("failing", ["view", "extension"])
    def test_a_scene_that_will_not_let_go_leaves_the_engine_holding_nothing(
        self, failing, caplog
    ):
        """An engine still holding a view or a scene that failed to close
        steps physics for nobody while reporting healthy, so letting go is
        total however badly it goes."""
        world = World(renders=False)
        launcher = _launcher(world)
        world.add("alpha", _known(), world.free_spot())
        launcher.rebind()

        def refuse():
            raise RuntimeError("this view will not close")

        if failing == "view":
            launcher._view.close = refuse  # pylint: disable=W0212
        else:
            launcher._extension.shutdown = refuse  # pylint: disable=W0212
        # What the scene before this one counted, which only a scene that
        # shuts down hands on.
        launcher._camera_counters = {("alpha", "wrist"): object()}  # pylint: disable=W0212

        with caplog.at_level(logging.ERROR):
            launcher.unbind()

        assert launcher._extension is None  # pylint: disable=W0212
        assert isinstance(launcher._view, _NoView)  # pylint: disable=W0212
        assert "did not" in caplog.text
        # A scene that did not shut down counted frames nobody carries on.
        assert launcher._camera_counters == {}  # pylint: disable=W0212

    def test_a_view_that_cannot_open_leaves_the_scene_stepping(self, caplog, monkeypatch):
        """The robots standing keep their physics, their state and the clock
        this engine publishes; the operator is told why there is no picture."""

        def refuse(*_args, **_kwargs):
            raise RuntimeError("the scene draws from no camera")

        monkeypatch.setattr(launcher_module, "_BrowserView", refuse)
        world = World(renders=False)
        launcher = _watched(world, _viewer_server())
        launcher.rebind()
        world.add("alpha", _known(), world.free_spot())

        with caplog.at_level(logging.ERROR):
            launcher.unbind()
            launcher.rebind()

        assert "stepped unwatched" in caplog.text
        assert isinstance(launcher._view, _NoView)  # pylint: disable=W0212
        before = launcher._extension.engine_time_s()  # pylint: disable=W0212
        launcher._extension.step()  # pylint: disable=W0212
        assert launcher._extension.engine_time_s() > before  # pylint: disable=W0212

    def test_a_joint_the_scene_leaves_unnamed_is_left_where_its_model_puts_it(self):
        """A model may leave a joint unnamed, and an unnamed joint is one the
        scene before this one has no name to look up."""
        unnamed = _ARM.replace('<joint name="bob" type="ball"/>', '<joint type="ball"/>')
        (mujoco_models.ASSETS_DIR / "arm" / "unnamed.xml").write_text(unnamed)
        known = known_model("arm", _ARM_ENTRY, mujoco_models.ASSETS_DIR / "arm" / "unnamed.xml", "base")
        world = World(renders=False)
        launcher = _launcher(world)
        launcher.rebind()
        world.add("alpha", known, world.free_spot())
        launcher.unbind()
        launcher.rebind()

        world.add("bravo", known, world.free_spot())
        launcher.unbind()
        launcher.rebind()

        assert {robot.instance for robot in world.robots()} == {"alpha", "bravo"}

    def test_the_servo_gains_a_robot_was_tuned_with_survive_another_joining(self):
        """Its damping comes from the scene's home configuration, so a robot
        carried into a scene composed around a robot that joined keeps the
        control law it was standing under."""
        servoed = _known(arm_gains={"arm": [(20.0, 0.0), (20.0, 0.0)]})
        world = World(renders=False)
        launcher = _launcher(world)
        launcher.rebind()
        _stand(launcher, world, "alpha", servoed)
        model, data = launcher._model, launcher._data  # pylint: disable=W0212
        lift = model.actuator("alpha/lift").id
        tuned = float(model.actuator_biasprm[lift][2])
        # The damping is raised to critical against the joint's inertia.
        assert tuned < 0.0
        data.qpos[model.joint("alpha/lift").qposadr[0]] = 1.4
        data.qpos[model.joint("alpha/flex").qposadr[0]] = -1.4

        _stand(launcher, world, "bravo", servoed)

        moved = launcher._model  # pylint: disable=W0212
        assert float(moved.actuator_biasprm[moved.actuator("alpha/lift").id][2]) == pytest.approx(
            tuned
        )


@pytest.mark.usefixtures("mjviser")
class TestTheSceneOnTheViewersServer:
    """Every scene is shown on the one server the node bound when it set up,
    so the address the node announced shows the scene that stands now."""

    def test_a_scene_no_robot_stands_in_is_shown_too(self):
        viewer_server = _viewer_server()
        launcher = _watched(World(renders=False), viewer_server)

        launcher.rebind()

        assert isinstance(launcher._view, launcher_module._BrowserView)  # pylint: disable=W0212
        assert viewer_server.viser.nodes == ["/fixed_bodies"]

    def test_a_scene_composed_again_is_drawn_on_the_same_server_in_place_of_the_one_before(self):
        """A browser open on the viewer stays on the server it reached, and
        sees the scene a robot joined and nothing of the one before it."""
        viewer_server = _viewer_server()
        world = World(renders=False)
        launcher = _watched(world, viewer_server)
        launcher.rebind()

        _stand(launcher, world, "alpha")

        viser = viewer_server.viser
        assert viser.events == ["draw", "render", "wipe", "draw", "render"]
        assert not viser.stopped
        # What one scene leaves on the server, once: what mjviser shows and
        # registers, and the callback of the view itself.
        assert viser.nodes == ["/fixed_bodies"]
        assert viser.controls == ["Controls"]
        assert len(viser.click_callbacks) == 1
        assert len(viser.connect_callbacks) == 2

    def test_the_free_joint_of_a_robot_that_rolls_is_no_prop_to_reset(self):
        """The button that restages the scene's props puts each free joint
        it holds back at its spawn pose: a robot's own free joint is left to
        its wheels, so a scene standing no prop shows no button."""
        viewer_server = _viewer_server()
        world = World(renders=False)
        launcher = _watched(world, viewer_server)
        launcher.rebind()

        _stand(launcher, world, "india", _wheeled())

        assert viewer_server.viser.controls == ["Controls"]

    def test_a_scene_the_engine_lets_go_of_leaves_nothing_on_the_server(self):
        viewer_server = _viewer_server()
        launcher = _watched(World(renders=False), viewer_server)
        launcher.rebind()

        launcher.unbind()

        assert viewer_server.viser.left_on_it() == []
        assert not viewer_server.viser.stopped

    def test_a_browser_view_that_cannot_finish_wipes_what_it_drew(self, monkeypatch):
        """The server outlives the view, so what a view drew before it failed
        is taken off it: the view of the scene composed next starts on a
        server that holds nothing."""

        class _Unfinished(viewer_fakes.Viewer):
            def _setup_gui(self) -> None:
                super()._setup_gui()
                raise RuntimeError("the scene draws from no camera")

        monkeypatch.setitem(sys.modules, "mjviser", types.SimpleNamespace(Viewer=_Unfinished))
        viewer_server = _viewer_server()
        world = World(renders=False)
        world.add("alpha", _known(), world.free_spot())
        model = world.compose().compile()

        with pytest.raises(RuntimeError, match="the scene draws from no camera"):
            # The view fails before the scene is ever stepped through the
            # extension, so it needs none.
            launcher_module._BrowserView(  # pylint: disable=W0212
                model, mujoco.MjData(model), None, viewer_server, {"alpha"}
            )

        assert viewer_server.viser.left_on_it() == []
        assert not viewer_server.viser.stopped

    def test_a_browser_already_open_on_the_viewer_is_sent_the_scene_once_it_is_whole(self):
        """viser runs a connect callback at once for a browser already open,
        on a thread of its own, and the callback reads the scene: it is
        registered after the first frame, when the scene is whole."""
        viewer_server = _viewer_server()
        viewer_server.viser.clients.append("a browser")
        launcher = _watched(World(renders=False), viewer_server)

        launcher.rebind()

        assert viewer_server.viser.events == ["draw", "render", "refresh"]

    def test_the_server_stops_once_the_thread_that_steps_the_scene_ends(self):
        """The scene is wiped off the server before it stops, so nothing
        draws on a server that is gone."""
        viewer_server = _viewer_server()
        launcher = _watched(World(renders=False), viewer_server)
        launcher._stop.set()  # pylint: disable=W0212

        launcher.run()

        assert viewer_server.viser.events == ["draw", "render", "wipe", "stop"]

    def test_a_server_that_will_not_stop_does_not_hide_what_ended_the_loop(
        self, caplog, monkeypatch
    ):
        viewer_server = _viewer_server()
        launcher = _watched(World(renders=False), viewer_server)

        def compile_nothing():
            raise RuntimeError("this scene compiles nothing")

        def refuse():
            raise RuntimeError("this server will not stop")

        monkeypatch.setattr(launcher, "rebind", compile_nothing)
        monkeypatch.setattr(viewer_server, "stop", refuse)

        with caplog.at_level(logging.ERROR):
            with pytest.raises(RuntimeError, match="this scene compiles nothing"):
                launcher.run()

        assert "the viewer's server did not stop" in caplog.text


class _CountingScene:
    """The bridge extension as a view steps it: each step is counted, then
    handed to `after_step` with the count so far."""

    def __init__(self, after_step=lambda _steps: None) -> None:
        self.steps = 0
        self._after_step = after_step

    def step(self) -> None:
        self.steps += 1
        self._after_step(self.steps)


def _open_view(kind: str, scene: _CountingScene):
    """A view of `kind` on a scene one robot stands in, stepping `scene`."""
    world = World(renders=False)
    world.add("alpha", _known(), world.free_spot())
    model = world.compose().compile()
    data = mujoco.MjData(model)
    if kind == "none":
        return launcher_module._NoView(model, scene)  # pylint: disable=W0212
    return launcher_module._BrowserView(  # pylint: disable=W0212
        model, data, scene, _viewer_server(), {"alpha"}
    )


def _owe_a_whole_turn(view) -> None:
    """Puts the view's scene an hour behind the wall clock, far more than one
    turn of the loop may catch up, so the next turn owes exactly
    `_MAX_CATCHUP_STEPS` steps whatever the host's speed."""
    view._pacer._stepped_to_s -= 3600.0  # pylint: disable=W0212


@pytest.mark.usefixtures("mjviser")
@pytest.mark.parametrize("kind", ["none", "browser"])
class TestTheStepsAViewOwes:
    """A view takes the steps its scene owes the wall clock one at a time,
    and stops before the first one the scene is changing at. On a box that
    steps the scene slower than real time the whole of them lasts seconds,
    so a robot leaving waits for one step, not for all of them."""

    def test_every_step_owed_is_taken_while_the_scene_stands_as_it_is(self, kind):
        launcher = _launcher(World(renders=False))
        scene = _CountingScene()
        view = _open_view(kind, scene)
        _owe_a_whole_turn(view)

        view.step(launcher._scene_is_changing)  # pylint: disable=W0212

        assert scene.steps == launcher_module._MAX_CATCHUP_STEPS  # pylint: disable=W0212

    def test_a_change_asked_during_the_steps_owed_ends_them_at_the_next_one(self, kind):
        launcher = _launcher(World(renders=False))

        def ask_for_a_change(steps: int) -> None:
            if steps == 3:
                launcher._edits.submit(lambda: None)  # pylint: disable=W0212

        scene = _CountingScene(after_step=ask_for_a_change)
        view = _open_view(kind, scene)
        _owe_a_whole_turn(view)

        view.step(launcher._scene_is_changing)  # pylint: disable=W0212

        assert scene.steps == 3

    def test_an_engine_stopping_takes_no_step_owed(self, kind):
        launcher = _launcher(World(renders=False))
        launcher._stop.set()  # pylint: disable=W0212
        scene = _CountingScene()
        view = _open_view(kind, scene)
        _owe_a_whole_turn(view)

        view.step(launcher._scene_is_changing)  # pylint: disable=W0212

        assert scene.steps == 0


def test_a_new_views_first_steps_are_the_ones_the_wall_clock_owes(monkeypatch):
    """mjviser's tick steps the scene for the time since its last tick. A view
    is new at the launch and at each rebuild, and starts that time as it shows
    the scene, so its first turn steps what the wall clock owes since then."""
    now = [1_000.0]
    monkeypatch.setattr(
        launcher_module,
        "time",
        types.SimpleNamespace(
            perf_counter=lambda: now[0], monotonic=lambda: now[0], sleep=lambda _seconds: None
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "mjviser",
        types.SimpleNamespace(
            Viewer=lambda model, data, server, step_fn: viewer_fakes.TimedViewer(
                model, data, server, step_fn, lambda: now[0]
            )
        ),
    )
    scene = _CountingScene()
    view = _open_view("browser", scene)
    owed = 10
    now[0] += owed * view._model.opt.timestep  # pylint: disable=W0212

    view.step(lambda: False)

    # The wall clock's time reads within one step of `owed` steps.
    assert owed - 1 <= scene.steps <= owed, scene.steps
