"""What a robot joining or leaving does to the scene already running: it is
composed again, and the robots that stay carry onto it what they were doing,
while the one that arrived starts in its model's posture. The scenes are
real (tiny) MJCF composed and compiled by MuJoCo.
"""

from __future__ import annotations

import logging
import sys
import threading
import types
from pathlib import Path

import mujoco
import mujoco.viewer
import pytest
from sim_robot_core.models import EngineModel, parse_entry

import runtime_fakes  # noqa: F401  pylint: disable=W0611  (installs the runtime the launcher imports)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))

import mujoco_models  # noqa: E402  pylint: disable=C0413
import _launcher as launcher_module  # noqa: E402  pylint: disable=C0413
from _launcher import SimLauncher, _NoView  # noqa: E402  pylint: disable=C0413
from edits import Edits  # noqa: E402  pylint: disable=C0413
from exts import camera_sensor  # noqa: E402  pylint: disable=C0413
from mujoco_models import parse  # noqa: E402  pylint: disable=C0413
from world import World  # noqa: E402  pylint: disable=C0413

_SCENE = "arm/arm.xml"
_STATE_RATE_HZ = 50
_POSTURE = {"lift": 0.25, "flex": -0.5}

# A two-joint arm with a prop beside it, so a rebuild has a free joint to
# carry as well as the robot's own.
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
    </body>
    <body name="prop" pos="0.4 0 0.3">
      <freejoint name="prop"/>
      <geom type="box" size="0.02 0.02 0.02" mass="0.05"/>
    </body>
  </worldbody>
  <actuator>
    <position name="lift" joint="lift" kp="20" kv="1" ctrlrange="-1.5 1.5"/>
    <position name="flex" joint="flex" kp="20" kv="1" ctrlrange="-1.5 1.5"/>
  </actuator>
</mujoco>"""
_ARM_ENTRY = {
    "arms": [{"name": "arm", "joints": ["lift", "flex"]}],
    "start_posture": _POSTURE,
}
_CAMERA_ARM_ENTRY = {
    **_ARM_ENTRY,
    "cameras": [{
        "name": "wrist",
        "parent_link": "fore",
        "pos": [0.0, 0.0, 0.1],
        "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        "fovy_deg": 60.0,
        "color": {"width": 8, "height": 4},
        "fps": 10,
    }],
}


class FakeIO:
    """The transport the bridge publishes on, which these tests only need to
    swallow."""

    def __init__(self) -> None:
        self.engine_times = []

    def record_engine_time(self, seconds):
        self.engine_times.append(seconds)

    def publish_clock_tick(self):
        return None

    def publish_arm_states(self, robot, arm, positions, velocities):
        return None

    def publish_gripper_states(self, robot, gripper, opening):
        return None

    def latest_arm_command(self, robot, arm):
        return None

    def latest_gripper_command(self, robot, gripper):
        return None


class _Unwatched(SimLauncher):
    """A launcher whose scene nobody watches, so these tests need neither a
    browser nor a display."""

    def _open_view(self, model, data, extension):
        return _NoView(model, extension)


@pytest.fixture(name="assets", autouse=True)
def assets_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(mujoco_models, "ASSETS_DIR", tmp_path)
    scene = tmp_path / _SCENE
    scene.parent.mkdir()
    scene.write_text(_ARM)
    return tmp_path


def _known(entry=None, **engine):
    parsed = parse_entry("arm", "arm.json5", entry or _ARM_ENTRY)
    return parse(EngineModel(entry=parsed, engine={"scene": _SCENE, **engine}))


def _launcher(world: World) -> SimLauncher:
    return _Unwatched(
        world,
        Edits(),
        threading.Event(),
        FakeIO(),
        _STATE_RATE_HZ,
        headless=True,
        viewer_host="0.0.0.0",
        viewer_port=8080,
    )


def _frame_ids(launcher: SimLauncher) -> dict:
    """Each standing camera's frame counter in the scene the engine holds."""
    return launcher._extension._camera_sensor.counters()  # pylint: disable=W0212


def _stand(launcher: SimLauncher, world: World, name: str) -> None:
    """A robot joining, the way an edit from the contract server does it."""
    world.add(name, _known(), world.free_spot())
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
        world = World(head_camera_pack=None, renders=False)
        launcher = _launcher(world)

        launcher.rebind()

        assert launcher._model.njnt == 0  # pylint: disable=W0212
        launcher._view.step(lambda: False)  # pylint: disable=W0212

    def test_a_robot_that_joins_starts_in_its_models_posture(self):
        world = World(head_camera_pack=None, renders=False)
        launcher = _launcher(world)
        launcher.rebind()

        _stand(launcher, world, "alpha")

        assert _qpos(launcher, "alpha/lift") == pytest.approx(_POSTURE["lift"])
        assert _ctrl(launcher, "alpha/lift") == pytest.approx(_POSTURE["lift"])

    def test_a_robot_standing_keeps_its_own_state_when_another_joins(self):
        """Its joints, their speeds and the targets its actuators hold are
        carried onto the scene composed around the robot that arrived."""
        world = World(head_camera_pack=None, renders=False)
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

    def test_a_prop_a_robot_carries_keeps_its_pose_across_a_rebuild(self):
        """A free joint carries seven positions and six speeds, which is a
        width the model states and the carry reads."""
        world = World(head_camera_pack=None, renders=False)
        launcher = _launcher(world)
        launcher.rebind()
        _stand(launcher, world, "alpha")
        model, data = launcher._model, launcher._data  # pylint: disable=W0212
        address = model.joint("alpha/prop").qposadr[0]
        data.qpos[address : address + 7] = [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]

        _stand(launcher, world, "bravo")

        moved, moved_data = launcher._model, launcher._data  # pylint: disable=W0212
        carried = moved.joint("alpha/prop").qposadr[0]
        assert moved_data.qpos[carried : carried + 3].tolist() == pytest.approx([0.1, 0.2, 0.3])

    def test_a_robot_that_leaves_takes_only_itself_out(self):
        world = World(head_camera_pack=None, renders=False)
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
        world = World(head_camera_pack=None, renders=False)
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
        world = World(head_camera_pack=None, renders=False)
        edits = Edits()
        stop = threading.Event()
        launcher = _Unwatched(
            world, edits, stop, FakeIO(), _STATE_RATE_HZ,
            headless=True, viewer_host="0.0.0.0", viewer_port=8080,
        )
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
        world = World(head_camera_pack=None, renders=False)
        edits = Edits()
        stop = threading.Event()
        stop.set()
        launcher = _Unwatched(
            world, edits, stop, FakeIO(), _STATE_RATE_HZ,
            headless=True, viewer_host="0.0.0.0", viewer_port=8080,
        )
        waiting = edits.submit(lambda: None)

        launcher.run()

        with pytest.raises(RuntimeError, match="the engine stopped"):
            waiting.result(timeout=0)

    def test_a_camera_counts_its_frames_on_across_a_scene_composed_again(self, monkeypatch):
        """A consumer pairs a frame with its depth by the id it carries, so
        the count runs on into the scene its robot is carried onto."""
        monkeypatch.setattr(camera_sensor.MujocoCameraSensor, "start", lambda self: None)
        world = World(head_camera_pack=None, renders=True)
        launcher = _launcher(world)
        world.add("alpha", _known(_CAMERA_ARM_ENTRY), world.free_spot())
        launcher.rebind()
        counted = [_frame_ids(launcher)[("alpha", "wrist")].next() for _ in range(3)]

        world.add("delta", _known(_CAMERA_ARM_ENTRY), world.free_spot())
        launcher.unbind()
        launcher.rebind()

        assert counted == [0, 1, 2]
        assert _frame_ids(launcher)[("alpha", "wrist")].next() == 3
        # The robot that just joined starts its own count.
        assert _frame_ids(launcher)[("delta", "wrist")].next() == 0

    def test_a_scene_that_does_not_compose_takes_the_node_down(self, monkeypatch):
        """An engine holding no scene steps nothing and publishes no clock.
        The node stops saying so, and the robots standing in it join the
        engine that comes back."""
        world = World(head_camera_pack=None, renders=False)
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
        world = World(head_camera_pack=None, renders=False)
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

    def test_a_scene_no_robot_stands_in_is_shown_to_nobody(self):
        launcher = _launcher(World(head_camera_pack=None, renders=False))

        launcher.rebind()

        assert isinstance(launcher._view, _NoView)  # pylint: disable=W0212

    def test_a_view_that_cannot_open_leaves_the_scene_stepping(self, caplog, monkeypatch):
        """The robots standing keep their physics, their state and the clock
        this engine publishes; the operator is told why there is no picture."""

        def refuse(*_args, **_kwargs):
            raise RuntimeError("the viewer port is taken")

        monkeypatch.setattr(launcher_module, "_BrowserView", refuse)
        world = World(head_camera_pack=None, renders=False)
        launcher = SimLauncher(
            world, Edits(), threading.Event(), FakeIO(), _STATE_RATE_HZ,
            headless=True, viewer_host="0.0.0.0", viewer_port=8080,
        )
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

    def test_a_browser_view_that_cannot_finish_hands_back_its_server(self, monkeypatch):
        """The server holds the viewer's address and its own non-daemon
        threads from the moment it is built, so a view that cannot finish
        gives them up: the view of the scene composed next takes the same
        address, and the process still exits once the sim loop ends."""
        servers = []

        class _Server:
            def __init__(self, host, port):
                self.host, self.port, self.stopped = host, port, False
                servers.append(self)

            def stop(self):
                self.stopped = True

        def _refuse(*_args, **_kwargs):
            raise RuntimeError("the scene draws from no camera")

        monkeypatch.setitem(sys.modules, "viser", types.SimpleNamespace(ViserServer=_Server))
        monkeypatch.setitem(sys.modules, "mjviser", types.SimpleNamespace(Viewer=_refuse))
        world = World(head_camera_pack=None, renders=False)
        world.add("alpha", _known(), world.free_spot())
        model = world.compose().compile()

        with pytest.raises(RuntimeError, match="the scene draws from no camera"):
            # The view fails before the scene is ever stepped through the
            # extension, so it needs none.
            launcher_module._BrowserView(  # pylint: disable=W0212
                model, mujoco.MjData(model), None, "0.0.0.0", 8080
            )

        assert [server.stopped for server in servers] == [True]

    def test_a_scene_with_a_robot_in_it_is_shown_in_a_browser_when_headless(self, monkeypatch):
        """Which view a scene is watched through is the engine's launch: a
        browser one where it runs headless, a window where it does not."""
        opened = []
        monkeypatch.setattr(
            launcher_module, "_BrowserView", lambda *args: opened.append("browser") or _NoView()
        )
        monkeypatch.setattr(
            launcher_module, "_WindowView", lambda *args: opened.append("window") or _NoView()
        )
        for headless in (True, False):
            world = World(head_camera_pack=None, renders=False)
            launcher = SimLauncher(
                world, Edits(), threading.Event(), FakeIO(), _STATE_RATE_HZ,
                headless=headless, viewer_host="0.0.0.0", viewer_port=8080,
            )
            launcher.rebind()  # nothing stands: no view of either kind
            world.add("alpha", _known(), world.free_spot())
            launcher.rebind()

        assert opened == ["browser", "window"]

    def test_a_joint_the_scene_leaves_unnamed_is_left_where_its_model_puts_it(self):
        """A model may leave a joint unnamed, and an unnamed joint is one the
        scene before this one has no name to look up."""
        unnamed = _ARM.replace('<freejoint name="prop"/>', "<freejoint/>")
        (mujoco_models.ASSETS_DIR / "arm" / "unnamed.xml").write_text(unnamed)
        known = parse(
            EngineModel(
                entry=parse_entry("arm", "arm.json5", _ARM_ENTRY),
                engine={"scene": "arm/unnamed.xml"},
            )
        )
        world = World(head_camera_pack=None, renders=False)
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
        world = World(head_camera_pack=None, renders=False)
        launcher = _launcher(world)
        launcher.rebind()
        _stand(launcher, world, "alpha")
        model, data = launcher._model, launcher._data  # pylint: disable=W0212
        lift = model.actuator("alpha/lift").id
        tuned = float(model.actuator_biasprm[lift][2])
        data.qpos[model.joint("alpha/lift").qposadr[0]] = 1.4
        data.qpos[model.joint("alpha/flex").qposadr[0]] = -1.4

        _stand(launcher, world, "bravo")

        moved = launcher._model  # pylint: disable=W0212
        assert float(moved.actuator_biasprm[moved.actuator("alpha/lift").id][2]) == pytest.approx(
            tuned
        )


class _CountingScene:
    """The bridge extension as a view steps it: each step is counted, then
    handed to `after_step` with the count so far."""

    def __init__(self, after_step=lambda _steps: None) -> None:
        self.steps = 0
        self._after_step = after_step

    def step(self) -> None:
        self.steps += 1
        self._after_step(self.steps)


class _Window:
    """MuJoCo's passive window, open and shown to nobody."""

    def is_running(self) -> bool:
        return True

    def sync(self) -> None:
        return None

    def close(self) -> None:
        return None


class _Server:
    """viser's server as the browser view uses it, serving nobody."""

    def __init__(self, host, port) -> None:
        self.host, self.port = host, port
        self.gui = types.SimpleNamespace(
            add_button=lambda _label: types.SimpleNamespace(on_click=lambda handler: handler)
        )

    def on_client_connect(self, handler):
        return handler

    def stop(self) -> None:
        return None


class _Viewer:
    """mjviser's viewer as the browser view drives it: a tick steps the scene
    through the callback the view handed it, and nothing is drawn."""

    def __init__(self, model, data, server, step_fn) -> None:
        self._model, self._data, self._server, self._step_fn = model, data, server, step_fn

    def _tick(self) -> None:
        self._step_fn(self._model, self._data)

    def _setup_gui(self) -> None:
        return None

    def _render(self) -> None:
        return None

    def _refresh_scene_from_gui(self) -> None:
        return None


def _open_view(kind: str, scene: _CountingScene, monkeypatch):
    """A view of `kind` on a scene one robot stands in, stepping `scene`,
    with its window or its browser server faked."""
    world = World(head_camera_pack=None, renders=False)
    world.add("alpha", _known(), world.free_spot())
    model = world.compose().compile()
    data = mujoco.MjData(model)
    if kind == "none":
        return launcher_module._NoView(model, scene)  # pylint: disable=W0212
    if kind == "window":
        monkeypatch.setattr(mujoco.viewer, "launch_passive", lambda _model, _data: _Window())
        return launcher_module._WindowView(model, data, scene)  # pylint: disable=W0212
    monkeypatch.setitem(sys.modules, "viser", types.SimpleNamespace(ViserServer=_Server))
    monkeypatch.setitem(sys.modules, "mjviser", types.SimpleNamespace(Viewer=_Viewer))
    return launcher_module._BrowserView(  # pylint: disable=W0212
        model, data, scene, "0.0.0.0", 8080
    )


def _owe_a_whole_turn(view) -> None:
    """Puts the view's scene an hour behind the wall clock, far more than one
    turn of the loop may catch up, so the next turn owes exactly
    `_MAX_CATCHUP_STEPS` steps whatever the host's speed."""
    view._pacer._stepped_to_s -= 3600.0  # pylint: disable=W0212


@pytest.mark.parametrize("kind", ["none", "window", "browser"])
class TestTheStepsAViewOwes:
    """A view takes the steps its scene owes the wall clock one at a time,
    and stops before the first one the scene is changing at. On a box that
    steps the scene slower than real time the whole of them lasts seconds,
    so a robot leaving waits for one step, not for all of them."""

    def test_every_step_owed_is_taken_while_the_scene_stands_as_it_is(self, kind, monkeypatch):
        launcher = _launcher(World(head_camera_pack=None, renders=False))
        scene = _CountingScene()
        view = _open_view(kind, scene, monkeypatch)
        _owe_a_whole_turn(view)

        view.step(launcher._scene_is_changing)  # pylint: disable=W0212

        assert scene.steps == launcher_module._MAX_CATCHUP_STEPS  # pylint: disable=W0212

    def test_a_change_asked_during_the_steps_owed_ends_them_at_the_next_one(
        self, kind, monkeypatch
    ):
        launcher = _launcher(World(head_camera_pack=None, renders=False))

        def ask_for_a_change(steps: int) -> None:
            if steps == 3:
                launcher._edits.submit(lambda: None)  # pylint: disable=W0212

        scene = _CountingScene(after_step=ask_for_a_change)
        view = _open_view(kind, scene, monkeypatch)
        _owe_a_whole_turn(view)

        view.step(launcher._scene_is_changing)  # pylint: disable=W0212

        assert scene.steps == 3

    def test_an_engine_stopping_takes_no_step_owed(self, kind, monkeypatch):
        launcher = _launcher(World(head_camera_pack=None, renders=False))
        launcher._stop.set()  # pylint: disable=W0212
        scene = _CountingScene()
        view = _open_view(kind, scene, monkeypatch)
        _owe_a_whole_turn(view)

        view.step(launcher._scene_is_changing)  # pylint: disable=W0212

        assert scene.steps == 0
