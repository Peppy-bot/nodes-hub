"""Letting go of the stage and taking it up again, and what each state tick
carries.

A view of a prim that is edited under it stops reading, so the bridge drops
every view before the stage changes and builds them again after. These cover
that the drop reaches every view of every robot, that what a view cached goes
with it, that the camera's render products, which ride USD prims, are left
alone, and that every state tick carries the spawned objects' snapshot.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest

_ENGINE_DIR = Path(__file__).resolve().parents[1] / "engine"


def _view():
    """One PhysX-backed view, ready the moment it is asked."""
    ext = Mock(spec=["setup", "teardown", "write_targets", "set_force_limit",
                     "get_gripper_state"])
    ext.setup.return_value = True
    ext.get_gripper_state.return_value = None
    return ext


def _limbs_of(module, instance):
    """One robot's views, standing in for what RobotLimbs builds from a stage."""
    limbs = Mock(spec=["exts", "teardown", "setup", "ready", "robot"])
    views = [_view() for _ in range(4)]
    limbs.exts.return_value = views
    limbs.setup.return_value = True
    limbs.ready = True
    limbs.robot = Mock(instance=instance)
    return limbs


@pytest.fixture
def bridge(monkeypatch):
    # The typed transport is not under test; the module only needs its names.
    topics = ModuleType("sim_topics")
    topics.SimTopicIO = object
    monkeypatch.setitem(sys.modules, "sim_topics", topics)
    monkeypatch.syspath_prepend(str(_ENGINE_DIR))
    spec = importlib.util.spec_from_file_location(
        "_bridge_under_test", _ENGINE_DIR / "bridge_extension.py"
    )
    bridge_extension = importlib.util.module_from_spec(spec)
    # Its dataclasses resolve postponed annotations through sys.modules.
    monkeypatch.setitem(sys.modules, spec.name, bridge_extension)
    spec.loader.exec_module(bridge_extension)
    monkeypatch.setattr(bridge_extension, "IsaacCameraSensor", Mock())

    extension = bridge_extension.IsaacBridgeExtension(
        Mock(), Mock(), Mock(), Mock(), state_rate_hz=60, renders=True
    )
    # Two robots on the stage, each with views of its own.
    extension._limbs = {
        "alpha": _limbs_of(bridge_extension, "alpha"),
        "bravo": _limbs_of(bridge_extension, "bravo"),
    }
    camera = Mock(spec=["setup", "teardown", "step"])
    camera.setup.return_value = True
    extension._camera_sensors = {"alpha": camera}
    # Which robots pair a camera is the transport's answer; this fixture
    # stands two robots and rigs alpha unless a test says otherwise.
    extension._world.robots.return_value = [Mock(instance="alpha"), Mock(instance="bravo")]
    extension._io.camera_robots.return_value = {"alpha"}
    return extension


def test_letting_go_drops_every_view_of_every_robot(bridge):
    held = list(bridge._limbs.values())
    assert bridge.is_ready

    bridge.unbind()

    assert not bridge.is_ready
    assert bridge._limbs == {}
    for limbs in held:
        limbs.teardown.assert_called_once_with()


def test_the_cameras_outlive_the_views_they_stand_beside(bridge, monkeypatch):
    module = sys.modules["_bridge_under_test"]
    monkeypatch.setattr(module, "RobotLimbs", lambda robot: _limbs_of(None, robot.instance))
    held_camera = bridge._camera_sensors["alpha"]
    bridge.unbind()
    bridge.bind()

    # Render products ride USD prims, so a stage edit costs a rigged robot
    # neither its rig nor a second one.
    held_camera.teardown.assert_not_called()
    assert bridge._camera_sensors["alpha"] is held_camera
    module.IsaacCameraSensor.assert_not_called()


def test_the_rig_of_a_robot_gone_from_the_stage_is_dropped(bridge):
    held_camera = bridge._camera_sensors["alpha"]
    bridge._world.robots.return_value = [Mock(instance="bravo")]

    bridge._reconcile_rigs()

    held_camera.teardown.assert_called_once_with()
    assert "alpha" not in bridge._camera_sensors


def test_taking_the_stage_up_again_builds_a_view_of_every_robot_on_it(bridge, monkeypatch):
    # What a robot's views are made of is RobotLimbs' business; this covers
    # that bind builds one set per robot the stage stands.
    built = []

    def limbs_for(robot):
        built.append(robot.instance)
        return _limbs_of(None, robot.instance)

    monkeypatch.setattr(
        sys.modules["_bridge_under_test"], "RobotLimbs", limbs_for
    )
    bridge._world.robots.return_value = [
        Mock(instance="alpha"), Mock(instance="bravo"), Mock(instance="charlie")
    ]
    bridge.unbind()

    bridge.bind()

    assert built == ["alpha", "bravo", "charlie"]
    assert sorted(bridge._limbs) == ["alpha", "bravo", "charlie"]


def test_resolving_steps_the_app_until_every_view_reads(bridge):
    alpha, bravo = _limbs_of(None, "alpha"), _limbs_of(None, "bravo")
    alpha.setup.side_effect = [False, False, True, True]
    bridge._limbs = {"alpha": alpha, "bravo": bravo}
    update = Mock()

    bridge.resolve(update, 10)

    assert update.call_count == 2
    assert alpha.setup.call_count == 3


def test_resolving_raises_for_a_robot_whose_joints_the_engine_cannot_drive(bridge):
    alpha = _limbs_of(None, "alpha")
    alpha.setup.side_effect = RuntimeError("the so101 entry names joints not on the articulation")
    bridge._limbs = {"alpha": alpha}

    with pytest.raises(RuntimeError, match="joints not on the articulation"):
        bridge.resolve(Mock(), 10)


def test_a_rig_follows_its_robots_camera_pair(bridge, monkeypatch):
    # alpha's rig stands from the fixture; bravo holds no camera pair yet.
    module = sys.modules["_bridge_under_test"]
    built = []
    module.IsaacCameraSensor.side_effect = lambda instance, *_: built.append(instance) or Mock(
        spec=["setup", "teardown", "step"]
    )
    monkeypatch.setattr(type(bridge), "_engine_time_s", lambda self: 0.0)
    bridge._state_pacer = Mock(take_if_due=Mock(return_value=True))
    bridge._objects.capture_object_states.return_value = None
    bridge._robots.standing.return_value = {}
    alpha_rig = bridge._camera_sensors["alpha"]

    # bravo's relays pair in after it stood: its rig is mounted on the
    # next state tick, alpha's is left as it is.
    bridge._io.camera_robots.return_value = {"alpha", "bravo"}
    bridge.step()
    assert built == ["bravo"]
    assert bridge._camera_sensors["alpha"] is alpha_rig

    # alpha's relays stop: its rig goes, bravo's stays.
    bridge._io.camera_robots.return_value = {"bravo"}
    bridge.step()
    alpha_rig.teardown.assert_called_once_with()
    assert sorted(bridge._camera_sensors) == ["bravo"]
    assert built == ["bravo"], "a rig is mounted once per pair"


def test_each_state_tick_publishes_the_object_snapshot_it_captured(bridge, monkeypatch):
    monkeypatch.setattr(type(bridge), "_engine_time_s", lambda self: 0.0)
    bridge._state_pacer = Mock(take_if_due=Mock(side_effect=[False, True, True, False]))
    captured = Mock(name="snapshot")
    bridge._objects.capture_object_states.side_effect = [captured, None]
    bridge._objects.reset_mock()
    bridge._io.reset_mock()
    bridge._robots.standing.return_value = {}

    for _ in range(4):
        bridge.step()

    assert bridge._objects.capture_object_states.call_count == 2, (
        "once per state tick, never between"
    )
    # A tick without a snapshot publishes none; the clock goes out ahead of
    # the state it stamps.
    bridge._io.publish_object_states.assert_called_once_with(captured)
    assert [
        name
        for name, _, _ in bridge._io.mock_calls
        if name in ("publish_clock_tick", "publish_object_states")
    ] == ["publish_clock_tick", "publish_object_states", "publish_clock_tick"]


def test_the_engine_clock_is_recorded_while_the_bridge_takes_the_stage_up_again(
    bridge, monkeypatch
):
    monkeypatch.setattr(type(bridge), "_engine_time_s", lambda self: 4.5)
    bridge.unbind()
    bridge._objects.reset_mock()
    bridge._io.reset_mock()

    bridge.step()

    assert not bridge.is_ready
    bridge._io.record_engine_time.assert_called_once_with(4.5)
    bridge._objects.capture_object_states.assert_not_called()
