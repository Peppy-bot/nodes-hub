"""Letting go of the stage and taking it up again.

A view of a prim that is edited under it stops reading, so the bridge drops
every view before the stage changes and builds them again after. These cover
that the drop reaches every view of every robot, that what a view cached goes
with it, that the camera's render products, which ride USD prims rather than
PhysX views, are left alone, and that the state tick carries the spawned
objects' snapshot.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest

_ROBOT_DIR = Path(__file__).resolve().parents[1] / "robots" / "openarm"


def _view():
    """One PhysX-backed view, ready the moment it is asked."""
    ext = Mock(spec=["setup", "teardown", "write_targets", "set_force_limit",
                     "get_gripper_state"])
    ext.setup.return_value = True
    ext.get_gripper_state.return_value = None
    return ext


def _limbs_of(module, instance):
    """One robot's views, standing in for what RobotLimbs builds from a stage."""
    limbs = Mock(spec=["exts", "teardown", "setup", "ready", "robot", "articulation"])
    views = [_view() for _ in range(4)]
    limbs.exts.return_value = views
    limbs.setup.return_value = True
    limbs.ready = True
    limbs.robot = Mock(instance=instance)
    # Nothing to read yet: a state tick publishes no joint states for it.
    limbs.articulation.get_joint_states.return_value = None
    return limbs


@pytest.fixture
def bridge(monkeypatch):
    # The typed transport is not under test; the module only needs its names.
    topics = ModuleType("sim_topics")
    topics.COLOR_CAMERA_SLOT_NAMES = ()
    topics.RGBD_CAMERA_SLOT_NAMES = ()
    topics.SimTopicIO = object
    monkeypatch.setitem(sys.modules, "sim_topics", topics)
    monkeypatch.syspath_prepend(str(_ROBOT_DIR))
    spec = importlib.util.spec_from_file_location(
        "_bridge_under_test", _ROBOT_DIR / "bridge_extension.py"
    )
    bridge_extension = importlib.util.module_from_spec(spec)
    # Its dataclasses resolve postponed annotations through sys.modules.
    monkeypatch.setitem(sys.modules, spec.name, bridge_extension)
    spec.loader.exec_module(bridge_extension)
    monkeypatch.setattr(bridge_extension, "IsaacCameraSensor", Mock())

    extension = bridge_extension.IsaacBridgeExtension(
        Mock(), Mock(), Mock(), Mock(), Mock(), Mock(), state_rate_hz=60, cameras=[Mock()]
    )
    # Two robots on the stage, each with views of its own.
    extension._limbs = {
        "alpha": _limbs_of(bridge_extension, "alpha"),
        "bravo": _limbs_of(bridge_extension, "bravo"),
    }
    camera = Mock(spec=["setup", "teardown", "step"])
    camera.setup.return_value = True
    extension._camera_sensor = camera
    # Neither robot holds a seat and the layout names no limb, so a step
    # applies and reads nothing of them; what a state tick carries is the
    # objects' snapshot.
    extension._seats.standing.return_value = {}
    extension._layout.arms = []
    extension._layout.grippers = []
    monkeypatch.setattr(extension, "_engine_time_s", lambda: 0.0)
    return extension


def test_letting_go_drops_every_view_of_every_robot(bridge):
    held = list(bridge._limbs.values())
    assert bridge.is_ready

    bridge.unbind()

    assert not bridge.is_ready
    assert bridge._limbs == {}
    for limbs in held:
        limbs.teardown.assert_called_once_with()


def test_the_cameras_outlive_the_views_they_stand_beside(bridge):
    bridge.unbind()

    # Render products ride USD prims, not PhysX views, so a stage edit does
    # not cost them.
    bridge._camera_sensor.teardown.assert_not_called()


def test_taking_the_stage_up_again_builds_a_view_of_every_robot_on_it(bridge, monkeypatch):
    # What a robot's views are made of is RobotLimbs' business; this covers
    # that bind builds one set per robot the stage stands.
    built = []

    def limbs_for(robot, layout):
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


def test_each_state_tick_publishes_the_object_snapshot_it_captured(bridge):
    bridge._state_pacer = Mock(take_if_due=Mock(side_effect=[False, True, True, False]))
    captured = Mock(name="snapshot")
    bridge._objects.capture_object_states.side_effect = [captured, None]
    bridge._objects.reset_mock()
    bridge._io.reset_mock()

    for _ in range(4):
        bridge.step()

    assert bridge._objects.capture_object_states.call_count == 2, "once per state tick, never between"
    # A tick without a snapshot publishes none; the clock goes out ahead of
    # the state it stamps.
    bridge._io.publish_object_states.assert_called_once_with(captured)
    assert [name for name, _, _ in bridge._io.mock_calls if name in ("publish_clock_tick", "publish_object_states")] == [
        "publish_clock_tick", "publish_object_states", "publish_clock_tick",
    ]


def test_the_engine_clock_is_recorded_while_no_robot_reads_yet(bridge):
    # Every robot's views are still being built: the step records the
    # engine clock for the captures that stamp from it, and nothing else.
    for limbs in bridge._limbs.values():
        limbs.setup.return_value = False
        limbs.ready = False
    bridge._objects.reset_mock()
    bridge._io.reset_mock()

    bridge.step()

    assert not bridge.is_ready
    bridge._io.record_engine_time.assert_called_once_with(0.0)
    bridge._objects.capture_object_states.assert_not_called()
    bridge._io.publish_clock_tick.assert_not_called()
