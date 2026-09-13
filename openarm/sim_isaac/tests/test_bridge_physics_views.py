"""The bridge drops its articulation handles on request and re-creates them on
the next step, keeping the camera render products."""

import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest

_ROBOT_DIR = Path(__file__).resolve().parents[1] / "robots" / "openarm"


@pytest.fixture
def bridge(monkeypatch):
    # The typed transport is not under test; the module only needs its names.
    topics = ModuleType("sim_topics")
    topics.COLOR_CAMERA_SLOT_NAMES = ()
    topics.RGBD_CAMERA_SLOT_NAMES = ()
    topics.SimTopicIO = object
    monkeypatch.setitem(sys.modules, "sim_topics", topics)
    monkeypatch.syspath_prepend(str(_ROBOT_DIR))
    spec = importlib.util.spec_from_file_location("_bridge_under_test", _ROBOT_DIR / "bridge_extension.py")
    bridge_extension = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge_extension)
    monkeypatch.setattr(bridge_extension, "IsaacCameraSensor", Mock())
    monkeypatch.setattr(bridge_extension, "load_camera_configs", Mock(return_value=[]))
    monkeypatch.setattr(bridge_extension, "validate_camera_slots", Mock())
    extension = bridge_extension.IsaacBridgeExtension(Mock(), state_rate_hz=60, cameras_enabled=True)

    joints = [j for arm in extension._arms for j in arm["joints"]] + [
        f for g in extension._grippers for f in g["fingers"]
    ]
    articulation = Mock(spec=["setup", "teardown", "get_joint_names", "get_joint_limits", "get_joint_states"])
    articulation.setup.return_value = True
    articulation.get_joint_names.return_value = joints
    articulation.get_joint_limits.return_value = ([-0.01] * len(joints), [0.05] * len(joints))
    articulation.get_joint_states.return_value = None
    extension._articulation = articulation
    for table in (extension._arm_actuators, extension._gripper_actuators, extension._gripper_sensors):
        for key in table:
            ext = Mock(spec=["setup", "teardown", "write_targets", "set_force_limit", "get_gripper_state"])
            ext.setup.return_value = True
            ext.get_gripper_state.return_value = None
            table[key] = ext
    camera = Mock(spec=["setup", "teardown", "step"])
    camera.setup.return_value = True
    extension._camera_sensor = camera
    extension._io.latest_arm_command.return_value = None
    extension._io.latest_gripper_command.return_value = None
    monkeypatch.setattr(extension, "_engine_time_s", lambda: 0.0)
    assert extension.step() is None
    assert extension.is_ready
    return extension


def _physics_exts(extension):
    return [
        extension._articulation,
        *extension._arm_actuators.values(),
        *extension._gripper_actuators.values(),
        *extension._gripper_sensors.values(),
    ]


def test_invalidation_tears_down_articulation_handles_and_the_next_step_recreates_them(bridge, caplog):
    caplog.set_level(logging.INFO)
    bridge._applied_effort = {1: 5.0}
    for ext in _physics_exts(bridge):
        ext.setup.reset_mock()

    bridge.invalidate_physics_views()

    assert not bridge.is_ready
    for ext in _physics_exts(bridge):
        ext.teardown.assert_called_once_with()
    bridge._camera_sensor.teardown.assert_not_called()
    assert bridge._applied_effort == {}
    assert bridge._joint_index == {}
    assert bridge._gripper_travels == {}
    assert "re-initialises on its next step" in caplog.text

    bridge.step()

    assert bridge.is_ready
    for ext in _physics_exts(bridge):
        ext.setup.assert_called_once_with()
    assert bridge._joint_index
    assert set(bridge._gripper_travels) == {g["gripper_id"] for g in bridge._grippers}


def test_invalidation_before_readiness_is_silent_and_still_drops_partial_handles(bridge, caplog):
    caplog.set_level(logging.INFO)
    bridge.invalidate_physics_views()
    for ext in _physics_exts(bridge):
        ext.teardown.reset_mock()
    caplog.clear()

    bridge.invalidate_physics_views()

    for ext in _physics_exts(bridge):
        ext.teardown.assert_called_once_with()
    assert caplog.records == []
