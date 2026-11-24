import importlib

import pytest
from control_core_py.runtime import SetpointError
from so101 import units

from so101_backbone import model
from so101_backbone.wire import parse_pose_setpoint


def test_a_pose_setpoint_is_a_pose_in_its_frame_at_its_time():
    setpoint = parse_pose_setpoint("odom", 12.5, [0.1, 0.2, 0.3], [0.0, 0.0, 0.0, 1.0])
    assert (setpoint.frame, setpoint.timestamp) == ("odom", 12.5)
    assert setpoint.value.position.position == (0.1, 0.2, 0.3)
    assert setpoint.value.orientation.xyzw == (0.0, 0.0, 0.0, 1.0)


def test_a_pose_setpoint_with_no_frame_lies_in_the_robot_frame():
    setpoint = parse_pose_setpoint(None, 12.5, [0.1, 0.2, 0.3], [0.0, 0.0, 0.0, 1.0])
    assert (setpoint.frame, setpoint.timestamp) == (model.ROBOT_FRAME, 12.5)


@pytest.mark.parametrize(
    ("frame", "position", "orientation", "refusal"),
    [
        ("odom", [0.1, 0.2, 0.3], [0.0, 0.0, 0.0, 0.5], "squared norm"),
        ("odom", [0.1, float("nan"), 0.3], [0.0, 0.0, 0.0, 1.0], "must be finite"),
        ("", [0.1, 0.2, 0.3], [0.0, 0.0, 0.0, 1.0], "the frame is empty"),
    ],
)
def test_a_pose_setpoint_no_contract_takes_is_refused(frame, position, orientation, refusal):
    with pytest.raises(SetpointError, match=refusal):
        parse_pose_setpoint(frame, None, position, orientation)


def test_an_arm_the_followers_wire_does_not_carry_in_its_order_stops_the_launch(monkeypatch):
    """The follower's joint messages carry the so101 library's bus order,
    which the backbone reads as the arm's joints, so the model the backbone
    loads as it launches refuses a description whose arm moves another
    order."""
    monkeypatch.setattr(units, "JOINT_NAMES", tuple(reversed(units.JOINT_NAMES)))
    try:
        with pytest.raises(RuntimeError, match="and the follower's wire carries"):
            importlib.reload(model)
    finally:
        monkeypatch.undo()
        importlib.reload(model)
