import importlib

import pytest
from control_core_py.runtime import SetpointError
from so101 import units

from so101_backbone import model
from so101_backbone.wire import parse_pose_setpoint


def test_pose_validated():
    position, orientation = parse_pose_setpoint([0.1, 0.2, 0.3], [0.0, 0.0, 0.0, 1.0])
    assert position == (0.1, 0.2, 0.3)
    assert orientation == (0.0, 0.0, 0.0, 1.0)
    with pytest.raises(SetpointError, match="unit quaternion"):
        parse_pose_setpoint([0.1, 0.2, 0.3], [0.0, 0.0, 0.0, 0.5])


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
