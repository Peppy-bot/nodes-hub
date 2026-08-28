import pytest

from so101_backbone.setpoints import SetpointError, parse_pose_setpoint


def test_pose_validated():
    position, orientation = parse_pose_setpoint([0.1, 0.2, 0.3], [0.0, 0.0, 0.0, 1.0])
    assert position == (0.1, 0.2, 0.3)
    assert orientation == (0.0, 0.0, 0.0, 1.0)
    with pytest.raises(SetpointError, match="unit quaternion"):
        parse_pose_setpoint([0.1, 0.2, 0.3], [0.0, 0.0, 0.0, 0.5])
