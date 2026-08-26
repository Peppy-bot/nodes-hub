"""The posture constants hold against the embedded kinematics URDF, the
same validation setup runs at startup, plus the FK claims the poses are
documented with."""

import math

from so101_description import limits as limits_mod
from so101_description import postures
from so101_description.kinematics import Kinematics
from so101_description.model import KINEMATICS_URDF_PATH
from so101_description.units import NUM_JOINTS


def test_postures_are_five_entry_joint_targets():
    for posture in (postures.HOME_POSITIONS_RAD, postures.READY_POSITIONS_RAD):
        assert len(posture) == NUM_JOINTS


def test_ready_is_the_calibration_midpoint():
    assert postures.READY_POSITIONS_RAD == (0.0,) * NUM_JOINTS


def test_postures_sit_inside_the_embedded_urdf_limits():
    limits = limits_mod.from_urdf(KINEMATICS_URDF_PATH)
    assert limits.contains(postures.HOME_POSITIONS_RAD)
    assert limits.contains(postures.READY_POSITIONS_RAD)


def test_home_is_collapsed_low_and_close_relative_to_ready():
    kinematics = Kinematics(KINEMATICS_URDF_PATH)
    home_position, _ = kinematics.forward_kinematics(postures.HOME_POSITIONS_RAD)
    ready_position, _ = kinematics.forward_kinematics(postures.READY_POSITIONS_RAD)
    home_reach = math.hypot(home_position[0], home_position[1])
    ready_reach = math.hypot(ready_position[0], ready_position[1])
    # Park is folded onto the base: closer in and lower than the working pose.
    assert home_reach < ready_reach
    assert home_position[2] < ready_position[2]
