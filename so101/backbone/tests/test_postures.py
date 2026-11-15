"""The postures: the FK claims they are documented with, and the real
solver taking its planning bars at them."""

import math

from so101_backbone import postures
from so101_backbone.kinematics import Kinematics
from so101_backbone.model import urdf_path


def test_home_is_lower_and_closer_in_than_ready():
    kinematics = Kinematics(urdf_path())
    home_position, _ = kinematics.forward_kinematics(postures.HOME_POSITIONS_RAD)
    ready_position, _ = kinematics.forward_kinematics(postures.READY_POSITIONS_RAD)
    home_reach = math.hypot(home_position[0], home_position[1])
    ready_reach = math.hypot(ready_position[0], ready_position[1])
    # Home folds the arm toward the base: closer in and lower than the working pose.
    assert home_reach < ready_reach
    assert home_position[2] < ready_position[2]


def test_the_solver_accepts_the_planning_bars():
    """The fakes in this suite accept the tolerance keywords by construction,
    so nothing else here would notice a solver whose signature lacks them.
    The real node would TypeError on its first move_arm instead."""
    solver = Kinematics(urdf_path())
    target = solver.forward_kinematics(postures.READY_POSITIONS_RAD)
    solver.inverse_kinematics(
        postures.READY_POSITIONS_RAD,
        *target,
        position_tolerance_m=0.01,
        orientation_tolerance_rad=0.5,
    )


def test_the_ready_grasp_point_stands_in_front_of_the_base_in_the_robot_frame():
    """The contracts define the robot frame with +x the way the robot
    faces, and this backbone reports poses in the URDF's base_link frame as
    they are: so the working posture's grasp point, out in front of the
    robot, must have a positive x there, and stand within 10 mm of the
    robot's centre line."""
    kinematics = Kinematics(urdf_path())
    ready_position, _ = kinematics.forward_kinematics(postures.READY_POSITIONS_RAD)
    assert ready_position[0] > 0.3
    assert abs(ready_position[1]) < 0.01
