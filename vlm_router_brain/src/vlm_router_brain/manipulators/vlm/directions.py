"""Where the robot frame's axes point in a camera's picture, in words a
model reads: clock hours, 12 up and 3 right.

A camera's pose is its optical frame in the robot frame (camera_mounts: x
right in the picture, y down, z forward). A robot-frame direction turned
into the camera frame has a component across the picture, right and up,
and one along the optical axis, which the picture does not show. The hour
is the direction of the across part.
"""

from __future__ import annotations

import math

from brain_core.perception.camera import rotate
from brain_core.ports import Quat, Vec3

ROBOT_X: Vec3 = (1.0, 0.0, 0.0)
ROBOT_Y: Vec3 = (0.0, 1.0, 0.0)


def clock_hour(right: float, up: float) -> int:
    """The hour a clock hand pointing `right` and `up` shows: 12 up, 3
    right, 6 down, 9 left. A direction with no across part shows 12."""
    if abs(right) < 1e-12 and abs(up) < 1e-12:
        return 12
    hour = int(round(math.degrees(math.atan2(right, up)) / 30.0)) % 12
    return 12 if hour == 0 else hour


def picture_direction(camera_orientation: Quat, direction: Vec3) -> int:
    """The hour `direction`, a robot-frame direction, points toward in the
    picture of a camera whose optical frame has `camera_orientation` in the
    robot frame."""
    x, y, z, w = camera_orientation
    right, down, _along = rotate((-x, -y, -z, w), direction)
    return clock_hour(right, -down)


def axes_in_picture(camera_orientation: Quat) -> tuple[int, int]:
    """The hours the robot's +x and +y point toward in that camera's
    picture."""
    return picture_direction(camera_orientation, ROBOT_X), picture_direction(camera_orientation, ROBOT_Y)
