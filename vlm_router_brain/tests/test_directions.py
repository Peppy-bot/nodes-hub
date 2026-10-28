"""Where the robot's axes point in a camera's picture, as clock hours."""

import math

import pytest

from vlm_router_brain.manipulators.vlm.directions import axes_in_picture, clock_hour, picture_direction


def quaternion(axis, degrees):
    """The unit quaternion [x, y, z, w] of a turn of `degrees` about `axis`."""
    half = math.radians(degrees) / 2
    s = math.sin(half)
    norm = math.sqrt(sum(a * a for a in axis))
    return (axis[0] / norm * s, axis[1] / norm * s, axis[2] / norm * s, math.cos(half))


def multiply(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


# A camera looking straight down (optical z along the robot's -z) with the
# picture's right along the robot's +y: its x axis is the robot's +y, its y
# axis (down in the picture) the robot's +x. Columns x = (0, 1, 0),
# y = (1, 0, 0), z = (0, 0, -1): a 180 degree turn about the axis (1, 1, 0).
LOOKING_DOWN = quaternion((1.0, 1.0, 0.0), 180.0)
# The chest camera of the brain_core tests: the optical frame is the robot
# frame, looking along +z, its right the robot's +x, its down the robot's +y.
IDENTITY = (0.0, 0.0, 0.0, 1.0)


@pytest.mark.parametrize("right, up, hour", [
    (0.0, 1.0, 12), (1.0, 0.0, 3), (0.0, -1.0, 6), (-1.0, 0.0, 9),
    (0.5, 0.866, 1), (0.866, 0.5, 2), (0.0, 0.0, 12),
])
def test_a_clock_hour_is_12_up_and_3_right(right, up, hour):
    assert clock_hour(right, up) == hour


def test_looking_down_with_the_right_along_y_reads_x_at_6_and_y_at_3():
    # Robot +x is the picture's down (6), robot +y its right (3).
    assert axes_in_picture(LOOKING_DOWN) == (6, 3)


def test_turning_that_camera_90_degrees_about_the_vertical_turns_the_hours_by_three():
    # The camera rolled a quarter turn about the robot's +z, counter-clockwise
    # seen from above: what was at 6 moves to 9, what was at 3 to 6.
    turned = multiply(quaternion((0.0, 0.0, 1.0), 90.0), LOOKING_DOWN)
    assert axes_in_picture(turned) == (9, 6)


def test_the_identity_camera_reads_x_at_3_and_y_at_6():
    assert axes_in_picture(IDENTITY) == (3, 6)


def test_a_direction_along_the_optical_axis_alone_reads_12():
    assert picture_direction(IDENTITY, (0.0, 0.0, 1.0)) == 12
