"""The arm's joint limits as SO-101's description gives them, and what
JointLimits holds, clamps and names outside them."""

import math
import xml.etree.ElementTree as ET

import pytest
from conftest import WIDE_LIMITS
from robot_description_py import RobotDescription

from so101_backbone import limits
from so101_backbone.model import ARM, DESCRIPTION


def described_with_limits(bounds: dict[str, tuple[float, float]]) -> RobotDescription:
    """SO-101's description with the limit of each joint of `bounds` set to
    its (lower, upper), and the joint's value in each named state moved
    inside them, where the description holds every state."""
    robot = ET.fromstring(DESCRIPTION.urdf)
    for joint in robot.findall("joint"):
        if joint.get("name") in bounds:
            lower, upper = bounds[joint.get("name")]
            joint.find("limit").set("lower", repr(lower))
            joint.find("limit").set("upper", repr(upper))
    semantics = ET.fromstring(DESCRIPTION.srdf)
    for joint in semantics.findall("group_state/joint"):
        if joint.get("name") in bounds:
            lower, upper = bounds[joint.get("name")]
            joint.set("value", repr(min(max(float(joint.get("value")), lower), upper)))
    return RobotDescription.parse(
        ET.tostring(robot, encoding="unicode"), ET.tostring(semantics, encoding="unicode")
    )


def test_the_limits_are_the_urdfs_of_the_arms_joints_in_the_arms_order():
    bounds = {joint: (-0.5 - n, 0.5 + n) for n, joint in enumerate(ARM.joints)}
    parsed = limits.from_description(described_with_limits(bounds))
    assert parsed.lower == (-0.5, -1.5, -2.5, -3.5, -4.5)
    assert parsed.upper == (0.5, 1.5, 2.5, 3.5, 4.5)


@pytest.mark.parametrize(
    "lower, upper",
    [(0.5, 0.5), (-math.inf, 0.5), (-0.5, math.inf)],
    ids=repr,
)
def test_a_joint_whose_limits_are_no_finite_range_fails(lower, upper):
    described = described_with_limits({"wrist_flex": (lower, upper)})
    with pytest.raises(ValueError, match="URDF joint wrist_flex has unusable limits"):
        limits.from_description(described)


def test_contains_and_clamp():
    assert WIDE_LIMITS.contains((0.0, 1.0, -3.1, 3.1, 0.5))
    assert not WIDE_LIMITS.contains((0.0, 0.0, 0.0, 3.2, 0.0))
    assert WIDE_LIMITS.clamp((9.0, -9.0, 0.5, 0.0, 0.0)) == (3.1, -3.1, 0.5, 0.0, 0.0)


def test_a_joint_a_hair_outside_its_limits_is_not_outside_and_one_farther_out_is_named():
    assert WIDE_LIMITS.first_outside((0.0, 0.0, -3.1009, 3.1009, 0.0)) is None
    outside = WIDE_LIMITS.first_outside((0.0, 0.0, 0.0, 3.1011, -3.2))
    assert outside == limits.OutsideLimits(joint=3, position=3.1011, lower=-3.1, upper=3.1)
    assert str(outside) == "joint 4 at 3.1011 rad, outside its limits [-3.1, 3.1]"
    assert WIDE_LIMITS.first_outside((float("nan"),) * 5).joint == 0


def test_wrong_cardinality_is_refused():
    assert not WIDE_LIMITS.contains(())
    assert not WIDE_LIMITS.contains((0.0, 0.0))
    with pytest.raises(ValueError, match="one position per joint"):
        WIDE_LIMITS.clamp((0.0,))
    with pytest.raises(ValueError, match="one position per joint"):
        WIDE_LIMITS.first_outside((0.0,))
