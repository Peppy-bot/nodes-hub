"""The arm's joint position limits, read from SO-101's description at startup.

The backbone enforces them: goals more than LIMIT_SLACK_RAD beyond a limit
are rejected and closer ones run at the limit, and streamed targets and IK
solutions are clamped or refused.
Parse, don't validate: a joint of the arm whose limits are no finite range
fails the launch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from robot_description_py import RobotDescription

# How far outside a joint limit a target may lie and still be taken, at that
# limit (rad). A measured position sent back as a target lies that close.
LIMIT_SLACK_RAD = 0.001


@dataclass(frozen=True)
class OutsideLimits:
    """A joint position farther outside its limits than LIMIT_SLACK_RAD: the
    joint counted from 0, its position and its limits."""

    joint: int
    position: float
    lower: float
    upper: float

    def __str__(self) -> str:
        """The joint counted from 1, with every digit of its position and its
        limits, so that a position a hair outside a limit reads apart from it."""
        return (
            f"joint {self.joint + 1} at {self.position!r} rad, "
            f"outside its limits [{self.lower!r}, {self.upper!r}]"
        )


@dataclass(frozen=True)
class JointLimits:
    lower: tuple[float, ...]
    upper: tuple[float, ...]

    def contains(self, positions: tuple[float, ...]) -> bool:
        if len(positions) != len(self.lower):
            return False
        return all(
            lo <= p <= hi
            for p, lo, hi in zip(positions, self.lower, self.upper, strict=True)
        )

    def first_outside(self, positions: tuple[float, ...]) -> OutsideLimits | None:
        """The first joint of `positions` farther than LIMIT_SLACK_RAD outside
        its limits, or None when `clamp` takes every joint at most that far."""
        if len(positions) != len(self.lower):
            raise ValueError("expected one position per joint")
        for joint, (p, lo, hi) in enumerate(zip(positions, self.lower, self.upper, strict=True)):
            if not lo - LIMIT_SLACK_RAD <= p <= hi + LIMIT_SLACK_RAD:
                return OutsideLimits(joint=joint, position=p, lower=lo, upper=hi)
        return None

    def clamp(self, positions: tuple[float, ...]) -> tuple[float, ...]:
        if len(positions) != len(self.lower):
            raise ValueError("expected one position per joint")
        return tuple(
            min(max(p, lo), hi)
            for p, lo, hi in zip(positions, self.lower, self.upper, strict=True)
        )


def from_description(description: RobotDescription) -> JointLimits:
    """The position limits (rad) the URDF of `description` gives each joint
    of its one arm, in the arm's order. Raises ValueError for a joint whose
    limits are not a finite range."""
    (arm,) = description.arms
    ranges = [(bound.lower, bound.upper) for bound in map(description.joint_limits, arm.joints)]
    for joint, (lo, hi) in zip(arm.joints, ranges, strict=True):
        if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
            raise ValueError(f"URDF joint {joint} has unusable limits [{lo}, {hi}]")
    lower, upper = zip(*ranges, strict=True)
    return JointLimits(lower=lower, upper=upper)
