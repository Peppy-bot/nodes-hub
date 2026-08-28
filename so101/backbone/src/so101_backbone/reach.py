"""The reachable-workspace ball the streamed pose targets are clamped into.

lerobot's EE teleop pipeline clips every target into configured bounds
before IK reaches it (EEBoundsAndSafety), so the solver is never asked for
an out-of-reach pose. Here the bound is fitted to the model instead of
configured: the smallest ball centered on the pan axis that holds the
sampled reachable set, shrunk by a margin that keeps clamped targets off
the ill-conditioned full-extension rim.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

# How far inside the sampled maximum reach clamped targets land.
REACH_MARGIN_M = 0.02
# Grid resolution per swept joint; timid grids under-sample the rim.
_GRID_POINTS = 9
# Joints swept by the fit: lift, elbow, wrist_flex, wrist_roll. Pan alone is
# excluded, because it turns the arm about the very axis the center sits on
# and so cannot change any sample's distance to it. Roll is not excluded:
# the grasp point sits off the roll axis, so rolling swings it outward.
_SWEPT_JOINTS = (1, 2, 3, 4)
# Axis search: a coarse-to-fine descent for the center height that minimises
# the enclosing radius.
_SEARCH_START_M = 0.15
_SEARCH_SPAN_M = 0.35
_SEARCH_STEPS = 6
_SEARCH_ROUNDS = 4
# Lift, elbow, wrist_flex, wrist_roll for the pan-axis probe: any bent,
# non-singular arm, since a folded or fully extended pose puts the sampled
# points too close together to fix a circle through them.
_PAN_PROBE_POSE = (0.3, 0.5, 0.2, 0.0)
# Pan angles the probe sweeps to fix the circle the grasp point rides.
_PAN_PROBE_ANGLES = (-1.0, 0.0, 1.0)
# Smallest circumcircle determinant the probe accepts. The deployed model
# yields order 1e-2; anything approaching zero means the samples did not
# sweep a circle, and the centre it implies is meaningless.
_MIN_PAN_CIRCLE_DETERMINANT = 1e-9


@dataclass(frozen=True)
class ReachBall:
    center: tuple[float, float, float]
    radius: float

    def clamp(self, position):
        """The position itself when inside the ball, else its projection
        onto the ball's surface along the ray from the center."""
        distance = math.dist(position, self.center)
        if distance <= self.radius:
            return position
        scale = self.radius / distance
        return tuple(c + (p - c) * scale for c, p in zip(self.center, position, strict=True))


def _pan_axis_xy(kinematics) -> tuple[float, float]:
    """Where the pan axis crosses the xy-plane, from the circle the EE
    sweeps when only pan moves (the axis is offset from the base origin)."""
    (x1, y1), (x2, y2), (x3, y3) = (
        kinematics.forward_kinematics((pan, *_PAN_PROBE_POSE))[0][:2]
        for pan in _PAN_PROBE_ANGLES
    )
    d = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    # Near-collinear samples are the real hazard, not exactly collinear ones:
    # a tiny determinant throws the centre kilometres away, and a non-finite
    # one poisons every distance computed against it. Either way the model is
    # not the arm this fit is meant to bound.
    if not math.isfinite(d) or abs(d) < _MIN_PAN_CIRCLE_DETERMINANT:
        raise ValueError(f"pan sweep is degenerate (determinant {d}); cannot locate the pan axis")
    ux = (
        (x1**2 + y1**2) * (y2 - y3)
        + (x2**2 + y2**2) * (y3 - y1)
        + (x3**2 + y3**2) * (y1 - y2)
    ) / d
    uy = (
        (x1**2 + y1**2) * (x3 - x2)
        + (x2**2 + y2**2) * (x1 - x3)
        + (x3**2 + y3**2) * (x2 - x1)
    ) / d
    return ux, uy


def fit_reach_ball(kinematics, limits) -> ReachBall:
    """Sample every joint but pan across its limits at pan 0 and fit the
    smallest ball centered on the pan axis holding every sample; with the
    center on the axis, pan symmetry extends the fit to the full
    workspace."""
    ax, ay = _pan_axis_xy(kinematics)
    grids = [
        [
            limits.lower[j] + (limits.upper[j] - limits.lower[j]) * i / (_GRID_POINTS - 1)
            for i in range(_GRID_POINTS)
        ]
        for j in _SWEPT_JOINTS
    ]
    points = [
        kinematics.forward_kinematics((0.0, *swept))[0]
        for swept in itertools.product(*grids)
    ]

    def worst(cz: float) -> float:
        return max(
            math.sqrt((x - ax) ** 2 + (y - ay) ** 2 + (z - cz) ** 2)
            for x, y, z in points
        )

    cz, span = _SEARCH_START_M, _SEARCH_SPAN_M
    radius = worst(cz)
    for _ in range(_SEARCH_ROUNDS):
        for step in range(-_SEARCH_STEPS, _SEARCH_STEPS + 1):
            candidate = cz + step * span / _SEARCH_STEPS
            r = worst(candidate)
            if r < radius:
                radius, cz = r, candidate
        span /= _SEARCH_STEPS
    return ReachBall(center=(ax, ay, cz), radius=radius - REACH_MARGIN_M)
