"""The reach ball: clamp geometry, and the fit against the deployed model."""

import itertools
import math

import pytest
from so101_description import limits as limits_mod
from so101_description.kinematics import Kinematics
from so101_description.model import KINEMATICS_URDF_PATH

from so101_backbone.reach import (
    REACH_MARGIN_M,
    ReachBall,
    _pan_axis_xy,
    fit_reach_ball,
)

BALL = ReachBall(center=(0.1, 0.0, 0.2), radius=0.3)


def test_inside_positions_pass_through_as_the_same_object():
    inside = (0.2, 0.1, 0.3)
    assert BALL.clamp(inside) is inside


def test_outside_positions_project_onto_the_surface():
    outside = (0.1, 0.0, 1.2)  # straight above the center, 1.0 away
    clamped = BALL.clamp(outside)
    assert clamped == pytest.approx((0.1, 0.0, 0.5))
    assert math.dist(clamped, BALL.center) == pytest.approx(BALL.radius)


def test_fit_covers_the_deployed_models_workspace():
    kinematics = Kinematics(KINEMATICS_URDF_PATH)
    limits = limits_mod.from_urdf(KINEMATICS_URDF_PATH)
    ball = fit_reach_ball(kinematics, limits)
    # Every reachable sample sits inside the fit before its margin shrink,
    # pan included: the fit never sweeps pan, trusting the center's placement
    # on that axis to make it irrelevant, and this is what checks that.
    def span(joint, points):
        lo, hi = limits.lower[joint], limits.upper[joint]
        return [lo + (hi - lo) * i / (points - 1) for i in range(points)]

    # Dense enough on the reaching joints to actually land on the furthest
    # pose; pan only has to prove the symmetry the center placement claims.
    reaches = [
        math.dist(kinematics.forward_kinematics(q)[0], ball.center)
        for q in itertools.product(span(0, 3), span(1, 9), span(2, 9), span(3, 9), span(4, 9))
    ]
    assert max(reaches) <= ball.radius + REACH_MARGIN_M + 1e-6
    # And the margin is real, not already spent: a clamped target lands the
    # full margin inside the furthest the arm can actually reach.
    assert max(reaches) >= ball.radius + REACH_MARGIN_M - 1e-3
    # A sane bound for this arm: near its ~0.5 m full extension, not a
    # degenerate fit and not a wild overestimate.
    assert 0.3 < ball.radius < 0.55


def test_a_model_whose_pan_sweep_is_not_a_circle_is_refused():
    # The circumcentre of three collinear or non-finite samples is
    # meaningless, and a ball centred there would bound nothing.
    class Fixed:
        def forward_kinematics(self, positions_rad):
            return (0.1, 0.2, 0.3), (0.0, 0.0, 0.0, 1.0)

    class NonFinite:
        def forward_kinematics(self, positions_rad):
            return (float("nan"), 0.2, 0.3), (0.0, 0.0, 0.0, 1.0)

    for model in (Fixed(), NonFinite()):
        with pytest.raises(ValueError, match="degenerate"):
            _pan_axis_xy(model)


def test_rolling_the_wrist_moves_the_grasp_point_off_the_pan_axis():
    # The fit sweeps wrist_roll because the grasp point sits off the roll
    # axis: rolling swings it, so a fit that held roll at zero would be
    # bounding a slice of the workspace and trusting an invariance the model
    # does not have.
    kinematics = Kinematics(KINEMATICS_URDF_PATH)
    limits = limits_mod.from_urdf(KINEMATICS_URDF_PATH)
    ball = fit_reach_ball(kinematics, limits)
    lo, hi = limits.lower[4], limits.upper[4]
    reaches = [
        math.dist(
            kinematics.forward_kinematics((0.0, 1.75, 0.0, 1.66, lo + (hi - lo) * i / 8))[0],
            ball.center,
        )
        for i in range(9)
    ]
    assert max(reaches) - min(reaches) > 0.005
