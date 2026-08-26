import math

import pytest

from so101_backbone.governor import EEGovernor, rate_step


class LinearKinematics:
    """FK whose EE position is the first three joints in meters and whose
    orientation rotates about z by the fourth joint, making EE speeds exact
    functions of joint deltas."""

    def forward_kinematics(self, positions_rad):
        position = (positions_rad[0], positions_rad[1], positions_rad[2])
        half = positions_rad[3] / 2.0
        orientation = (0.0, 0.0, math.sin(half), math.cos(half))
        return position, orientation


def make_governor(max_linear=1.0, max_angular=10.0, period_s=0.1):
    return EEGovernor(LinearKinematics(), max_linear, max_angular, period_s)


def test_within_caps_returns_the_target_object():
    governor = make_governor()
    current = (0.0, 0.0, 0.0, 0.0, 0.0)
    target = (0.05, 0.0, 0.0, 0.0, 0.0)  # 0.5 m/s, under the 1.0 cap
    assert governor.govern(current, target) is target


def test_identical_target_passes_through():
    governor = make_governor()
    joints = (0.1, 0.2, 0.3, 0.4, 0.5)
    assert governor.govern(joints, joints) is joints


def test_linear_step_beyond_the_cap_is_scaled_to_cap_speed():
    governor = make_governor(max_linear=1.0, period_s=0.1)
    current = (0.0, 0.0, 0.0, 0.0, 0.0)
    target = (0.4, 0.0, 0.0, 0.0, 0.0)  # 4 m/s commanded
    governed = governor.govern(current, target)
    # Scaled to 0.1 m this tick (cap * period), same direction.
    assert governed[0] == 0.1
    assert governed[1:] == (0.0, 0.0, 0.0, 0.0)


def test_angular_step_beyond_the_cap_is_scaled():
    governor = make_governor(max_linear=1e9, max_angular=1.0, period_s=0.1)
    current = (0.0, 0.0, 0.0, 0.0, 0.0)
    target = (0.0, 0.0, 0.0, 1.0, 0.0)  # 10 rad/s about z
    governed = governor.govern(current, target)
    assert governed[3] == pytest.approx(0.1)


def test_tightest_cap_wins():
    governor = make_governor(max_linear=1.0, max_angular=1.0, period_s=0.1)
    current = (0.0, 0.0, 0.0, 0.0, 0.0)
    # Linear asks for scale 0.5, angular for scale 0.1: angular binds.
    target = (0.2, 0.0, 0.0, 1.0, 0.0)
    governed = governor.govern(current, target)
    assert governed[0] == pytest.approx(0.02)
    assert governed[3] == pytest.approx(0.1)


def test_convergence_over_ticks_reaches_the_target_exactly():
    governor = make_governor(max_linear=1.0, period_s=0.1)
    current = (0.0, 0.0, 0.0, 0.0, 0.0)
    target = (0.25, 0.0, 0.0, 0.0, 0.0)
    for _ in range(10):
        current = governor.govern(current, target)
        if current is target:
            break
    assert current is target


def test_rate_step_lands_exactly_and_caps_beyond():
    assert rate_step(0.0, 1.0, 0.25) == 0.25
    assert rate_step(1.0, 0.0, 0.25) == 0.75
    # Within one step: the target itself, not an arithmetic neighbor.
    assert rate_step(0.9, 1.0, 0.25) == 1.0
    third = 0.1 + 0.2  # 0.30000000000000004: adversarial float target
    assert rate_step(0.3, third, 1.0) == third
