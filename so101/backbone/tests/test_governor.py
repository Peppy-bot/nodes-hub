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


class PlanarArmKinematics:
    """Two revolute links in the xy-plane: the end effector rides a circle,
    so joint-space interpolation bows away from the straight line between
    endpoints. LinearKinematics above cannot show this, which is why the
    chord-scaling defect survived every test in this file."""

    L1 = 0.4
    L2 = 0.3

    def forward_kinematics(self, positions_rad):
        q0, q1 = positions_rad[0], positions_rad[1]
        x = self.L1 * math.cos(q0) + self.L2 * math.cos(q0 + q1)
        y = self.L1 * math.sin(q0) + self.L2 * math.sin(q0 + q1)
        return (x, y, 0.0), (0.0, 0.0, 0.0, 1.0)


def test_a_bowing_path_is_governed_by_measurement_not_by_the_chord():
    # The chord between two joint configurations is not a bound on the path
    # the joints trace between them. Scaling by cap/chord therefore leaves a
    # step whose real displacement is larger, and on the one limiter this
    # stream has, larger means faster than the operator asked for.
    kinematics = PlanarArmKinematics()
    cap_m_s, period_s = 1.0, 0.1
    cap = cap_m_s * period_s
    governor = EEGovernor(
        kinematics, max_linear_m_s=cap_m_s, max_angular_rad_s=1e9, period_s=period_s
    )
    # Half a turn of the shoulder: endpoints near each other, path a wide arc.
    current = (0.0, 2.4, 0.0, 0.0, 0.0)
    target = (math.pi, 2.4, 0.0, 0.0, 0.0)

    here = kinematics.forward_kinematics(current)[0]
    chord = math.dist(here, kinematics.forward_kinematics(target)[0])
    naive = tuple(c + (cap / chord) * (t - c) for c, t in zip(current, target, strict=True))
    naive_step = math.dist(here, kinematics.forward_kinematics(naive)[0])
    assert naive_step > cap * 1.5, (
        "this case must actually defeat chord scaling, or it guards nothing"
    )

    governed = governor.govern(current, target)
    stepped = math.dist(here, kinematics.forward_kinematics(governed)[0])
    assert stepped <= cap + 1e-9
    # And not so conservative that the arm crawls: the search keeps most of
    # the budget it is allowed to spend.
    assert stepped > cap * 0.9
