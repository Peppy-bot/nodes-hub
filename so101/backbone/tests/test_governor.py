import math

import numpy as np
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

    def jacobian(self, positions_rad):
        # Constant, because this chain is linear: the first three joints
        # translate the tool one for one and the fourth spins it about z.
        return np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
            ]
        )


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
    """Two revolute links in the xy-plane: the tool rides a circle, so
    joint-space interpolation bows away from the straight line between
    endpoints. LinearKinematics cannot show that, which is why chord scaling
    survived every test in this file."""

    L1 = 0.4
    L2 = 0.3

    def forward_kinematics(self, positions_rad):
        q0, q1 = positions_rad[0], positions_rad[1]
        x = self.L1 * math.cos(q0) + self.L2 * math.cos(q0 + q1)
        y = self.L1 * math.sin(q0) + self.L2 * math.sin(q0 + q1)
        return (x, y, 0.0), (0.0, 0.0, 0.0, 1.0)

    def jacobian(self, positions_rad):
        q0, q1 = positions_rad[0], positions_rad[1]
        tip = self.L2 * math.sin(q0 + q1), self.L2 * math.cos(q0 + q1)
        jacobian = np.zeros((6, 5))
        jacobian[0, 0] = -self.L1 * math.sin(q0) - tip[0]
        jacobian[1, 0] = self.L1 * math.cos(q0) + tip[1]
        jacobian[0, 1] = -tip[0]
        jacobian[1, 1] = tip[1]
        # Orientation is constant in this double, so its angular rows are zero
        # and the double stays consistent with its own forward kinematics.
        return jacobian


def test_a_curved_path_is_capped_on_speed_not_on_the_endpoint_chord():
    # The chord between two joint configurations is not proportional to the
    # step that joins them, so scaling by cap/chord leaves a step that moves
    # the tool faster than asked. The Jacobian is proportional by
    # construction, which is why the cap is met by one division.
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
    naive = tuple(
        c + (cap / chord) * (t - c) for c, t in zip(current, target, strict=True)
    )
    naive_speed = np.linalg.norm(
        (kinematics.jacobian(current) @ (np.array(naive) - np.array(current)))[:3]
    )
    assert naive_speed > cap * 1.5, (
        "this case must actually defeat chord scaling, or it guards nothing"
    )

    governed = governor.govern(current, target)
    delta = np.array(governed) - np.array(current)
    speed = np.linalg.norm((kinematics.jacobian(current) @ delta)[:3])
    # The capped quantity is met exactly, with no search.
    assert speed == pytest.approx(cap, rel=1e-12)
    # And the distance actually travelled cannot exceed it either: a chord is
    # never longer than the arc the joints trace to it.
    assert math.dist(here, kinematics.forward_kinematics(governed)[0]) <= cap + 1e-12
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_step_that_cannot_be_measured_freezes_rather_than_passing_through(bad):
    # min(1.0, nan) is 1.0 in Python, so a governor that scales by a computed
    # fraction lets a non-finite target through untouched. A basis that
    # cannot be evaluated is not a licence to run uncapped.
    governor = make_governor(max_linear=1.0, period_s=0.1)
    current = (0.0, 0.0, 0.0, 0.0, 0.0)
    governed = governor.govern(current, (bad, 0.0, 0.0, 0.0, 0.0))
    assert governed == current
