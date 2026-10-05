"""Shared fixtures:

- a config;
- wide limits;
- the measured joints;
- a fake kinematics;
- a follower whose liveness does not ride the host's clock;
- one gripper control tick;
- the targets and checks of the workspace tests.
"""

from __future__ import annotations

import math

import pytest
from so101_description.limits import JointLimits

from so101_backbone.params import Config, UpstreamMode
from so101_backbone.reach import ReachBall

WIDE_LIMITS = JointLimits(lower=(-3.1,) * 5, upper=(3.1,) * 5)
WIDE_REACH = ReachBall(center=(0.0, 0.0, 0.0), radius=1e9)

# The joints the follower measures in the tests. They are not equal to a
# target of the tests. Thus the grasp point of the measurement is different
# from the grasp point of each target.
MEASURED = (0.0, 0.1, 0.2, 0.3, 0.4)

# Targets of the robot frame (m). The arm reaches NEAR pointing down. Its grasp
# point gets to CLOSE, but in no grasp direction: forward points 0.27 rad away
# from the plane of the arm there, and down is more than 0.1 m out of reach.
# FAR is out of the reach of the arm: farther than the arm is long (about
# 0.5 m from the pan axis), so it is short by more than FAR_SHORT_BY_MORE_THAN.
NEAR = (0.2, 0.0, 0.04)
CLOSE = (0.4, 0.1, 0.1)
FAR = (1.0, 0.0, 0.2)
FAR_SHORT_BY_MORE_THAN = 0.4

# The sentence each workspace answer of this robot, which has no perception
# camera, ends with.
NOT_CHECKED = "The view is not checked: the robot has no perception camera."


class FakeKinematics:
    """Records solve calls; scripted to succeed (echoing a fixed solution) or
    fail like a corrupted solver output."""

    def __init__(self):
        self.solution = (0.1, 0.2, 0.3, 0.4, 0.5)
        self.corrupted = False
        self.solve_calls: list[tuple] = []
        # The (position, orientation) bars each solve was handed, so a test
        # can prove the caller's slack reached the solver.
        self.bars: list[tuple] = []

    def inverse_kinematics(
        self,
        seed,
        position,
        orientation,
        *,
        position_tolerance_m=None,
        orientation_tolerance_rad=None,
    ):
        self.solve_calls.append(("solve", seed, position, orientation))
        self.bars.append((position_tolerance_m, orientation_tolerance_rad))
        return None if self.corrupted else self.solution

    def inverse_kinematics_streaming(self, seed, position, orientation):
        self.solve_calls.append(("stream", seed, position, orientation))
        return None if self.corrupted else self.solution

    def jacobian(self, positions_rad):
        """The exact derivative of this fake's own forward kinematics, so the
        governor's speed measure and the pose it reports agree."""
        import numpy as np

        jacobian = np.zeros((6, 5))
        jacobian[0, :] = 0.01
        jacobian[1, 0] = 0.02
        jacobian[2, 1] = 0.03
        return jacobian

    def forward_kinematics(self, positions_rad):
        """Deterministic, and dependent on every joint. A fake that answered
        a constant would let a readout publishing a stale pose pass."""
        return (
            (
                0.1 + sum(positions_rad) * 0.01,
                positions_rad[0] * 0.02,
                0.2 + positions_rad[1] * 0.03,
            ),
            (0.0, 0.0, 0.0, 1.0),
        )


def make_config(**overrides) -> Config:
    base = {
        "upstream_mode": UpstreamMode.JOINTS,
        "control_rate_hz": 100,
        "max_joint_velocity_rad_s": (2.0,) * 5,
        # Transparent by default: parity tests must see pure pass-through.
        # Governor- and rate-specific tests override these downward.
        "max_ee_velocity_m_s": 1e9,
        "max_ee_angular_velocity_rad_s": 1e9,
        "max_gripper_rate_frac_s": 1e9,
    }
    return Config(**{**base, **overrides})


def assert_rectangle_inside_reach(answer) -> None:
    """The workable rectangle of a describe_workspace answer has an area and
    lies inside the answer's reach bounds, both (x_min, x_max, y_min, y_max)."""
    x_min, x_max, y_min, y_max = answer.reach
    r_x_min, r_x_max, r_y_min, r_y_max = answer.rectangle
    assert x_min <= r_x_min < r_x_max <= x_max
    assert y_min <= r_y_min < r_y_max <= y_max


def gripper_step(coordinator, measured: float, now: float) -> float | None:
    """One control tick of the gripper at the instant `now`, with the follower
    reporting `measured`: the opening that went out, or None for silence."""
    coordinator.measured_gripper.set(measured)
    opening = coordinator.gripper_tick(now)
    if opening is not None:
        coordinator.gripper_published(True, opening)
    return opening


@pytest.fixture
def fake_kinematics():
    return FakeKinematics()


@pytest.fixture
def follower_never_stale(monkeypatch):
    """Keep every measured sample fresh however long the host takes. The
    gripper settle tests hand gripper_tick their own instants, while follower
    liveness is judged on the host's clock; pinning it keeps a scheduling
    pause between a sample and the tick that reads it from failing the plan
    as stale."""
    monkeypatch.setattr("so101_backbone.coordinator.STALE_FOLLOWER_TIMEOUT_S", math.inf)
