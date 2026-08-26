"""Shared fixtures: a config, wide limits, and a fake kinematics."""

from __future__ import annotations

import pytest
from so101_description.limits import JointLimits

from so101_backbone.params import Config, UpstreamMode
from so101_backbone.reach import ReachBall

WIDE_LIMITS = JointLimits(lower=(-3.1,) * 5, upper=(3.1,) * 5)
WIDE_REACH = ReachBall(center=(0.0, 0.0, 0.0), radius=1e9)


class FakeKinematics:
    """Records solve calls; scripted to succeed (echoing a fixed solution) or
    fail like a corrupted solver output."""

    def __init__(self):
        self.solution = (0.1, 0.2, 0.3, 0.4, 0.5)
        self.corrupted = False
        self.solve_calls: list[tuple] = []

    def inverse_kinematics(self, seed, position, orientation):
        self.solve_calls.append(("solve", seed, position, orientation))
        return None if self.corrupted else self.solution

    def inverse_kinematics_streaming(self, seed, position, orientation):
        self.solve_calls.append(("stream", seed, position, orientation))
        return None if self.corrupted else self.solution

    def forward_kinematics(self, positions_rad):
        return (0.1, 0.0, 0.2), (0.0, 0.0, 0.0, 1.0)


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


@pytest.fixture
def fake_kinematics():
    return FakeKinematics()
