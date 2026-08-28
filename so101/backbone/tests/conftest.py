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


@pytest.fixture
def fake_kinematics():
    return FakeKinematics()
