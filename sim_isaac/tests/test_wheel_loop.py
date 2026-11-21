"""The rate loop of a base's wheel, engine/wheel_loop.py: the PI action the
drive delivers through its target velocity, the integral of the rate's error
and its clamp, and the angle counted across PhysX's wrap and across a stage
that let go of its joints."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))

from wheel_loop import RateLoopGains, WheelLoop  # noqa: E402  pylint: disable=C0413

GAINS = RateLoopGains(kp=128.0, ki=128.0, max_torque=16.0)
STEP_S = 1.0 / 60.0


def _anchored(read: float = 0.0) -> WheelLoop:
    return WheelLoop().advanced(read, STEP_S, GAINS)


def test_a_first_read_anchors_the_loop_at_rest_where_the_wheel_stands():
    loop = WheelLoop(angle=2.0, integral=0.1, rate=1.0).advanced(0.7, STEP_S, GAINS)
    assert loop == WheelLoop(angle=2.0, integral=0.0, rate=1.0, read=0.7)


def test_the_drive_delivers_the_loops_proportional_and_integral_action():
    """damping * (target - velocity) is kp * (rate - velocity) + ki * integral."""
    loop = WheelLoop(integral=0.05, rate=1.5, read=0.0)
    velocity = 1.2
    torque = GAINS.kp * (loop.velocity_target(GAINS) - velocity)
    assert torque == pytest.approx(GAINS.kp * (1.5 - velocity) + GAINS.ki * 0.05)


def test_the_integral_is_the_angle_the_rates_carry_the_wheel_to_less_its_turn():
    loop = _anchored().driven_at(2.0)
    loop = loop.advanced(0.02, STEP_S, GAINS)
    assert loop.integral == pytest.approx(2.0 * STEP_S - 0.02)
    assert loop.angle == pytest.approx(0.02)


@pytest.mark.parametrize("rate", [40.0, -40.0])
def test_the_integral_is_clamped_at_what_the_torque_cap_holds(rate):
    """A stalled wheel stores no more than the cap over ki, either way."""
    loop = _anchored().driven_at(rate)
    for _ in range(60):
        loop = loop.advanced(0.0, STEP_S, GAINS)
    assert loop.integral == pytest.approx(math.copysign(GAINS.max_torque / GAINS.ki, rate))


def test_a_turn_across_the_wrap_counts_on_without_a_jump():
    """PhysX reads a joint without limits wrapped at a full turn."""
    loop = _anchored(math.pi - 0.05)
    loop = loop.advanced(-math.pi + 0.05, STEP_S, GAINS)
    assert loop.angle == pytest.approx(0.1)
    loop = loop.advanced(-math.pi + 0.15, STEP_S, GAINS)
    assert loop.angle == pytest.approx(0.2)


def test_a_wheel_turning_backward_across_the_wrap_counts_down():
    loop = _anchored(-math.pi + 0.05).advanced(math.pi - 0.05, STEP_S, GAINS)
    assert loop.angle == pytest.approx(-0.1)


def test_a_stage_that_let_go_of_its_joints_is_read_afresh_and_the_angle_runs_on():
    """The timeline sets the wheel back to its stage's angle, 0, when it
    stops: the loop counts no turn for it and starts at rest."""
    loop = _anchored(0.0).driven_at(1.0)
    loop = loop.advanced(1.5, STEP_S, GAINS)
    loop = loop.let_go()
    assert loop.read is None and loop.integral == 0.0
    loop = loop.advanced(0.0, STEP_S, GAINS)
    assert loop.angle == pytest.approx(1.5)
    loop = loop.advanced(0.25, STEP_S, GAINS)
    assert loop.angle == pytest.approx(1.75)


def test_a_step_of_no_time_anchors_the_loop():
    """A timeline that started over reads an earlier time than the last step."""
    loop = _anchored(0.3).driven_at(1.0).advanced(0.9, -2.0, GAINS)
    assert loop.read == 0.9 and loop.angle == 0.0 and loop.integral == 0.0


def test_the_windup_is_the_cap_over_the_integral_gain():
    assert RateLoopGains(kp=1.0, ki=4.0, max_torque=16.0).windup() == 4.0
