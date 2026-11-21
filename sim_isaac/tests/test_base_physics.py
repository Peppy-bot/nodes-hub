"""The rate loop of a base's wheel (engine/wheel_loop.py) on a wheel as PhysX
drives it, without Isaac: one revolute degree of freedom whose drive is the
velocity drive the stage authors, damping * (target velocity - velocity)
capped at its maximum force, implicit over the step, its target velocity the
loop's at each step, as the bridge writes it, and its angle read wrapped at a
full turn. A free wheel follows its rate, a wheel against a steady load
reaches its rate through the integral, and a stalled wheel pushes at its cap
and, released, holds where it stands. Every outcome is read after a step
count, so none depends on the host's speed.

What needs Isaac itself runs in the Isaac image: PhysX's articulation with the
stage's drives, the tires on the floor, and the base driving, lifted off the
floor and stalled against an obstacle."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))

from wheel_loop import RateLoopGains, WheelLoop  # noqa: E402  pylint: disable=C0413

# A wheel's loop as a test stage authors it, and the inertia its drive turns
# (kg m^2): test inputs.
GAINS = RateLoopGains(kp=128.0, ki=128.0, max_torque=16.0)
INERTIA_KG_M2 = 0.05
# The engine's step at a 60 Hz timeline (s).
STEP_S = 1.0 / 60.0
RATE_RAD_S = 5.0


@dataclass(frozen=True)
class Wheel:
    """One wheel's degree of freedom: its angle, unwrapped (rad), its rate
    (rad/s), and whether something holds it still."""

    angle: float = 0.0
    velocity: float = 0.0
    stalled: bool = False

    def torque(self, target: float, load: float) -> float:
        """The drive's torque over the next step (N m): damping * (target -
        velocity) at the velocity the step ends at, implicit as PhysX's is,
        capped at the maximum force."""
        if self.stalled:
            return max(-GAINS.max_torque, min(GAINS.kp * (target - 0.0), GAINS.max_torque))
        ended = (INERTIA_KG_M2 * self.velocity + STEP_S * (GAINS.kp * target + load)) / (
            INERTIA_KG_M2 + STEP_S * GAINS.kp
        )
        return max(-GAINS.max_torque, min(GAINS.kp * (target - ended), GAINS.max_torque))

    def stepped(self, target: float, load: float = 0.0) -> Wheel:
        """The wheel a step later, its drive at `target` and `load` (N m) on
        it besides."""
        if self.stalled:
            return self
        velocity = self.velocity + STEP_S * (self.torque(target, load) + load) / INERTIA_KG_M2
        return replace(self, angle=self.angle + STEP_S * velocity, velocity=velocity)

    def read(self) -> float:
        """Its angle as PhysX reads a revolute joint without limits, wrapped at
        a full turn."""
        return math.remainder(self.angle, math.tau)


def run(seconds: float, rate: float, load: float = 0.0, loop=None, wheel=None):
    """The loop and the wheel after `seconds` of steps, each reading the
    wheel, setting `rate` as the loop's target, and driving the wheel at the
    loop's target velocity."""
    loop, wheel = loop or WheelLoop(), wheel or Wheel()
    for _ in range(round(seconds / STEP_S)):
        loop = loop.advanced(wheel.read(), STEP_S, GAINS).driven_at(rate)
        wheel = wheel.stepped(loop.velocity_target(GAINS), load)
    return loop, wheel


def test_a_free_wheel_follows_its_rate_and_counts_its_turns():
    loop, wheel = run(2.0, RATE_RAD_S)

    assert wheel.velocity == pytest.approx(RATE_RAD_S, abs=1e-3)
    # The loop last read the wheel a step before it ended, and counted every
    # turn across the wrap up to there.
    assert loop.angle == pytest.approx(wheel.angle - STEP_S * wheel.velocity, abs=1e-9)
    assert wheel.angle > math.tau, "past a full turn"


def test_a_wheel_against_a_steady_load_reaches_its_rate_through_the_integral():
    """A drag of 4 N m holds a proportional loop short of its rate by the
    drag over kp; the integral makes the drag up, standing at the drag over
    ki."""
    drag = -4.0
    loop, wheel = run(5.0, RATE_RAD_S, load=drag)

    assert wheel.velocity == pytest.approx(RATE_RAD_S, abs=1e-3)
    assert loop.integral == pytest.approx(-drag / GAINS.ki, abs=1e-3)


def test_a_stalled_wheel_pushes_at_its_cap_and_released_holds_where_it_stands():
    loop, wheel = run(2.0, RATE_RAD_S, wheel=Wheel(angle=0.3, stalled=True))
    assert loop.integral == pytest.approx(GAINS.windup())
    assert wheel.torque(loop.velocity_target(GAINS), 0.0) == GAINS.max_torque

    # Released where it stands, its rate 0, and free again.
    loop, wheel = run(1.0, 0.0, loop=loop.let_go(), wheel=replace(wheel, stalled=False))

    assert wheel.angle == pytest.approx(0.3, abs=1e-9)
    assert wheel.velocity == pytest.approx(0.0, abs=1e-9)
    assert loop.integral == pytest.approx(0.0, abs=1e-9)
