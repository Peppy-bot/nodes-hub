"""The rate loop of a base's wheel, as the base's driver, the ZLAC8015D, runs
each wheel in profile velocity mode: a loop on the wheel's rate whose target
is the rate its latest setpoint gives, its action proportional to the rate's
error and to that error's integral, capped together at the motor's torque,
with the integral clamped at what the cap holds.

PhysX drives a joint as a PD controller, implicit over the step: a torque of
stiffness * (target position - position) + damping * (target velocity -
velocity), capped at the drive's maximum force. A wheel's drive has no
stiffness and the loop's proportional gain as its damping, and each step the
engine sets its target velocity to the rate plus the integral term over that
gain:

    damping * (rate + ki / kp * integral - velocity)
        = kp * (rate - velocity) + ki * integral

which is the loop's action, capped at the drive's maximum force, the motor's
torque. The integral of the rate's error is the angle the target rates carry
the wheel to less the angle it turned; the engine sums it at each step from
the angles the articulation reads, and clamps it at the cap over ki either
way, as a drive clamps its integral at its current limit.

PhysX wraps the angle of a revolute joint without limits at a full turn, and
sets every joint back to its stage's state when the timeline stops. The loop
counts each step's turn, the change of the read angle less whole turns, so
the angle a wheel reports runs on without a jump, and a stage that let go of
its joints is read afresh, the loop starting at rest where the wheel stands.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional


@dataclass(frozen=True)
class RateLoopGains:
    """The gains of a wheel's rate loop and the torque it is capped at."""

    # The proportional gain, on the rate's error (N*m*s/rad): the drive's
    # damping.
    kp: float
    # The integral gain, on the error's integral (N*m/rad).
    ki: float
    # The motor's torque (N*m): the drive's maximum force.
    max_torque: float

    def windup(self) -> float:
        """How far the integral may stand either way (rad): what the cap
        holds through the integral gain alone."""
        return self.max_torque / self.ki


@dataclass(frozen=True)
class WheelLoop:
    """One wheel's loop between two steps."""

    # The angle the wheel turned to since it joined (rad), unwrapped.
    angle: float = 0.0
    # The integral of the rate's error (rad).
    integral: float = 0.0
    # The target rate written at the last step, in force since (rad/s).
    rate: float = 0.0
    # The angle the articulation read at the last step, None on a stage the
    # loop has not read since it let go of its joints.
    read: Optional[float] = None

    def advanced(self, read: float, dt: float, gains: RateLoopGains) -> WheelLoop:
        """The loop once the wheel reads `read` (rad), `dt` (s) after the last
        step: the wheel's turn, less whole turns, added to its angle, and the
        rate's error over the step to the integral, clamped. A first read
        anchors the loop there at rest."""
        if self.read is None or dt <= 0.0:
            return replace(self, integral=0.0, read=read)
        turned = math.remainder(read - self.read, math.tau)
        bound = gains.windup()
        integral = min(max(self.integral + self.rate * dt - turned, -bound), bound)
        return replace(self, angle=self.angle + turned, integral=integral, read=read)

    def driven_at(self, rate: float) -> WheelLoop:
        """The loop with `rate` as its target from this step on."""
        return replace(self, rate=rate)

    def velocity_target(self, gains: RateLoopGains) -> float:
        """The target velocity of the wheel's drive: the rate, and the integral
        term over the drive's damping."""
        return self.rate + gains.ki / gains.kp * self.integral

    def let_go(self) -> WheelLoop:
        """The loop of a wheel whose stage lets go of its joints: the next read
        anchors it afresh, at rest, and its angle runs on from there."""
        return replace(self, integral=0.0, read=None)
