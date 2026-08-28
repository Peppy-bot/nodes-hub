"""The stream's only limiter: a max end-effector-speed governor.

A streamed target whose commanded step keeps the end effector under the caps
passes through as the same object, no arithmetic residue; a step that would
exceed them is scaled down in joint space, so the end effector traverses the
same path at cap speed and converges over the following ticks.
"""

from __future__ import annotations

import math

from so101_description.transforms import relative_rotation_rad


def rate_step(current: float, target: float, max_step: float) -> float:
    """The gripper's scalar form: the target itself once within one step's
    cap (exact landing, no float residue), else one cap-sized step."""
    delta = target - current
    if abs(delta) <= max_step:
        return target
    return current + math.copysign(max_step, delta)


# How many times the search halves its bracket. The step it returns is within
# 2^-SAFE_STEP_HALVINGS of the largest safe one, so at the 60 Hz tick and a
# 2 m/s cap the shortfall is under a hundredth of a millimetre; the cost is
# that many forward-kinematics evaluations, and only on ticks that clamp.
SAFE_STEP_HALVINGS = 12


class EEGovernor:
    """Scales one tick's joint-space step so the end effector never exceeds
    the linear or angular speed caps.

    The scale is found by measuring, not by proportion. Forward kinematics is
    nonlinear, so a joint step scaled by `cap / chord` does not move the end
    effector by `cap`: measured over the model, 89% of clamped steps came out
    above the cap, half of them by more than a third and the worst by 13x.
    The chord only sets the bracket; the step returned is one whose own
    forward kinematics was checked against both caps."""

    def __init__(
        self,
        kinematics,
        max_linear_m_s: float,
        max_angular_rad_s: float,
        period_s: float,
    ):
        self._kinematics = kinematics
        self._max_linear_step_m = max_linear_m_s * period_s
        self._max_angular_step_rad = max_angular_rad_s * period_s

    def govern(
        self, current: tuple[float, ...], target: tuple[float, ...]
    ) -> tuple[float, ...]:
        if current == target:
            return target
        position_now, orientation_now = self._kinematics.forward_kinematics(current)
        position_next, orientation_next = self._kinematics.forward_kinematics(target)
        linear_step = math.dist(position_now, position_next)
        angular_step = relative_rotation_rad(orientation_now, orientation_next)
        scale = min(
            1.0,
            self._max_linear_step_m / linear_step if linear_step > 0.0 else 1.0,
            self._max_angular_step_rad / angular_step if angular_step > 0.0 else 1.0,
        )
        if scale >= 1.0:
            return target
        return self._largest_safe_step(
            current, target, scale, position_now, orientation_now
        )

    def _largest_safe_step(
        self, current, target, bracket: float, position_now, orientation_now
    ) -> tuple[float, ...]:
        """The furthest point along the joint-space segment whose end-effector
        displacement is inside both caps.

        Bisection rather than a formula: the displacement is monotone in the
        scale for a short segment but has no closed form. Zero is safe by
        construction (it is where the arm already is), so the bracket always
        holds a safe endpoint and the search returns a measured step."""
        safe, unsafe = 0.0, bracket
        if self._within_caps(current, target, bracket, position_now, orientation_now):
            safe = bracket
        else:
            for _ in range(SAFE_STEP_HALVINGS):
                middle = (safe + unsafe) / 2.0
                if self._within_caps(
                    current, target, middle, position_now, orientation_now
                ):
                    safe = middle
                else:
                    unsafe = middle
        return self._step(current, target, safe)

    def _within_caps(
        self, current, target, scale: float, position_now, orientation_now
    ) -> bool:
        position, orientation = self._kinematics.forward_kinematics(
            self._step(current, target, scale)
        )
        return (
            math.dist(position_now, position) <= self._max_linear_step_m
            and relative_rotation_rad(orientation_now, orientation)
            <= self._max_angular_step_rad
        )

    @staticmethod
    def _step(current, target, scale: float) -> tuple[float, ...]:
        return tuple(
            c + scale * (t - c) for c, t in zip(current, target, strict=True)
        )
