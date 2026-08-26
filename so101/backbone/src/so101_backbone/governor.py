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


class EEGovernor:
    """Scales one tick's joint-space step so the end effector never exceeds
    the linear or angular speed caps. Linearized: the scaled step's true EE
    motion is re-measured next tick, so convergence is monotone."""

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
        return tuple(c + scale * (t - c) for c, t in zip(current, target))
