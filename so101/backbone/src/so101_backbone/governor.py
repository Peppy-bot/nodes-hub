"""The stream's only limiter: a max end-effector-speed governor.

Deliberately the only one. The SO-101 teleop stream is meant to match
lerobot's, so a streamed target that keeps the tool under the caps goes
downstream as the same object, and the caps are the single stated divergence.
Per-joint rate limiting belongs to the pose path, where the targets are
machine-generated, and the coordinator holds it there.

The cap is measured through the Jacobian, following `openarm_backbone`'s
ee-speed limiter, because that is the quantity a speed cap can actually be
enforced through.
"""

from __future__ import annotations

import math

import numpy as np


def rate_step(current: float, target: float, max_step: float) -> float:
    """The gripper's scalar form: the target itself once within one step's
    cap (exact landing, no float residue), else one cap-sized step."""
    delta = target - current
    if abs(delta) <= max_step:
        return target
    return current + math.copysign(max_step, delta)


class EEGovernor:
    """Scales one tick's joint step so the tool stays inside the linear and
    angular speed caps.

    The scale is exact, not searched for. `J @ dq` is the twist a joint step
    produces and is linear in that step, so halving the step halves the speed
    and one division meets the cap. Differencing forward kinematics instead
    gives a chord that is not proportional to the step: enforced that way the
    cap held on one clamped step in ten, and the worst exceeded it 13-fold.
    """

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
        delta = np.asarray(target, dtype=float) - np.asarray(current, dtype=float)
        # Refused before the product rather than after: a non-finite step
        # reaches the same answer by way of a numpy warning on every tick,
        # and `0.0 * inf` would carry it into the result regardless.
        if not np.all(np.isfinite(delta)):
            return current
        twist = self._kinematics.jacobian(current) @ delta
        linear = float(np.linalg.norm(twist[:3]))
        angular = float(np.linalg.norm(twist[3:]))
        if not (math.isfinite(linear) and math.isfinite(angular)):
            # A basis that cannot be evaluated is not a licence to run
            # uncapped.
            return current
        scale = min(
            1.0,
            self._max_linear_step_m / linear if linear > 0.0 else 1.0,
            self._max_angular_step_rad / angular if angular > 0.0 else 1.0,
        )
        if scale >= 1.0:
            return target
        return tuple(
            c + scale * (t - c) for c, t in zip(current, target, strict=True)
        )
