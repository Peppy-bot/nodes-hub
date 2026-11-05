"""The stream's end-effector speed cap, as a governor.

A streamed target that keeps the tool under the cap goes downstream as the
same object. The coordinator holds each stream to the joint velocity limits
ahead of it.

The cap is measured through the Jacobian, following `openarm_backbone`'s
ee-speed limiter, because that is the quantity a speed cap can actually be
enforced through.
"""

from __future__ import annotations

import math

import numpy as np


def rate_step(current: float, target: float, max_step: float) -> float:
    """One scalar step toward `target`: the target itself once within the
    cap (exact landing, no float residue), else one cap-sized step. The
    gripper's ramp and each joint of the pose path step through it."""
    delta = target - current
    if abs(delta) <= max_step:
        return target
    return current + math.copysign(max_step, delta)


class EEGovernor:
    """Scales one tick's joint step so the tool stays inside the speed cap.

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
        period_s: float,
    ):
        self._kinematics = kinematics
        self._max_linear_step_m = max_linear_m_s * period_s

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
        if not math.isfinite(linear):
            # A basis that cannot be evaluated is not a licence to run
            # uncapped.
            return current
        if linear <= self._max_linear_step_m:
            return target
        scale = self._max_linear_step_m / linear
        return tuple(
            c + scale * (t - c) for c, t in zip(current, target, strict=True)
        )
