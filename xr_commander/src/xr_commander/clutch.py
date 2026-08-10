"""One hand's clutch: controller motion in, pose setpoint out.

Relative mapping: squeezing the grip snapshots hand and end-effector, and the
command is that snapshot displaced by the hand's motion since. The snapshot
never moves, so the target is a pure function of the hand pose and returning
the hand returns the arm. Commands are best effort: the follower's governor
and rate caps decide what actually happens.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from xr_commander.frames import Pose, apply_offset, quat_axis_angle, relative_rotation

# A measured pose that moved this far between consecutive state reports did not
# travel, it was re-anchored or re-paired. Assumes a state stream fast enough
# (>= ~10 Hz at the follower's speed caps) that real travel per report stays
# well under this.
_MEASURED_JUMP_M = 0.10
# The rotational analog: a wrist that swung this far in one report flipped, it
# did not track. Far above per-report travel at any sane slew cap.
_MEASURED_JUMP_RAD = 0.5


@dataclass(frozen=True)
class _Engagement:
    """State captured at the squeeze and carried until release."""

    hand_at_engage: Pose
    ee_at_engage: Pose
    # Last measured pose seen, so a discontinuity can be told from tracking.
    last_measured: Pose


class HandClutch:
    """The clutch for a single hand. Not thread-safe; step it from one task."""

    def __init__(self, motion_scale: float) -> None:
        # Hand travel to end-effector travel, parsed by config; orientation is
        # never scaled.
        self._motion_scale = motion_scale
        self._engagement: _Engagement | None = None

    @property
    def engaged(self) -> bool:
        return self._engagement is not None

    def release(self) -> None:
        self._engagement = None

    def step(
        self,
        *,
        squeezing: bool,
        hand: Pose | None,
        measured_ee: Pose | None,
    ) -> Pose | None:
        """This tick's pose setpoint, or None to command nothing.

        None is the deadman: the caller publishes nothing and the follower
        holds. Engaging before the follower has reported a pose is refused;
        the relative mapping has nothing to hang on.
        """
        if not squeezing or hand is None or measured_ee is None:
            self.release()
            return None

        if self._engagement is None:
            # The engage tick commands the measured pose: a no-op that starts
            # the stream.
            self._engagement = _Engagement(
                hand_at_engage=hand, ee_at_engage=measured_ee, last_measured=measured_ee
            )
            return measured_ee

        engagement = self._engagement
        # A teleported measurement means a re-anchor underneath us; honouring
        # the old snapshot would drag the arm back across the discontinuity.
        # Drop the engagement and re-snapshot next tick.
        jump_m = float(
            np.linalg.norm(measured_ee.position - engagement.last_measured.position)
        )
        _axis, jump_rad = quat_axis_angle(
            relative_rotation(measured_ee, engagement.last_measured)
        )
        if jump_m > _MEASURED_JUMP_M or jump_rad > _MEASURED_JUMP_RAD:
            self.release()
            return None
        self._engagement = replace(engagement, last_measured=measured_ee)

        return apply_offset(
            engagement.ee_at_engage,
            self._motion_scale * (hand.position - engagement.hand_at_engage.position),
            relative_rotation(hand, engagement.hand_at_engage),
        )
