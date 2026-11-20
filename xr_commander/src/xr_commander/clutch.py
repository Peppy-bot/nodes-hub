"""One hand's clutch: controller motion in, pose setpoint out.

Relative mapping: squeezing the grip snapshots hand and end-effector, and the
command is that snapshot displaced by the hand's motion since. The snapshot
never moves, so the target is a pure function of the hand pose and returning
the hand returns the arm. Commands are best effort: the follower's governor
and rate caps decide what actually happens.

The end-effector snapshot is a measured pose in the frame its pose_states
message names, the follower's robot frame, and every setpoint names that frame;
the stream sends each with the time of its tick (publish.py), so the follower
places it where the robot frame stood then (pose_link).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from control_core_py.geometry import (
    Point,
    Pose,
    Quaternion,
    Stamped,
    UnitQuaternion,
    Vector,
    angle_shortest_path,
    inverse,
    normalized,
    product,
)

# A measured pose that moved this far between consecutive state reports did not
# travel, it was re-anchored or re-paired. Assumes a state stream fast enough
# (>= ~10 Hz at the follower's speed caps) that real travel per report stays
# well under this.
_MEASURED_JUMP_M = 0.10
# The rotational analog: a wrist that swung this far in one report flipped, it
# did not track. Far above per-report travel at any sane slew cap.
_MEASURED_JUMP_RAD = 0.5


@dataclass(frozen=True)
class Setpoint:
    """A pose setpoint: `pose` in `frame`, which the stream sends with the
    time of its tick."""

    frame: str
    pose: Pose


def _rotation_between(current: UnitQuaternion, reference: UnitQuaternion) -> Quaternion:
    """The rotation carrying `reference` onto `current`, about the axes both
    are given in."""
    return product(current.quaternion, inverse(reference.quaternion))


def _travel(current: Point, reference: Point, scale: float) -> Vector:
    """`scale` times the displacement from `reference` to `current`."""
    (x, y, z), (rx, ry, rz) = current.position, reference.position
    return (scale * (x - rx), scale * (y - ry), scale * (z - rz))


def _displaced(base: Pose, translation: Vector, rotation: Quaternion) -> Pose:
    """`base` moved by `translation` and turned by `rotation`, both about the
    axes `base` is given in."""
    (x, y, z), (dx, dy, dz) = base.position.position, translation
    return Pose(
        Point((x + dx, y + dy, z + dz)),
        UnitQuaternion(normalized(product(rotation, base.orientation.quaternion))),
    )


def _jumped(last: Pose, measured: Pose) -> bool:
    """Whether the end-effector moved from `last` to `measured`, both in one
    frame, farther than one state report's travel."""
    jump_m = math.dist(measured.position.position, last.position.position)
    jump_rad = angle_shortest_path(
        measured.orientation.quaternion, last.orientation.quaternion
    )
    return jump_m > _MEASURED_JUMP_M or jump_rad > _MEASURED_JUMP_RAD


@dataclass(frozen=True)
class _Engagement:
    """State captured at the squeeze and carried until release."""

    hand_at_engage: Pose
    ee_at_engage: Stamped[Pose]
    # Last measured pose seen, in the frame of `ee_at_engage`, so a
    # discontinuity can be told from tracking.
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
        measured_ee: Stamped[Pose] | None,
    ) -> Setpoint | None:
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
                hand_at_engage=hand,
                ee_at_engage=measured_ee,
                last_measured=measured_ee.value,
            )
            return Setpoint(measured_ee.frame, measured_ee.value)

        engagement = self._engagement
        anchor = engagement.ee_at_engage
        # A measured pose in another frame than the snapshot's, or one that
        # teleported, is a re-anchor or a re-pair underneath the snapshot: the
        # engagement drops, this tick commands nothing, and the next tick
        # re-snapshots where the arm now is, in the frame it is now measured
        # in.
        if measured_ee.frame != anchor.frame or _jumped(
            engagement.last_measured, measured_ee.value
        ):
            self.release()
            return None
        self._engagement = replace(engagement, last_measured=measured_ee.value)

        return Setpoint(
            anchor.frame,
            _displaced(
                anchor.value,
                _travel(
                    hand.position,
                    engagement.hand_at_engage.position,
                    self._motion_scale,
                ),
                _rotation_between(
                    hand.orientation, engagement.hand_at_engage.orientation
                ),
            ),
        )
