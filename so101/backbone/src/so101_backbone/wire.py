"""Wire-input parsing beyond what so101_description covers: the leader's pose
stream and the follower's state streams. Parse, don't validate: a pose
setpoint either becomes a typed target or a SetpointError naming the fault,
and a state message becomes the value it measures or None."""

from __future__ import annotations

import math

from control_core_py.runtime import SetpointError
from so101_description import units
from so101_description.transforms import PoseError, matrix_from_pose


def parse_pose_setpoint(position, orientation):
    """(position, orientation) validated as a rigid pose."""
    try:
        matrix_from_pose(position, orientation)
    except PoseError as e:
        raise SetpointError(str(e)) from e
    return tuple(float(v) for v in position), tuple(float(v) for v in orientation)


def parse_arm_states(m) -> tuple[float, ...] | None:
    """The joint positions of a follower joint_states message, or None for
    one of the wrong width or with a non-finite value."""
    positions = tuple(m.positions)
    if (
        len(positions) != units.NUM_JOINTS
        or len(m.velocities) not in (0, units.NUM_JOINTS)
        or len(m.efforts) not in (0, units.NUM_JOINTS)
        or not all(math.isfinite(v) for v in (*positions, *m.velocities, *m.efforts))
    ):
        return None
    return positions


def parse_gripper_states(m) -> float | None:
    """The opening of a follower gripper_states message, clamped into the
    contract range, or None for one with a non-finite value."""
    if not all(math.isfinite(v) for v in (m.opening, m.effort, m.max_effort)):
        return None
    # One truth: the contract-range opening anchors motion and relays.
    return min(max(float(m.opening), 0.0), 1.0)
