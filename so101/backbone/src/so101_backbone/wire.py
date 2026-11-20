"""Wire-input parsing beyond what the so101 library covers: the leader's pose
stream and the follower's state streams, and how long the node waits for a
service's answer. Parse, don't validate: a pose setpoint either becomes a pose
in its frame at its time or a SetpointError naming the fault, and a state
message becomes the value it measures or None."""

from __future__ import annotations

import math

from control_core_py.geometry import GeometryError, Pose, Stamped
from control_core_py.runtime import SetpointError
from so101 import units

from so101_backbone.model import frame_of

# How long the node waits for a service's answer on the wire, beyond the time
# the provider holds the request: the answers of the robot's initializer carry
# the collision meshes.
ANSWER_WAIT_S = 10.0


def parse_pose_setpoint(frame, timestamp, position, orientation) -> Stamped[Pose]:
    """The setpoint as the geometry types hold it, in its frame or, with none,
    in the robot frame, its orientation normalized, or a SetpointError saying
    what to send: a value that is not finite, an orientation whose squared
    norm lies 0.01 or more from 1, or an empty frame."""
    try:
        return Stamped.parse(frame_of(frame), timestamp, Pose.parse(position, orientation))
    except GeometryError as e:
        raise SetpointError(str(e)) from e


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
