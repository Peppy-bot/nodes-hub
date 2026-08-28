"""Wire-input parsing for the upstream leader streams beyond what
so101_description covers: the pose stream. Parse, don't validate: a message either
becomes a typed target or a SetpointError naming the fault."""

from __future__ import annotations

from control_core_py.runtime import SetpointError
from so101_description.transforms import PoseError, matrix_from_pose


def parse_pose_setpoint(position, orientation):
    """(position, orientation) validated as a rigid pose."""
    try:
        matrix_from_pose(position, orientation)
    except PoseError as e:
        raise SetpointError(str(e)) from e
    return tuple(float(v) for v in position), tuple(float(v) for v in orientation)
