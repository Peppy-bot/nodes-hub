"""The named whole-arm postures, in joint_link radians, in the arm's order:
the `ready` and `home` states of the arm in SO-101's SRDF, which says why
each stands where it does. A posture is a fact of the arm, the same for every
deployment, and the description holds each within the URDF's joint limits.
"""

from __future__ import annotations

from robot_description_py import HOME, READY

from so101_backbone.model import ARM, DESCRIPTION


def _posture(name: str) -> tuple[float, ...]:
    """The arm's state `name`; the description raises KeyError, naming the arm
    and the state, when the SRDF gives the arm no such state."""
    return tuple(DESCRIPTION.state(ARM.name, name))


# Where the arm rests and parks.
HOME_POSITIONS_RAD = _posture(HOME)

# Where work starts.
READY_POSITIONS_RAD = _posture(READY)
