"""The robot model this backbone drives: SO-101's model in robot_description,
the URDF and SRDF every node reads, from which the names of the arm and the
gripper, the joint limits, the postures and the tool frame every wire pose
names come."""

from __future__ import annotations

import functools
import tempfile
from pathlib import Path

import robot_description_py
from so101 import units

MODEL = robot_description_py.model("so101")
DESCRIPTION = MODEL.description()
# The SO-101 is one arm with one gripper.
(ARM,) = DESCRIPTION.arms

# The follower's joint_states and joint_setpoints carry the servos in the so101
# library's bus order, which this backbone reads as the arm's joints.
if ARM.joints != units.JOINT_NAMES:
    raise RuntimeError(
        f"the description's arm moves {list(ARM.joints)}, and the follower's wire carries "
        f"{list(units.JOINT_NAMES)}: build the backbone from one public-peppy-libs commit"
    )

# A fixed frame on the fixed-jaw body near the jaw region; on this
# single-moving-jaw gripper the pad midpoint shifts with the opening, so no
# static frame is exactly the contracts' grasp point.
END_EFFECTOR_FRAME = ARM.grasp_link


@functools.cache
def _model_directory() -> tempfile.TemporaryDirectory:
    """The model's files on disk, written once per process into a directory of
    its own, which the process removes as it exits."""
    directory = tempfile.TemporaryDirectory(prefix="so101_backbone_")
    MODEL.write_to(Path(directory.name))
    return directory


def urdf_path() -> str:
    """The model's URDF on disk, its collision meshes beside it under
    meshes/, for a reader that takes a file."""
    return str(Path(_model_directory().name) / f"{MODEL.id}.urdf")
