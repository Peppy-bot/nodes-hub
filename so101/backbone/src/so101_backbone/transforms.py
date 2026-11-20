"""The transforms this backbone publishes: at the timestamp of each
limb_states, the transform of each joint of the arm at its measured position
and of the gripper's joint where its measured opening puts it, from SO-101's
model in robot_description. The robot's initializer publishes the transforms
of the fixed joints."""

from __future__ import annotations

from peppygen.emitted_topics.transforms import transforms

from so101_backbone.model import ARM, DESCRIPTION


def transforms_message(
    timestamp_s: float, joints_rad: tuple[float, ...], opening: float, started_s: float
) -> bytes:
    """The transforms message of one snapshot: `joints_rad` the arm's measured
    joints in ARM.joints order and `opening` the gripper's measured opening,
    both measured at `timestamp_s`, and `started_s` when this process started,
    on the wall clock, when every transform it publishes started over."""
    positions = {
        **dict(zip(ARM.joints, joints_rad, strict=True)),
        **ARM.gripper.finger_positions_at(opening),
    }
    moving = DESCRIPTION.transforms_at(positions)
    return transforms.build_message(
        timestamp_s,
        [
            transforms.MessageTransformsItem(
                frame=transform.parent,
                child_frame=transform.child,
                position=list(transform.position),
                orientation=list(transform.orientation),
            )
            for transform in moving
        ],
        started_s,
    )
