"""The transforms message of a snapshot: one transform per joint that moves,
at the snapshot's joints and the gripper's opening, which composed down the
arm with the fixed joints' transforms place the grasp link where the
backbone's kinematics place the grasp point."""

from __future__ import annotations

import pytest
from conftest import MEASURED, TREE_TOLERANCE, place_in_robot_frame
from control_core_py.geometry import angle_shortest_path

# The decoder the harness reads the node's transforms with.
from peppygen.fixtures.emitted_topics.transforms.transforms import (
    _deserialize_message as decode,  # pylint: disable=protected-access
)

from so101_backbone.kinematics import Kinematics
from so101_backbone.limits import from_description
from so101_backbone.model import ARM, DESCRIPTION, urdf_path
from so101_backbone.transforms import transforms_message

TIMESTAMP_S = 1_700_000_123.25
STARTED_S = 1_700_000_000.5

LIMITS = from_description(DESCRIPTION)
# The arm's joints: the follower's test measurement, zero, and each end of
# every joint's range.
CONFIGURATIONS = (MEASURED, (0.0,) * 5, LIMITS.lower, LIMITS.upper)
OPENINGS = (0.0, 0.5, 1.0)


def _transforms(message):
    """The transforms of a decoded message, one (parent, child, position,
    orientation) each."""
    return [
        (record.frame, record.child_frame, tuple(record.position), tuple(record.orientation))
        for record in message.transforms
    ]


def _fixed():
    return [
        (transform.parent, transform.child, transform.position, transform.orientation)
        for transform in DESCRIPTION.fixed_transforms()
    ]


@pytest.fixture(scope="module", name="kinematics")
def _kinematics():
    return Kinematics(urdf_path())


def test_a_message_carries_its_snapshots_time_the_process_start_and_each_moving_joint_once():
    message = decode(transforms_message(TIMESTAMP_S, MEASURED, 0.5, STARTED_S))
    moving = DESCRIPTION.transforms_at(
        {**dict(zip(ARM.joints, MEASURED)), **ARM.gripper.finger_positions_at(0.5)}
    )

    assert message.timestamp == pytest.approx(TIMESTAMP_S, abs=1e-9)
    assert message.start_timestamp == pytest.approx(STARTED_S, abs=1e-9)
    children = [record.child_frame for record in message.transforms]
    assert children == [transform.child for transform in moving]
    assert [record.frame for record in message.transforms] == [transform.parent for transform in moving]
    assert len(set(children)) == len(ARM.joints) + len(ARM.gripper.fingers)
    assert not {transform.child for transform in DESCRIPTION.fixed_transforms()} & set(children)


@pytest.mark.parametrize("joints", CONFIGURATIONS)
@pytest.mark.parametrize("opening", OPENINGS)
def test_the_transforms_place_the_grasp_link_where_the_kinematics_place_the_grasp_point(kinematics, joints, opening):
    message = decode(transforms_message(TIMESTAMP_S, tuple(joints), opening, STARTED_S))
    position, orientation = place_in_robot_frame(_fixed() + _transforms(message), ARM.grasp_link)
    expected_position, expected_orientation = kinematics.forward_kinematics(tuple(joints))

    assert max(abs(a - b) for a, b in zip(position, expected_position)) <= TREE_TOLERANCE
    assert angle_shortest_path(orientation, expected_orientation) <= TREE_TOLERANCE


def test_the_gripper_opening_moves_the_jaw_alone():
    closed = _transforms(decode(transforms_message(TIMESTAMP_S, MEASURED, 0.0, STARTED_S)))
    opened = _transforms(decode(transforms_message(TIMESTAMP_S, MEASURED, 1.0, STARTED_S)))
    (finger,) = ARM.gripper.fingers

    changed = {child for (_, child, *before), (_, _, *after) in zip(closed, opened) if before != after}
    assert changed == {finger.link}


def test_a_snapshot_of_another_joint_count_is_refused():
    with pytest.raises(ValueError):
        transforms_message(TIMESTAMP_S, MEASURED[:4], 0.0, STARTED_S)
