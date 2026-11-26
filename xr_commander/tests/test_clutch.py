import math

import pytest
from control_core_py.geometry import Pose, product

from tests.helpers import (
    OTHER_FRAME,
    ROBOT_FRAME,
    X,
    Y,
    Z,
    pose_at,
    quat_from_axis_angle,
    reported,
)
from xr_commander.clutch import (
    _MEASURED_JUMP_M,
    _MEASURED_JUMP_RAD,
    HandClutch,
    _displaced,
    _rotation_between,
    Setpoint,
)

HOME = pose_at((0.4, 0.0, 0.3))
HAND = pose_at((0.0, 0.0, 1.0))
AT_HOME = reported(HOME)


def at(pose: Pose, offset) -> Pose:
    """`pose` moved by `offset` (m), its orientation kept."""
    moved = tuple(p + o for p, o in zip(pose.position.position, offset))
    return pose_at(moved, pose.orientation.xyzw)


def position_of(target):
    return target.pose.position.position


def orientation_of(target):
    return target.pose.orientation.xyzw


def engaged_clutch(motion_scale=1.0, measured=AT_HOME, hand=HAND):
    """A clutch already past its engage tick."""
    clutch = HandClutch(motion_scale)
    assert clutch.step(squeezing=True, hand=hand, measured_ee=measured) is not None
    return clutch


def test_a_released_grip_commands_nothing():
    clutch = HandClutch(1.0)
    assert clutch.step(squeezing=False, hand=HAND, measured_ee=AT_HOME) is None
    assert not clutch.engaged


def test_engaging_without_a_measured_pose_is_refused_not_guessed():
    clutch = HandClutch(1.0)
    assert clutch.step(squeezing=True, hand=HAND, measured_ee=None) is None
    assert not clutch.engaged


def test_an_untracked_hand_commands_nothing_and_disengages():
    clutch = engaged_clutch()
    assert clutch.step(squeezing=True, hand=None, measured_ee=AT_HOME) is None
    assert not clutch.engaged


def test_the_engage_tick_commands_the_measured_pose_in_its_frame():
    clutch = HandClutch(1.0)
    first = clutch.step(squeezing=True, hand=HAND, measured_ee=AT_HOME)
    assert first == Setpoint(ROBOT_FRAME, HOME)
    assert clutch.engaged


def test_every_setpoint_names_the_frame_of_the_measured_pose_it_anchors_on():
    clutch = engaged_clutch(measured=reported(HOME, OTHER_FRAME))
    target = clutch.step(
        squeezing=True,
        hand=at(HAND, [0.05, 0.0, 0.0]),
        measured_ee=reported(HOME, OTHER_FRAME),
    )
    assert target.frame == OTHER_FRAME


def test_hand_translation_carries_to_the_end_effector():
    clutch = engaged_clutch()
    target = clutch.step(
        squeezing=True, hand=at(HAND, [0.05, 0.0, 0.0]), measured_ee=AT_HOME
    )
    assert position_of(target) == pytest.approx((0.45, 0.0, 0.3))


def test_motion_scale_shrinks_translation_but_never_rotation():
    clutch = engaged_clutch(motion_scale=0.5)
    turned = pose_at((0.2, 0.0, 1.0), quat_from_axis_angle(Y, 0.4))
    target = clutch.step(squeezing=True, hand=turned, measured_ee=AT_HOME)
    assert position_of(target) == pytest.approx((0.5, 0.0, 0.3))
    assert orientation_of(target) == pytest.approx(
        quat_from_axis_angle(Y, 0.4), abs=1e-9
    )


def test_hand_rotation_carries_to_the_end_effector():
    clutch = engaged_clutch()
    turned = pose_at(HAND.position.position, quat_from_axis_angle(Y, 0.3))
    target = clutch.step(squeezing=True, hand=turned, measured_ee=AT_HOME)
    assert orientation_of(target) == pytest.approx(
        quat_from_axis_angle(Y, 0.3), abs=1e-9
    )


def test_a_hand_turn_turns_the_end_effector_about_the_same_axis_of_the_robot_frame():
    # The hand's turn is applied about the frame's axes, before the
    # end-effector's own orientation: a quarter turn about z on a wrist
    # turned a quarter about x.
    wrist = quat_from_axis_angle(X, math.pi / 2)
    quarter_z = quat_from_axis_angle(Z, math.pi / 2)
    clutch = engaged_clutch(measured=reported(pose_at(HOME.position.position, wrist)))
    turned = pose_at(HAND.position.position, quarter_z)
    target = clutch.step(
        squeezing=True,
        hand=turned,
        measured_ee=reported(pose_at(HOME.position.position, wrist)),
    )
    assert orientation_of(target) == pytest.approx(product(quarter_z, wrist), abs=1e-12)


def test_a_pinned_arm_is_led_best_effort_and_returning_the_hand_returns_it():
    # The arm is stuck at HOME; the command still asks for the full excursion
    # (the follower's governor decides what happens), and because the snapshot
    # never moves, bringing the hand back commands HOME again, not beyond it.
    clutch = engaged_clutch()
    for _ in range(50):
        target = clutch.step(
            squeezing=True, hand=at(HAND, [1.0, 0.0, 0.0]), measured_ee=AT_HOME
        )
    assert position_of(target) == pytest.approx((1.4, 0.0, 0.3))
    target = clutch.step(squeezing=True, hand=HAND, measured_ee=AT_HOME)
    assert position_of(target) == pytest.approx(HOME.position.position, abs=1e-12)


def test_re_engaging_snapshots_the_arm_where_it_now_is():
    clutch = engaged_clutch()
    clutch.step(squeezing=False, hand=HAND, measured_ee=AT_HOME)
    moved = at(HOME, [0.2, -0.1, 0.05])
    target = clutch.step(squeezing=True, hand=HAND, measured_ee=reported(moved))
    assert target.pose == moved


def test_a_stale_gap_then_resumption_does_not_jump():
    # Headset freezes (hand None, clutch releases), operator re-squeezes with
    # the hand somewhere else entirely: nothing may move.
    clutch = engaged_clutch()
    clutch.step(squeezing=True, hand=None, measured_ee=AT_HOME)
    elsewhere = pose_at((5.0, -3.0, 2.0), quat_from_axis_angle(X, 1.1))
    target = clutch.step(squeezing=True, hand=elsewhere, measured_ee=AT_HOME)
    assert target.pose == HOME


def test_the_arm_tracking_normally_does_not_read_as_a_jump():
    clutch = engaged_clutch()
    crept = at(HOME, [_MEASURED_JUMP_M * 0.5, 0.0, 0.0])
    assert (
        clutch.step(squeezing=True, hand=HAND, measured_ee=reported(crept)) is not None
    )
    assert clutch.engaged


def test_a_teleporting_end_effector_drops_the_engagement():
    # A re-anchor or re-pair moves the arm without it traveling: the
    # engagement drops.
    clutch = engaged_clutch()
    teleported = at(HOME, [_MEASURED_JUMP_M * 10.0, 0.0, 0.0])
    assert (
        clutch.step(squeezing=True, hand=HAND, measured_ee=reported(teleported)) is None
    )
    assert not clutch.engaged


def test_a_flipping_end_effector_drops_the_engagement_too():
    # The rotational discontinuity: same spot, wrist suddenly swung around.
    clutch = engaged_clutch()
    flipped = pose_at(
        HOME.position.position, quat_from_axis_angle(Y, _MEASURED_JUMP_RAD * 2.0)
    )
    assert clutch.step(squeezing=True, hand=HAND, measured_ee=reported(flipped)) is None
    assert not clutch.engaged


def test_the_wrist_slewing_normally_does_not_read_as_a_flip():
    clutch = engaged_clutch()
    slewed = pose_at(
        HOME.position.position, quat_from_axis_angle(Y, _MEASURED_JUMP_RAD * 0.5)
    )
    assert (
        clutch.step(squeezing=True, hand=HAND, measured_ee=reported(slewed)) is not None
    )
    assert clutch.engaged


def test_an_orientation_reported_with_its_sign_flipped_is_the_same_rotation():
    # q and -q are one rotation: a follower that reports either is tracking.
    turned = quat_from_axis_angle(Y, 0.2)
    negated = tuple(-c for c in turned)
    clutch = engaged_clutch(measured=reported(pose_at(HOME.position.position, turned)))
    assert (
        clutch.step(
            squeezing=True,
            hand=HAND,
            measured_ee=reported(pose_at(HOME.position.position, negated)),
        )
        is not None
    )
    assert clutch.engaged


def test_an_end_effector_measured_twice_at_one_orientation_is_no_jump():
    # The rotation between these two equal orientations has a w that rounds
    # to 1.0000000000000004: its angle reads 0, never a domain error.
    still = pose_at(HOME.position.position, (-0.218, -0.128, 0.941, -0.216))
    clutch = engaged_clutch(measured=reported(still))
    assert (
        clutch.step(squeezing=True, hand=HAND, measured_ee=reported(still)) is not None
    )
    assert clutch.engaged


def test_after_a_jump_the_next_tick_re_snapshots_where_the_arm_now_is():
    clutch = engaged_clutch()
    teleported = reported(at(HOME, [1.0, 0.0, 0.0]))
    clutch.step(squeezing=True, hand=HAND, measured_ee=teleported)
    target = clutch.step(squeezing=True, hand=HAND, measured_ee=teleported)
    assert target.pose == teleported.value


def test_a_measured_pose_in_another_frame_drops_the_engagement():
    # The same numbers in another frame: a re-anchor of the follower's frame.
    clutch = engaged_clutch()
    assert (
        clutch.step(squeezing=True, hand=HAND, measured_ee=reported(HOME, OTHER_FRAME))
        is None
    )
    assert not clutch.engaged


def test_after_a_change_of_frame_the_next_tick_re_snapshots_in_the_new_frame():
    clutch = engaged_clutch()
    moved = at(HAND, [0.05, 0.0, 0.0])
    elsewhere = reported(HOME, OTHER_FRAME)
    clutch.step(squeezing=True, hand=moved, measured_ee=elsewhere)
    target = clutch.step(squeezing=True, hand=moved, measured_ee=elsewhere)
    assert target == Setpoint(OTHER_FRAME, HOME)


def test_displacing_by_the_delta_between_two_poses_rebuilds_the_second():
    base = pose_at((0.1, 0.2, 0.3), quat_from_axis_angle(X, 0.4))
    moved = pose_at((1.1, 0.2, -0.7), quat_from_axis_angle(Z, 0.9))
    delta = tuple(
        m - b for m, b in zip(moved.position.position, base.position.position)
    )
    rebuilt = _displaced(
        base, delta, _rotation_between(moved.orientation, base.orientation)
    )
    assert rebuilt.position.position == pytest.approx(
        moved.position.position, abs=1e-12
    )
    assert rebuilt.orientation.xyzw == pytest.approx(
        moved.orientation.xyzw, abs=1e-12
    )
