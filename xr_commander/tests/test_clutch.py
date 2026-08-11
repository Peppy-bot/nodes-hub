import numpy as np
import pytest

from tests.helpers import IDENTITY, X, Y
from xr_commander.clutch import _MEASURED_JUMP_M, _MEASURED_JUMP_RAD, HandClutch
from xr_commander.frames import Pose, quat_axis_angle, quat_from_axis_angle

HOME = Pose(np.array([0.4, 0.0, 0.3]), IDENTITY)
HAND = Pose(np.array([0.0, 0.0, 1.0]), IDENTITY)


def at(pose: Pose, offset) -> Pose:
    return Pose(pose.position + np.asarray(offset, dtype=float), pose.orientation)


def engaged_clutch(motion_scale=1.0, measured=HOME, hand=HAND):
    """A clutch already past its engage tick."""
    clutch = HandClutch(motion_scale)
    assert clutch.step(squeezing=True, hand=hand, measured_ee=measured) is not None
    return clutch


def test_a_released_grip_commands_nothing():
    clutch = HandClutch(1.0)
    assert clutch.step(squeezing=False, hand=HAND, measured_ee=HOME) is None
    assert not clutch.engaged


def test_engaging_without_a_measured_pose_is_refused_not_guessed():
    clutch = HandClutch(1.0)
    assert clutch.step(squeezing=True, hand=HAND, measured_ee=None) is None
    assert not clutch.engaged


def test_an_untracked_hand_commands_nothing_and_disengages():
    clutch = engaged_clutch()
    assert clutch.step(squeezing=True, hand=None, measured_ee=HOME) is None
    assert not clutch.engaged


def test_the_engage_tick_commands_the_measured_pose():
    clutch = HandClutch(1.0)
    first = clutch.step(squeezing=True, hand=HAND, measured_ee=HOME)
    assert np.allclose(first.position, HOME.position)
    assert np.allclose(first.orientation, HOME.orientation)
    assert clutch.engaged


def test_hand_translation_carries_to_the_end_effector():
    clutch = engaged_clutch()
    target = clutch.step(
        squeezing=True, hand=at(HAND, [0.05, 0.0, 0.0]), measured_ee=HOME
    )
    assert np.allclose(target.position, HOME.position + np.array([0.05, 0.0, 0.0]))


def test_motion_scale_shrinks_translation_but_never_rotation():
    clutch = engaged_clutch(motion_scale=0.5)
    turned = Pose(
        HAND.position + np.array([0.2, 0.0, 0.0]), quat_from_axis_angle(Y, 0.4)
    )
    target = clutch.step(squeezing=True, hand=turned, measured_ee=HOME)
    assert np.allclose(target.position, HOME.position + np.array([0.1, 0.0, 0.0]))
    _axis, angle = quat_axis_angle(target.orientation)
    assert angle == pytest.approx(0.4, abs=1e-9)


def test_hand_rotation_carries_to_the_end_effector():
    clutch = engaged_clutch()
    turned = Pose(HAND.position, quat_from_axis_angle(Y, 0.3))
    target = clutch.step(squeezing=True, hand=turned, measured_ee=HOME)
    axis, angle = quat_axis_angle(target.orientation)
    assert angle == pytest.approx(0.3, abs=1e-9)
    assert np.allclose(axis, Y, atol=1e-9)


def test_a_pinned_arm_is_led_best_effort_and_returning_the_hand_returns_it():
    # The arm is stuck at HOME; the command still asks for the full excursion
    # (the follower's governor decides what happens), and because the snapshot
    # never moves, bringing the hand back commands HOME again, not beyond it.
    clutch = engaged_clutch()
    for _ in range(50):
        target = clutch.step(
            squeezing=True, hand=at(HAND, [1.0, 0.0, 0.0]), measured_ee=HOME
        )
    assert np.allclose(target.position, HOME.position + np.array([1.0, 0.0, 0.0]))
    target = clutch.step(squeezing=True, hand=HAND, measured_ee=HOME)
    assert np.allclose(target.position, HOME.position, atol=1e-12)


def test_re_engaging_snapshots_the_arm_where_it_now_is():
    clutch = engaged_clutch()
    clutch.step(squeezing=False, hand=HAND, measured_ee=HOME)
    moved = at(HOME, [0.2, -0.1, 0.05])
    target = clutch.step(squeezing=True, hand=HAND, measured_ee=moved)
    assert np.allclose(target.position, moved.position)


def test_a_stale_gap_then_resumption_does_not_jump():
    # Headset freezes (hand None, clutch releases), operator re-squeezes with
    # the hand somewhere else entirely: nothing may move.
    clutch = engaged_clutch()
    clutch.step(squeezing=True, hand=None, measured_ee=HOME)
    elsewhere = Pose(np.array([5.0, -3.0, 2.0]), quat_from_axis_angle(X, 1.1))
    target = clutch.step(squeezing=True, hand=elsewhere, measured_ee=HOME)
    assert np.allclose(target.position, HOME.position)
    assert np.allclose(target.orientation, HOME.orientation)


def test_the_arm_tracking_normally_does_not_read_as_a_jump():
    clutch = engaged_clutch()
    crept = at(HOME, [_MEASURED_JUMP_M * 0.5, 0.0, 0.0])
    assert clutch.step(squeezing=True, hand=HAND, measured_ee=crept) is not None
    assert clutch.engaged


def test_a_teleporting_end_effector_drops_the_engagement():
    # A re-anchor or re-pair moves the arm without it travelling; honouring the
    # old snapshot would drag it back across the discontinuity.
    clutch = engaged_clutch()
    teleported = at(HOME, [_MEASURED_JUMP_M * 10.0, 0.0, 0.0])
    assert clutch.step(squeezing=True, hand=HAND, measured_ee=teleported) is None
    assert not clutch.engaged


def test_a_flipping_end_effector_drops_the_engagement_too():
    # The rotational discontinuity: same spot, wrist suddenly swung around.
    clutch = engaged_clutch()
    flipped = Pose(HOME.position, quat_from_axis_angle(Y, _MEASURED_JUMP_RAD * 2.0))
    assert clutch.step(squeezing=True, hand=HAND, measured_ee=flipped) is None
    assert not clutch.engaged


def test_the_wrist_slewing_normally_does_not_read_as_a_flip():
    clutch = engaged_clutch()
    slewed = Pose(HOME.position, quat_from_axis_angle(Y, _MEASURED_JUMP_RAD * 0.5))
    assert clutch.step(squeezing=True, hand=HAND, measured_ee=slewed) is not None
    assert clutch.engaged


def test_after_a_jump_the_next_tick_re_snapshots_where_the_arm_now_is():
    clutch = engaged_clutch()
    teleported = at(HOME, [1.0, 0.0, 0.0])
    clutch.step(squeezing=True, hand=HAND, measured_ee=teleported)
    target = clutch.step(squeezing=True, hand=HAND, measured_ee=teleported)
    assert np.allclose(target.position, teleported.position)
