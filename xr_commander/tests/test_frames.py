import math

import numpy as np
import pytest

from tests.helpers import IDENTITY, X, Y
from xr_commander.frames import (
    Pose,
    apply_offset,
    quat_axis_angle,
    quat_conjugate,
    quat_from_axis_angle,
    quat_multiply,
    quat_normalize,
    relative_rotation,
)

Z = np.array([0.0, 0.0, 1.0])


def test_quat_normalize_rejects_the_shapes_that_would_poison_downstream_math():
    with pytest.raises(ValueError, match="shape"):
        quat_normalize(np.array([0.0, 0.0, 1.0]))
    with pytest.raises(ValueError, match="finite"):
        quat_normalize(np.array([0.0, 0.0, 0.0, np.nan]))
    with pytest.raises(ValueError, match="finite"):
        quat_normalize(np.array([np.inf, 0.0, 0.0, 1.0]))
    with pytest.raises(ValueError, match="zero length"):
        quat_normalize(np.zeros(4))


def test_pose_rejects_a_non_finite_position():
    with pytest.raises(ValueError, match="finite"):
        Pose(np.array([0.0, np.nan, 0.0]), IDENTITY)
    with pytest.raises(ValueError, match="shape"):
        Pose(np.array([0.0, 0.0]), IDENTITY)


def test_pose_normalises_and_canonicalises_its_orientation():
    pose = Pose(np.zeros(3), np.array([0.0, 0.0, 0.0, -2.0]))
    # -2w is the identity rotation written with a negative scalar and a
    # non-unit length; both are fixed at construction.
    assert np.allclose(pose.orientation, IDENTITY)
    assert pose.orientation[3] >= 0.0


def test_quaternion_multiply_composes_right_to_left():
    quarter_z = quat_from_axis_angle(Z, math.pi / 2)
    half_z = quat_multiply(quarter_z, quarter_z)
    axis, angle = quat_axis_angle(half_z)
    assert np.allclose(axis, Z, atol=1e-12)
    assert angle == pytest.approx(math.pi, abs=1e-12)


def test_conjugate_inverts():
    q = quat_from_axis_angle(np.array([1.0, 2.0, 3.0]), 0.7)
    assert np.allclose(quat_multiply(q, quat_conjugate(q)), IDENTITY, atol=1e-12)


def test_axis_angle_round_trips_and_stays_in_the_upper_half():
    for angle in (0.0, 0.01, 1.0, math.pi - 1e-6):
        q = quat_from_axis_angle(Y, angle)
        axis, recovered = quat_axis_angle(q)
        assert recovered == pytest.approx(angle, abs=1e-9)
        if angle > 0.0:
            assert np.allclose(axis, Y, atol=1e-9)
    # A rotation written with a negative scalar is the same rotation, and must
    # decompose to the same angle rather than its 2*pi complement.
    q = quat_from_axis_angle(Y, 1.0)
    _axis, from_negated = quat_axis_angle(-q)
    assert from_negated == pytest.approx(1.0, abs=1e-9)


def test_axis_angle_of_identity_is_a_zero_rotation_that_recomposes_to_identity():
    axis, angle = quat_axis_angle(IDENTITY)
    assert angle == 0.0
    assert np.allclose(quat_from_axis_angle(axis, angle), IDENTITY)


def test_apply_offset_matches_the_delta_it_was_built_from():
    base = Pose(np.array([0.1, 0.2, 0.3]), quat_from_axis_angle(X, 0.4))
    moved = Pose(np.array([1.1, 0.2, -0.7]), quat_from_axis_angle(Z, 0.9))
    rebuilt = apply_offset(
        base, moved.position - base.position, relative_rotation(moved, base)
    )
    assert np.allclose(rebuilt.position, moved.position, atol=1e-12)
    assert np.allclose(rebuilt.orientation, moved.orientation, atol=1e-12)
