"""Pose algebra, on quaternions only, in xyzw (scalar-last) wire order."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Below this a vector carries no usable direction, and normalising it would be
# numerically meaningless rather than merely imprecise.
_EPS = 1e-12


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """`q` at unit length; ValueError on misshapen, non-finite, or zero input."""
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError(f"quaternion must have shape (4,), got {q.shape}")
    if not np.all(np.isfinite(q)):
        raise ValueError(f"quaternion is not finite: {q}")
    norm = float(np.linalg.norm(q))
    if norm < _EPS:
        raise ValueError("quaternion has zero length")
    return q / norm


def quat_canonical(q: np.ndarray) -> np.ndarray:
    """The non-negative-scalar representative: q and -q are one rotation."""
    return -q if q[3] < 0.0 else q


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """The inverse of a unit quaternion."""
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product `a * b`: apply `b` first, then `a`."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float64,
    )


def quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation of `angle` about `axis`; degenerate axis or zero angle is identity."""
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm < _EPS or angle == 0.0:
        return np.array([0.0, 0.0, 0.0, 1.0])
    half = 0.5 * angle
    return np.concatenate([(axis / norm) * math.sin(half), [math.cos(half)]])


def quat_axis_angle(q: np.ndarray) -> tuple[np.ndarray, float]:
    """(axis, angle) with angle in [0, pi]; identity yields (+Z, 0)."""
    q = quat_canonical(q)
    vec_norm = float(np.linalg.norm(q[:3]))
    if vec_norm < _EPS:
        return np.array([0.0, 0.0, 1.0]), 0.0
    # atan2 keeps its significant digits at the small per-tick angles.
    return q[:3] / vec_norm, 2.0 * math.atan2(vec_norm, float(q[3]))


@dataclass(frozen=True)
class Pose:
    """A rigid pose: meters + unit xyzw quaternion, validated at construction."""

    position: np.ndarray
    orientation: np.ndarray

    def __post_init__(self) -> None:
        position = np.asarray(self.position, dtype=np.float64)
        if position.shape != (3,):
            raise ValueError(f"position must have shape (3,), got {position.shape}")
        if not np.all(np.isfinite(position)):
            raise ValueError(f"position is not finite: {position}")
        object.__setattr__(self, "position", position)
        object.__setattr__(
            self, "orientation", quat_canonical(quat_normalize(self.orientation))
        )

    @classmethod
    def from_xyz_quat(cls, position, orientation) -> Pose:
        """Build from two wire sequences: [x, y, z] and [x, y, z, w]."""
        return cls(
            np.asarray(position, dtype=np.float64),
            np.asarray(orientation, dtype=np.float64),
        )

    def as_wire(self) -> tuple[list[float], list[float]]:
        """The pair of plain float lists pose_link carries."""
        return [float(x) for x in self.position], [float(x) for x in self.orientation]


def relative_rotation(current: Pose, reference: Pose) -> np.ndarray:
    """The rotation carrying `reference`'s orientation onto `current`'s."""
    return quat_multiply(current.orientation, quat_conjugate(reference.orientation))


def apply_offset(base: Pose, translation: np.ndarray, rotation: np.ndarray) -> Pose:
    """`base` displaced by a frame-aligned translation and rotation."""
    return Pose(base.position + translation, quat_multiply(rotation, base.orientation))
