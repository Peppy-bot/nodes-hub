"""The camera model: what turns a pixel and its depth into a world
position. The camera contracts carry image size and encoding only, so
the field of view and the camera's pose come from the node's parameters.

Conventions, the same the engines use for their cameras: the camera
looks along its own -Z with +Y as image up and +X as image right; a
depth sample is the distance along the optical axis; the pose is the
optical frame in the world frame limb_motion uses, a unit quaternion
[x, y, z, w].

A pixel becomes a ray through `pixel_to_ray`, in camera_geometry:v1's
terms: the pixel through fx, fy, cx, cy to a normalised point, then the
lens taken off. The distortion models are the contract's: "none";
"plumb_bob", OpenCV's k1, k2, p1, p2, k3 taking an undistorted point to
the distorted one, so undistorting inverts it by iteration; and
"inverse_plumb_bob", the same polynomial run from the distorted point to
the undistorted one, applied once. Any other name is refused.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Optional

from ..ports import Quat, Vec3

# The models camera_geometry:v1 names, and how a pixel is undistorted
# under each.
NONE = "none"
PLUMB_BOB = "plumb_bob"
INVERSE_PLUMB_BOB = "inverse_plumb_bob"
DISTORTION_MODELS = (NONE, PLUMB_BOB, INVERSE_PLUMB_BOB)

# Iterations of the fixed-point inverse of "plumb_bob". OpenCV's
# undistortPoints runs a handful; the lenses this meets converge long before.
_UNDISTORT_ITERATIONS = 20


@dataclass(frozen=True)
class Intrinsics:
    """Where the pixels of an image of `width` x `height` point: OpenCV's
    pinhole through fx, fy, cx, cy, with the lens named by `distortion_model`
    as camera_geometry:v1 reports it."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion_model: str = NONE
    distortion: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"intrinsics need a positive image size, got {self.width}x{self.height}")
        if not all(math.isfinite(v) for v in (self.fx, self.fy, self.cx, self.cy)) or self.fx <= 0.0 or self.fy <= 0.0:
            raise ValueError(f"intrinsics need positive finite focal lengths, got fx {self.fx} fy {self.fy}")
        if self.distortion_model not in DISTORTION_MODELS:
            raise ValueError(f"unknown distortion model {self.distortion_model!r}; known: {', '.join(DISTORTION_MODELS)}")

    @classmethod
    def from_fovy(cls, fovy_deg: float, width: int, height: int) -> "Intrinsics":
        """A rendered camera with no lens: the vertical field of view gives
        the focal length and the image centre is the principal point. For
        tests and fixtures; a real camera answers get_color_intrinsics."""
        if not (0.0 < fovy_deg < 180.0):
            raise ValueError(f"the field of view must be between 0 and 180 degrees, got {fovy_deg}")
        focal = (height / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
        return cls(width, height, focal, focal, width / 2.0, height / 2.0)

    def scaled_to(self, width: int, height: int) -> "Intrinsics":
        """The same camera at another image size: focal lengths and the
        principal point follow the pixels (pixel centres, so (c + 0.5) s - 0.5)."""
        if (width, height) == (self.width, self.height):
            return self
        sx, sy = width / self.width, height / self.height
        return Intrinsics(width, height, self.fx * sx, self.fy * sy, (self.cx + 0.5) * sx - 0.5, (self.cy + 0.5) * sy - 0.5, self.distortion_model, self.distortion)


@dataclass(frozen=True)
class CameraModel:
    """The camera the brain looks through: its pose in the robot's world
    frame, from the camera_pose parameter, and its intrinsics, from the
    camera itself once it has answered. Not ready until it has."""

    position: Vec3
    orientation: Quat
    intrinsics: Optional[Intrinsics] = None

    @classmethod
    def from_parameters(cls, pose_text: str) -> "CameraModel":
        """`pose_text` is "x y z qx qy qz qw", as the camera_pose parameter
        spells it. The intrinsics come later, with `with_intrinsics`."""
        parts = pose_text.replace(",", " ").split()
        if len(parts) != 7:
            raise ValueError("camera_pose must hold seven numbers: x y z qx qy qz qw")
        try:
            values = [float(part) for part in parts]
        except ValueError:
            raise ValueError("camera_pose must hold seven numbers: x y z qx qy qz qw") from None
        if any(not math.isfinite(value) for value in values):
            raise ValueError("camera_pose must be finite")
        x, y, z, qx, qy, qz, qw = values
        norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if norm < 1e-9:
            raise ValueError("camera_pose quaternion must not be zero")
        return cls((x, y, z), (qx / norm, qy / norm, qz / norm, qw / norm))

    def with_intrinsics(self, intrinsics: Intrinsics) -> "CameraModel":
        return CameraModel(self.position, self.orientation, intrinsics)

    @property
    def ready(self) -> bool:
        return self.intrinsics is not None

    def deproject(self, u: float, v: float, depth_m: float, width: int, height: int) -> Vec3:
        """The world position of pixel (u, v), x right and y down, seen at
        `depth_m` along the optical axis in an image of `width` x `height`:
        the pixel through the intrinsics, scaled to that image size when the
        stream is not at the size the intrinsics were given for, the lens
        taken off, then the ray placed by the pose."""
        if self.intrinsics is None:
            raise ValueError("the camera's intrinsics are not known yet")
        k = self.intrinsics.scaled_to(width, height)
        x, y = pixel_to_ray(u, v, k.fx, k.fy, k.cx, k.cy, k.distortion_model, k.distortion)
        camera = (x * depth_m, -y * depth_m, -depth_m)
        rotated = rotate(self.orientation, camera)
        return (
            self.position[0] + rotated[0],
            self.position[1] + rotated[1],
            self.position[2] + rotated[2],
        )

    def forward(self) -> Vec3:
        """The world direction the camera looks along."""
        return rotate(self.orientation, (0.0, 0.0, -1.0))


def pixel_to_ray(
    u: float,
    v: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    distortion_model: str,
    distortion: Sequence[float],
) -> tuple[float, float]:
    """The ideal image point (x, y) at unit depth that pixel (u, v) shows,
    x right and y down as camera_geometry:v1 has them: the pixel through
    fx, fy, cx, cy, then the lens taken off under `distortion_model`."""
    xd, yd = (u - cx) / fx, (v - cy) / fy
    return undistort(distortion_model, distortion, xd, yd)


def undistort(model: str, coefficients: Sequence[float], xd: float, yd: float) -> tuple[float, float]:
    """The undistorted normalised point behind the distorted one (xd, yd)."""
    if model == NONE:
        if len(coefficients) != 0:
            raise ValueError(f"distortion model {model!r} carries no coefficients, got {len(coefficients)}")
        return xd, yd
    if model not in (PLUMB_BOB, INVERSE_PLUMB_BOB):
        raise ValueError(f"distortion model {model!r} is not one camera_geometry names: {DISTORTION_MODELS}")
    if len(coefficients) != 5:
        raise ValueError(f"distortion model {model!r} takes k1, k2, p1, p2, k3, got {len(coefficients)} coefficients")
    if any(not math.isfinite(c) for c in coefficients):
        raise ValueError(f"distortion coefficients must be finite, got {list(coefficients)}")
    if model == INVERSE_PLUMB_BOB:
        # The polynomial already runs from the distorted point to the
        # undistorted one: one application, nothing to invert.
        return _apply(coefficients, xd, yd)
    # "plumb_bob" runs the other way, so the undistorted point is found by
    # iterating the forward model from the distorted point.
    x, y = xd, yd
    for _ in range(_UNDISTORT_ITERATIONS):
        radial, dx, dy = _terms(coefficients, x, y)
        x = (xd - dx) / radial
        y = (yd - dy) / radial
    return x, y


def distort(model: str, coefficients: Sequence[float], x: float, y: float) -> tuple[float, float]:
    """The distorted normalised point the lens makes of the ideal one (x, y):
    the exact inverse of `undistort`, for projecting and for tests."""
    if model == NONE:
        return x, y
    if model not in (PLUMB_BOB, INVERSE_PLUMB_BOB):
        raise ValueError(f"distortion model {model!r} is not one camera_geometry names: {DISTORTION_MODELS}")
    if model == PLUMB_BOB:
        return _apply(coefficients, x, y)
    # Under "inverse_plumb_bob" the polynomial undistorts, so distorting is
    # the iteration, from the ideal point.
    xd, yd = x, y
    for _ in range(_UNDISTORT_ITERATIONS):
        radial, dx, dy = _terms(coefficients, xd, yd)
        xd = (x - dx) / radial
        yd = (y - dy) / radial
    return xd, yd


def _apply(coefficients: Sequence[float], x: float, y: float) -> tuple[float, float]:
    """The Brown-Conrady polynomial applied once to (x, y)."""
    radial, dx, dy = _terms(coefficients, x, y)
    return x * radial + dx, y * radial + dy


def _terms(coefficients: Sequence[float], x: float, y: float) -> tuple[float, float, float]:
    """The radial factor and the tangential shift of the polynomial at
    (x, y), with k1, k2, p1, p2, k3 as OpenCV orders them."""
    k1, k2, p1, p2, k3 = coefficients
    r2 = x * x + y * y
    radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
    dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return radial, dx, dy


def rotate(q: Quat, v: Vec3) -> Vec3:
    """`v` rotated by the unit quaternion `q` = [x, y, z, w]."""
    qx, qy, qz, qw = q
    vx, vy, vz = v
    # t = 2 * cross(q.xyz, v); v' = v + w * t + cross(q.xyz, t)
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    )
