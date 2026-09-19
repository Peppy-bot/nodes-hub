"""The camera model: what turns a pixel and its depth into a world
position. The camera contracts carry image size and encoding only, so
the field of view and the camera's pose come from the node's parameters.

Conventions, the same the engines use for their cameras: the camera
looks along its own -Z with +Y as image up and +X as image right; a
depth sample is the distance along the optical axis; the pose is the
optical frame in the world frame limb_motion uses, a unit quaternion
[x, y, z, w].
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..ports import Quat, Vec3


@dataclass(frozen=True)
class CameraModel:
    fovy_deg: float
    position: Vec3
    orientation: Quat

    @classmethod
    def from_parameters(cls, fovy_deg: float, pose_text: str) -> "CameraModel":
        """`pose_text` is "x y z qx qy qz qw", as the camera_pose parameter
        spells it."""
        if not (0.0 < fovy_deg < 180.0):
            raise ValueError(f"camera_fovy_deg must be between 0 and 180, got {fovy_deg}")
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
        return cls(fovy_deg, (x, y, z), (qx / norm, qy / norm, qz / norm, qw / norm))

    def focal_px(self, height: int) -> float:
        """Pixels per unit of tangent, from the vertical field of view;
        square pixels, so the same horizontally."""
        return (height / 2.0) / math.tan(math.radians(self.fovy_deg) / 2.0)

    def deproject(self, u: float, v: float, depth_m: float, width: int, height: int) -> Vec3:
        """The world position of pixel (u, v), x right and y down, seen at
        `depth_m` along the optical axis in an image of `width` x `height`."""
        focal = self.focal_px(height)
        cx, cy = width / 2.0, height / 2.0
        camera = ((u - cx) / focal * depth_m, -(v - cy) / focal * depth_m, -depth_m)
        rotated = rotate(self.orientation, camera)
        return (
            self.position[0] + rotated[0],
            self.position[1] + rotated[1],
            self.position[2] + rotated[2],
        )

    def forward(self) -> Vec3:
        """The world direction the camera looks along."""
        return rotate(self.orientation, (0.0, 0.0, -1.0))


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
