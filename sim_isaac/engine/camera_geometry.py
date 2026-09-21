"""Where a rendered camera's pixels point, in the terms of camera_geometry:v1.

The model's entry gives a camera its vertical field of view and the sizes it
renders at; this turns them into the pinhole model the geometry topic of the
camera pairings carries. The conventions are the contract's, which are
OpenCV's: the centre of the first pixel is (0, 0), and the optical frame has
+X to the right of the image, +Y down it and +Z along the view.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# What the geometry topic says of a rendered camera, in the camera_geometry
# contract's terms: an ideal pinhole, its depth the distance to the image
# plane, and its depth seen from the colour camera's own optical frame.
DISTORTION_MODEL = "none"
DEPTH_MODEL = "z"
DEPTH_TO_COLOR_POSITION = (0.0, 0.0, 0.0)
DEPTH_TO_COLOR_ORIENTATION = (0.0, 0.0, 0.0, 1.0)


@dataclass(frozen=True)
class Pinhole:
    """The pinhole model of one published grid, in the camera_geometry
    contract's pixel convention, which is OpenCV's: the centre of the top-left
    pixel is (0, 0), so pixel centres sit on whole numbers. Focal lengths and
    principal point are in pixels."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


def pinhole(fovy_deg: float, width: int, height: int) -> Pinhole:
    """The model of a grid rendered `width` x `height` under a vertical field
    of view of `fovy_deg`: square pixels and the optical axis through the image
    centre, which is what an MJCF or a USD camera renders. With pixel centres
    on whole numbers the middle of a `width` x `height` image is
    ((width - 1) / 2, (height - 1) / 2)."""
    focal = (height / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
    return Pinhole(width, height, focal, focal, (width - 1) / 2.0, (height - 1) / 2.0)
