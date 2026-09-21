"""The pinhole model the geometry topic carries for a rendered camera."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))

from camera_geometry import (  # noqa: E402
    DEPTH_MODEL,
    DEPTH_TO_COLOR_ORIENTATION,
    DEPTH_TO_COLOR_POSITION,
    DISTORTION_MODEL,
    pinhole,
)


def test_the_chest_cameras_pinhole_follows_its_field_of_view():
    # The OpenArm v2 chest camera: 52 degrees of vertical field of view.
    color = pinhole(52.0, 1280, 720)
    assert color.fx == pytest.approx(738.1, abs=0.05)
    assert color.fy == color.fx
    # Pixel centres sit on whole numbers, so the middle of 1280 is 639.5.
    assert (color.cx, color.cy) == (639.5, 359.5)
    # The same camera rendered at the depth stream's own size.
    depth = pinhole(52.0, 640, 360)
    assert depth.fx == pytest.approx(369.06, abs=0.05)
    assert (depth.cx, depth.cy) == (319.5, 179.5)


def test_a_wrist_cameras_pinhole_follows_its_field_of_view():
    # The OpenArm v2 wrist cameras: 66 degrees at 960x600.
    wrist = pinhole(66.0, 960, 600)
    assert wrist.fx == pytest.approx(461.96, abs=0.05)
    assert (wrist.cx, wrist.cy) == (479.5, 299.5)
    assert (wrist.width, wrist.height) == (960, 600)


def test_a_rendered_camera_is_an_ideal_pinhole_aligned_with_its_depth():
    assert DISTORTION_MODEL == "none"
    assert DEPTH_MODEL == "z"
    assert DEPTH_TO_COLOR_POSITION == (0.0, 0.0, 0.0)
    assert DEPTH_TO_COLOR_ORIENTATION == (0.0, 0.0, 0.0, 1.0)
