"""What the suites read out of sim_base_images: where the base images of both
simulation nodes are built from, the value a shell file of it pins, and the
stages the Isaac base image bakes."""

import os
import platform
import re
import sys
from pathlib import Path

import pytest

IMAGES_DIR = Path(__file__).resolve().parents[2] / "sim_base_images"

# The bundles are built on Linux x86_64, the platform of the Isaac image.
# MuJoCo rounds what it compiles differently on another architecture, so only
# there is a stage built here its bundle's byte for byte.
only_on_linux_x86_64 = pytest.mark.skipif(
    not (sys.platform == "linux" and platform.machine() == "x86_64"),
    reason="the bundle is built on Linux x86_64, and MuJoCo rounds differently elsewhere",
)


def shell_value(script: str, name: str) -> str:
    return re.search(rf'^{name}="([^"]*)"', script, re.MULTILINE).group(1)


def baked_stage(robot: str, filename: str) -> Path | None:
    """The stage `filename` the base image bakes for `robot`, where
    PEPPY_ROBOT_ASSETS_DIR holds one."""
    assets = os.environ.get("PEPPY_ROBOT_ASSETS_DIR")
    if not assets:
        return None
    path = Path(assets) / robot / filename
    return path if path.is_file() else None
