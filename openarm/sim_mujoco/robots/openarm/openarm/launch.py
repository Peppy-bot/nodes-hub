#!/usr/bin/env python3
"""MuJoCo launch script for the openarm sim engine node."""

# pylint: disable=C0413
from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading
from pathlib import Path

from peppylib.runtime import NodeBuilder

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True
)
logger = logging.getLogger(__name__)

_ASSETS_DIR = Path(
    os.environ.get("PEPPY_ROBOT_ASSETS_DIR", str(Path(__file__).parent / "assets"))
)
def _version(hardware_version: str) -> str:
    version = hardware_version.lower()
    if version not in ("v1", "v2"):
        raise ValueError(f"hardware_version must be v1 or v2, got {hardware_version!r}")
    return version


def _scene_path(hardware_version: str) -> Path:
    # The v1 and v2 scenes are separate MJCF files (openarm_bimanual_v1.xml /
    # _v2.xml) in the base image's assets dir. A missing scene fails loudly at
    # load rather than silently simulating the other geometry.
    return _ASSETS_DIR / f"openarm_bimanual_{_version(hardware_version)}.xml"


_MUJOCO_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(_MUJOCO_DIR))
from _launcher import SimLauncher
from camera_common import CameraConfig, load_camera_configs, validate_camera_slots
from sim_topics import COLOR_CAMERA_SLOT_NAMES, RGBD_CAMERA_SLOT_NAMES, SimTopicIO

_CAMERAS_CONFIG_PATH = _MUJOCO_DIR / "config" / "cameras.json5"

_ready = threading.Event()
_stop = threading.Event()


def _camera_configs(params) -> list[CameraConfig]:
    """The cameras this launch renders, empty when the parameter is off. Parsed
    here so a bad config fails node setup, before the scene is compiled around
    it."""
    if not params.cameras_enabled:
        return []
    if _version(params.hardware_version) != "v2":
        raise ValueError(
            "cameras_enabled requires hardware_version v2: the camera geometry "
            "in config/cameras.json5 mounts on v2 links only"
        )
    cameras = load_camera_configs(_CAMERAS_CONFIG_PATH)
    validate_camera_slots(cameras, COLOR_CAMERA_SLOT_NAMES, RGBD_CAMERA_SLOT_NAMES)
    return cameras


async def _run_sim(params, node_runner) -> list:
    # Typed peppygen pub/sub lives on this loop; declare publishers and start the
    # command-consume tasks before the sim thread starts reading from them.
    cameras = _camera_configs(params)
    loop = asyncio.get_running_loop()
    io = SimTopicIO(node_runner, loop)
    await io.start()

    async def _run_sim_task() -> None:
        try:
            await loop.run_in_executor(
                None,
                SimLauncher(
                    _scene_path(params.hardware_version),
                    _ready,
                    _stop,
                    io,
                    params.state_rate_hz,
                    params.headless,
                    params.viewer_host,
                    params.viewer_port,
                    cameras,
                ).run,
            )
        finally:
            # Belt-and-braces against asyncio cancellation paths that race the
            # on_shutdown hook below; idempotent.
            _stop.set()

    async def _shutdown_hook() -> None:
        # Drive the sim executor to exit inside the runtime grace window so
        # SimLauncher's finally runs extension.shutdown(), then cancel the
        # consume tasks.
        _stop.set()
        await io.stop()

    node_runner.on_shutdown(_shutdown_hook)

    return [
        asyncio.create_task(_run_sim_task()),
    ]


def main() -> None:
    NodeBuilder().run(_run_sim)


if __name__ == "__main__":
    main()
