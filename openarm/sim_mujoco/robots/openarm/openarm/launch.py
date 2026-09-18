#!/usr/bin/env python3
"""MuJoCo launch script for the openarm sim engine node."""

# pylint: disable=C0413
from __future__ import annotations

import asyncio
import logging
import sys
import threading
from pathlib import Path

from peppylib.runtime import NodeBuilder

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True
)
logger = logging.getLogger(__name__)

_MUJOCO_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(_MUJOCO_DIR))
import head_camera
from _launcher import SimLauncher
from bridge_extension import Layout
from camera_common import CameraConfig, load_camera_configs, validate_camera_slots
from robots import Limbs, Registry
from robots_io import RobotsIO
from scenes import Catalogue
from sim_topics import COLOR_CAMERA_SLOT_NAMES, RGBD_CAMERA_SLOT_NAMES, SimTopicIO
from stands import Stands

_CAMERAS_CONFIG_PATH = _MUJOCO_DIR / "config" / "cameras.json5"

# The head camera pack apptainer.def stages at image build (head_camera.py
# fetch), beside the node's code. Setup reads it, and a robot standing as a
# model that carries the head camera draws it.
_HEAD_CAMERA_DIR = Path(__file__).parent / "assets" / "head_camera"

_stop = threading.Event()


def _camera_configs(params) -> list[CameraConfig]:
    """The cameras this launch renders, empty when the parameter is off. Parsed
    here so a bad config fails node setup, before a scene is compiled around
    it."""
    if not params.cameras_enabled:
        return []
    cameras = load_camera_configs(_CAMERAS_CONFIG_PATH)
    validate_camera_slots(cameras, COLOR_CAMERA_SLOT_NAMES, RGBD_CAMERA_SLOT_NAMES)
    return cameras


async def _run_sim(params, node_runner) -> list:
    # Typed peppygen pub/sub lives on this loop; declare publishers and start the
    # command-consume tasks before the sim thread starts reading from them.
    cameras = _camera_configs(params)
    head_camera_pack = head_camera.load(_HEAD_CAMERA_DIR, _CAMERAS_CONFIG_PATH)
    layout = Layout.read()
    loop = asyncio.get_running_loop()
    robots = Registry()
    stands = Stands()
    io = SimTopicIO(node_runner, loop, robots)
    await io.start()
    robots_io = RobotsIO(
        node_runner,
        loop,
        Catalogue.baked(head_camera_pack=head_camera_pack),
        robots,
        stands,
        io,
        Limbs(
            arm_names=tuple(layout.arm_names()),
            arm_joints=tuple(layout.arm_joint_counts()),
            gripper_names=tuple(layout.gripper_names()),
        ),
        params.robot_lease_ms / 1000.0,
        renders=bool(cameras),
    )
    await robots_io.start()
    launcher = SimLauncher(
        stands,
        _stop,
        io,
        layout,
        params.state_rate_hz,
        params.headless,
        params.viewer_host,
        params.viewer_port,
        cameras,
    )

    async def _run_sim_task() -> None:
        try:
            await loop.run_in_executor(None, launcher.run)
        finally:
            # Belt-and-braces against asyncio cancellation paths that race the
            # on_shutdown hook below; idempotent.
            _stop.set()

    async def _shutdown_hook() -> None:
        # Drive the sim executor to exit inside the runtime grace window so
        # the standing scene is shut down, then end the robot's stay and
        # cancel the consume tasks.
        _stop.set()
        await robots_io.stop()
        await io.stop()

    node_runner.on_shutdown(_shutdown_hook)

    return [asyncio.create_task(_run_sim_task())]


def main() -> None:
    NodeBuilder().run(_run_sim)


if __name__ == "__main__":
    main()
