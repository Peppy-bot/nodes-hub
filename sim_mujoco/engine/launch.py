#!/usr/bin/env python3
"""MuJoCo launch script for the simulation node."""

from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path

from peppylib.runtime import NodeBuilder

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True
)
logger = logging.getLogger(__name__)

import head_camera
from _launcher import SimLauncher
from edits import Edits
from mujoco_models import MujocoModels
from robots_io import RobotsIO
from sim_robot_core.registry import Registry
from sim_topics import SimTopicIO
from world import World

# The head camera pack apptainer.def stages at image build (head_camera.py
# fetch), beside the node's code. Setup reads it, and a robot standing as a
# model that carries the head camera draws it.
_HEAD_CAMERA_DIR = Path(__file__).parent / "assets" / "head_camera"

_stop = threading.Event()


async def _run_sim(params, node_runner) -> list:
    # Every model's entry is parsed here, so a bad one fails node setup, before
    # a robot attaches as it.
    models = MujocoModels.read()
    head_camera_pack = head_camera.load_for(models, _HEAD_CAMERA_DIR)
    # Typed peppygen pub/sub lives on this loop; declare publishers and start the
    # command-consume tasks before the sim thread starts reading from them.
    loop = asyncio.get_running_loop()
    robots = Registry()
    edits = Edits()
    world = World(head_camera_pack, renders=params.cameras_enabled)
    io = SimTopicIO(node_runner, loop, robots)
    await io.start()
    robots_io = RobotsIO(
        node_runner,
        loop,
        models,
        robots,
        world,
        edits,
        io,
        params.robot_lease_ms / 1000.0,
    )
    await robots_io.start()
    launcher = SimLauncher(
        world,
        edits,
        _stop,
        io,
        params.state_rate_hz,
        params.headless,
        params.viewer_host,
        params.viewer_port,
    )
    # Standing a robot and taking one out both compose the scene again, on
    # the thread that steps it.
    robots_io.binds_with(launcher.rebind, launcher.unbind)

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
