#!/usr/bin/env python3
"""Isaac Sim launch script for the simulation node."""

# pylint: disable=C0413

from __future__ import annotations

import asyncio
import ctypes
import logging
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from peppylib.runtime import NodeBuilder


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
)

logger = logging.getLogger(__name__)


# The engine's modules sit beside this script.
_ENGINE_DIR = Path(__file__).resolve().parent

# The head camera pack apptainer.def stages at image build (head_camera.py
# fetch), beside the node's code. Setup reads it, and a robot standing as a
# model that carries the head camera draws it.
_HEAD_CAMERA_DIR = Path(__file__).parent / "assets" / "head_camera"

# The loop and livestream share a target independent of state publication limits.
_FRAME_RATE_HZ = 60
_EXPERIENCE_PATH = _ENGINE_DIR / "config" / "sim_isaac.kit"

# The viewport renders with RTX Real-Time 2.0 (`RealTimePathTracing`) and DLSS
# at 720p. That renderer denoises only through DLSS Ray Reconstruction, which
# runs on the NGX core library the base image carries (see
# sim_base_images/Dockerfile.isaac); Peppy's `--nv` binding does not
# bring the host's copy in. Kit falls back to TAA without a word when the
# library is missing and streams raw path-tracing noise, so the launcher checks
# the effective profile once the first frames have rendered.
_RENDER_CONFIG = {
    "renderer": "RealTimePathTracing",
    "anti_aliasing": 3,
    "width": 1280,
    "height": 720,
}

_ready = threading.Event()
_stop = threading.Event()


@dataclass
class _SimHandoff:
    """Resolved Peppy parameters passed to the Isaac main thread.

    SimulationApp must be constructed on the main thread before
    importing omni.* modules.

    The Peppy node thread resolves the node parameters and IO first,
    stores them here, then signals the main thread to continue.
    """

    io: object
    scene_actions: object
    robots: object
    robots_io: object
    world: object
    edits: object
    state_rate_hz: int
    headless: bool
    # Whether the robots' cameras are rendered: a rig is its robot's model's.
    renders: bool


_handoff: dict[str, _SimHandoff] = {}
_setup_error: dict[str, Exception] = {}
_handoff_ready = threading.Event()


async def setup(params, node_runner) -> list:
    """Set up Peppy IO and hand resolved parameters to Isaac."""

    # A setup failure must reach main() promptly: record it and release the
    # handoff wait, so the process dies on the real error instead of the
    # 30s parameter timeout.
    try:
        return await _node_setup(params, node_runner)
    except Exception as exc:
        _setup_error["value"] = exc
        _handoff_ready.set()
        raise


async def _node_setup(params, node_runner) -> list:
    # Typed peppygen pub/sub lives on this loop; declare publishers and start
    # the command-consume tasks before the sim thread starts reading from them.
    sys.path.insert(
        0,
        str(_ENGINE_DIR),
    )
    import head_camera
    from edits import Edits
    from isaac_models import IsaacModels
    from robots_io import RobotsIO
    from scene_actions import SceneActionIO
    from sim_robot_core.registry import Registry
    from sim_topics import SimTopicIO
    from world import World

    # Every model's entry is parsed here, so a bad one fails node setup, before
    # the stage is opened and a robot attaches as it.
    models = IsaacModels.read()
    world = World(head_camera.load_for(models, _HEAD_CAMERA_DIR))
    robots = Registry()
    edits = Edits()

    loop = asyncio.get_running_loop()

    io = SimTopicIO(
        node_runner,
        loop,
        robots,
    )

    await io.start()

    scene_actions = SceneActionIO(
        node_runner,
        loop,
        io,
        world,
    )

    await scene_actions.start()

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

    _handoff["value"] = _SimHandoff(
        io=io,
        scene_actions=scene_actions,
        robots=robots,
        robots_io=robots_io,
        world=world,
        edits=edits,
        state_rate_hz=params.state_rate_hz,
        headless=params.headless,
        renders=params.cameras_enabled,
    )

    _handoff_ready.set()

    async def _shutdown_hook() -> None:
        # Tell the Isaac main-thread simulation loop to exit,
        # then stop Peppy topic IO.

        _stop.set()
        await robots_io.stop()
        await scene_actions.stop()
        await io.stop()

    node_runner.on_shutdown(
        _shutdown_hook
    )

    return []


def _run_node_builder() -> None:
    """Run the Peppy node runtime in its own thread."""

    try:
        NodeBuilder().run(
            setup
        )

    finally:
        _stop.set()


def _preflight_nvidia_driver() -> None:
    """Check NVML access to the host driver, not Vulkan renderer health."""

    guidance = (
        "Verify nvidia-smi works on the host, check the host NVIDIA driver, "
        "and launch the container with --nv."
    )
    try:
        nvml = ctypes.CDLL("libnvidia-ml.so.1")
    except OSError as exc:
        raise RuntimeError(
            f"NVIDIA driver preflight failed: cannot load libnvidia-ml.so.1: {exc}. "
            f"{guidance}"
        ) from exc

    try:
        nvml_init = nvml.nvmlInit_v2
        nvml_error_string = nvml.nvmlErrorString
        nvml_shutdown = nvml.nvmlShutdown
    except AttributeError as exc:
        raise RuntimeError(
            f"NVIDIA driver preflight failed: missing required NVML API symbol: {exc}. "
            f"{guidance}"
        ) from exc

    nvml_init.argtypes = []
    nvml_init.restype = ctypes.c_int
    nvml_error_string.argtypes = [ctypes.c_int]
    nvml_error_string.restype = ctypes.c_char_p
    nvml_shutdown.argtypes = []
    nvml_shutdown.restype = ctypes.c_int

    # Shutdown is called only after init succeeds, and must also succeed.
    for name, operation in (("nvmlInit_v2", nvml_init), ("nvmlShutdown", nvml_shutdown)):
        result = operation()
        if result != 0:
            detail = nvml_error_string(result)
            detail = detail.decode("utf-8", errors="replace") if detail else "unknown NVML error"
            mismatch_guidance = (
                " The NVIDIA user-space library and loaded kernel driver do not match. "
                "Reboot the host after a driver update."
                if result == 18 else ""
            )
            raise RuntimeError(
                f"NVIDIA driver preflight failed: {name} returned NVML error {result} "
                f"({detail}).{mismatch_guidance} {guidance}"
            )


def main() -> None:
    """Launch Peppy and Isaac Sim."""

    _preflight_nvidia_driver()

    threading.Thread(
        target=_run_node_builder,
        daemon=True,
    ).start()

    if not _handoff_ready.wait(
        timeout=30
    ):
        raise RuntimeError(
            "node parameters not resolved within 30s"
        )

    if "value" in _setup_error:
        raise RuntimeError(
            "node setup failed"
        ) from _setup_error["value"]
    handoff = _handoff["value"]

    # --------------------------------------------------------------
    # WebRTC configuration
    # --------------------------------------------------------------
    #
    # PEPPY_ISAAC_PUBLIC_IP is optional.
    #
    # Leave it unset to allow WebRTC/ICE to determine the address
    # automatically. Set it explicitly only when the advertised
    # address must be fixed, for example when connecting from
    # another host.
    #
    # Example:
    #
    #   export PEPPY_ISAAC_PUBLIC_IP=<YOUR_HOST_IP>
    #
    # Optional port overrides:
    #
    #   export PEPPY_ISAAC_SIGNAL_PORT=49100
    #   export PEPPY_ISAAC_STREAM_PORT=47998

    public_ip = os.environ.get(
        "PEPPY_ISAAC_PUBLIC_IP",
        "",
    ).strip()

    signal_port = os.environ.get(
        "PEPPY_ISAAC_SIGNAL_PORT",
        "49100",
    ).strip()

    stream_port = os.environ.get(
        "PEPPY_ISAAC_STREAM_PORT",
        "47998",
    ).strip()

    if handoff.headless:
        logger.info(
            "WebRTC streaming configuration: "
            "publicIp=%s signalPort=%s streamPort=%s",
            public_ip or "<auto>",
            signal_port,
            stream_port,
        )

        streaming_args = [
            "--enable",
            "omni.kit.livestream.app",
            (
                "--/exts/omni.kit.livestream.app/"
                "primaryStream/targetFps="
                f"{_FRAME_RATE_HZ}"
            ),
            (
                "--/exts/omni.kit.livestream.app/"
                "primaryStream/signalPort="
                f"{signal_port}"
            ),
            (
                "--/exts/omni.kit.livestream.app/"
                "primaryStream/streamPort="
                f"{stream_port}"
            ),
        ]

        if public_ip:
            streaming_args.append(
                (
                    "--/exts/omni.kit.livestream.app/"
                    "primaryStream/publicIp="
                    f"{public_ip}"
                )
            )

        sys.argv.extend(
            streaming_args
        )

    if handoff.renders:
        sys.argv.extend(["--enable", "omni.replicator.core"])

    # SimulationApp must be imported only after all launch arguments
    # have been prepared.
    sys.argv.extend([
        "--/log/channels/omni.usd.multitick.render=warn",
        "--/log/fileLogLevel=warn",
    ])

    from isaacsim import SimulationApp

    launch_config = {
        "headless": handoff.headless,
        **_RENDER_CONFIG,
    }

    simulation_app = SimulationApp(
        launch_config,
        experience=str(_EXPERIENCE_PATH),
    )

    sys.path.insert(
        0,
        str(_ENGINE_DIR),
    )

    from _launcher import SimLauncher
    from bridge_extension import IsaacBridgeExtension

    extension = IsaacBridgeExtension(
        handoff.world,
        handoff.io,
        handoff.robots,
        handoff.scene_actions,
        handoff.state_rate_hz,
        handoff.renders,
    )

    launcher = SimLauncher(
        simulation_app,
        handoff.world,
        handoff.edits,
        extension,
        _ready,
        _stop,
        handoff.io,
        handoff.scene_actions,
        frame_rate_hz=_FRAME_RATE_HZ,
        render_mode=_RENDER_CONFIG["renderer"],
        anti_aliasing=_RENDER_CONFIG["anti_aliasing"],
    )
    # A robot that attached before now waits in the edits queue, and the loop
    # below stands it on this thread, which is the only one that may.
    handoff.robots_io.binds_with(launcher.rebind, launcher.unbind)
    launcher.run()


if __name__ == "__main__":
    main()
