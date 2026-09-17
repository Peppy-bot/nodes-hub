#!/usr/bin/env python3
"""Isaac Sim launch script for the openarm sim engine node."""

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


# The model whose links the camera rig mounts on.
_CAMERA_MODEL = "openarm_v2"


def _camera_configs(params) -> list:
    """The cameras this launch renders, empty when the parameter is off.
    Parsed here so a bad config fails node setup, before the stage is opened
    around it."""
    if not params.cameras_enabled:
        return []
    if params.model != _CAMERA_MODEL:
        raise ValueError(
            f"cameras_enabled needs model set to {_CAMERA_MODEL}: the camera geometry "
            "in config/cameras.json5 mounts on that model's links, and it is rendered "
            "from the robot this engine stands"
        )
    sys.path.insert(0, str(_ROBOTS_DIR))
    from camera_common import load_camera_configs, validate_camera_slots
    from sim_topics import COLOR_CAMERA_SLOT_NAMES, RGBD_CAMERA_SLOT_NAMES

    cameras = load_camera_configs(_ROBOTS_DIR / "config" / "cameras.json5")
    validate_camera_slots(cameras, COLOR_CAMERA_SLOT_NAMES, RGBD_CAMERA_SLOT_NAMES)
    return cameras


_ROBOTS_DIR = Path(__file__).resolve().parents[1]

# The loop and livestream share a target independent of state publication limits.
_FRAME_RATE_HZ = 60
_EXPERIENCE_PATH = _ROBOTS_DIR / "config" / "openarm.sim.kit"

# The viewport renders with RTX Real-Time 2.0 (`RealTimePathTracing`) and DLSS
# at 720p. That renderer denoises only through DLSS Ray Reconstruction, which
# runs on the NGX core library the base image carries (see
# scripts/Dockerfile.isaac); Peppy's `--nv` binding does not
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
    seats: object
    seat_io: object
    world: object
    edits: object
    layout: object
    state_rate_hz: int
    headless: bool
    cameras: list


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
        str(_ROBOTS_DIR),
    )
    from bridge_extension import Layout
    from edits import Edits
    from scene_actions import SceneActionIO
    from seat_io import SeatIO
    from seats import Limbs, Registry
    from sim_topics import SimTopicIO
    from world import Catalogue, Placement, Robot, World

    cameras = _camera_configs(params)

    catalogue = Catalogue.baked()
    standing = None
    if params.model:
        catalogue.scene(params.model)
        standing = Robot(
            instance="",
            model=params.model,
            placement=Placement.of((0.0, 0.0, 0.0), 0.0),
        )
    layout = Layout.read()
    world = World(catalogue, standing)
    seats = Registry()
    edits = Edits()

    loop = asyncio.get_running_loop()

    io = SimTopicIO(
        node_runner,
        loop,
    )

    await io.start()

    scene_actions = SceneActionIO(
        node_runner,
        loop,
        io,
        world,
    )

    await scene_actions.start()

    seat_io = SeatIO(
        node_runner,
        loop,
        world,
        seats,
        edits,
        Limbs(
            arm_names=tuple(layout.arm_names()),
            arm_joints=tuple(layout.arm_joint_counts()),
            gripper_names=tuple(layout.gripper_names()),
        ),
        params.robot_lease_ms / 1000.0,
    )
    await seat_io.start()

    _handoff["value"] = _SimHandoff(
        io=io,
        scene_actions=scene_actions,
        seats=seats,
        seat_io=seat_io,
        world=world,
        edits=edits,
        layout=layout,
        state_rate_hz=params.state_rate_hz,
        headless=params.headless,
        cameras=cameras,
    )

    _handoff_ready.set()

    async def _shutdown_hook() -> None:
        # Tell the Isaac main-thread simulation loop to exit,
        # then stop Peppy topic IO.

        _stop.set()
        await seat_io.stop()
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

    if handoff.cameras:
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
        str(_ROBOTS_DIR),
    )

    from _launcher import SimLauncher
    from bridge_extension import IsaacBridgeExtension

    extension = IsaacBridgeExtension(
        handoff.world,
        handoff.io,
        handoff.scene_actions,
        handoff.seats,
        handoff.seat_io,
        handoff.layout,
        handoff.state_rate_hz,
        handoff.cameras,
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
        bool(handoff.cameras),
        frame_rate_hz=_FRAME_RATE_HZ,
        render_mode=_RENDER_CONFIG["renderer"],
        anti_aliasing=_RENDER_CONFIG["anti_aliasing"],
    )
    # A robot that attached before now waits in the edits queue, and the loop
    # below stands it on this thread, which is the only one that may.
    handoff.seat_io.binds_with(launcher.rebind, launcher.unbind)
    launcher.run()


if __name__ == "__main__":
    main()
