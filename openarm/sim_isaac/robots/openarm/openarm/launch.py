#!/usr/bin/env python3
"""Isaac Sim launch script for the openarm sim engine node."""

# pylint: disable=C0413

from __future__ import annotations

import asyncio
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


_ASSETS_DIR = Path(
    os.environ.get(
        "PEPPY_ROBOT_ASSETS_DIR",
        str(Path(__file__).parent / "assets"),
    )
)


def _version(hardware_version: str) -> str:
    version = hardware_version.lower()

    if version not in ("v1", "v2"):
        raise ValueError(
            "hardware_version must be v1 or v2, "
            f"got {hardware_version!r}"
        )

    return version


def _scene_path(hardware_version: str) -> Path:
    # The v1 and v2 scenes are separate USD files:
    #
    # openarm_bimanual_v1.usd
    # openarm_bimanual_v2.usd
    #
    # A missing scene fails loudly rather than silently
    # simulating a different hardware version.

    return (
        _ASSETS_DIR
        / f"openarm_bimanual_{_version(hardware_version)}.usd"
    )


_ROBOTS_DIR = Path(__file__).resolve().parents[1]

# The loop and livestream share a target independent of state publication limits.
_FRAME_RATE_HZ = 60
_EXPERIENCE_PATH = _ROBOTS_DIR / "config" / "openarm.sim.kit"

# The viewport renders with RTX Real-Time (`RaytracedLighting`) and TAA at
# 720p. Isaac Sim 6.0 defaults to RTX Real-Time 2.0, whose only denoiser is
# DLSS Ray Reconstruction; that runs on the host driver's NGX library, which
# Peppy's stock `--nv` binding does not carry into the container, and without
# it the stream is raw path-tracing noise. RTX Real-Time denoises on its own,
# so this profile needs nothing beyond the ordinary GPU binding. Kit ships the
# pipeline switched off (config/openarm.sim.kit turns it on) and renders a
# disabled mode with RTX Real-Time 2.0 without a word, hence the check in
# `_require_render_mode`.
_RENDER_CONFIG = {
    "renderer": "RaytracedLighting",
    "anti_aliasing": 1,
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
    state_rate_hz: int
    headless: bool
    hardware_version: str
    cameras_enabled: bool


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
    from sim_topics import SimTopicIO
    from scene_actions import SceneActionIO

    if params.cameras_enabled and _version(params.hardware_version) != "v2":
        raise ValueError(
            "cameras_enabled requires hardware_version v2: the camera geometry "
            "in config/cameras.json5 mounts on v2 links only"
        )

    loop = asyncio.get_running_loop()

    io = SimTopicIO(
        node_runner,
        loop,
    )

    await io.start()

    scene_actions = SceneActionIO(
        node_runner,
        loop,
    )

    await scene_actions.start()

    _handoff["value"] = _SimHandoff(
        io=io,
        scene_actions=scene_actions,
        state_rate_hz=params.state_rate_hz,
        headless=params.headless,
        hardware_version=params.hardware_version,
        cameras_enabled=params.cameras_enabled,
    )

    _handoff_ready.set()

    async def _shutdown_hook() -> None:
        # Tell the Isaac main-thread simulation loop to exit,
        # then stop Peppy topic IO.

        _stop.set()
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


def _require_render_mode(expected: str) -> None:
    """Refuse a renderer Kit substituted for the requested one.

    Kit only honours `/rtx/rendermode` values whose pipeline is enabled and
    falls back to RTX Real-Time 2.0 otherwise, silently. That fallback needs
    NGX to denoise, so a launch on it would stream noise at full frame rate.
    """
    import carb.settings

    actual = carb.settings.get_settings().get("/rtx/rendermode")

    if actual != expected:
        raise RuntimeError(
            f"Isaac is rendering with {actual!r} instead of the requested "
            f"{expected!r}; the pipeline must be enabled through "
            "persistent.rtx.modes in config/openarm.sim.kit"
        )


def main() -> None:
    """Launch Peppy and Isaac Sim."""

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

    if handoff.cameras_enabled:
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

    _require_render_mode(
        _RENDER_CONFIG["renderer"]
    )

    sys.path.insert(
        0,
        str(_ROBOTS_DIR),
    )

    from _launcher import SimLauncher

    SimLauncher(
        simulation_app,
        _scene_path(
            handoff.hardware_version
        ),
        _ready,
        _stop,
        handoff.io,
        handoff.scene_actions,
        handoff.state_rate_hz,
        handoff.cameras_enabled,
        frame_rate_hz=_FRAME_RATE_HZ,
    ).run()


if __name__ == "__main__":
    main()
