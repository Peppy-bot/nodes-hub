#!/usr/bin/env python3
"""Peppy wrapper for the Isaac Sim WebRTC browser viewer."""

from __future__ import annotations

import asyncio
import logging
import signal

from peppylib.runtime import NodeBuilder


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
)

logger = logging.getLogger(__name__)


async def setup(params, node_runner) -> list:
    """Start and supervise the Vite WebRTC viewer."""

    del params

    logger.info(
        "Starting Isaac Sim browser WebRTC viewer on 0.0.0.0:8210"
    )

    process = await asyncio.create_subprocess_exec(
        "npx",
        "vite",
        "preview",
        "--host",
        "0.0.0.0",
        "--port",
        "8210",
        cwd="/app",
    )

    logger.info(
        "Isaac Sim browser viewer started with PID %s",
        process.pid,
    )

    async def shutdown() -> None:
        if process.returncode is not None:
            return

        logger.info("Stopping Isaac Sim browser viewer")

        process.send_signal(signal.SIGTERM)

        try:
            await asyncio.wait_for(
                process.wait(),
                timeout=5.0,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Viewer did not terminate gracefully; killing it"
            )
            process.kill()
            await process.wait()

    node_runner.on_shutdown(shutdown)

    async def watch_process() -> None:
        return_code = await process.wait()

        logger.info(
            "Isaac Sim browser viewer exited with code %s",
            return_code,
        )

    asyncio.create_task(watch_process())

    return []


def main() -> None:
    """Run the Peppy viewer node."""

    NodeBuilder().run(setup)


if __name__ == "__main__":
    main()
