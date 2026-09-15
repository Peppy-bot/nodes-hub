#!/usr/bin/env python3
"""Peppy wrapper for the Isaac Sim WebRTC browser viewer."""

from __future__ import annotations

import asyncio
import logging
import threading

from peppylib.runtime import NodeBuilder
from viewer_server import ViewerServer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
)

logger = logging.getLogger(__name__)


async def setup(params, node_runner) -> list:
    """Serve the WebRTC viewer and log its browser diagnostics."""

    server = ViewerServer(("0.0.0.0", params.http_port))
    try:
        thread = threading.Thread(target=server.serve_forever, name="isaac-viewer", daemon=True)
        thread.start()
    except Exception:
        server.server_close()
        raise

    def stop_server() -> None:
        server.shutdown()
        server.server_close()
        thread.join()

    async def shutdown() -> None:
        logger.info("Stopping Isaac Sim browser viewer")
        await asyncio.to_thread(stop_server)

    try:
        node_runner.on_shutdown(shutdown)
    except Exception:
        await shutdown()
        raise

    logger.info(
        "Isaac Sim browser WebRTC viewer listening on 0.0.0.0:%s; "
        "browser warnings and errors are forwarded to this node log",
        params.http_port,
    )
    return []


def main() -> None:
    """Run the Peppy viewer node."""

    NodeBuilder().run(setup)


if __name__ == "__main__":
    main()
