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

    del params
    server = ViewerServer(("0.0.0.0", 8210))
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
        # The daemon renders the URLs an operator opens from this
        # announcement, one per address of the machine.
        host, port = server.server_address[:2]
        node_runner.announce_endpoint("viewer", "http", host, port)
    except Exception:
        await shutdown()
        raise

    logger.info(
        "Isaac Sim browser WebRTC viewer listening on %s:%d; "
        "browser warnings and errors are forwarded to this node log",
        host,
        port,
    )
    return []


def main() -> None:
    """Run the Peppy viewer node."""

    NodeBuilder().run(setup)


if __name__ == "__main__":
    main()
