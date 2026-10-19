"""Where the headset's page is, on the wire: the headset contract's page
topic, the URLs the page answers on.

A consumer that is not wearing the headset reads it: the simulation's panel
offers the link that opens the page in the headset.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import ModuleType

from peppygen import NodeRunner

from xr_commander.bus import log


async def publish_page(
    node_runner: NodeRunner, topic_module: ModuleType, urls: Sequence[str]
) -> None:
    """Publish the page's URLs once; the topic retains them for every
    consumer that reads them later.

    Loud and fail-safe: a failure is logged and the headset drives on, with
    no panel seeing it.
    """
    try:
        publisher = await topic_module.declare_publisher(node_runner)
        await publisher.publish(topic_module.build_message(list(urls)))
    except Exception as e:
        log(f"failed to publish the page's URLs: {e!r}")
