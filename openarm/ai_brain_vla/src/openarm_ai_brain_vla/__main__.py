"""openarm_ai_brain_vla: the entry point. Wiring only: read the settings,
build the brain with the backends the launcher selected, and start one
loop per exposed member plus the camera follower. The rules live in the
core, the moves and the model behind the two ports."""

from __future__ import annotations

import asyncio
import logging
import sys
from functools import partial

from peppygen import NodeBuilder, NodeRunner, clock
from peppygen.exposed_actions.item_manipulation import abort, drop_item, grab_item, place_item
from peppygen.exposed_actions.item_perception import identify_item, scan_items
from peppygen.exposed_services.item_manipulation import get_state
from peppygen.parameters import Parameters

from .brain import Brain
from .handlers import abort as abort_handler
from .handlers import drop_item as drop_item_handler
from .handlers import get_state as get_state_handler
from .handlers import grab_item as grab_item_handler
from .handlers import identify_item as identify_item_handler
from .handlers import place_item as place_item_handler
from .handlers import scan_items as scan_items_handler
from .serve import serve_action, serve_service


async def setup(params: Parameters, node_runner: NodeRunner) -> list[asyncio.Task]:
    await clock.init(node_runner)
    token = node_runner.cancellation_token()
    brain = Brain(params, node_runner)
    await brain.start()
    node_runner.on_shutdown(brain.shutdown)
    print(
        f"[brain] grippers {brain.state.gripper_names()}, perception '{brain.perceiver.detector.name}', "
        f"manipulation '{brain.manipulator.name}'"
    )
    # Per-goal tasks live here so a goal in flight is never collected
    # before it completes.
    goal_tasks: set[asyncio.Task] = set()
    actions = [
        (scan_items, scan_items_handler.run),
        (identify_item, identify_item_handler.run),
        (grab_item, grab_item_handler.run),
        (drop_item, drop_item_handler.run),
        (place_item, place_item_handler.run),
        (abort, abort_handler.run),
    ]
    loops = [
        asyncio.create_task(serve_action(module, node_runner, token, goal_tasks, partial(run, brain)))
        for module, run in actions
    ]
    loops.append(
        asyncio.create_task(serve_service(get_state, node_runner, token, partial(get_state_handler.handle, brain)))
    )
    loops.extend(brain.background(token))
    return loops


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """The node's own log lines reach the daemon's log. Python drops INFO
    by default, which hid the backends' "loaded" lines inside the container;
    only this package's logger is raised, so the model libraries stay quiet."""
    logger = logging.getLogger("openarm_ai_brain_vla")
    if not any(getattr(h, "_brain_handler", False) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("[brain] %(message)s"))
        handler._brain_handler = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def main() -> None:
    configure_logging()
    NodeBuilder().run(setup)


if __name__ == "__main__":
    main()
