"""ai_brain_vla: the entry point. Wiring only: the node's generated modules
for brain_core, the models the parameters name, and brain_core's shared
setup."""

from __future__ import annotations

import asyncio
from typing import Callable, Mapping

import peppygen
from peppygen import NodeBuilder, NodeRunner
from peppygen.parameters import Parameters

from brain_core import node
from brain_core.bindings import Bindings
from brain_core.manipulators import Manipulator
from brain_core.ports import Detector

from .models import MANIPULATORS, VlaModels
from .perception import DETECTORS

BINDINGS = Bindings.from_generated(peppygen)


async def setup(
    params: Parameters,
    node_runner: NodeRunner,
    *,
    detectors: Mapping[str, Callable[[str], Detector]] = DETECTORS,
    manipulators: Mapping[str, Callable[[], Manipulator]] = MANIPULATORS,
) -> list[asyncio.Task]:
    """The node's entry point. `detectors` and `manipulators` stand in for
    the models the brain runs when given, which is how the whole node runs
    over the wire against fakes."""
    source = VlaModels(
        params.perception_model, params.manipulation_model, params.perception_gallery,
        detectors=detectors, manipulators=manipulators,
    )
    return await node.setup(params, node_runner, BINDINGS, source)


def main() -> None:
    node.configure_logging("ai_brain_vla")
    NodeBuilder().run(setup)


if __name__ == "__main__":
    main()
