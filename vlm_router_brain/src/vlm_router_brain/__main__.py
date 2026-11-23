"""vlm_router_brain: the entry point. Wiring only: the node's generated
modules for brain_core, the OpenAI-compatible API the environment names,
the catalog's models through any-llm, and brain_core's shared setup."""

from __future__ import annotations

import asyncio
import os
from typing import Optional

import peppygen
from peppygen import NodeBuilder, NodeRunner
from peppygen.parameters import Parameters

from brain_core import node
from brain_core.bindings import Bindings

from .catalog import load_catalog
from .models import RouterModels
from .openai_compatible import compatible_api
from .provider import AnyLlmProvider, Provider

BINDINGS = Bindings.from_generated(peppygen)


async def setup(params: Parameters, node_runner: NodeRunner, *, provider: Optional[Provider] = None) -> list[asyncio.Task]:
    """The node's entry point. An OpenAI-compatible API half set in the
    environment stops the node here, before it serves anything; an API set
    takes every model call. `provider` stands in for any-llm when given,
    which is how the whole node runs over the wire against a scripted
    fake."""
    # The process environment, looked up at each start: the API, the keys
    # and the provider's clients all read this one mapping.
    environ = os.environ
    api = compatible_api(environ)
    source = RouterModels(load_catalog(), provider or AnyLlmProvider(environ, api=api), environ, api=api)
    return await node.setup(params, node_runner, BINDINGS, source)


def main() -> None:
    node.configure_logging("vlm_router_brain")
    NodeBuilder().run(setup)


if __name__ == "__main__":
    main()
