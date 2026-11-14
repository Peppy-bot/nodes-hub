"""The check that the robot's initializer serves the robot this backbone was
built for: the model it names against so101, then the digest of the
description it serves against the digest of the model this backbone links. The
backbone checks once its setup has returned, while it serves, since the
initializer may start after it, and asks until the initializer answers."""

from __future__ import annotations

from dataclasses import dataclass

from control_core_py import runtime
from peppygen import NodeRunner
from peppygen.consumed_services.robot_identity import get_description, get_identity

from so101_backbone.model import MODEL

# How long the initializer may take to answer each question: the description
# carries the collision meshes.
ANSWER_TIMEOUT_S = 10.0


@dataclass(frozen=True)
class _Served:
    """What the initializer answers: the robot's model and the digest of the
    description it serves."""

    model: str
    digest: str


async def _served(node_runner: NodeRunner) -> _Served:
    identity = await get_identity.poll(
        node_runner, get_identity.bound_producer(node_runner), ANSWER_TIMEOUT_S
    )
    description = await get_description.poll(
        node_runner, get_description.bound_producer(node_runner), ANSWER_TIMEOUT_S
    )
    return _Served(model=identity.data.model, digest=description.data.digest)


async def check(node_runner: NodeRunner) -> None:
    """Raises ValueError when the initializer serves another model, or another
    description of so101, which ends the node; returns once it serves this
    backbone's, or once the node is cancelled before it answers."""
    served = await runtime.ask(
        "robot_identity", node_runner.cancellation_token(), lambda: _served(node_runner)
    )
    if served is None:
        return
    if served.model != MODEL.id:
        raise ValueError(
            f"the robot's initializer serves model {served.model}, and this backbone drives "
            f"{MODEL.id}: link robot_identity to the initializer of an {MODEL.id} robot"
        )
    if served.digest != MODEL.digest:
        raise ValueError(
            f"the robot's initializer serves the description {served.digest} of {MODEL.id}, "
            f"and this backbone was built with {MODEL.digest}: build the backbone and the "
            "initializer from one public-peppy-libs commit"
        )
