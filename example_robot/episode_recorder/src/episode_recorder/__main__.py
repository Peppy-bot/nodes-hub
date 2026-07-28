"""Captures what the robot actually did, for later training.

Taps the executor side of a `deliberation` pairing without joining it. An
observer claims no endpoint and holds no peer, so however this node behaves it
cannot perturb the control loop it is recording, and any number of recorders
can watch the same executor at once.
"""

import asyncio

from peppygen import NodeBuilder, NodeRunner
from peppygen.paired_topics.observed_execution import situation
from peppygen.parameters import Parameters


async def record(params: Parameters, node_runner: NodeRunner) -> None:
    """Records every situation the observed executor publishes.

    Subscribing before the source resolves is legal: the subscription is
    pinned to the source instance and stays silent until it emits. It follows
    that instance's lifecycle rather than any pairing, so a re-pair on the
    executor's side changes nothing here, and a restart is picked up as a new
    incarnation with no messages leaking across it.
    """
    subscription = await situation.subscribe(node_runner)

    captured = 0
    escalated = 0
    while True:
        received = await subscription.next()
        if received is None:
            break
        producer, message = received

        captured += 1
        if message.escalated:
            escalated += 1

        if captured == 1:
            print(
                f"[episode_recorder] observing execution on "
                f"{producer.core_node}/{producer.instance_id}"
            )
        elif captured % params.report_every == 0:
            print(
                f"[episode_recorder] captured {captured} samples "
                f"({escalated} escalated), last positions="
                f"{[round(p, 3) for p in message.joint_positions]}"
            )


async def setup(params: Parameters, node_runner: NodeRunner) -> list[asyncio.Task]:
    async def announce_shutdown():
        print("[episode_recorder] Shutdown signal received")

    node_runner.on_shutdown(announce_shutdown)

    return [asyncio.create_task(record(params, node_runner))]


def main():
    NodeBuilder().run(setup)


if __name__ == "__main__":
    main()
