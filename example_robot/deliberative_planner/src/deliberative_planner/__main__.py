"""The slow half of a split-compute manipulation stack.

Plans over tens of seconds against a camera and an executor that typically run
on another machine. It thinks at roughly 1 Hz, so a WAN round trip costs it a
small fraction of its own cycle: latency is nearly free here, which is what
makes it the half that can move to a datacenter.
"""

import asyncio
import time

from peppygen import NodeBuilder, NodeRunner
from peppygen.consumed_topics.scene import video_stream
from peppygen.paired_topics.deliberation import situation, subgoal
from peppygen.parameters import Parameters
from peppylib import CancellationToken

# How far each planning step nudges the executor toward the goal. A real
# planner would run a world model; a mock only has to produce subgoals that
# move, and move visibly.
STEP = 0.25


class Scene:
    """The planner's view of the world, written by its two subscriptions."""

    def __init__(self) -> None:
        self.frames_seen = 0
        self.executor_positions = [0.0, 0.0, 0.0]
        self.escalations = 0


async def watch_scene(node_runner: NodeRunner, scene: Scene) -> None:
    """Consumes the camera.

    The producer link this reads is bound to the same camera instance the
    reactive policy reads locally, and both name it identically. That the
    camera is on another machine appears nowhere in this node: placement is
    declared once, on the instance.
    """
    subscription = await video_stream.subscribe(node_runner)

    while True:
        received = await subscription.next()
        if received is None:
            break
        producer, message = received
        scene.frames_seen += 1
        if scene.frames_seen == 1:
            print(
                f"[deliberative_planner] first frame received across the boundary "
                f"from {producer.core_node}/{producer.instance_id}: "
                f"{message.width}x{message.height} {message.encoding}"
            )


async def watch_executor(node_runner: NodeRunner, scene: Scene) -> None:
    subscription = await situation.subscribe(node_runner)

    peer = await situation.wait_paired(node_runner)
    print(
        f"[deliberative_planner] paired with executor "
        f"{peer.producer.core_node}/{peer.producer.instance_id}"
    )

    while True:
        received = await subscription.next()
        if received is None:
            break
        _producer, message = received
        scene.executor_positions = list(message.joint_positions)
        if message.escalated:
            scene.escalations += 1


async def plan(
    params: Parameters, node_runner: NodeRunner, scene: Scene, token: CancellationToken
) -> None:
    """Publishes a subgoal per planning cycle.

    Planning starts before the first frame arrives and before the pair is
    established. Publishing on an unpaired slot is a legal no-op, so there is
    nothing to synchronize here: the executor receives every subgoal published
    while the pair is live, and none published before it.
    """
    publisher = await subgoal.declare_publisher(node_runner)
    interval = 1.0 / params.plan_rate_hz
    print(
        f"[deliberative_planner] planning at {params.plan_rate_hz} Hz "
        f"over a {params.horizon_s}s horizon"
    )

    subgoal_id = 0
    while not token.is_cancelled():
        subgoal_id += 1
        target = [round(position + STEP, 3) for position in scene.executor_positions]
        await publisher.publish(
            subgoal.build_message(time.time(), subgoal_id, target, params.horizon_s)
        )
        print(
            f"[deliberative_planner] subgoal {subgoal_id} target={target} "
            f"(frames={scene.frames_seen}, escalations={scene.escalations})"
        )
        await asyncio.sleep(interval)


async def setup(params: Parameters, node_runner: NodeRunner) -> list[asyncio.Task]:
    scene = Scene()
    token = node_runner.cancellation_token()

    async def announce_shutdown():
        print("[deliberative_planner] Shutdown signal received")

    node_runner.on_shutdown(announce_shutdown)

    return [
        asyncio.create_task(watch_scene(node_runner, scene)),
        asyncio.create_task(watch_executor(node_runner, scene)),
        asyncio.create_task(plan(params, node_runner, scene, token)),
    ]


def main():
    NodeBuilder().run(setup)


if __name__ == "__main__":
    main()
