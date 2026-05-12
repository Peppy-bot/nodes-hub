import asyncio
import math
import time

from peppygen import NodeBuilder, NodeRunner
from peppygen.consumed_topics import joint_commands
from peppygen.emitted_topics import joint_states
from peppygen.parameters import Parameters


async def publish_joint_states(node_runner: NodeRunner):
    while True:
        now = time.time()
        positions = [math.sin(now), math.sin(now * 1.3), math.sin(now * 0.7)]
        velocities = [0.1, 0.2, 0.1]
        try:
            await joint_states.emit(node_runner, positions, velocities, now)
        except Exception as e:
            print(f"[arm] emit joint_states error: {e!r}")
            break
        print(
            f"[arm] published joint_states: positions={[round(p, 3) for p in positions]}"
        )
        await asyncio.sleep(0.5)


async def receive_joint_commands(node_runner: NodeRunner):
    while True:
        try:
            _instance_id, msg = await joint_commands.on_next_message_received(
                node_runner, None, None
            )
            print(
                f"[arm] received joint_commands: "
                f"target_positions={[round(p, 3) for p in msg.target_positions]} "
                f"max_velocity={msg.max_velocity}"
            )
        except Exception as e:
            print(f"[arm] joint_commands subscription error: {e!r}")
            break


async def setup(params: Parameters, node_runner: NodeRunner) -> list[asyncio.Task]:
    return [
        asyncio.create_task(publish_joint_states(node_runner)),
        asyncio.create_task(receive_joint_commands(node_runner)),
    ]


def main():
    NodeBuilder().run(setup)


if __name__ == "__main__":
    main()
