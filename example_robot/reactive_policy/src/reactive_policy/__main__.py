"""The fast half of a split-compute manipulation stack.

Closes a servo loop against hardware it can reach in microseconds, and
escalates anything it cannot resolve within its own horizon to a planner that
may be a continent away. The uplink is an enhancement path, never a
dependency: when subgoals stop arriving, this node keeps running on its own
local behavior.
"""

import asyncio
import time

from peppygen import NodeBuilder, NodeRunner
from peppygen.consumed_topics.arm import joint_states
from peppygen.consumed_topics.camera import video_stream
from peppygen.paired_topics.deliberation import situation, subgoal
from peppygen.parameters import Parameters
from peppylib import CancellationToken

# How often the current situation is pushed up to the planner. Deliberately
# decoupled from `control_rate_hz`: the servo loop is local and runs hot, while
# this direction may cross a WAN and only has to keep a ~1 Hz planner supplied
# with recent state.
SITUATION_RATE_HZ = 10

# The local fallback the policy servos toward when no subgoal is authoritative.
# A real policy would compute this; a mock only has to be honest about the fact
# that losing the uplink degrades behavior instead of stopping it.
HOME_POSITION = [0.0, 0.0, 0.0]


class Situation:
    """Everything the control loop knows, shared across this node's tasks.

    Plain state rather than a queue: every consumer here wants the *latest*
    value, and none of them wants a backlog. A subgoal that has expired is
    exactly as useful as one that never arrived.
    """

    def __init__(self, subgoal_ttl_ms: int) -> None:
        self._subgoal_ttl_s = subgoal_ttl_ms / 1000.0
        self.joint_positions = list(HOME_POSITION)
        self.frames_seen = 0
        self._target = None
        self._adopted_at = None

    def adopt(self, target: list[float], adopted_at: float) -> None:
        self._target = list(target)
        self._adopted_at = adopted_at

    def authoritative_target(self, now: float):
        """The subgoal to servo toward, or None once it has gone stale.

        This is the whole staleness watchdog. Peppy guarantees a dissolution
        notice when a peer instance dies cleanly, but an unreachable daemon
        cannot send one, so a node whose correctness depends on freshness owns
        the bound itself.
        """
        if self._adopted_at is None:
            return None
        if now - self._adopted_at > self._subgoal_ttl_s:
            return None
        return self._target


async def track_arm(node_runner: NodeRunner, state: Situation) -> None:
    subscription = await joint_states.subscribe(node_runner)
    while True:
        received = await subscription.next()
        if received is None:
            break
        _producer, message = received
        state.joint_positions = list(message.positions)


async def track_camera(node_runner: NodeRunner, state: Situation) -> None:
    subscription = await video_stream.subscribe(node_runner)
    while True:
        received = await subscription.next()
        if received is None:
            break
        _producer, message = received
        state.frames_seen += 1
        if state.frames_seen == 1:
            print(
                f"[reactive_policy] first frame from the local camera: "
                f"{message.width}x{message.height} {message.encoding}"
            )


async def adopt_subgoals(node_runner: NodeRunner, state: Situation) -> None:
    """Takes subgoals from the paired planner.

    Subscribing before the pair exists is legal: the subscription follows the
    slot's live pin and stays silent until a planner is paired.
    """
    subscription = await subgoal.subscribe(node_runner)

    peer = await subgoal.wait_paired(node_runner)
    print(
        f"[reactive_policy] paired with planner "
        f"{peer.producer.core_node}/{peer.producer.instance_id}"
    )

    while True:
        received = await subscription.next()
        if received is None:
            break
        _producer, message = received
        state.adopt(message.target_position, time.time())
        print(
            f"[reactive_policy] adopted subgoal {message.subgoal_id} "
            f"target={[round(p, 3) for p in message.target_position]} "
            f"horizon={message.horizon_s}s"
        )


async def report_situation(node_runner: NodeRunner, state: Situation) -> None:
    """Publishes what this node is facing, up to the paired planner.

    Runs whether or not anything is escalated, so the planner always has recent
    state to plan against rather than only hearing from the robot when it is
    already stuck.
    """
    publisher = await situation.declare_publisher(node_runner)
    interval = 1.0 / SITUATION_RATE_HZ

    while True:
        now = time.time()
        escalated = state.authoritative_target(now) is None
        await publisher.publish(
            situation.build_message(now, list(state.joint_positions), escalated)
        )
        await asyncio.sleep(interval)


async def control_loop(
    params: Parameters, state: Situation, token: CancellationToken
) -> None:
    """The servo loop.

    Every input it reads is on this machine by construction: the launcher
    places the camera, the arm, and this node on one core node, so nothing here
    can be made slower by a bad link.
    """
    interval = 1.0 / params.control_rate_hz
    print(f"[reactive_policy] servo loop at {params.control_rate_hz} Hz")

    running_on_fallback = None
    while not token.is_cancelled():
        target = state.authoritative_target(time.time())
        on_fallback = target is None

        # Log only on transitions. At 200 Hz, logging per tick would bury the
        # one line that matters in tens of thousands that do not.
        if on_fallback != running_on_fallback:
            running_on_fallback = on_fallback
            if on_fallback:
                print(
                    "[reactive_policy] no authoritative subgoal; "
                    "falling back to local behavior"
                )
            else:
                print("[reactive_policy] servoing to the planner's subgoal")

        _ = target if target is not None else HOME_POSITION
        await asyncio.sleep(interval)


async def setup(params: Parameters, node_runner: NodeRunner) -> list[asyncio.Task]:
    state = Situation(params.subgoal_ttl_ms)
    token = node_runner.cancellation_token()

    async def announce_shutdown():
        print("[reactive_policy] Shutdown signal received")

    node_runner.on_shutdown(announce_shutdown)

    return [
        asyncio.create_task(track_arm(node_runner, state)),
        asyncio.create_task(track_camera(node_runner, state)),
        asyncio.create_task(adopt_subgoals(node_runner, state)),
        asyncio.create_task(report_situation(node_runner, state)),
        asyncio.create_task(control_loop(params, state, token)),
    ]


def main():
    NodeBuilder().run(setup)


if __name__ == "__main__":
    main()
