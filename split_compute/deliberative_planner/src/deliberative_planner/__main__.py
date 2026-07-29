"""Slow deliberative planner: advises, never commands.

A mock in the shape of a large vision-language-action model. It emits a canned
subgoal on a timer rather than running inference, so the E2E that drives it is
deterministic, needs no model weights, and does not depend on host speed.

What it does faithfully reproduce is the SHAPE of the real thing: it reads the
same camera the reactive policy reads (across a daemon boundary, named
identically), it plans two orders of magnitude slower than the control loop, and
it has no way to touch an actuator. Nothing here can stall the robot.
"""

import asyncio
import itertools

from peppygen import NodeBuilder, NodeRunner
from peppygen.parameters import Parameters
from peppygen.consumed_topics.scene import video_stream
from peppygen.consumed_topics.deliberation import situation as situation_topic
from peppygen.emitted_topics.deliberation import subgoal as subgoal_topic

# Cycled through so consecutive subgoals differ. A test asserting that a subgoal
# was applied needs to tell a NEW one from a redelivery of the old one.
CANNED_DIRECTIVES = [
    "approach_object",
    "grasp_object",
    "lift_object",
    "place_object",
]

# How long each subgoal stays authoritative on the executor, in milliseconds.
# Comfortably longer than a plan cycle so the normal case is a fresh subgoal
# arriving before the previous one lapses; a severed link is what makes the
# executor's watchdog fire.
SUBGOAL_VALIDITY_MS = 5000


async def watch_scene(runner: NodeRunner, seen: dict) -> None:
    """Drain the camera across the daemon boundary.

    This is the cross-daemon PRODUCER LINK. It is written exactly as a local
    subscription would be, because the producer address already carries its core
    node: placement was declared once, on the instance, in the launcher.
    """
    token = runner.cancellation_token()
    subscription = await video_stream.subscribe(runner)
    while not token.is_cancelled():
        received = await subscription.next()
        if received is None:
            break
        seen["frames"] = seen.get("frames", 0) + 1
        if seen["frames"] == 1:
            print("[planner] first frame received across the boundary", flush=True)


async def watch_situation(runner: NodeRunner, seen: dict) -> None:
    """Track what the executor is facing, including its escalations."""
    token = runner.cancellation_token()
    subscription = await situation_topic.subscribe(runner)
    while not token.is_cancelled():
        received = await subscription.next()
        if received is None:
            break
        _producer, message = received
        seen["situations"] = seen.get("situations", 0) + 1
        if message.escalating:
            print(
                f"[planner] executor escalated: {message.observation}",
                flush=True,
            )


async def plan_loop(runner: NodeRunner, seen: dict, plan_rate_hz: int, horizon_s: int) -> None:
    """Emit one subgoal per plan cycle."""
    token = runner.cancellation_token()
    period = 1.0 / plan_rate_hz

    for cycle in itertools.count():
        if token.is_cancelled():
            break
        directive = CANNED_DIRECTIVES[cycle % len(CANNED_DIRECTIVES)]
        subgoal_id = f"subgoal-{cycle}"

        await subgoal_topic.emit(
            runner,
            subgoal_topic.Subgoal(
                stamp=runner.now(),
                subgoal_id=subgoal_id,
                directive=directive,
                valid_for_ms=SUBGOAL_VALIDITY_MS,
            ),
        )
        print(
            f"[planner] issued {subgoal_id}: {directive} "
            f"(horizon {horizon_s}s, frames={seen.get('frames', 0)})",
            flush=True,
        )
        await asyncio.sleep(period)


async def run(runner: NodeRunner) -> None:
    parameters = Parameters(runner)
    seen: dict = {}

    print(
        f"[planner] up at {parameters.plan_rate_hz} Hz, "
        f"horizon {parameters.horizon_s}s",
        flush=True,
    )

    await asyncio.gather(
        watch_scene(runner, seen),
        watch_situation(runner, seen),
        plan_loop(runner, seen, parameters.plan_rate_hz, parameters.horizon_s),
    )


def main() -> None:
    NodeBuilder().build().run(run)


if __name__ == "__main__":
    main()
