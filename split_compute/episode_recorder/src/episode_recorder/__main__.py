"""Records what the robot actually did, by observing rather than participating.

This node taps the executor side of the `task_delegation` pairing without
joining it. That distinction is the whole point of an observer slot: it claims
no slot on the source, holds no peer, and the source is not even aware of it, so
however badly this node behaves it cannot perturb control. The worst it can do
to the robot is fall behind.

It observes ACROSS a daemon boundary here, and nothing in this file reflects
that. What differs is invisible from this side: the source's lifecycle
transitions reach this node's daemon as notifications from the source's own
daemon rather than from local lifecycle events, which is what makes a source
restart still drop and redeclare this node's subscription.
"""

import asyncio

from peppygen import NodeBuilder, NodeRunner
from peppygen.consumed_topics.observed_execution import situation as situation_topic


async def run(runner: NodeRunner) -> None:
    token = runner.cancellation_token()
    print("[recorder] up, waiting for the executor's source pin", flush=True)

    subscription = await situation_topic.subscribe(runner)
    print("[recorder] observing execution", flush=True)

    recorded = 0
    while not token.is_cancelled():
        received = await subscription.next()
        if received is None:
            break
        _producer, message = received
        recorded += 1
        # Log the subgoal the executor reports servoing on. That is the field
        # that proves the pairing is live end to end: it originated on the
        # planner, was adopted by the executor, and is being read here by a
        # third node that is party to neither side.
        print(
            f"[recorder] episode step {recorded}: "
            f"active_subgoal_id={message.active_subgoal_id or 'none'} "
            f"escalating={message.escalating} observation={message.observation!r}",
            flush=True,
        )


def main() -> None:
    NodeBuilder().build().run(run)


if __name__ == "__main__":
    main()
