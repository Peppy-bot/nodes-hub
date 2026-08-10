"""Small helpers shared by every stream task: logging and bus receive."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any, Protocol


class CancellationToken(Protocol):
    """The runner's shutdown signal, polled or awaited."""

    def is_cancelled(self) -> bool: ...
    async def cancelled(self) -> None: ...


class Subscription(Protocol):
    """The receive side of a generated topic subscription."""

    async def next(self) -> Any: ...


# Backoff after a failed receive, so a broken subscription cannot spin a loop.
_RECEIVE_RETRY_S = 0.1

# Bus deadlines shared by every goal-firing task: sending and cancelling are
# near-instant round trips.
GOAL_SEND_TIMEOUT_S = 2.0
GOAL_CANCEL_TIMEOUT_S = 2.0


def log(message: str) -> None:
    """Stdout with the node prefix, flushed so a dev run buffers nothing."""
    print(f"[xr_commander] {message}", flush=True)


def select_producer(module, node_runner, label: str):
    """The slot's first bound producer, or None when none is bound; a surplus
    is logged so the unused producers are not a silent mystery."""
    producers = module.bound_producers(node_runner)
    if not producers:
        return None
    if len(producers) > 1:
        log(f"several {label} producers bound; using {producers[0].instance_id}")
    return producers[0]


class Latch:
    """Log a recurring condition once, re-arming once it clears."""

    def __init__(self) -> None:
        self._tripped = False

    def trip(self, message: str) -> None:
        if not self._tripped:
            self._tripped = True
            log(f"{message}; suppressing repeats")

    def clear(self) -> None:
        self._tripped = False


async def next_message(subscription: Subscription, token: CancellationToken) -> Any:
    """The subscription's next (producer, message), or None once the token
    cancels or the subscription closes."""
    cancelled = asyncio.ensure_future(token.cancelled())
    receive = None
    try:
        receive = asyncio.ensure_future(subscription.next())
        await asyncio.wait([cancelled, receive], return_when=asyncio.FIRST_COMPLETED)
        if not receive.done():
            # The watcher ended this call, so it is done; a broken one must
            # not read as a clean close. Logged here, once per ended stream.
            if cancelled.cancelled():
                log("cancellation watcher failed: cancelled itself")
            elif cancelled.exception():
                log(f"cancellation watcher failed: {cancelled.exception()!r}")
            return None
        try:
            return receive.result()
        except asyncio.CancelledError:
            # The transport cancelled its own receive (caller cancellation is
            # always delivered at the wait above): the subscription is done.
            return None
    finally:
        # Both: cancelling the caller must not orphan a pending future.
        cancelled.cancel()
        if receive is not None:
            receive.cancel()


async def ticks(period_s: float, token: CancellationToken) -> AsyncIterator[None]:
    """Yield once per `period_s` until the token cancels.

    Deadline-paced so the cadence does not drift; a slipped tick resyncs
    instead of bursting, and a saturated loop still yields to the rest of the
    event loop.
    """
    deadline = time.monotonic()
    while not token.is_cancelled():
        deadline += period_s
        delay = deadline - time.monotonic()
        if delay > 0.0:
            await asyncio.sleep(delay)
        else:
            deadline = time.monotonic()
            await asyncio.sleep(0)
        # Re-checked after the sleep: a shutdown that lands mid-wait must not
        # buy one more tick of work.
        if token.is_cancelled():
            return
        yield


async def messages(
    subscription: Subscription, token: CancellationToken, label: str
) -> AsyncIterator[Any]:
    """Every (producer, message) until the token cancels or the subscription
    closes; a failed receive logs and backs off instead of ending the stream."""
    failing = Latch()
    while not token.is_cancelled():
        try:
            received = await next_message(subscription, token)
        except Exception as e:
            failing.trip(f"{label} receive error: {e!r}")
            await asyncio.sleep(_RECEIVE_RETRY_S)
            continue
        failing.clear()
        if received is None:
            return
        yield received
