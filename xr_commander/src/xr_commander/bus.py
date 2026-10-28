"""Small helpers shared by every stream task: the clock, logging, firing a
goal, the outcomes of service calls and bus receive."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable
from dataclasses import dataclass
from typing import Any, Protocol


class Clock(Protocol):
    """Where the node reads time: the ticks of its loops, the freshness of a
    frame and of a measured pose, the release burst, the panel notes and the
    retries toward the daemon. Tests inject their own. The alert, motor
    health and sensor hold stores read the time through a `monotonic`
    callable of their own."""

    def monotonic(self) -> float: ...
    async def sleep(self, seconds: float) -> None: ...


class RealClock:
    """The host's monotonic clock and the event loop's sleep."""

    def monotonic(self) -> float:
        return time.monotonic()

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


REAL_CLOCK = RealClock()

# The tasks that `spawn` started, until each ends: the event loop keeps a weak
# reference to a task only.
_SPAWNED: set[asyncio.Task] = set()


def spawn(coroutine) -> asyncio.Task:
    """Run `coroutine` as a task that nothing awaits, kept until it ends."""
    task = asyncio.create_task(coroutine)
    _SPAWNED.add(task)
    task.add_done_callback(_SPAWNED.discard)
    return task


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


def latch_for(latches: dict[Any, Latch], key: Any) -> Latch:
    """`key`'s latch in `latches`, made on first use."""
    latch = latches.get(key)
    if latch is None:
        latch = latches[key] = Latch()
    return latch


def producer_key(producer) -> tuple[str, str]:
    """A producer's (core node, instance id): unique across the mesh, where
    an instance id alone is unique only within one stack."""
    return (producer.core_node, producer.instance_id)


# How long an outcome flashes on a panel row.
FLASH_S = 3.0


class FlashNote:
    """A line that shows for `FLASH_S` after it is set, then clears."""

    def __init__(self, clock: Clock = REAL_CLOCK) -> None:
        self._clock = clock
        self._text = ""
        self._until_monotonic = 0.0

    def set(self, text: str) -> None:
        self._text = text
        self._until_monotonic = self._clock.monotonic() + FLASH_S

    def text(self, now_monotonic: float | None = None) -> str:
        now = self._clock.monotonic() if now_monotonic is None else now_monotonic
        return self._text if now < self._until_monotonic else ""


def cause(message: str) -> str:
    """The clause of the message of a refusal or a failure that names the
    cause, ahead of the remedy a backbone appends after a semicolon."""
    return (message or "no reason given").split(";", 1)[0]


async def fire_goal(node_runner, action_module, target, name: str, **request_fields):
    """The goal handle, accepted or refused with the refusal logged, or None
    with the send failure logged. Builds the request too, so a constructor
    that raises is logged as a send failure and the button loop goes on."""
    try:
        request = action_module.GoalRequest(**request_fields)
        fired = await action_module.ActionHandle.fire_goal(
            node_runner, target, request, GOAL_SEND_TIMEOUT_S
        )
    except Exception as e:
        log(f"{name} failed to send: {e!r}")
        return None
    if not fired.accepted:
        log(f"{name} refused: {fired.refusal}")
    return fired


@dataclass(frozen=True)
class Answered:
    """The answer of a call."""

    response: object


@dataclass(frozen=True)
class Failed:
    """The error reply of a call, by its cause."""

    cause: str


@dataclass(frozen=True)
class Unanswered:
    """A call that got no answer: it timed out, or its producer is gone."""


Outcome = Answered | Failed | Unanswered


async def outcome_of(call: Awaitable, failure: type[Exception], what: str) -> Outcome:
    """How a service call ended; a call that got no answer is logged as
    `what`."""
    try:
        return Answered(await call)
    except failure as error:
        return Failed(cause(error.message))
    except Exception as e:
        log(f"{what} unanswered: {e!r}")
        return Unanswered()


def refusal_of(outcome: Outcome, unanswered: str) -> str | None:
    """The cause of a call that was not answered with a yes, None for a yes;
    `unanswered` says a call that got no answer."""
    match outcome:
        case Answered():
            return None
        case Failed(cause=refused):
            return refused
        case Unanswered():
            return unanswered


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


async def ticks(
    period_s: float, token: CancellationToken, clock: Clock = REAL_CLOCK
) -> AsyncIterator[None]:
    """Yield once per `period_s` until the token cancels.

    Deadline-paced so the cadence does not drift; a slipped tick resyncs
    instead of bursting, and a saturated loop still yields to the rest of the
    event loop.
    """
    deadline = clock.monotonic()
    while not token.is_cancelled():
        deadline += period_s
        delay = deadline - clock.monotonic()
        if delay > 0.0:
            await clock.sleep(delay)
        else:
            deadline = clock.monotonic()
            await asyncio.sleep(0)
        # Re-checked after the sleep: a shutdown that lands mid-wait must not
        # buy one more tick of work.
        if token.is_cancelled():
            return
        yield


async def subscribe_or_log(node_runner, topic_module, label: str):
    """`topic_module`'s subscription, or None with the failure logged as
    `label`'s."""
    try:
        return await topic_module.subscribe(node_runner)
    except Exception as e:
        log(f"{label} subscribe failed: {e!r}")
        return None


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
