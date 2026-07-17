"""Bounded hand-off for live camera frames.

The camera subscription must be drained continuously so its transport buffer
cannot fill while the brain waits for arm actions.  A live camera is
latest-value data: retaining an old backlog only adds latency, so this mailbox
keeps at most one pending value and replaces it when a newer one arrives.
"""

import asyncio
from typing import Generic, TypeVar, cast


T = TypeVar("T")
_CLOSED = object()


class LatestValueMailbox(Generic[T]):
    """A single-consumer, latest-value mailbox for one asyncio event loop.

    ``offer`` never blocks.  If the consumer is busy and a value is already
    pending, the pending value is discarded in favor of the new one.  ``close``
    wakes a consumer parked in ``get``; a pending final value is delivered
    before ``get`` returns ``None``.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[T | object] = asyncio.Queue(maxsize=1)
        self._closed = False

    def offer(self, value: T) -> bool:
        """Offer ``value`` without blocking, returning false after close."""
        if self._closed:
            return False

        if self._queue.full():
            self._queue.get_nowait()
        self._queue.put_nowait(value)
        return True

    async def get(self) -> T | None:
        """Return the newest pending value, or ``None`` once closed and empty."""
        if self._closed and self._queue.empty():
            return None

        value = await self._queue.get()
        if value is _CLOSED:
            return None
        return cast(T, value)

    def close(self) -> None:
        """Close the mailbox and wake a consumer if no value is pending."""
        if self._closed:
            return

        self._closed = True
        if self._queue.empty():
            self._queue.put_nowait(_CLOSED)
