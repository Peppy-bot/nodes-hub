"""The one way a loop of the node waits for something unless the node's
cancellation token fires first."""

from __future__ import annotations

import asyncio
from typing import Awaitable, Optional, TypeVar

T = TypeVar("T")


async def unless_cancelled(token, awaitable: Awaitable[T]) -> Optional[T]:
    """Awaits `awaitable` and returns its result, or None when `token`
    fires first, the awaitable cancelled. What the awaitable raises is
    raised here."""
    pending = asyncio.ensure_future(awaitable)
    cancelled = asyncio.ensure_future(token.cancelled())
    try:
        await asyncio.wait([cancelled, pending], return_when=asyncio.FIRST_COMPLETED)
    except asyncio.CancelledError:
        pending.cancel()
        raise
    finally:
        cancelled.cancel()
    if not pending.done():
        pending.cancel()
        return None
    return pending.result()
