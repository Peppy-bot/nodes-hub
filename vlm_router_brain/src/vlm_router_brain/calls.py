"""One call to a model under a deadline, asked again once when it may be:
the rule the detector and the step model share.

A call gets the budget left, never under the route's `min_call_timeout_s`,
the least deadline the provider takes. A call that failed is asked again
once, after `RETRY_WAIT_S`, when the failure is one that comes back at
once (`ProviderFailure.retryable`: a 429, a gateway timeout, a 5xx) and
the budget has room for the wait. A budget gone before a call raises
`SearchTimeout` from the deadline, so the caller refuses or fails its
goal.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Sequence

from brain_core.ports import Deadline

from .catalog import Route
from .provider import Provider, ProviderFailure, Reply

logger = logging.getLogger(__name__)

RETRY_WAIT_S = 1.0

Sleep = Callable[[float], Awaitable[None]]


async def ask_once_more(
    provider: Provider,
    route: Route,
    images: Sequence[bytes],
    prompt: str,
    schema: object,
    deadline: Deadline,
    *,
    name: str,
    sleep: Sleep = asyncio.sleep,
) -> Reply:
    """One answer within `deadline`, asked again once after a failure that
    may be asked again, when the budget has room for the wait. `name` is
    the model's catalog name, for the log."""
    try:
        return await _call(provider, route, images, prompt, schema, deadline)
    except ProviderFailure as failure:
        if not failure.retryable or deadline.remaining_s() <= RETRY_WAIT_S:
            raise
        logger.info("%s: %s; asking again in %g s", name, failure.message, RETRY_WAIT_S)
    await sleep(RETRY_WAIT_S)
    return await _call(provider, route, images, prompt, schema, deadline)


async def _call(provider: Provider, route: Route, images: Sequence[bytes], prompt: str, schema: object, deadline: Deadline) -> Reply:
    deadline.check()
    timeout_s = max(deadline.remaining_s(), route.min_call_timeout_s)
    return await provider.ask(route, images, prompt, schema, timeout_s)
