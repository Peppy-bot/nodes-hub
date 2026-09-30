"""The loops' one way of waiting on the node's token, and what a handler
that raises leaves in the log."""

import asyncio
import logging

import pytest

from conftest import FakeToken
from openarm_ai_brain_vla.serve import report_failure
from openarm_ai_brain_vla.waiting import unless_cancelled


async def test_the_awaitable_wins_when_it_finishes_first():
    async def answer():
        return "goal"

    assert await unless_cancelled(FakeToken(), answer()) == "goal"


async def test_the_token_wins_and_the_awaitable_is_cancelled():
    token = FakeToken()
    reached = asyncio.Event()
    cancelled = asyncio.Event()

    async def wait_forever():
        reached.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    waiting = asyncio.create_task(unless_cancelled(token, wait_forever()))
    await reached.wait()
    token.cancel()
    assert await asyncio.wait_for(waiting, 1.0) is None
    await asyncio.wait_for(cancelled.wait(), 1.0)


async def test_what_the_awaitable_raises_is_raised():
    async def fail():
        raise ValueError("no")

    with pytest.raises(ValueError, match="no"):
        await unless_cancelled(FakeToken(), fail())


@pytest.fixture
def serve_log(caplog):
    """The serve module's log records, whatever the package logger's
    propagation is set to."""
    logger = logging.getLogger("openarm_ai_brain_vla.serve")
    propagated = logger.propagate
    logger.propagate = False
    logger.addHandler(caplog.handler)
    try:
        yield caplog
    finally:
        logger.removeHandler(caplog.handler)
        logger.propagate = propagated


async def test_a_handler_that_raises_is_reported_with_its_traceback(serve_log):
    async def handler():
        raise RuntimeError("the goal context is gone")

    task = asyncio.create_task(handler())
    with pytest.raises(RuntimeError):
        await task
    report_failure(task)
    [record] = serve_log.records
    assert record.levelno == logging.ERROR
    assert record.getMessage() == "a goal handler failed and its goal was never completed"
    assert "RuntimeError: the goal context is gone" in serve_log.text


async def test_a_cancelled_or_clean_handler_is_not_reported(serve_log):
    async def clean():
        return None

    async def hang():
        await asyncio.Event().wait()

    done = asyncio.create_task(clean())
    await done
    cancelled = asyncio.create_task(hang())
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    report_failure(done)
    report_failure(cancelled)
    assert serve_log.records == []
