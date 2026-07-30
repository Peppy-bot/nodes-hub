import asyncio

import pytest

from tests.helpers import FakeSubscription, FakeToken
from xr_commander.bus import Latch, messages, next_message


class HangingSubscription:
    """Never yields; records whether its pending receive was cancelled."""

    def __init__(self):
        self.receive_cancelled = asyncio.Event()

    async def next(self):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.receive_cancelled.set()
            raise


class NeverCancelledToken:
    def is_cancelled(self):
        return False

    async def cancelled(self):
        await asyncio.Event().wait()


class SyncBrokenSubscription:
    def next(self):
        raise RuntimeError("broken transport")


def test_cancelling_the_caller_cancels_the_pending_receive():
    # An orphaned receive would outlive its loop and die noisily at shutdown.
    async def scenario():
        subscription = HangingSubscription()
        task = asyncio.create_task(next_message(subscription, NeverCancelledToken()))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(subscription.receive_cancelled.wait(), 1.0)

    asyncio.run(scenario())


def test_a_synchronously_broken_subscription_leaks_no_tasks():
    async def scenario():
        with pytest.raises(RuntimeError, match="broken transport"):
            await next_message(SyncBrokenSubscription(), NeverCancelledToken())
        # Give cancellation a few ticks to finalize, then demand a clean loop.
        for _ in range(3):
            await asyncio.sleep(0)
        assert [t for t in asyncio.all_tasks() if t is not asyncio.current_task()] == []

    asyncio.run(scenario())


class SelfCancellingSubscription:
    async def next(self):
        raise asyncio.CancelledError


class CancelledToken(NeverCancelledToken):
    def is_cancelled(self):
        return True


class BrokenToken(NeverCancelledToken):
    async def cancelled(self):
        raise RuntimeError("token backend gone")


class SelfCancellingToken(NeverCancelledToken):
    async def cancelled(self):
        raise asyncio.CancelledError


def test_a_transport_side_cancel_reads_as_a_clean_close():
    async def scenario():
        subscription = SelfCancellingSubscription()
        assert await next_message(subscription, NeverCancelledToken()) is None

    asyncio.run(scenario())


def test_messages_ends_before_receiving_on_a_cancelled_token():
    async def scenario():
        stream = messages(HangingSubscription(), CancelledToken(), "test")
        assert [item async for item in stream] == []

    asyncio.run(scenario())


def test_a_broken_cancellation_watcher_is_loud_not_a_clean_close(capsys):
    async def scenario():
        assert await next_message(HangingSubscription(), BrokenToken()) is None

    asyncio.run(scenario())
    assert "cancellation watcher failed" in capsys.readouterr().out


def test_a_self_cancelling_watcher_is_loud_too(capsys):
    async def scenario():
        subscription = HangingSubscription()
        assert await next_message(subscription, SelfCancellingToken()) is None

    asyncio.run(scenario())
    assert "cancellation watcher failed" in capsys.readouterr().out


def test_a_broken_watcher_logs_once_at_stream_end_not_per_message(capsys):
    async def scenario():
        stream = messages(FakeSubscription([1, 2, 3]), BrokenToken(), "test")
        assert [item async for item in stream] == [1, 2, 3]

    asyncio.run(scenario())
    reported = [
        line
        for line in capsys.readouterr().out.splitlines()
        if "cancellation watcher" in line
    ]
    assert len(reported) == 1


class RaiseOnceSubscription:
    """Fails one receive, then yields its item, then pends forever."""

    def __init__(self, item):
        self._item = item
        self._raised = False

    async def next(self):
        if not self._raised:
            self._raised = True
            raise RuntimeError("transient receive fault")
        if self._item is not None:
            item, self._item = self._item, None
            return item
        await asyncio.Event().wait()


def test_messages_survives_a_failed_receive_and_resumes(capsys):
    async def scenario():
        subscription = RaiseOnceSubscription(("producer", "payload"))
        stream = messages(subscription, NeverCancelledToken(), "test")
        received = await asyncio.wait_for(anext(stream), 2.0)
        assert received == ("producer", "payload")
        await stream.aclose()

    asyncio.run(scenario())
    assert "test receive error" in capsys.readouterr().out


def test_a_normal_token_fire_ends_the_stream_quietly(capsys):
    async def scenario():
        token = FakeToken()
        token.cancel()
        assert await next_message(HangingSubscription(), token) is None

    asyncio.run(scenario())
    assert "cancellation watcher" not in capsys.readouterr().out


def test_a_latch_logs_once_until_cleared(capsys):
    latch = Latch()
    latch.trip("stream broken")
    latch.trip("stream still broken")
    latch.clear()
    latch.trip("stream broken anew")
    reported = [
        line for line in capsys.readouterr().out.splitlines() if "suppressing" in line
    ]
    assert reported == [
        "[xr_commander] stream broken; suppressing repeats",
        "[xr_commander] stream broken anew; suppressing repeats",
    ]
