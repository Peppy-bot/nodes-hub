import asyncio
import contextlib
import time

import pytest

from peppygen.consumed_actions.postures import move_to_home as postures_move_to_home
from peppygen.consumed_actions.recorder import record_episode as recorder_record_episode

from tests.helpers import FakeSubscription, FakeToken, boot
from xr_commander.bus import Latch, messages, next_message, select_producer, ticks


async def test_select_producer_picks_the_first_and_reports_a_surplus(capsys):
    async with boot(postures_instances=2) as h:
        bound = postures_move_to_home.bound_producers(h.node_runner)
        target = select_producer(postures_move_to_home, h.node_runner, "postures")
        assert target.instance_id == bound[0].instance_id
    assert "several postures producers" in capsys.readouterr().out


async def test_select_producer_is_none_and_silent_on_an_empty_slot(capsys):
    async with boot() as h:  # the recorder slot boots unwired
        assert (
            select_producer(recorder_record_episode, h.node_runner, "recorder") is None
        )
    # Silent: nothing the node logs, in particular no surplus line.
    assert "[xr_commander]" not in capsys.readouterr().out


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


async def collect_ticks(period_s, token, limit, body_s=0.0):
    """Run the pacer `limit` times, optionally with a body that takes time."""
    started = time.monotonic()
    count = 0
    async for _ in ticks(period_s, token):
        if body_s:
            await asyncio.sleep(body_s)
        count += 1
        if count >= limit:
            break
    return count, time.monotonic() - started


def test_the_pacer_holds_its_cadence_rather_than_adding_the_body_time():
    # 20 paced ticks take ~0.40s; a naive sleep(period) per iteration would
    # add the body and take limit * (period + body) = 0.60s. The bound sits
    # between with margin for a loaded runner.
    _count, elapsed = asyncio.run(
        collect_ticks(0.02, NeverCancelledToken(), 20, body_s=0.01)
    )
    assert elapsed < 0.50, f"cadence drifted with the body: {elapsed:.3f}s"


def test_a_cancel_landing_mid_sleep_buys_no_further_tick():
    async def scenario():
        token = FakeToken()
        asyncio.get_running_loop().call_later(0.005, token.cancel)
        count, _elapsed = await collect_ticks(0.05, token, limit=5)
        return count

    # The cancel lands during the first period's sleep, before any yield.
    assert asyncio.run(scenario()) == 0


def test_a_slipped_tick_resyncs_instead_of_bursting():
    # One long body must not leave the pacer owing several instant ticks.
    async def scenario():
        token = NeverCancelledToken()
        stream = ticks(0.02, token)
        await anext(stream)
        await asyncio.sleep(0.1)  # overrun five periods
        started = time.monotonic()
        await anext(stream)
        await anext(stream)
        await stream.aclose()
        return time.monotonic() - started

    # Bursting would return both ticks immediately; resyncing paces the second.
    assert asyncio.run(scenario()) > 0.01


def test_a_saturated_pacer_still_lets_other_tasks_run():
    # The setpoint streams run on this pacer; starving the loop would stall
    # the WebSocket ingest the grip deadman depends on.
    async def scenario():
        ran = []

        async def competitor():
            for _ in range(5):
                await asyncio.sleep(0)
                ran.append(1)

        task = asyncio.create_task(competitor())
        count = 0
        async for _ in ticks(0.0, NeverCancelledToken()):
            count += 1
            if count >= 20:
                break
        progressed = len(ran)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        return progressed

    assert asyncio.run(scenario()) > 0, "the pacer never yielded"


def test_a_cancelled_token_ends_the_pacer_before_any_tick():
    async def scenario():
        return [tick async for tick in ticks(0.01, CancelledToken())]

    assert asyncio.run(scenario()) == []
