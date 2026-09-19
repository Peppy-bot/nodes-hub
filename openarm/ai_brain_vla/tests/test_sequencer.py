"""One sequence per lane, and stopping the one that runs."""

import asyncio

import pytest

from openarm_ai_brain_vla.ports import Cancelled, Refusal
from openarm_ai_brain_vla.sequencer import MANIPULATION, PERCEPTION, Sequencer


async def test_a_second_sequence_in_a_lane_is_refused_by_name():
    sequencer = Sequencer()
    job = sequencer.start(MANIPULATION, "grab_item", now_ns=0)
    with pytest.raises(Refusal, match="grab_item is running"):
        sequencer.start(MANIPULATION, "drop_item", now_ns=1)
    # The other lane is free: a search may run beside a manipulation.
    search = sequencer.start(PERCEPTION, "scan_items", now_ns=1)
    assert sequencer.running(PERCEPTION) is search
    sequencer.finish(job)
    assert sequencer.running(MANIPULATION) is None
    sequencer.start(MANIPULATION, "drop_item", now_ns=2)


async def test_stopping_an_idle_lane_reports_nothing():
    assert await Sequencer().stop(MANIPULATION, "aborted") is None


async def test_stopping_a_running_job_sets_its_token_calls_the_stopper_and_waits():
    stopped_lanes: list[str] = []

    async def stopper(lane: str) -> None:
        stopped_lanes.append(lane)

    sequencer = Sequencer(stopper=stopper)
    job = sequencer.start(MANIPULATION, "grab_item", now_ns=0)

    async def worker():
        await job.cancel.wait()
        with pytest.raises(Cancelled) as raised:
            job.cancel.check()
        assert raised.value.by_caller is False
        assert raised.value.reason == "aborted: operator"
        await asyncio.sleep(0.01)
        sequencer.finish(job)

    task = asyncio.create_task(worker())
    stopped = await asyncio.wait_for(sequencer.stop(MANIPULATION, "aborted: operator"), 1.0)
    await task
    assert stopped == "grab_item"
    assert stopped_lanes == [MANIPULATION]
    assert sequencer.running(MANIPULATION) is None


async def test_a_callers_cancel_is_marked_as_such():
    sequencer = Sequencer()
    job = sequencer.start(PERCEPTION, "scan_items", now_ns=0)
    await sequencer.cancel(job, "cancelled by the caller", by_caller=True)
    assert job.cancel.cancelled and job.cancel.by_caller
    assert job.elapsed_s(now_ns=2_500_000_000) == 2.5
