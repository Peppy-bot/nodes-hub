"""One sequence at a time per lane, and the way to stop the one running.

Two lanes, because the contracts allow a search while a manipulation
runs: `PERCEPTION` for scan_items and identify_item, `MANIPULATION` for
grab_item, drop_item and place_item. get_state reports the manipulation
lane only; a running search never appears there, so a caller that reads
"identify_item is running" is never talked out of a grab.

A stop is the same whether the caller cancelled its own goal or anyone
sent abort: the job's token is set, the lane's stopper halts the backend
and the move in flight, and the job finishes on its own.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

from .ports import CancelToken, Refusal

PERCEPTION = "perception"
MANIPULATION = "manipulation"

Stopper = Callable[[str], Awaitable[None]]


@dataclass
class Job:
    lane: str
    action: str
    started_ns: int
    cancel: CancelToken = field(default_factory=CancelToken)
    done: asyncio.Event = field(default_factory=asyncio.Event)

    def elapsed_s(self, now_ns: int) -> float:
        return max(0.0, (now_ns - self.started_ns) / 1e9)


class Sequencer:
    def __init__(self, stopper: Optional[Stopper] = None) -> None:
        self._running: dict[str, Job] = {}
        self._stopper = stopper

    def running(self, lane: str) -> Optional[Job]:
        return self._running.get(lane)

    def start(self, lane: str, action: str, now_ns: int) -> Job:
        """Admits a sequence into its lane, or refuses it naming the one
        that runs, as the contracts ask."""
        running = self._running.get(lane)
        if running is not None:
            raise Refusal(f"{running.action} is running")
        job = Job(lane=lane, action=action, started_ns=now_ns)
        self._running[lane] = job
        return job

    def finish(self, job: Job) -> None:
        if self._running.get(job.lane) is job:
            del self._running[job.lane]
        job.done.set()

    async def cancel(self, job: Job, reason: str, *, by_caller: bool) -> None:
        """Stops one job: sets its token and halts the lane's backend."""
        if job.done.is_set():
            return
        job.cancel.cancel(reason, by_caller=by_caller)
        if self._stopper is not None:
            await self._stopper(job.lane)

    async def stop(self, lane: str, reason: str) -> Optional[str]:
        """abort: stops whatever runs in the lane and waits until it has
        finished, which for manipulation means the robot has come to rest.
        Returns the stopped action's name, or none when the lane was idle."""
        job = self._running.get(lane)
        if job is None:
            return None
        await self.cancel(job, reason, by_caller=False)
        await job.done.wait()
        return job.action
