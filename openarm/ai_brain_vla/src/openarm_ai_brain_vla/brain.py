"""The brain assembled: the core parts, the two backends the launcher
selected, and the one way every action runs a sequence and completes its
goal.

`run_guarded` is where the contracts' result rules are applied for all
five sequences: a refusal or a failure completes with success false and
the message; a cancel by the caller completes as cancelled; an abort
completes with success false and "aborted: <reason>"; the fields the
contract says are zero or empty on failure are the `zero` dict each
handler passes. It is also where each goal of the five leaves its one line
in the log, so a search that found nothing says there what it looked at.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Optional, Sequence

from peppygen import clock

from .manipulation import make_manipulator
from .perception import make_detector
from .perception.camera import CameraModel
from .perception.frames import FrameStore
from .perception.geometry import learn_camera_pose, learn_intrinsics
from .perception.perceiver import Look, Perceiver
from .ports import Cancelled, Manipulator, Refusal
from .robot import Robot
from .sequencer import MANIPULATION, PERCEPTION, Job, Sequencer
from .state import State, new_run_token

logger = logging.getLogger(__name__)

Body = Callable[[Job], Awaitable[dict]]


class Brain:
    def __init__(
        self,
        params,
        node_runner,
        *,
        detector=None,
        manipulator: Optional[Manipulator] = None,
        now: Optional[Callable[[], int]] = None,
        run_token: Optional[str] = None,
    ) -> None:
        self.params = params
        self.node_runner = node_runner
        self.state = State(params.gripper_names.split(","), new_run_token() if run_token is None else run_token)
        self.robot = Robot(node_runner)
        self.frames = FrameStore()
        self.camera = CameraModel()
        self.perceiver = Perceiver(detector or make_detector(params.perception_backend), self.frames, self.camera)
        if params.perception_confidence > 0.0:
            self.perceiver.detector.min_confidence = params.perception_confidence
        self.manipulator: Manipulator = manipulator or make_manipulator(params.manipulation_backend)
        self.sequencer = Sequencer(stopper=self._stop_lane)
        self._now = now or clock.now_ns

    def now(self) -> int:
        return self._now()

    async def start(self) -> None:
        """Hands the manipulator the robot and begins loading the perception
        model, once. The load runs in the background, since a backend that
        takes a minute to load must not hold the node's start; searches are
        refused as still loading until it ends."""
        self.perceiver.start_loading(self.params.perception_model, self.params.perception_gallery)
        await self.manipulator.start(self.robot)

    def background(self, token) -> list[asyncio.Task]:
        """The loops that run beside the action loops: the camera follower,
        the one asking the camera where its pixels point, and the one
        asking the robot where the camera stands."""
        return [
            asyncio.create_task(self.frames.run(self.node_runner, token)),
            asyncio.create_task(learn_intrinsics(self.node_runner, token, self.perceiver)),
            asyncio.create_task(learn_camera_pose(self.node_runner, token, self.perceiver, self.params.camera_name)),
        ]

    async def shutdown(self) -> None:
        """The shutdown hook: stop what runs while the messenger is still
        connected, so in-flight moves end as cancelled rather than
        abandoned."""
        await self.sequencer.stop(MANIPULATION, "aborted: the node is shutting down")
        await self.sequencer.stop(PERCEPTION, "aborted: the node is shutting down")
        await self.robot.stop()

    async def look(self, job: Job, phrases: Sequence[str], timeout_s: float) -> Look:
        """One look at the camera for the search `job` runs, as
        `Perceiver.scan` gives it; what the look saw goes on the job's notes
        for the log."""
        seen = await self.perceiver.scan(phrases, job.cancel, timeout_s)
        job.notes.append(seen.describe(job.started_ns))
        return seen

    def manipulator_or_refuse(self) -> Manipulator:
        """The manipulation backend, or the refusal every sequence gets
        when the launcher selected none."""
        if not self.manipulator.available:
            raise Refusal(f"no manipulation backend: manipulation_backend is '{self.manipulator.name}'")
        return self.manipulator

    async def _stop_lane(self, lane: str) -> None:
        if lane == MANIPULATION:
            await self.manipulator.stop()
            await self.robot.stop()

    async def run_guarded(self, ctx, lane: str, action: str, body: Body, zero: dict, asked: str = "") -> None:
        """Admits the goal into its lane, runs `body` under a cancel
        watcher, and completes the goal the way the contract wants. Before
        it completes, the goal leaves its line in the log: the action and
        `asked`, what the goal asked for in a few words, how the goal ended
        and after how long, then the notes the body put on its job."""
        started = self.now()
        goal = f"{action} {asked}" if asked else action
        try:
            job = self.sequencer.start(lane, action, started)
        except Refusal as refusal:
            self._log_end(goal, "refused", started, refusal.message)
            await ctx.complete(success=False, message=refusal.message, action_time=0.0, **zero)
            return
        watcher = asyncio.create_task(self._watch_cancel(ctx, job))
        try:
            fields = await body(job)
            job.cancel.check()
            self._log_end(goal, "succeeded", started, "", job.notes)
            await ctx.complete(success=True, message="", action_time=self._elapsed(started), **fields)
        except Refusal as refusal:
            self._log_end(goal, "refused", started, refusal.message, job.notes)
            await ctx.complete(success=False, message=refusal.message, action_time=0.0, **zero)
        except Cancelled as cancelled:
            self._log_end(goal, "stopped", started, cancelled.reason, job.notes)
            if cancelled.by_caller:
                await ctx.complete_cancelled(success=False, message=cancelled.reason, action_time=self._elapsed(started), **zero)
            else:
                await ctx.complete(success=False, message=cancelled.reason, action_time=self._elapsed(started), **zero)
        except Exception as error:
            self._log_end(goal, "failed", started, repr(error), job.notes, error=error)
            await ctx.complete(success=False, message=f"{action} failed: {error!r}", action_time=0.0, **zero)
        finally:
            watcher.cancel()
            self.sequencer.finish(job)

    def _log_end(
        self, goal: str, ended: str, started_ns: int, reason: str, notes: Sequence[str] = (), error: Optional[Exception] = None
    ) -> None:
        """The one line a goal leaves in the log: `goal`, how it `ended`
        and after how long, the reason when there is one, then `notes`. A
        failure, `error`, logs at error level with its traceback."""
        line = f"{goal} {ended} after {self._elapsed(started_ns):.2f} s"
        if reason:
            line += f": {reason}"
        line = "; ".join([line, *notes])
        if error is None:
            logger.info("%s", line)
        else:
            logger.error("%s", line, exc_info=error)

    async def _watch_cancel(self, ctx, job: Job) -> None:
        await ctx.cancel_signal()
        await self.sequencer.cancel(job, "cancelled by the caller", by_caller=True)

    def _elapsed(self, started_ns: int) -> float:
        return max(0.0, (self.now() - started_ns) / 1e9)
