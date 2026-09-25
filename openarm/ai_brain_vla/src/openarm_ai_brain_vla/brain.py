"""The brain assembled: the core parts, the two backends the launcher
selected, and the one way every action runs a sequence and completes its
goal.

`run_guarded` is where the contracts' result rules are applied for all
five sequences: a refusal or a failure completes with success false and
the message; a cancel by the caller completes as cancelled; an abort
completes with success false and "aborted: <reason>"; the fields the
contract says are zero or empty on failure are the `zero` dict each
handler passes.
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Optional

from peppygen import clock

from .manipulation import make_manipulator
from .perception import make_detector
from .perception.camera import CameraModel
from .perception.frames import FrameStore
from .perception.perceiver import Perceiver
from .ports import Cancelled, Manipulator, Refusal
from .robot import Robot
from .sequencer import MANIPULATION, PERCEPTION, Job, Sequencer
from .state import State

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
    ) -> None:
        self.params = params
        self.node_runner = node_runner
        self.state = State(params.gripper_names.split(","))
        self.robot = Robot(node_runner)
        self.frames = FrameStore()
        self.camera = CameraModel.from_parameters(params.camera_fovy_deg, params.camera_pose)
        self.perceiver = Perceiver(detector or make_detector(params.perception_backend), self.frames, self.camera)
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
        self.perceiver.start_loading(self.params.perception_model)
        await self.manipulator.start(self.robot)

    def background(self, token) -> list[asyncio.Task]:
        """The loops that run beside the action loops: the camera follower."""
        return [asyncio.create_task(self.frames.run(self.node_runner, token))]

    async def shutdown(self) -> None:
        """The shutdown hook: stop what runs while the messenger is still
        connected, so in-flight moves end as cancelled rather than
        abandoned."""
        await self.sequencer.stop(MANIPULATION, "aborted: the node is shutting down")
        await self.sequencer.stop(PERCEPTION, "aborted: the node is shutting down")
        await self.robot.stop()

    async def _stop_lane(self, lane: str) -> None:
        if lane == MANIPULATION:
            await self.manipulator.stop()
            await self.robot.stop()

    async def run_guarded(self, ctx, lane: str, action: str, body: Body, zero: dict) -> None:
        """Admits the goal into its lane, runs `body` under a cancel
        watcher, and completes the goal the way the contract wants."""
        started = self.now()
        try:
            job = self.sequencer.start(lane, action, started)
        except Refusal as refusal:
            await ctx.complete(success=False, message=refusal.message, action_time=0.0, **zero)
            return
        watcher = asyncio.create_task(self._watch_cancel(ctx, job))
        try:
            fields = await body(job)
            job.cancel.check()
            await ctx.complete(success=True, message="", action_time=self._elapsed(started), **fields)
        except Refusal as refusal:
            await ctx.complete(success=False, message=refusal.message, action_time=0.0, **zero)
        except Cancelled as cancelled:
            if cancelled.by_caller:
                await ctx.complete_cancelled(success=False, message=cancelled.reason, action_time=self._elapsed(started), **zero)
            else:
                await ctx.complete(success=False, message=cancelled.reason, action_time=self._elapsed(started), **zero)
        except Exception as error:
            await ctx.complete(success=False, message=f"{action} failed: {error!r}", action_time=0.0, **zero)
        finally:
            watcher.cancel()
            self.sequencer.finish(job)

    async def _watch_cancel(self, ctx, job: Job) -> None:
        await ctx.cancel_signal()
        await self.sequencer.cancel(job, "cancelled by the caller", by_caller=True)

    def _elapsed(self, started_ns: int) -> float:
        return max(0.0, (self.now() - started_ns) / 1e9)
