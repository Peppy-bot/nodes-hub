"""The SO-101's answer to workspace:v1, from the robot's design alone.

Reach. The arm, "arm", reaches a target by workspace_core's rule
(workspace_core_py.reach) over the kinematics module's ApproachSolver, which
meets the approach direction alone (ApproachArm): joints that bring the grasp
point within REACH_TOLERANCE of it with the gripper's approach axis within
GRASP_ANGLE_TOLERANCE of a grasp direction. On a surface, they reach it only in
a pose clear of the block under the surface (workspace_core_py.ArmBodies).
reach_of gives the reach of a target, reached by the arm or short by its least
distance.

The least distance of a target is the distance from it to the nearest grasp
point that the solver's search finds, in any orientation. The search is
local, so the least distance can be longer than the true one, and thus it does
not decide whether a target is reached. ApproachSolver and
ApproachSolver.least_distance give how far from the true least distance the
answer can be, and the measured rates at which the solver reaches targets.

The grasp point is the origin of gripper_frame_link, the fixed frame that
limb_state reports. On this gripper the midpoint between the pads moves with
the opening, so the frame is an approximation of the contract's grasp point.
The robot frame is the frame of the model's base_link.

The parsing of a request and its refusals, the grid of a surface, the reach
memo and the composition of each answer with its messages are
workspace_core_py's. The SO-101 has no perception camera, so every answer is
on reach alone and says that the view is not checked.

Where the work runs. The solver work for the reach of a surface lasts many
ticks of the control loop (ApproachSolver gives its measured time). This
process runs the control loop, and only one of its threads runs Python at
a time, so the solver runs in one worker process of its own (ReachWorker),
never in this process. The requests send their targets to the worker in
chunks of one surface (CHUNK_TARGETS), and the worker holds one chunk at a
time. Thus a request waits for at most one chunk of each request ahead of it,
and when the node stops or a request is cancelled, the worker measures only
the chunk that it holds. The reach memo, one surface reach per height, stays
in this process.
"""

from __future__ import annotations

import asyncio
import contextlib
import multiprocessing
import os
import threading
from collections.abc import Awaitable, Callable, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from multiprocessing.connection import wait
from pathlib import Path

from workspace_core_py import (
    GRASP_ANGLE_TOLERANCE,
    REACH_TOLERANCE,
    ArmBodies,
    Positions,
    PositionsAnswer,
    Reach,
    ReachMemo,
    SurfaceAnswer,
    SurfaceHeight,
    SurfaceReach,
    Solves,
    check_without_perception_camera,
    describe_without_perception_camera,
    reach,
)

from so101_backbone.failures import FailureKind, Refusal
from so101_backbone.kinematics import ApproachSolver
from so101_backbone.model import ARM

# A point of the robot frame (m): (x, y, z).
Target = tuple[float, float, float]

# How many targets the worker measures at a time: the grid targets of one
# surface (663).
CHUNK_TARGETS = len(SurfaceHeight.from_wire(0.0).grid_targets())


class ApproachArm:
    """The arm of an ApproachSolver as workspace_core's reach rule asks it: one
    arm, "arm", at the base frame's origin, whose solver meets the approach
    direction alone. A target the reach sphere rules out is ruled out by its
    distance outside the sphere; a target no pose reaches falls short by the
    solver's least distance."""

    def __init__(self, solver: ApproachSolver):
        self._solver = solver

    def solves(self) -> Solves:
        return Solves.APPROACHES

    def arm_names(self) -> list[str]:
        return [ARM.name]

    def arm_base(self, arm: int) -> Target:
        return (0.0, 0.0, 0.0)

    def reach_bound(self, arm: int, target: Target) -> float:
        return self._solver.reach_sphere.short_of(target)

    def position_shortfall(self, arm: int, target: Target) -> float:
        return self._solver.least_distance(target)

    def reach_pose(self, arm: int, target: Target, rotation: tuple[float, ...]):
        approach = (rotation[2], rotation[5], rotation[8])
        joints = self._solver.approach(
            target,
            approach,
            position_tolerance_m=REACH_TOLERANCE,
            angle_tolerance_rad=GRASP_ANGLE_TOLERANCE,
        )
        if joints is None:
            return None
        return (joints, *self._solver.grasp_frame(joints))


def reach_of(solver: ApproachSolver, target: Target) -> Reach:
    """Whether the arm reaches `target`, a point in the robot frame (m), with
    nothing in the way: reached by "arm" in the first grasp direction, down
    then forward, for which the solver finds joints, else short by the
    solver's least distance."""
    return reach(ApproachArm(solver), target)


# The solver and the arm bodies of the worker process. _start_worker sets
# them there; this process never sets them.
_worker_solver: ApproachSolver | None = None
_worker_bodies: ArmBodies | None = None


def _start_worker(urdf_path: str) -> None:
    """The first call in the worker process: it ends the worker with this
    process and builds the worker's own solver and arm bodies, from the
    model's files beside `urdf_path`."""
    global _worker_solver, _worker_bodies
    _end_with_parent()
    _worker_solver = ApproachSolver(urdf_path)
    urdf = Path(urdf_path)
    _worker_bodies = ArmBodies(
        urdf.read_text(), urdf.with_suffix(".srdf").read_text(), str(urdf.parent / "meshes")
    )


def _measure_reaches(surface_m: float | None, targets: list[Target]) -> list[Reach]:
    """The reach of each target, in order, in the worker process: on the
    surface `surface_m` metres high in a pose clear of it, or, with none, with
    nothing in the way."""
    assert _worker_solver is not None and _worker_bodies is not None, (
        "the worker's initializer builds the solver and the arm bodies first"
    )
    if surface_m is None:
        return [reach_of(_worker_solver, target) for target in targets]
    arm = ApproachArm(_worker_solver)
    surface = SurfaceHeight.from_wire(surface_m)
    return [_worker_bodies.reach_over(arm, surface, target) for target in targets]


def _end_with_parent() -> None:
    """End this worker process when its parent process ends, however it ends.
    A parent that is killed cannot stop its worker, and the worker then waits
    for its next request with no end. Thus a thread waits on the sentinel of
    the parent, which becomes ready only when the parent has ended."""
    parent = multiprocessing.parent_process()
    assert parent is not None, "only a worker process has a parent process"
    threading.Thread(
        target=_exit_when_ready, args=(parent.sentinel,), name="end-with-parent", daemon=True
    ).start()


def _exit_when_ready(sentinel: int) -> None:
    wait([sentinel])
    os._exit(0)


class ReachWorker:
    """The one process that measures reach: started with the spawn method at
    the first request, with its own solver. A request does not run in this
    process, so the control loop keeps its rate while a surface is measured.

    Each request sends its targets in chunks of CHUNK_TARGETS, the next chunk
    when the worker has measured the one before. The chunks take turns, in the
    order the requests send them, and a chunk goes to the worker only when the
    worker is done with the chunk before it, also when the request of that
    chunk was cancelled. Thus the worker holds one chunk at a time, and a
    request that is cancelled sends no more chunks.

    A worker that ends while it holds a chunk (killed, crashed) refuses the
    request of that chunk. A worker that ends while it holds no chunk refuses
    the next request only when that request sends its chunk before the
    process pool has seen the end. Else the next chunk starts another worker,
    and so does the next chunk after each of these refusals. After close(),
    every request is refused."""

    def __init__(self, urdf_path: str):
        self._urdf_path = urdf_path
        self._pool: ProcessPoolExecutor | None = None
        self._closed = False
        self._turn = asyncio.Lock()
        # The chunk sent to the worker last. It stays in the worker until it
        # is measured, also when its request was cancelled.
        self._sent_last: Future | None = None

    async def reaches(
        self, targets: Sequence[Target], surface: SurfaceHeight | None
    ) -> list[Reach]:
        """The reach of each target, in order, on `surface` when one is given.
        Raises Refusal when the worker cannot measure them."""
        targets = list(targets)
        surface_m = None if surface is None else surface.metres
        reaches: list[Reach] = []
        for start in range(0, len(targets), CHUNK_TARGETS):
            chunk = targets[start : start + CHUNK_TARGETS]
            reaches += await self._measured_chunk(surface_m, chunk)
        return reaches

    def close(self) -> None:
        """Stop the worker: it ends when it has measured the chunk it holds.
        No chunk goes to it after this."""
        self._closed = True
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None

    async def _measured_chunk(self, surface_m: float | None, chunk: list[Target]) -> list[Reach]:
        """The reach of each target of `chunk`, on the surface `surface_m`
        metres high when one is given, measured by the worker in the chunk's
        turn."""
        async with self._turn:
            await self._last_chunk_measured()
            in_worker = self._sent(surface_m, chunk)
            self._sent_last = in_worker
            try:
                return await asyncio.wrap_future(in_worker)
            except BrokenProcessPool as e:
                raise Refusal(FailureKind.ATTEMPT_FAILED, f"the reach worker ended: {e}") from e
            except Exception as e:
                raise Refusal(
                    FailureKind.ATTEMPT_FAILED, f"the reach could not be measured: {e!r}"
                ) from e

    async def _last_chunk_measured(self) -> None:
        """Wait until the worker is done with the chunk sent to it last. Only
        a chunk whose request was cancelled can still be in the worker here;
        its result is for nobody."""
        sent_last = self._sent_last
        if sent_last is None or sent_last.done():
            return
        with contextlib.suppress(Exception):
            await asyncio.wrap_future(sent_last)

    def _sent(self, surface_m: float | None, chunk: list[Target]) -> Future:
        """`chunk`, sent to the worker. A pool that has seen its worker end
        is broken: a new pool, with a new worker, takes the chunk."""
        try:
            return self._running_pool().submit(_measure_reaches, surface_m, chunk)
        except BrokenProcessPool:
            self._pool = None
            return self._running_pool().submit(_measure_reaches, surface_m, chunk)

    def _running_pool(self) -> ProcessPoolExecutor:
        if self._closed:
            raise Refusal(FailureKind.RESOURCE_UNAVAILABLE, "the node is shutting down")
        if self._pool is None:
            self._pool = ProcessPoolExecutor(
                max_workers=1,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_start_worker,
                initargs=(self._urdf_path,),
            )
        return self._pool


class Workspace:
    """The workspace answers of the robot, from requests as the wire gives
    them, around `measure_reaches`, the reach of targets in order, on a
    surface or with nothing in the way (a ReachWorker's in the node). It
    keeps the reach memo: a surface height asked again answers from the
    reach measured for it."""

    def __init__(
        self,
        measure_reaches: Callable[
            [Sequence[Target], SurfaceHeight | None], Awaitable[list[Reach]]
        ],
    ):
        self._measure_reaches = measure_reaches
        self._memo = ReachMemo()

    async def describe(self, surface_height: float) -> SurfaceAnswer:
        """The describe_workspace answer of the surface at `surface_height`
        (m). Raises Refusal for a height that SurfaceHeight.from_wire
        refuses, with its reason, and when the reach cannot be measured."""
        height = _parsed(SurfaceHeight.from_wire, surface_height)
        return describe_without_perception_camera(await self._surface_reach(height))

    async def check(self, positions: Sequence[float]) -> PositionsAnswer:
        """The check_positions answer of `positions`, 3 values (x, y, z, m)
        per point. Raises Refusal for values that are not points of a
        request, and when the reach cannot be measured."""
        points = _parsed(Positions.from_wire, positions)
        reaches = await self._measure_reaches(points.points, None)
        return check_without_perception_camera(points, reaches)

    async def _surface_reach(self, height: SurfaceHeight) -> SurfaceReach:
        stored = self._memo.get(height)
        if stored is not None:
            return stored
        reaches = await self._measure_reaches(height.grid_targets(), height)
        return self._memo.insert(SurfaceReach.from_grid_order(height, reaches))


def _parsed(parse, wire):
    """What `parse` makes of `wire`; Refusal, `rejected`, with its reason
    when it raises ValueError."""
    try:
        return parse(wire)
    except ValueError as e:
        raise Refusal(FailureKind.REJECTED, str(e)) from None
