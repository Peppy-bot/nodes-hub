"""The SO-101's answer to workspace:v1, from the robot's design alone.

Reach. The arm, "arm", reaches a target when so101_description's
ApproachSolver finds joints that bring the grasp point within REACH_TOLERANCE
of it with the gripper's approach axis within GRASP_ANGLE_TOLERANCE of a grasp
direction. reach_of gives the reach of a target, reached by the arm or short
by its least distance. Its docstring gives the order in which the grasp
directions are tried.

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

from so101_description.kinematics import ApproachSolver
from so101_description.limbs import ARM_LIMB
from workspace_core_py import (
    GRASP_ANGLE_TOLERANCE,
    GRASP_DIRECTIONS,
    REACH_TOLERANCE,
    Positions,
    PositionsAnswer,
    Reach,
    ReachMemo,
    SurfaceAnswer,
    SurfaceHeight,
    SurfaceReach,
    check_without_perception_camera,
    describe_without_perception_camera,
)

# A point of the robot frame (m): (x, y, z).
Target = tuple[float, float, float]

# How many targets the worker measures at a time: the grid targets of one
# surface (663).
CHUNK_TARGETS = len(SurfaceHeight.from_wire(0.0).grid_targets())


class Refused(Exception):
    """A workspace request the robot refuses: the message is the reason the
    answer carries."""


def reach_of(solver: ApproachSolver, target: Target) -> Reach:
    """Whether the arm reaches `target`, a point in the robot frame (m).

    Every target is tried in the order of GRASP_DIRECTIONS, down then
    forward, and is reached by "arm" in the first direction for which the
    solver finds joints. Else it is short by its least distance, which is
    computed only for a target that is short. For a target that the solver's
    reach sphere rules out at REACH_TOLERANCE, ApproachSolver.approach
    answers None at once, with no descent (see ApproachSolver)."""
    if _approached_in_a_grasp_direction(solver, target):
        return Reach.reached_by(ARM_LIMB)
    return Reach.short(solver.least_distance(target))


def _approached_in_a_grasp_direction(solver: ApproachSolver, target: Target) -> bool:
    """Whether the solver finds joints for `target` in a grasp direction,
    tried in the order of GRASP_DIRECTIONS up to the first that it finds
    joints for."""
    for direction in GRASP_DIRECTIONS:
        joints = solver.approach(
            target,
            direction.approach,
            position_tolerance_m=REACH_TOLERANCE,
            angle_tolerance_rad=GRASP_ANGLE_TOLERANCE,
        )
        if joints is not None:
            return True
    return False


# The solver of the worker process. _start_worker sets it there; this process
# never sets it.
_worker_solver: ApproachSolver | None = None


def _start_worker(urdf_path: str) -> None:
    """The first call in the worker process: it ends the worker with this
    process and builds the worker's own solver."""
    global _worker_solver
    _end_with_parent()
    _worker_solver = ApproachSolver(urdf_path)


def _measure_reaches(targets: list[Target]) -> list[Reach]:
    """The reach of each target, in order, in the worker process."""
    assert _worker_solver is not None, "the worker's initializer builds the solver first"
    return [reach_of(_worker_solver, target) for target in targets]


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

    async def reaches(self, targets: Sequence[Target]) -> list[Reach]:
        """The reach of each target, in order. Raises Refused when the worker
        cannot measure them."""
        targets = list(targets)
        reaches: list[Reach] = []
        for start in range(0, len(targets), CHUNK_TARGETS):
            reaches += await self._measured_chunk(targets[start : start + CHUNK_TARGETS])
        return reaches

    def close(self) -> None:
        """Stop the worker: it ends when it has measured the chunk it holds.
        No chunk goes to it after this."""
        self._closed = True
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None

    async def _measured_chunk(self, chunk: list[Target]) -> list[Reach]:
        """The reach of each target of `chunk`, measured by the worker in the
        chunk's turn."""
        async with self._turn:
            await self._last_chunk_measured()
            in_worker = self._sent(chunk)
            self._sent_last = in_worker
            try:
                return await asyncio.wrap_future(in_worker)
            except BrokenProcessPool as e:
                raise Refused(f"the reach worker ended: {e}") from e
            except Exception as e:
                raise Refused(f"the reach could not be measured: {e!r}") from e

    async def _last_chunk_measured(self) -> None:
        """Wait until the worker is done with the chunk sent to it last. Only
        a chunk whose request was cancelled can still be in the worker here;
        its result is for nobody."""
        sent_last = self._sent_last
        if sent_last is None or sent_last.done():
            return
        with contextlib.suppress(Exception):
            await asyncio.wrap_future(sent_last)

    def _sent(self, chunk: list[Target]) -> Future:
        """`chunk`, sent to the worker. A pool that has seen its worker end
        is broken: a new pool, with a new worker, takes the chunk."""
        try:
            return self._running_pool().submit(_measure_reaches, chunk)
        except BrokenProcessPool:
            self._pool = None
            return self._running_pool().submit(_measure_reaches, chunk)

    def _running_pool(self) -> ProcessPoolExecutor:
        if self._closed:
            raise Refused("the node is shutting down")
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
    them, around `measure_reaches`, the reach of targets in order (a
    ReachWorker's in the node). It keeps the reach memo: a surface height
    asked again answers from the reach measured for it."""

    def __init__(
        self,
        measure_reaches: Callable[[Sequence[Target]], Awaitable[list[Reach]]],
    ):
        self._measure_reaches = measure_reaches
        self._memo = ReachMemo()

    async def describe(self, surface_height: float) -> SurfaceAnswer:
        """The describe_workspace answer of the surface at `surface_height`
        (m). Raises Refused for a height that SurfaceHeight.from_wire
        refuses, with its reason, and when the reach cannot be measured."""
        height = _parsed(SurfaceHeight.from_wire, surface_height)
        return describe_without_perception_camera(await self._surface_reach(height))

    async def check(self, positions: Sequence[float]) -> PositionsAnswer:
        """The check_positions answer of `positions`, 3 values (x, y, z, m)
        per point. Raises Refused for values that are not points of a
        request, and when the reach cannot be measured."""
        points = _parsed(Positions.from_wire, positions)
        reaches = await self._measure_reaches(points.points)
        return check_without_perception_camera(points, reaches)

    async def _surface_reach(self, height: SurfaceHeight) -> SurfaceReach:
        stored = self._memo.get(height)
        if stored is not None:
            return stored
        reaches = await self._measure_reaches(height.grid_targets())
        return self._memo.insert(SurfaceReach.from_grid_order(height, reaches))


def _parsed(parse, wire):
    """What `parse` makes of `wire`; Refused with its reason when it raises
    ValueError."""
    try:
        return parse(wire)
    except ValueError as e:
        raise Refused(str(e)) from None
