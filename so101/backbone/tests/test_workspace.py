"""The workspace answer: the reach of a target from the approach solver, the
answers composed from requests as the wire gives them, the reach memo kept in
this process, and the worker process that measures reach in chunks."""

import asyncio
import ctypes
import math
import multiprocessing
import os
import signal
import subprocess
import sys
from concurrent.futures import Future, ProcessPoolExecutor
from multiprocessing.connection import wait

import pytest
from conftest import (
    CLOSE,
    FAR,
    FAR_SHORT_BY_MORE_THAN,
    NEAR,
    NOT_CHECKED,
    assert_rectangle_inside_reach,
)
from so101_description.kinematics import ApproachSolver, ReachSphere
from so101_description.model import KINEMATICS_URDF_PATH
from workspace_core_py import (
    GRASP_ANGLE_TOLERANCE,
    GRASP_DIRECTIONS,
    REACH_TOLERANCE,
    Positions,
    Reach,
    SurfaceHeight,
    SurfaceReach,
    check_without_perception_camera,
    describe_without_perception_camera,
)

from so101_backbone import workspace as workspace_module
from so101_backbone.workspace import (
    CHUNK_TARGETS,
    ReachWorker,
    Refused,
    Target,
    Workspace,
    reach_of,
)

DOWN, FORWARD = GRASP_DIRECTIONS
ORIGIN = (0.0, 0.0, 0.0)
# Reach spheres about the origin of the robot frame: one that holds NEAR,
# about 0.2 m from that origin, and one that rules NEAR out at
# REACH_TOLERANCE.
HOLDS_NEAR = ReachSphere(centre=ORIGIN, radius_m=0.5)
RULES_OUT_NEAR = ReachSphere(centre=ORIGIN, radius_m=0.1)
# A least distance far past the reach tolerance.
FAR_PAST_THE_TOLERANCE = 0.25
# A target that the arm reaches straight down.
REACHED_STRAIGHT_DOWN = (-0.06132, 0.2295, -0.13798)
# A hang guard, not a measure of speed: a process that must end, a chunk that
# must be sent and a request that must end each do so long before it. A
# broken rule thus fails its test with a timeout and does not hang it.
HANG_BOUND_S = 60.0


class FakeApproachSolver:
    """Has `reach_sphere`, gives `least_distance` for every position, and
    records each call of least_distance and approach. Its approach keeps the
    rule of ApproachSolver.approach: None for a position that the reach
    sphere rules out at the position tolerance. It approaches every other
    position in the directions named `fitting`."""

    def __init__(
        self, reach_sphere: ReachSphere, least_distance: float, fitting: tuple[str, ...] = ()
    ):
        self.reach_sphere = reach_sphere
        self._least_distance = least_distance
        self._fitting = [d.approach for d in GRASP_DIRECTIONS if d.name in fitting]
        self.calls: list[tuple] = []

    def least_distance(self, position):
        self.calls.append(("least_distance", position))
        return self._least_distance

    def approach(self, position, direction, *, position_tolerance_m, angle_tolerance_rad):
        self.calls.append(
            ("approach", position, direction, position_tolerance_m, angle_tolerance_rad)
        )
        if self.reach_sphere.rules_out(position, position_tolerance_m):
            return None
        return (0.0,) * 5 if direction in self._fitting else None


def approached(direction, target: Target = NEAR) -> tuple:
    """The call of an approach of `target` in `direction`, at the contract's
    bars."""
    return ("approach", target, direction.approach, REACH_TOLERANCE, GRASP_ANGLE_TOLERANCE)


def test_a_target_the_reach_sphere_rules_out_is_short_by_its_least_distance():
    solver = FakeApproachSolver(
        RULES_OUT_NEAR, least_distance=FAR_PAST_THE_TOLERANCE, fitting=("down", "forward")
    )
    assert reach_of(solver, NEAR) == Reach.short(FAR_PAST_THE_TOLERANCE)
    # Both directions fit inside the sphere. reach_of tries both, and the
    # solver approaches NEAR in neither: the sphere rules it out at the reach
    # tolerance.
    assert solver.calls == [approached(DOWN), approached(FORWARD), ("least_distance", NEAR)]


def test_a_target_is_reached_by_the_arm_in_the_first_direction_that_fits():
    solver = FakeApproachSolver(HOLDS_NEAR, least_distance=0.0, fitting=("down", "forward"))
    assert reach_of(solver, NEAR) == Reach.reached_by("arm")
    # The least distance of a reached target is not computed.
    assert solver.calls == [approached(DOWN)]


def test_forward_is_tried_when_down_does_not_fit():
    solver = FakeApproachSolver(HOLDS_NEAR, least_distance=0.004, fitting=("forward",))
    assert reach_of(solver, NEAR) == Reach.reached_by("arm")
    assert solver.calls == [approached(DOWN), approached(FORWARD)]


def test_a_target_in_the_reach_sphere_is_tried_whatever_its_least_distance():
    # The solver's least distance can be longer than the true one: a least
    # distance far past the reach tolerance does not keep the arm from a
    # direction that fits.
    solver = FakeApproachSolver(
        HOLDS_NEAR, least_distance=FAR_PAST_THE_TOLERANCE, fitting=("down",)
    )
    assert reach_of(solver, NEAR) == Reach.reached_by("arm")
    assert solver.calls == [approached(DOWN)]


def test_a_target_the_grasp_point_comes_close_to_in_no_direction_is_short_by_that_distance():
    solver = FakeApproachSolver(HOLDS_NEAR, least_distance=0.004)
    assert reach_of(solver, NEAR) == Reach.short(0.004)
    assert solver.calls == [approached(DOWN), approached(FORWARD), ("least_distance", NEAR)]


def test_a_target_the_arm_reaches_straight_down_is_reached():
    solver = ApproachSolver(KINEMATICS_URDF_PATH)
    down = solver.approach(
        REACHED_STRAIGHT_DOWN,
        DOWN.approach,
        position_tolerance_m=REACH_TOLERANCE,
        angle_tolerance_rad=GRASP_ANGLE_TOLERANCE,
    )
    assert down is not None
    assert reach_of(solver, REACHED_STRAIGHT_DOWN) == Reach.reached_by("arm")


class FakeReaches:
    """Measures the reach of targets by `rule`, or raises what `failures`
    holds first, and records the targets of each call."""

    def __init__(self, rule, failures: list[Exception] | None = None):
        self._rule = rule
        self._failures = failures or []
        self.calls: list[list] = []

    async def __call__(self, targets):
        self.calls.append(list(targets))
        if self._failures:
            raise self._failures.pop(0)
        return [self._rule(target) for target in targets]


def reached_near_the_middle(target) -> Reach:
    """A reach that changes across the grid: reached up to 0.3 m ahead and
    0.1 m to either side, short beyond."""
    x, y, _ = target
    if x <= 0.3 and abs(y) <= 0.1 + 1e-9:
        return Reach.reached_by("arm")
    return Reach.short(round(x + abs(y), 3))


def surface_fields(answer) -> tuple:
    return (
        answer.workable,
        answer.area,
        answer.rectangle,
        answer.reach,
        answer.view,
        answer.message,
    )


def positions_fields(answer) -> tuple:
    return (
        [(p.position, p.reach, p.view, p.in_view, p.workable, p.message) for p in answer.points],
        answer.all_workable,
        answer.message,
    )


async def test_describe_measures_the_grid_targets_of_the_height_and_answers_on_reach_alone():
    reaches = FakeReaches(reached_near_the_middle)
    answer = await Workspace(reaches).describe(0.1)

    height = SurfaceHeight.from_wire(0.1)
    assert reaches.calls == [height.grid_targets()]
    expected = describe_without_perception_camera(
        SurfaceReach.from_grid_order(
            height, [reached_near_the_middle(t) for t in height.grid_targets()]
        )
    )
    assert surface_fields(answer) == surface_fields(expected)
    assert answer.workable
    assert answer.view is None
    assert answer.message.endswith(NOT_CHECKED)


async def test_a_surface_height_asked_again_answers_from_the_reach_measured_for_it():
    reaches = FakeReaches(reached_near_the_middle)
    workspace = Workspace(reaches)
    first = await workspace.describe(0.1)
    # The same height to the millimetre.
    again = await workspace.describe(0.1004)
    assert len(reaches.calls) == 1
    assert surface_fields(again) == surface_fields(first)

    await workspace.describe(0.2)
    assert reaches.calls[1] == SurfaceHeight.from_wire(0.2).grid_targets()


async def test_a_reach_not_measured_refuses_the_request_and_stores_nothing():
    reaches = FakeReaches(reached_near_the_middle, failures=[Refused("the worker ended")])
    workspace = Workspace(reaches)
    with pytest.raises(Refused, match="^the worker ended$"):
        await workspace.describe(0.1)
    answer = await workspace.describe(0.1)
    assert len(reaches.calls) == 2
    assert answer.workable


async def test_check_measures_the_points_in_the_requests_order():
    reaches = FakeReaches(reached_near_the_middle)
    wire = [0.2, 0.0, 0.04, 0.5, 0.0, 0.04, 0.1, -0.05, 0.3]
    answer = await Workspace(reaches).check(wire)

    positions = Positions.from_wire(wire)
    assert reaches.calls == [positions.points]
    expected = check_without_perception_camera(
        positions, [reached_near_the_middle(p) for p in positions.points]
    )
    assert positions_fields(answer) == positions_fields(expected)
    assert [p.workable for p in answer.points] == [True, False, True]
    assert {p.view for p in answer.points} == {"no_camera"}
    assert answer.message.endswith(NOT_CHECKED)


NOT_FINITE = "surface_height must be a finite number"
TOO_FAR = "surface_height must be within 1000 m of the robot's base point"


@pytest.mark.parametrize(
    ("surface_height", "reason"),
    [
        (math.nan, NOT_FINITE),
        (math.inf, NOT_FINITE),
        (-math.inf, NOT_FINITE),
        (2e6, TOO_FAR),
        (-2e6, TOO_FAR),
    ],
)
async def test_a_height_that_is_not_a_surface_height_of_a_request_is_refused_unmeasured(
    surface_height, reason
):
    reaches = FakeReaches(reached_near_the_middle)
    with pytest.raises(Refused, match=f"^{reason}$"):
        await Workspace(reaches).describe(surface_height)
    assert reaches.calls == []


@pytest.mark.parametrize(
    ("wire", "reason"),
    [
        ([], "positions must hold at least one point"),
        ([0.2, 0.0], r"positions must hold 3 values \(x, y, z\) per point"),
        ([0.2, 0.0, math.nan], "positions must hold finite numbers only"),
        (
            [1000.5, 0.0, 0.0],
            "positions must hold coordinates within 1000 m of the robot's base point",
        ),
    ],
)
async def test_values_that_are_not_points_of_a_request_are_refused_unmeasured(wire, reason):
    reaches = FakeReaches(reached_near_the_middle)
    with pytest.raises(Refused, match=f"^{reason}$"):
        await Workspace(reaches).check(wire)
    assert reaches.calls == []


# ------------------------------------------------------------- the worker


@pytest.fixture
async def worker():
    worker = ReachWorker(KINEMATICS_URDF_PATH)
    yield worker
    worker.close()


def children_pids() -> set[int]:
    return {child.pid for child in multiprocessing.active_children()}


def spawned_since(before: set[int]) -> list[multiprocessing.Process]:
    """The child processes started since `before` was taken."""
    return [c for c in multiprocessing.active_children() if c.pid not in before]


def assert_ends(ended: int) -> None:
    """Wait for `ended`, a file descriptor that reads ready once a process
    has ended, and fail when it is not ready within the bound."""
    assert wait([ended], HANG_BOUND_S), f"the process still runs after {HANG_BOUND_S} s"


# pidfd_open(2). The Python builds uv installs have no os.pidfd_open, so the
# test calls it by its number, the same on every Linux architecture.
SYS_PIDFD_OPEN = 434


def open_pidfd(pid: int) -> int:
    """A pidfd of the running process `pid`, a process this one did not
    start: it reads ready once that process has ended."""
    libc = ctypes.CDLL(None, use_errno=True)
    pidfd = libc.syscall(SYS_PIDFD_OPEN, pid, 0)
    if pidfd < 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))
    return pidfd


async def test_the_worker_measures_what_the_solver_answers_in_this_process(worker):
    near, close, far = await worker.reaches([NEAR, CLOSE, FAR])

    solver = ApproachSolver(KINEMATICS_URDF_PATH)
    assert [near, close, far] == [reach_of(solver, target) for target in (NEAR, CLOSE, FAR)]
    assert near == Reach.reached_by("arm")
    assert not close.reached
    assert 0.0 <= close.short_by <= REACH_TOLERANCE
    assert far.short_by > FAR_SHORT_BY_MORE_THAN
    # The worker's solver was built in the worker: this process never built
    # one, and _measure_reaches refuses to run without it.
    assert workspace_module._worker_solver is None


async def test_one_spawned_worker_serves_every_request_and_ends_on_close(worker):
    before = children_pids()
    # Two requests at once: a second worker would take the second.
    await asyncio.gather(worker.reaches([NEAR]), worker.reaches([FAR]))
    await worker.reaches([CLOSE])
    (process,) = spawned_since(before)
    assert isinstance(process, multiprocessing.context.SpawnProcess)
    assert process.pid != os.getpid()

    worker.close()
    assert_ends(process.sentinel)
    with pytest.raises(Refused, match="^the node is shutting down$"):
        await worker.reaches([NEAR])
    assert spawned_since(before | {process.pid}) == []


async def test_a_target_the_solver_refuses_refuses_the_request_and_the_worker_serves_on(worker):
    before = children_pids()
    with pytest.raises(Refused, match=r"^the reach could not be measured: ValueError\("):
        await worker.reaches([NEAR, (math.nan, 0.0, 0.0)])
    (process,) = spawned_since(before)

    assert await worker.reaches([NEAR]) == [Reach.reached_by("arm")]
    assert spawned_since(before) == [process]


def wait_until_the_pool_is_broken(worker: ReachWorker) -> None:
    """Wait until the process pool of `worker` has marked itself broken. Its
    manager thread does so when it sees that the worker ended, and then
    ends."""
    manager = worker._pool._executor_manager_thread
    manager.join(HANG_BOUND_S)
    assert not manager.is_alive(), f"the pool still runs after {HANG_BOUND_S} s"


async def test_a_worker_that_ended_while_it_held_no_chunk_is_replaced_once_the_pool_has_seen_it(
    worker,
):
    before = children_pids()
    await worker.reaches([NEAR])
    (ended,) = spawned_since(before)
    os.kill(ended.pid, signal.SIGKILL)
    assert_ends(ended.sentinel)
    # The test waits until the pool has seen the end: a request that sends
    # its chunk before that is refused.
    wait_until_the_pool_is_broken(worker)

    # Another worker measures the next request.
    assert await worker.reaches([NEAR]) == [Reach.reached_by("arm")]
    (another,) = spawned_since(before)
    assert another.pid != ended.pid


async def next_sent(sent: asyncio.Queue):
    """The next item of `sent`, a queue of what was sent to the worker; fail
    when none comes within the bound."""
    return await asyncio.wait_for(sent.get(), HANG_BOUND_S)


async def outcome_of(request: asyncio.Task):
    """What `request`, a task of the test, returns, or raises when it raises;
    fail with TimeoutError when it does not end within the bound."""
    return await asyncio.wait_for(request, HANG_BOUND_S)


@pytest.fixture
def sent_chunks(monkeypatch) -> asyncio.Queue:
    """The future of each chunk the worker's process pools are sent, in the
    order they are sent."""
    sent: asyncio.Queue[Future] = asyncio.Queue()

    class RecordedPool(ProcessPoolExecutor):
        def submit(self, fn, /, *args, **kwargs):
            future = super().submit(fn, *args, **kwargs)
            sent.put_nowait(future)
            return future

    monkeypatch.setattr(workspace_module, "ProcessPoolExecutor", RecordedPool)
    return sent


async def test_a_worker_that_ends_while_it_holds_a_chunk_refuses_its_request(worker, sent_chunks):
    before = children_pids()
    await worker.reaches([NEAR])
    await next_sent(sent_chunks)
    (ended,) = spawned_since(before)
    # A stopped worker holds the chunk it is sent and cannot measure it.
    os.kill(ended.pid, signal.SIGSTOP)
    os.waitid(os.P_PID, ended.pid, os.WSTOPPED | os.WNOWAIT)
    request = asyncio.create_task(worker.reaches([NEAR]))
    await next_sent(sent_chunks)
    os.kill(ended.pid, signal.SIGKILL)

    with pytest.raises(Refused, match="^the reach worker ended: "):
        await outcome_of(request)
    # The next request starts another worker.
    assert await worker.reaches([NEAR]) == [Reach.reached_by("arm")]
    (another,) = spawned_since(before)
    assert another.pid != ended.pid


# The node, cut down to its worker: it measures once, names its worker's
# process, and waits for the test to kill it.
NODE_SCRIPT = """
import asyncio, multiprocessing, sys
from so101_description.model import KINEMATICS_URDF_PATH
from so101_backbone.workspace import ReachWorker

async def main():
    worker = ReachWorker(KINEMATICS_URDF_PATH)
    await worker.reaches([(0.2, 0.0, 0.04)])
    (process,) = multiprocessing.active_children()
    print(process.pid, flush=True)
    sys.stdin.read()

asyncio.run(main())
"""


def test_the_worker_ends_when_the_node_process_is_killed():
    node = subprocess.Popen(
        [sys.executable, "-c", NODE_SCRIPT],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )
    try:
        line = node.stdout.readline()
        assert line, "the node ended before it named its worker"
        pidfd = open_pidfd(int(line))
    finally:
        node.kill()
        node.wait()
        node.stdin.close()
        node.stdout.close()
    try:
        assert_ends(pidfd)
    finally:
        os.close(pidfd)


async def test_a_stored_height_answers_without_the_worker(worker):
    workspace = Workspace(worker.reaches)
    measured = await workspace.describe(0.0)
    assert measured.workable
    assert measured.view is None
    assert measured.message.endswith(NOT_CHECKED)
    assert_rectangle_inside_reach(measured)
    # Inside what the arm can reach: about 0.5 m from the pan axis.
    _, x_max, _, _ = measured.reach
    assert x_max < 0.5

    # A closed worker measures nothing: the stored height still answers, from
    # this process.
    worker.close()
    stored = await workspace.describe(0.0)
    assert surface_fields(stored) == surface_fields(measured)
    with pytest.raises(Refused, match="^the node is shutting down$"):
        await workspace.describe(0.1)


# ------------------------------------------------------------- the chunks


class HeldPool:
    """Stands in for the worker's process pool. It holds each chunk it is
    sent, as the worker holds the chunk it measures, until the test answers
    it, and it lists the chunks in the order they come."""

    def __init__(self):
        self.sent: asyncio.Queue[tuple[list, Future]] = asyncio.Queue()
        self.shutdowns: list[bool] = []

    def submit(self, fn, chunk):
        held = Future()
        # In the worker, a chunk can no longer be cancelled.
        held.set_running_or_notify_cancel()
        self.sent.put_nowait((chunk, held))
        return held

    def shutdown(self, wait=True):
        self.shutdowns.append(wait)


@pytest.fixture
def held_pool(monkeypatch) -> HeldPool:
    pool = HeldPool()
    monkeypatch.setattr(workspace_module, "ProcessPoolExecutor", lambda **_: pool)
    return pool


def targets(count: int) -> list[Target]:
    """`count` targets, each of its own."""
    return [(0.001 * i, 0.0, 0.0) for i in range(count)]


def answered(chunk) -> list[Reach]:
    """What the test answers for `chunk` in the held pool's worker: each
    target is short by its x."""
    return [Reach.short(x) for x, _, _ in chunk]


async def test_a_long_request_goes_to_the_worker_in_chunks_of_one_surface_one_after_another(
    held_pool,
):
    assert CHUNK_TARGETS == len(SurfaceHeight.from_wire(0.0).grid_targets()) == 663
    worker = ReachWorker(KINEMATICS_URDF_PATH)
    asked = targets(2 * CHUNK_TARGETS + 5)
    request = asyncio.create_task(worker.reaches(asked))

    chunks = []
    for _ in range(3):
        chunk, held = await next_sent(held_pool.sent)
        # The next chunk goes only when the worker has measured this one.
        assert held_pool.sent.empty()
        chunks.append(chunk)
        held.set_result(answered(chunk))
    assert chunks == [asked[:663], asked[663:1326], asked[1326:]]
    assert await outcome_of(request) == answered(asked)


async def test_a_request_waits_for_one_chunk_of_a_long_request_ahead_of_it(held_pool):
    worker = ReachWorker(KINEMATICS_URDF_PATH)
    long = targets(2 * CHUNK_TARGETS)
    long_request = asyncio.create_task(worker.reaches(long))
    short_request = asyncio.create_task(worker.reaches([NEAR]))

    # Both requests start before the worker has measured a chunk.
    order = []
    for _ in range(3):
        chunk, held = await next_sent(held_pool.sent)
        # The worker holds one chunk at a time.
        assert held_pool.sent.empty()
        order.append(chunk)
        held.set_result(answered(chunk))
    assert order == [long[:CHUNK_TARGETS], [NEAR], long[CHUNK_TARGETS:]]
    assert await outcome_of(short_request) == answered([NEAR])
    assert await outcome_of(long_request) == answered(long)


async def test_a_cancelled_request_sends_no_more_chunks_and_the_worker_ends_its_chunk_first(
    held_pool,
):
    worker = ReachWorker(KINEMATICS_URDF_PATH)
    cancelled = asyncio.create_task(worker.reaches(targets(2 * CHUNK_TARGETS)))
    next_request = asyncio.create_task(worker.reaches([NEAR]))
    first, held_first = await next_sent(held_pool.sent)

    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await outcome_of(cancelled)
    assert cancelled.cancelled()
    # The worker still measures the chunk of the cancelled request: no other
    # chunk goes to it before that one is measured.
    assert held_pool.sent.empty()
    held_first.set_result(answered(first))

    chunk, held = await next_sent(held_pool.sent)
    assert chunk == [NEAR]
    held.set_result(answered(chunk))
    assert await outcome_of(next_request) == answered([NEAR])
    assert held_pool.sent.empty()


async def test_close_lets_the_worker_end_after_the_chunk_it_holds_and_refuses_the_rest(
    held_pool,
):
    worker = ReachWorker(KINEMATICS_URDF_PATH)
    request = asyncio.create_task(worker.reaches(targets(2 * CHUNK_TARGETS)))
    first, held = await next_sent(held_pool.sent)
    assert held_pool.sent.empty()

    worker.close()
    # close() does not wait for the worker: the event loop goes on.
    assert held_pool.shutdowns == [False]
    held.set_result(answered(first))
    with pytest.raises(Refused, match="^the node is shutting down$"):
        await outcome_of(request)
    assert held_pool.sent.empty()
