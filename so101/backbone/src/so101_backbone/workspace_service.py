"""The workspace contract's two services, describe_workspace and
check_positions, answered for the life of the node by Workspace from the
robot's design. They read nothing of the arm and move nothing, so they answer
from bringup, follower state or not. This module is the thin edge between the
generated handlers and workspace.py: check_positions places the points of a
request, in its frame (the robot frame for none) at its timestamp, in the
robot frame by the robot's tree first, with one lookup (placement.py), the
list and every point parsed before they are placed, and a request that the
parse, the placement or Workspace refuses gets an error reply with the kind
of the refusal and its reason. Every answer is in the robot frame, which it
names: describe_workspace's bounds hold at every time, so its answer holds no
timestamp, and each result of check_positions gives its point as placed, at
the time of the robot frame's place, or with no timestamp when that place
holds at every time."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from control_core_py import runtime
from control_core_py.geometry import GeometryError, Point, Stamped
from peppygen import NodeRunner
from peppygen.exposed_services.workspace import check_positions, describe_workspace
from workspace_core_py import NO_POINTS, PointAnswer, PositionsAnswer, SurfaceAnswer

from so101_backbone.failures import FailureKind, Refusal
from so101_backbone.model import ROBOT_FRAME, frame_of
from so101_backbone.placement import Placed, Placer
from so101_backbone.workspace import ReachWorker, Workspace

# The answers' perception_camera: the SO-101 has none. Its one camera, wrist,
# rides the arm.
_NO_PERCEPTION_CAMERA = ""


async def serve(node_runner: NodeRunner, worker: ReachWorker, placer: Placer) -> None:
    """Answer both services until shutdown, each on its own loop, with the
    reach `worker` measures and the points `placer` places. The worker stops
    with them."""
    workspace = Workspace(worker.reaches)

    async def on_describe(request) -> describe_workspace.Response:
        return await _answer(
            "describe_workspace",
            lambda: workspace.describe(request.data.surface_height),
            _described,
            describe_workspace.Failure,
        )

    async def on_check(request) -> check_positions.Response:
        async def check():
            placed = await _placed_points(placer, request.data)
            answer = await workspace.check([point.position for point in placed.value])
            return answer, placed

        return await _answer(
            "check_positions",
            check,
            lambda checked: _checked(*checked),
            check_positions.Failure,
        )

    try:
        await asyncio.gather(
            runtime.serve(node_runner, describe_workspace, on_describe, "describe_workspace"),
            runtime.serve(node_runner, check_positions, on_check, "check_positions"),
        )
    finally:
        worker.close()


async def _answer(
    service: str,
    ask: Callable[[], Awaitable],
    respond: Callable,
    failure: Callable[[FailureKind, str], Exception],
):
    """The response of one request: `respond` to its answer. Raises the
    service's `failure`, its error reply, with the kind and the reason of a
    refusal, and with `attempt_failed` for a failure nobody foresaw, so the
    caller gets an answer and the service keeps serving."""
    try:
        return respond(await ask())
    except Refusal as e:
        raise failure(e.kind, str(e)) from None
    except Exception as e:
        runtime.log(f"{service} failed unexpectedly: {e!r}")
        raise failure(FailureKind.ATTEMPT_FAILED, f"{service} failed: {e!r}") from None


async def _placed_points(placer: Placer, data) -> Placed[tuple[Point, ...]]:
    """The points of a request, placed together in the robot frame by
    `placer`, in the request's order; Refusal with `rejected` for an empty
    list and for a value no contract takes, a point named by its place in the
    request, from 1, before any lookup, and for points their placement
    carries past the finite numbers, and with the failure kind of the lookup
    when the tree cannot place them."""
    if not data.points:
        raise Refusal(FailureKind.REJECTED, NO_POINTS)
    points = []
    for number, point in enumerate(data.points, start=1):
        try:
            points.append(Point.parse(point.position))
        except GeometryError as e:
            raise Refusal(FailureKind.REJECTED, f"point {number}: {e}") from None
    try:
        stamped = Stamped.parse(frame_of(data.frame), data.timestamp, tuple(points))
    except GeometryError as e:
        raise Refusal(FailureKind.REJECTED, str(e)) from None
    return await placer.place(stamped, "the points")


def _described(answer: SurfaceAnswer) -> describe_workspace.Response:
    return describe_workspace.Response(
        frame=ROBOT_FRAME,
        perception_camera=_NO_PERCEPTION_CAMERA,
        workable=answer.workable,
        area=answer.area,
        rectangle=_listed(answer.rectangle),
        reach=_listed(answer.reach),
        view=_listed(answer.view),
    )


def _checked(
    answer: PositionsAnswer, placed: Placed[tuple[Point, ...]]
) -> check_positions.Response:
    return check_positions.Response(
        perception_camera=_NO_PERCEPTION_CAMERA,
        all_workable=answer.all_workable,
        results=[_result(point, placed.timestamp) for point in answer.points],
    )


def _result(point: PointAnswer, timestamp: float | None) -> check_positions.ResponseResultsItem:
    return check_positions.ResponseResultsItem(
        frame=ROBOT_FRAME,
        timestamp=timestamp,
        position=list(point.position),
        workable=point.workable,
        reachable=point.reach.reached,
        arm=point.reach.arm,
        short_by=point.reach.short_by,
        in_view=point.in_view,
        view=point.view,
        message=point.message,
    )


def _listed(bounds: tuple[float, float, float, float] | None) -> list[float] | None:
    """Bounds as the wire carries them: a list, or absent."""
    return None if bounds is None else list(bounds)
