"""The workspace contract's two services, describe_workspace and
check_positions, answered for the life of the node by Workspace from the
robot's design. They read nothing of the arm and move nothing, so they answer
from bringup, follower state or not. This module is the thin edge between the
generated handlers and workspace.py: a refusal answers success false with its
reason as the message, and every other field empty or zero."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from control_core_py import runtime
from peppygen import NodeRunner
from peppygen.exposed_services.workspace import check_positions, describe_workspace
from workspace_core_py import PointAnswer, PositionsAnswer, SurfaceAnswer

from so101_backbone.workspace import ReachWorker, Refused, Workspace

# The answers' perception_camera: the SO-101 has none. Its one camera, wrist,
# rides the arm.
NO_PERCEPTION_CAMERA = ""


async def serve(node_runner: NodeRunner, worker: ReachWorker) -> None:
    """Answer both services until shutdown, each on its own loop, with the
    reach `worker` measures. The worker stops with them."""
    workspace = Workspace(worker.reaches)

    async def on_describe(request) -> describe_workspace.Response:
        return await _answer(
            "describe_workspace",
            lambda: workspace.describe(request.data.surface_height),
            _described,
            _describe_refused,
        )

    async def on_check(request) -> check_positions.Response:
        return await _answer(
            "check_positions",
            lambda: workspace.check(request.data.positions),
            _checked,
            _check_refused,
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
    refuse: Callable[[str], object],
):
    """The response of one request: `respond` to its answer, or `refuse`
    with the reason. A failure nobody foresaw refuses too, so the caller
    gets an answer and the service keeps serving."""
    try:
        return respond(await ask())
    except Refused as e:
        return refuse(str(e))
    except Exception as e:
        runtime.log(f"{service} failed unexpectedly: {e!r}")
        return refuse(f"{service} failed: {e!r}")


def _described(answer: SurfaceAnswer) -> describe_workspace.Response:
    return describe_workspace.Response(
        success=True,
        message=answer.message,
        perception_camera=NO_PERCEPTION_CAMERA,
        workable=answer.workable,
        area=answer.area,
        rectangle=_listed(answer.rectangle),
        reach=_listed(answer.reach),
        view=_listed(answer.view),
    )


def _describe_refused(reason: str) -> describe_workspace.Response:
    return describe_workspace.Response(
        success=False,
        message=reason,
        perception_camera="",
        workable=False,
        area=0.0,
        rectangle=None,
        reach=None,
        view=None,
    )


def _checked(answer: PositionsAnswer) -> check_positions.Response:
    return check_positions.Response(
        success=True,
        message=answer.message,
        perception_camera=NO_PERCEPTION_CAMERA,
        all_workable=answer.all_workable,
        results=[_result(point) for point in answer.points],
    )


def _check_refused(reason: str) -> check_positions.Response:
    return check_positions.Response(
        success=False,
        message=reason,
        perception_camera="",
        all_workable=False,
        results=[],
    )


def _result(point: PointAnswer) -> check_positions.ResponseResultsItem:
    return check_positions.ResponseResultsItem(
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
