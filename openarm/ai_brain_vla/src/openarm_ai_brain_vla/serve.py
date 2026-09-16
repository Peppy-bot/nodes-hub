"""The loops every exposed member runs, written once.

An action loop waits for a goal, admits it, hands it to its own task and
goes back to waiting, until the node's cancellation token fires. The
per-goal tasks are kept in a set the caller owns: asyncio holds tasks
weakly, and a task collected mid-await drops its goal context without
completing it, which the caller sees as abandoned.

A service loop answers one request at a time the same way.
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable


async def serve_action(action_module, node_runner, token, tasks: set, run: Callable[[object], Awaitable[None]]) -> None:
    """Serves one exposed action. Every goal is accepted at admission: the
    contracts report refusals in the result, so `run` decides."""
    action = await action_module.ActionHandle.expose(node_runner)

    def accept(_request):
        return action_module.GoalDecision.accept()

    cancelled = asyncio.ensure_future(token.cancelled())
    try:
        while not token.is_cancelled():
            next_goal = asyncio.ensure_future(action.handle_goal_next_request(accept))
            await asyncio.wait([cancelled, next_goal], return_when=asyncio.FIRST_COMPLETED)
            if not next_goal.done():
                next_goal.cancel()
                break
            ctx = next_goal.result()
            if ctx is None:
                break
            task = asyncio.create_task(run(ctx))
            tasks.add(task)
            task.add_done_callback(tasks.discard)
            task.add_done_callback(_report_failure)
    finally:
        cancelled.cancel()


def _report_failure(task: asyncio.Task) -> None:
    """A handler that raises leaves its goal uncompleted, which the caller
    sees as abandoned; the reason must at least reach the log."""
    if task.cancelled():
        return
    error = task.exception()
    if error is not None:
        import traceback

        print("[brain] a goal handler failed:")
        traceback.print_exception(type(error), error, error.__traceback__)


async def serve_service(service_module, node_runner, token, handler) -> None:
    """Serves one exposed service, one request after another."""
    cancelled = asyncio.ensure_future(token.cancelled())
    try:
        while not token.is_cancelled():
            next_request = asyncio.ensure_future(service_module.handle_next_request(node_runner, handler))
            await asyncio.wait([cancelled, next_request], return_when=asyncio.FIRST_COMPLETED)
            if not next_request.done():
                next_request.cancel()
                break
            error = next_request.exception()
            if error is not None:
                print(f"[brain] {service_module.SERVICE_NAME} request failed: {error!r}")
    finally:
        cancelled.cancel()
