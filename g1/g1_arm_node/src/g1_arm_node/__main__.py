"""g1_arm_node entry point: exposes preset arm gestures as a peppy action.

The G1ArmActionClient is synchronous, so its calls run through one single-worker
executor. setup() spawns the arm_gesture action handler and the get_arm_actions
service loop and returns them so the runtime owns the tasks.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

from peppygen import NodeBuilder, NodeRunner
from peppygen.exposed_actions import arm_gesture
from peppygen.exposed_services import get_arm_actions
from peppygen.parameters import Parameters

from .backend import ArmActionClientBackend, ArmBackend
from .gestures import UnknownGesture, action_id_for

SDK_CALL_TIMEOUT = 12.0


class ArmController:
    """Serializes G1ArmActionClient access through one worker."""

    def __init__(self, backend: ArmBackend) -> None:
        self._backend = backend
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="g1-arm")

    @property
    def backend(self) -> ArmBackend:
        return self._backend

    async def execute_action(self, action_id: int) -> int:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._backend.execute_action, action_id)

    def call_sync(self, fn, *args):
        return self._executor.submit(fn, *args).result(timeout=SDK_CALL_TIMEOUT)

    def close(self) -> None:
        self._executor.shutdown(wait=False)


async def _run_gesture_action(node_runner: NodeRunner, controller: ArmController, token):
    action = await arm_gesture.ActionHandle.expose(node_runner)
    busy = [False]
    drive_tasks: set[asyncio.Task] = set()

    def decide(request):
        if busy[0]:
            return arm_gesture.GoalResponse.reject("a gesture is in flight")
        try:
            action_id_for(request.data.gesture)
        except UnknownGesture as exc:
            return arm_gesture.GoalResponse.reject(str(exc))
        busy[0] = True
        return arm_gesture.GoalResponse.accept()

    cancelled = asyncio.ensure_future(token.cancelled())
    try:
        while not token.is_cancelled():
            next_goal = asyncio.ensure_future(action.handle_goal_next_request(decide))
            await asyncio.wait([cancelled, next_goal], return_when=asyncio.FIRST_COMPLETED)
            if not next_goal.done():
                next_goal.cancel()
                break
            ctx = next_goal.result()
            if ctx is None:
                break
            task = asyncio.create_task(_drive_gesture(ctx, controller, busy))
            drive_tasks.add(task)
            task.add_done_callback(drive_tasks.discard)
    finally:
        cancelled.cancel()


async def _drive_gesture(ctx, controller: ArmController, busy: list[bool]) -> None:
    action_id = action_id_for(ctx.request().data.gesture)  # validated in decide()
    try:
        await controller.execute_action(action_id)
        await ctx.complete(action_id)
    except Exception as error:
        print(f"[g1-arm] arm_gesture {action_id} failed: {error!r}")
    finally:
        busy[0] = False


async def _serve_get_actions(node_runner: NodeRunner, controller: ArmController, token):
    def handler(_request):
        return get_arm_actions.Response(
            actions_json=controller.call_sync(controller.backend.get_action_list)
        )

    cancelled = asyncio.ensure_future(token.cancelled())
    try:
        while not token.is_cancelled():
            req = asyncio.ensure_future(
                get_arm_actions.handle_next_request(node_runner, handler)
            )
            await asyncio.wait([cancelled, req], return_when=asyncio.FIRST_COMPLETED)
            if not req.done():
                req.cancel()
                break
            try:
                req.result()
            except Exception as exc:
                print(f"[g1-arm] service get_arm_actions error: {exc!r}")
    finally:
        cancelled.cancel()


async def setup(params: Parameters, node_runner: NodeRunner) -> list[asyncio.Task]:
    backend = ArmActionClientBackend(params.network_interface, params.dds_domain_id)
    controller = ArmController(backend)

    async def on_shutdown():
        controller.close()

    node_runner.on_shutdown(on_shutdown)

    token = node_runner.cancellation_token()
    return [
        asyncio.create_task(_run_gesture_action(node_runner, controller, token)),
        asyncio.create_task(_serve_get_actions(node_runner, controller, token)),
    ]


def main():
    NodeBuilder().run(setup)


if __name__ == "__main__":
    main()
