import asyncio

from peppygen import NodeBuilder, NodeRunner, QoSProfile
from peppygen.consumed_actions import (
    left_robot_arm_move_arm,
    right_robot_arm_move_arm,
)
from peppygen.exposed_actions import move_arm
from peppygen.parameters import Parameters

ARM_ID_LEFT = 0
ARM_ID_RIGHT = 1
ARM_MODULES = {
    ARM_ID_LEFT: left_robot_arm_move_arm,
    ARM_ID_RIGHT: right_robot_arm_move_arm,
}

GOAL_TIMEOUT = 5.0
CANCEL_TIMEOUT = 2.0
RESULT_TIMEOUT = 30.0


def _arm_side(arm_id: int) -> str:
    if arm_id == ARM_ID_LEFT:
        return "Left"
    if arm_id == ARM_ID_RIGHT:
        return "Right"
    return "Unknown"


async def _run_arm_action_safe(node_runner, token, active_handles, drive_tasks):
    try:
        await _run_arm_action(node_runner, token, active_handles, drive_tasks)
    except Exception as error:
        print(f"move_arm action error: {error}")


async def _run_arm_action(node_runner, token, active_handles, drive_tasks):
    print("[controller] move_arm action handler started")
    action = await move_arm.ActionHandle.expose(node_runner)
    busy_arms: set[int] = set()
    # The decider pre-fires the arm goal so it can mirror the arm's
    # accept/reject. On accept, the resulting handle lands here for drive_goal
    # to pick up — re-firing later would just produce a different goal_id.
    pending_handles: dict[int, object] = {}

    async def decide(request):
        arm_id = request.data.arm_id
        side = _arm_side(arm_id)
        print(f"[controller] {side} arm received goal: {request.data.desired_position}")
        if arm_id not in ARM_MODULES:
            return move_arm.GoalResponse.reject(f"unknown arm_id {arm_id}")
        if arm_id in busy_arms:
            return move_arm.GoalResponse.reject(f"arm {arm_id} is already moving")
        arm_module = ARM_MODULES[arm_id]
        arm_request = arm_module.GoalRequest(
            desired_position=request.data.desired_position
        )
        try:
            arm_handle = await arm_module.ActionHandle.fire_goal(
                node_runner,
                arm_module.bound_producer(node_runner),
                arm_request,
                GOAL_TIMEOUT,
                QoSProfile.Standard,
            )
        except Exception as e:
            return move_arm.GoalResponse.reject(f"{side} fire_goal error: {e!r}")
        if not arm_handle.data.accepted:
            reason = arm_handle.data.error_message or f"{side} arm rejected goal"
            print(f"[controller] {side} arm rejected forwarded goal: {reason}")
            return move_arm.GoalResponse.reject(reason)
        print(f"[controller] {side} arm accepted forwarded goal")
        busy_arms.add(arm_id)
        pending_handles[arm_id] = arm_handle
        # Tracked from the moment the forwarded goal exists, so the shutdown
        # hook in setup can cancel it; removed in _drive_goal's finally.
        active_handles[arm_id] = arm_handle
        return move_arm.GoalResponse.accept()

    cancelled = asyncio.ensure_future(token.cancelled())
    try:
        while not token.is_cancelled():
            next_goal = asyncio.ensure_future(action.handle_goal_next_request(decide))
            await asyncio.wait(
                [cancelled, next_goal], return_when=asyncio.FIRST_COMPLETED
            )
            if not next_goal.done():
                # Shutdown began; stop accepting goals. In-flight forwarded
                # goals are cancelled by the shutdown hook registered in setup.
                next_goal.cancel()
                break
            ctx = next_goal.result()
            if ctx is None:
                print("[controller] move_arm action handler closed")
                break
            arm_id = ctx.request().data.arm_id
            arm_handle = pending_handles.pop(arm_id)
            task = asyncio.create_task(
                _drive_goal(ctx, arm_handle, busy_arms, active_handles, arm_id, token)
            )
            drive_tasks.add(task)
            task.add_done_callback(drive_tasks.discard)
    finally:
        cancelled.cancel()


async def _drive_goal(
    backbone_ctx, arm_handle, busy_arms, active_handles, arm_id, token
):
    side = _arm_side(arm_id)
    arm_module = ARM_MODULES[arm_id]
    try:
        # Drain feedback until the arm closes its stream (the engine guarantees
        # closure on completion, cancel, or abandonment) or shutdown begins,
        # forwarding any backbone-side cancel to the arm meanwhile, then fetch
        # the authoritative typed result. Draining and get_result run
        # sequentially: the Python action handle serializes access, so the
        # feedback wait and the result poll must not run concurrently on the
        # same handle.
        await _pump_feedback(backbone_ctx, arm_handle, side, token)
        try:
            result = await arm_handle.get_result(RESULT_TIMEOUT)
        except Exception as e:
            # A genuine timeout or transport error. Abandon the forwarded goal:
            # leaving backbone_ctx uncompleted surfaces as Abandoned upstream.
            print(f"[controller] {side} get_result error: {e!r}")
            return
        await _relay_outcome(backbone_ctx, arm_module, side, result)
    finally:
        busy_arms.discard(arm_id)
        active_handles.pop(arm_id, None)


async def _relay_outcome(backbone_ctx, arm_module, side, result):
    # Mirror the arm's typed outcome onto our own goal. Completed/Cancelled carry
    # the final position; for Abandoned/Expired we leave backbone_ctx
    # uncompleted, which the engine reports to our client as Abandoned.
    if result.status == arm_module.ResultStatus.COMPLETED:
        fp = result.data.final_position
        print(f"[controller] {side} arm completed at position: {fp}")
        try:
            await backbone_ctx.complete(fp)
        except Exception as e:
            print(f"[controller] {side} complete error: {e!r}")
    elif result.status == arm_module.ResultStatus.CANCELLED:
        fp = result.data.final_position
        print(f"[controller] {side} arm cancelled at position: {fp}")
        try:
            await backbone_ctx.complete_cancelled(fp)
        except Exception as e:
            print(f"[controller] {side} complete error: {e!r}")
    elif result.status == arm_module.ResultStatus.ABANDONED:
        print(f"[controller] {side} arm abandoned its goal; abandoning forwarded goal")
    else:  # ResultStatus.EXPIRED
        print(f"[controller] {side} arm result expired; abandoning forwarded goal")


async def _pump_feedback(backbone_ctx, arm_handle, side, token):
    # Forward arm feedback to the backbone client, and forward a backbone-side
    # cancel to the arm. Returns when the arm closes its feedback stream (on
    # completion, cancel, or abandonment) or when shutdown begins, at which
    # point the caller fetches the result.
    async def cancel_watcher():
        await backbone_ctx.cancel_signal()
        try:
            await arm_handle.cancel_goal(CANCEL_TIMEOUT)
        except Exception as e:
            print(f"[controller] {side} cancel_goal error: {e!r}")

    cancel_task = asyncio.create_task(cancel_watcher())
    cancelled = asyncio.ensure_future(token.cancelled())
    try:
        while not token.is_cancelled():
            feedback = asyncio.ensure_future(arm_handle.on_next_feedback_message())
            await asyncio.wait(
                [cancelled, feedback], return_when=asyncio.FIRST_COMPLETED
            )
            if not feedback.done():
                # Shutdown began; the hook in setup cancels the arm goal, so
                # stop relaying feedback and let get_result fetch the
                # cancelled outcome.
                feedback.cancel()
                break
            try:
                msg = feedback.result()
            except Exception:
                break
            try:
                await backbone_ctx.publish_feedback(msg.current_position)
            except Exception:
                pass
    finally:
        cancelled.cancel()
        cancel_task.cancel()


async def _cancel_forwarded_goal(arm_id, arm_handle):
    side = _arm_side(arm_id)
    try:
        await arm_handle.cancel_goal(CANCEL_TIMEOUT)
        print(f"[controller] {side} arm goal cancelled for shutdown")
    except Exception as e:
        print(f"[controller] {side} shutdown cancel_goal error: {e!r}")


async def setup(params: Parameters, node_runner: NodeRunner) -> list[asyncio.Task]:
    token = node_runner.cancellation_token()
    # Forwarded goals currently executing on an arm (arm_id -> arm handle),
    # shared with the shutdown hook below so it can cancel them deliberately.
    active_handles: dict[int, object] = {}
    # asyncio.create_task only keeps a weak reference, so we hold the tasks
    # ourselves; otherwise GC can drop _drive_goal mid-await, which drops the
    # backbone GoalContext without completing and races our own get_result on
    # the arm side.
    drive_tasks: set[asyncio.Task] = set()

    async def cancel_forwarded_goals():
        # Only shutdown hooks are awaited by the runtime: forward the shutdown
        # as a deliberate cancel to every in-flight arm goal while the
        # messenger is still connected, then give the _drive_goal tasks
        # (cancelled only after the hooks) a bounded window to relay the
        # cancelled outcome upstream instead of abandoning it.
        await asyncio.gather(
            *(
                _cancel_forwarded_goal(arm_id, arm_handle)
                for arm_id, arm_handle in list(active_handles.items())
            )
        )
        if drive_tasks:
            await asyncio.wait(set(drive_tasks), timeout=CANCEL_TIMEOUT)

    node_runner.on_shutdown(cancel_forwarded_goals)

    # Log when the shutdown/cancel signal is received so it is visible in the
    # node's stdout.
    async def announce_shutdown():
        print("[controller] Shutdown signal received")

    node_runner.on_shutdown(announce_shutdown)

    return [
        asyncio.create_task(
            _run_arm_action_safe(node_runner, token, active_handles, drive_tasks)
        )
    ]


def main():
    NodeBuilder().run(setup)


if __name__ == "__main__":
    main()
