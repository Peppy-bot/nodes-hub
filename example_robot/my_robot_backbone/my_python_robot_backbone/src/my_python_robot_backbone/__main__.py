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


async def _run_arm_action_safe(node_runner):
    try:
        await _run_arm_action(node_runner)
    except Exception as error:
        print(f"move_arm action error: {error}")


async def _run_arm_action(node_runner):
    print("[controller] move_arm action handler started")
    action = await move_arm.ActionHandle.expose(node_runner)
    busy_arms: set[int] = set()
    # The decider pre-fires the arm goal so it can mirror the arm's
    # accept/reject. On accept, the resulting handle lands here for drive_goal
    # to pick up — re-firing later would just produce a different goal_id.
    pending_handles: dict[int, object] = {}
    # asyncio.create_task only keeps a weak reference, so we hold the task
    # ourselves; otherwise GC can drop _drive_goal mid-await, which drops the
    # backbone GoalContext without completing and races our own get_result on
    # the arm side.
    drive_tasks: set[asyncio.Task] = set()

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
                node_runner, arm_request, GOAL_TIMEOUT, QoSProfile.Standard
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
        return move_arm.GoalResponse.accept()

    while True:
        ctx = await action.handle_goal_next_request(decide)
        if ctx is None:
            print("[controller] move_arm action handler closed")
            break
        arm_id = ctx.request().data.arm_id
        arm_handle = pending_handles.pop(arm_id)
        task = asyncio.create_task(_drive_goal(ctx, arm_handle, busy_arms, arm_id))
        drive_tasks.add(task)
        task.add_done_callback(drive_tasks.discard)


async def _drive_goal(backbone_ctx, arm_handle, busy_arms, arm_id):
    side = _arm_side(arm_id)
    # Forward feedback on a side task and treat get_result as the authoritative
    # completion signal. Result delivery must not wait on the feedback stream
    # draining: the end-of-stream sentinel is an ordinary feedback message and
    # can be lost or delayed, which would otherwise wedge this goal forever
    # (get_result would never run) and time out the client.
    cancelled = [False]
    pump_task = asyncio.create_task(
        _pump_feedback(backbone_ctx, arm_handle, side, cancelled)
    )
    try:
        try:
            result = await arm_handle.get_result(RESULT_TIMEOUT)
            fp = result.data.final_position
            print(f"[controller] {side} arm completed at position: {fp}")
            try:
                if cancelled[0]:
                    await backbone_ctx.complete_cancelled(fp)
                else:
                    await backbone_ctx.complete(fp)
            except Exception as e:
                print(f"[controller] {side} complete error: {e!r}")
        except Exception as e:
            print(f"[controller] {side} get_result error: {e!r}")
            try:
                await backbone_ctx.complete_cancelled([0, 0, 0])
            except Exception:
                pass
    finally:
        pump_task.cancel()
        busy_arms.discard(arm_id)


async def _pump_feedback(backbone_ctx, arm_handle, side, cancelled):
    # Drain arm feedback, forwarding to the backbone client, and flip the shared
    # `cancelled` flag if the backbone client cancels (forwarding the cancel to
    # the arm). Runs as a side task to _drive_goal, which cancels it once the
    # result is in; the feedback loop also ends naturally when the arm closes
    # its stream. Because this no longer gates get_result, a lost feedback
    # end-of-stream cannot stall the goal.
    async def cancel_watcher():
        await backbone_ctx.cancel_signal()
        cancelled[0] = True
        try:
            await arm_handle.cancel_goal(CANCEL_TIMEOUT)
        except Exception as e:
            print(f"[controller] {side} cancel_goal error: {e!r}")

    cancel_task = asyncio.create_task(cancel_watcher())
    try:
        while True:
            try:
                msg = await arm_handle.on_next_feedback_message()
            except Exception:
                break
            try:
                await backbone_ctx.publish_feedback(msg.current_position)
            except Exception:
                pass
    finally:
        cancel_task.cancel()


async def setup(params: Parameters, node_runner: NodeRunner) -> list[asyncio.Task]:
    return [asyncio.create_task(_run_arm_action_safe(node_runner))]


def main():
    NodeBuilder().run(setup)


if __name__ == "__main__":
    main()
