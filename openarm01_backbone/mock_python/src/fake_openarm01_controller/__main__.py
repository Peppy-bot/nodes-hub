import asyncio
import time

from peppygen import NodeBuilder, NodeRunner
from peppygen.consumed_topics import (
    left_robot_arm_joint_states,
    right_robot_arm_joint_states,
)
from peppygen.emitted_topics import joint_positions
from peppygen.exposed_actions import move_arm
from peppygen.parameters import Parameters

ARM_ID_LEFT = 0
ARM_ID_RIGHT = 1
ARM_COUNT = 2


def _arm_side(arm_id: int) -> str:
    if arm_id == ARM_ID_LEFT:
        return "Left"
    if arm_id == ARM_ID_RIGHT:
        return "Right"
    return "Unknown"


def _arm_slot(arm_id: int) -> int:
    return arm_id if 0 <= arm_id < ARM_COUNT else 0


async def _receive_joint_states(node_runner: NodeRunner, side: str, topic_module):
    while True:
        try:
            _id, msg = await topic_module.on_next_message_received(node_runner, None)
            print(
                f"[controller] {side} joint_states update: "
                f"positions={[round(p, 3) for p in msg.positions]} "
                f"velocities={[round(v, 3) for v in msg.velocities]}"
            )
        except Exception as e:
            print(f"[controller] {side} joint_states subscription closed: {e!r}")
            break


async def setup(params: Parameters, node_runner: NodeRunner) -> list[asyncio.Task]:
    return [
        asyncio.create_task(
            _receive_joint_states(node_runner, "left", left_robot_arm_joint_states)
        ),
        asyncio.create_task(
            _receive_joint_states(node_runner, "right", right_robot_arm_joint_states)
        ),
        asyncio.create_task(_run_arm_action_safe(node_runner)),
    ]


async def _run_arm_action_safe(node_runner):
    try:
        await _run_arm_action(node_runner)
    except Exception as error:
        print(f"move_arm action error: {error}")


async def _run_arm_action(node_runner):
    print("[controller] move_arm action handler started")
    action = await move_arm.ActionHandle.expose(node_runner)
    last_positions: list[list[int]] = [[0, 0, 0] for _ in range(ARM_COUNT)]
    # asyncio is single-threaded, so a plain set is safe here.
    busy_arms: set[int] = set()

    def decide(request):
        arm_id = request.data.arm_id
        side = _arm_side(arm_id)
        print(f"[controller] {side} arm received goal: {request.data.desired_position}")
        # Reject a second goal for the same arm; _drive_goal releases the slot when done.
        if arm_id in busy_arms:
            return move_arm.GoalResponse.reject(f"arm {arm_id} is already moving")
        busy_arms.add(arm_id)
        return move_arm.GoalResponse.accept()

    while True:
        ctx = await action.handle_goal_next_request(decide)
        if ctx is None:
            print("[controller] move_arm action handler closed")
            break
        asyncio.create_task(_drive_goal(node_runner, ctx, last_positions, busy_arms))


async def _drive_goal(node_runner, ctx, last_positions, busy_arms):
    arm_id = ctx.request().data.arm_id
    side = _arm_side(arm_id)
    slot = _arm_slot(arm_id)
    desired_position = ctx.request().data.desired_position

    cmd_positions = [float(v) for v in desired_position]
    try:
        await joint_positions.emit(node_runner, arm_id, cmd_positions, 1.0)
        print(
            f"[controller] {side} published joint_positions: "
            f"arm_id={arm_id} target={[round(p, 3) for p in cmd_positions]} "
            f"max_vel=1.0"
        )
    except Exception as e:
        print(f"[controller] {side} emit joint_positions error: {e!r}")

    start_position = list(last_positions[slot])
    duration = _choose_action_duration()
    # _execute_goal updates this in place; the cancel branch reads the last stepped value.
    current_position = list(start_position)

    cancel_task = asyncio.ensure_future(ctx.cancel_signal())
    work_task = asyncio.ensure_future(
        _execute_goal(
            node_runner,
            ctx,
            arm_id,
            start_position,
            desired_position,
            duration,
            current_position,
        )
    )
    try:
        done, pending = await asyncio.wait(
            [cancel_task, work_task], return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        if cancel_task in done:
            last_known = list(current_position)
            last_positions[slot] = list(last_known)
            print(f"[controller] {side} arm cancelled at position: {last_known}")
            try:
                await ctx.complete_cancelled(last_known)
            except Exception as e:
                print(f"[controller] {side} complete_cancelled error: {e!r}")
        else:
            final_position = work_task.result()
            last_positions[slot] = list(final_position)
            print(f"[controller] {side} arm completed at position: {final_position}")
            try:
                await ctx.complete(final_position)
            except Exception as e:
                print(f"[controller] {side} complete error: {e!r}")
    finally:
        busy_arms.discard(arm_id)


async def _execute_goal(
    node_runner, ctx, arm_id, start, target, duration, current_position
):
    steps, step_duration = _feedback_plan(duration)
    for step in range(1, steps + 1):
        await asyncio.sleep(step_duration)
        ratio = step / steps
        current = _interpolate_position(start, target, ratio)
        current_position[:] = current
        cmd_positions = [float(v) for v in current]
        try:
            await joint_positions.emit(node_runner, arm_id, cmd_positions, 1.0)
        except Exception:
            pass
        try:
            await ctx.publish_feedback(current)
        except Exception:
            pass
    return list(target)


def _choose_action_duration():
    nanos = int(time.time() * 1_000_000_000) % 1_000_000_000
    millis = 1000 + (nanos % 2000)
    return millis / 1000.0


def _feedback_plan(duration):
    total_ms = max(duration * 1000, 1)
    steps = max(int(total_ms // 200), 1)
    step_s = max(total_ms / steps, 1) / 1000.0
    return steps, step_s


def _interpolate_position(start, target, ratio):
    return [round(s + (t - s) * ratio) for s, t in zip(start, target)]


def main():
    NodeBuilder().run(setup)


if __name__ == "__main__":
    main()
