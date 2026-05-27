import asyncio
import time
from collections import deque

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

    # Per-arm goal queues feed per-arm worker tasks so left and right run in
    # parallel. Completed final_positions land in `completed_results` and are
    # served back FIFO to result-request callers.
    arm_queues: list[asyncio.Queue] = [asyncio.Queue() for _ in range(ARM_COUNT)]
    completed_results: asyncio.Queue = asyncio.Queue()
    pending_results: deque = deque()
    workers_alive = ARM_COUNT

    worker_tasks = [
        asyncio.create_task(
            _arm_worker(
                node_runner,
                arm_id,
                arm_queues[arm_id],
                last_positions,
                completed_results,
            )
        )
        for arm_id in range(ARM_COUNT)
    ]

    goal_task = asyncio.create_task(_wait_for_goal(action))
    completion_task = asyncio.create_task(completed_results.get())

    try:
        while True:
            if pending_results:
                final_position = pending_results[0]
                try:
                    await asyncio.wait_for(
                        action.handle_result_next_request(
                            lambda _request, fp=final_position: move_arm.ResultResponse(
                                final_position=fp
                            )
                        ),
                        timeout=10.0,
                    )
                    pending_results.popleft()
                except asyncio.TimeoutError:
                    print(
                        "[controller] result request timed out, "
                        f"discarding final_position={final_position}"
                    )
                    pending_results.popleft()
                continue

            if workers_alive == 0:
                break

            done, _pending = await asyncio.wait(
                {goal_task, completion_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if goal_task in done:
                goal_request = goal_task.result()
                if goal_request is None:
                    print("[controller] move_arm action handler closed")
                    for queue in arm_queues:
                        queue.put_nowait(None)
                    goal_task = None
                else:
                    arm_queues[_arm_slot(goal_request.data.arm_id)].put_nowait(
                        goal_request
                    )
                    goal_task = asyncio.create_task(_wait_for_goal(action))
            if completion_task in done:
                final_position = completion_task.result()
                if final_position is None:
                    workers_alive -= 1
                else:
                    pending_results.append(final_position)
                completion_task = asyncio.create_task(completed_results.get())

            if goal_task is None and workers_alive == 0 and not pending_results:
                break
    finally:
        for task in (goal_task, completion_task, *worker_tasks):
            if task is not None and not task.done():
                task.cancel()


async def _arm_worker(node_runner, arm_id, queue, last_positions, completed_results):
    side = _arm_side(arm_id)
    slot = _arm_slot(arm_id)
    while True:
        goal_request = await queue.get()
        if goal_request is None:
            await completed_results.put(None)
            return
        desired_position = goal_request.data.desired_position
        print(f"[controller] {side} arm received goal: {desired_position}")

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

        start_position = last_positions[slot]
        duration = _choose_action_duration()

        final_position = await _execute_goal(
            node_runner, arm_id, start_position, desired_position, duration
        )
        print(f"[controller] {side} arm completed at position: {final_position}")
        last_positions[slot] = final_position
        await completed_results.put(final_position)


async def _wait_for_goal(action):
    goal_holder = []

    def on_goal(request):
        goal_holder.append(request)
        return move_arm.GoalResponse(accepted=True)

    await action.handle_goal_next_request(on_goal)
    return goal_holder[0] if goal_holder else None


def _choose_action_duration():
    nanos = int(time.time() * 1_000_000_000) % 1_000_000_000
    millis = 1000 + (nanos % 2000)
    return millis / 1000.0


async def _execute_goal(node_runner, arm_id, start, target, duration):
    steps, step_duration = _feedback_plan(duration)
    for step in range(1, steps + 1):
        await asyncio.sleep(step_duration)
        ratio = step / steps
        current = _interpolate_position(start, target, ratio)
        cmd_positions = [float(v) for v in current]
        try:
            await joint_positions.emit(node_runner, arm_id, cmd_positions, 1.0)
        except Exception:
            pass
    return list(target)


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
