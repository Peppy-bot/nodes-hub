import asyncio
import time

from peppygen import NodeBuilder, NodeRunner
from peppygen.emitted_topics import joint_states
from peppygen.exposed_actions import move_arm
from peppygen.parameters import Parameters


async def publish_joint_states(node_runner: NodeRunner, current_position: list[int]):
    while True:
        now = time.time()
        positions = [float(p) for p in current_position]
        velocities = [0.0, 0.0, 0.0]
        try:
            await joint_states.emit(node_runner, positions, velocities, now)
        except Exception as e:
            print(f"[arm] emit joint_states error: {e!r}")
            break
        print(
            f"[arm] published joint_states: positions={[round(p, 3) for p in positions]}"
        )
        await asyncio.sleep(0.5)


async def _run_action_safe(node_runner: NodeRunner, current_position: list[int]):
    try:
        await _run_action(node_runner, current_position)
    except Exception as error:
        print(f"move_arm action error: {error}")


async def _run_action(node_runner: NodeRunner, current_position: list[int]):
    print("[arm] move_arm action handler started")
    action = await move_arm.ActionHandle.expose(node_runner)
    # Single arm per instance, so just one in-flight goal at a time.
    busy = [False]

    def decide(request):
        print(f"[arm] received move_arm goal: {request.data.desired_position}")
        if busy[0]:
            return move_arm.GoalResponse.reject("arm is already moving")
        busy[0] = True
        return move_arm.GoalResponse.accept()

    while True:
        ctx = await action.handle_goal_next_request(decide)
        if ctx is None:
            print("[arm] move_arm action handler closed")
            break
        asyncio.create_task(_drive_goal(ctx, current_position, busy))


async def _drive_goal(ctx, current_position: list[int], busy: list[bool]):
    target = ctx.request().data.desired_position
    start = list(current_position)
    duration = _choose_action_duration()

    cancel_task = asyncio.ensure_future(ctx.cancel_signal())
    work_task = asyncio.ensure_future(
        _execute_goal(ctx, start, target, duration, current_position)
    )
    try:
        done, pending = await asyncio.wait(
            [cancel_task, work_task], return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        if cancel_task in done:
            last_known = list(current_position)
            print(f"[arm] move_arm cancelled at position: {last_known}")
            try:
                await ctx.complete_cancelled(last_known)
            except Exception as e:
                print(f"[arm] complete_cancelled error: {e!r}")
        else:
            final_position = work_task.result()
            print(f"[arm] move_arm completed at position: {final_position}")
            try:
                await ctx.complete(final_position)
            except Exception as e:
                print(f"[arm] complete error: {e!r}")
    finally:
        busy[0] = False


async def _execute_goal(ctx, start, target, duration, current_position):
    steps, step_duration = _feedback_plan(duration)
    for step in range(1, steps + 1):
        await asyncio.sleep(step_duration)
        ratio = step / steps
        current = _interpolate_position(start, target, ratio)
        current_position[:] = current
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


async def setup(params: Parameters, node_runner: NodeRunner) -> list[asyncio.Task]:
    current_position: list[int] = [0, 0, 0]
    return [
        asyncio.create_task(publish_joint_states(node_runner, current_position)),
        asyncio.create_task(_run_action_safe(node_runner, current_position)),
    ]


def main():
    NodeBuilder().run(setup)


if __name__ == "__main__":
    main()
