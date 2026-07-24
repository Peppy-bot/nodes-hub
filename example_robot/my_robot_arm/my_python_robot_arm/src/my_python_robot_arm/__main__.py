import asyncio
import time

from peppygen import NodeBuilder, NodeRunner
from peppygen.emitted_topics import joint_states
from peppygen.exposed_actions import move_arm
from peppygen.parameters import Parameters
from peppylib import CancellationToken


def _accept_goal():
    return move_arm.GoalDecision.accept(move_arm.GoalResponse(True, None))


def _reject_goal(reason: str):
    return move_arm.GoalDecision.reject(move_arm.GoalResponse(False, reason))


async def publish_joint_states(
    node_runner: NodeRunner,
    current_position: list[int],
    token: CancellationToken,
):
    publisher = await joint_states.declare_publisher(node_runner)
    cancelled = asyncio.ensure_future(token.cancelled())
    try:
        while not token.is_cancelled():
            now = time.time()
            positions = [float(p) for p in current_position]
            velocities = [0.0, 0.0, 0.0]
            try:
                await publisher.publish(
                    joint_states.build_message(positions, velocities, now)
                )
            except Exception as e:
                print(f"[arm] publish joint_states error: {e!r}")
                break
            print(
                f"[arm] published joint_states: positions={[round(p, 3) for p in positions]}"
            )
            # Sleep between publishes, waking early when shutdown begins.
            await asyncio.wait([cancelled], timeout=0.5)
    finally:
        cancelled.cancel()


async def _run_action_safe(
    node_runner: NodeRunner,
    current_position: list[int],
    token: CancellationToken,
):
    try:
        await _run_action(node_runner, current_position, token)
    except Exception as error:
        print(f"move_arm action error: {error}")


async def _run_action(
    node_runner: NodeRunner,
    current_position: list[int],
    token: CancellationToken,
):
    print("[arm] move_arm action handler started")
    action = await move_arm.ActionHandle.expose(node_runner)
    # Single arm per instance, so just one in-flight goal at a time.
    busy = [False]
    # asyncio.create_task only keeps a weak reference, so we hold the task
    # ourselves; otherwise GC can drop _drive_goal mid-await, which drops the
    # GoalContext without completing and evicts the slot from the registry.
    drive_tasks: set[asyncio.Task] = set()

    def decide(request):
        print(f"[arm] received move_arm goal: {request.data.desired_position}")
        if busy[0]:
            return _reject_goal("arm is already moving")
        busy[0] = True
        return _accept_goal()

    cancelled = asyncio.ensure_future(token.cancelled())
    try:
        while not token.is_cancelled():
            next_goal = asyncio.ensure_future(action.handle_goal_next_request(decide))
            await asyncio.wait(
                [cancelled, next_goal], return_when=asyncio.FIRST_COMPLETED
            )
            if not next_goal.done():
                # Shutdown began; stop accepting goals. In-flight _drive_goal
                # tasks are cancelled by the runtime's task teardown.
                next_goal.cancel()
                break
            ctx = next_goal.result()
            if ctx is None:
                print("[arm] move_arm action handler closed")
                break
            task = asyncio.create_task(_drive_goal(ctx, current_position, busy))
            drive_tasks.add(task)
            task.add_done_callback(drive_tasks.discard)
    finally:
        cancelled.cancel()


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
    token = node_runner.cancellation_token()

    # Log when the shutdown/cancel signal is received so it is visible in the
    # node's stdout.
    async def announce_shutdown():
        print("[arm] Shutdown signal received")

    node_runner.on_shutdown(announce_shutdown)

    return [
        asyncio.create_task(
            publish_joint_states(node_runner, current_position, token)
        ),
        asyncio.create_task(_run_action_safe(node_runner, current_position, token)),
    ]


def main():
    NodeBuilder().run(setup)


if __name__ == "__main__":
    main()
