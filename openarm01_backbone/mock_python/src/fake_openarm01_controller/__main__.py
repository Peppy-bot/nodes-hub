import asyncio
import time

from peppygen import NodeBuilder, NodeRunner
from peppygen.consumed_topics import robot_arm_joint_states
from peppygen.emitted_topics import joint_commands
from peppygen.exposed_actions import move_left_arm, move_right_arm
from peppygen.parameters import Parameters


async def _receive_joint_states(node_runner: NodeRunner):
    while True:
        try:
            _id, msg = await robot_arm_joint_states.on_next_message_received(
                node_runner, None, None
            )
            print(
                f"[controller] joint_states update: "
                f"positions={[round(p, 3) for p in msg.positions]} "
                f"velocities={[round(v, 3) for v in msg.velocities]}"
            )
        except Exception as e:
            print(f"[controller] joint_states subscription closed: {e!r}")
            break


async def setup(params: Parameters, node_runner: NodeRunner) -> list[asyncio.Task]:
    return [
        asyncio.create_task(_receive_joint_states(node_runner)),
        asyncio.create_task(_run_arm_action_safe(node_runner, "Left", move_left_arm)),
        asyncio.create_task(_run_arm_action_safe(node_runner, "Right", move_right_arm)),
    ]


async def _run_arm_action_safe(node_runner, side, arm_module):
    try:
        await _run_arm_action(node_runner, side, arm_module)
    except Exception as error:
        print(f"{side} arm action error: {error}")


async def _run_arm_action(node_runner, side, arm_module):
    print(f"[controller] {side} arm action handler started")
    action = await arm_module.ActionHandle.expose(node_runner)
    last_position = [0, 0, 0]

    while True:
        goal_request = await _wait_for_goal(action, arm_module)
        if goal_request is None:
            print(f"[controller] {side} arm action handler closed")
            break

        desired_position = goal_request.data.desired_position
        print(f"[controller] {side} arm received goal: {desired_position}")
        cmd_positions = [float(v) for v in desired_position]
        try:
            await joint_commands.emit(node_runner, cmd_positions, 1.0)
            print(
                f"[controller] {side} published joint_commands: "
                f"target={[round(p, 3) for p in cmd_positions]} max_vel=1.0"
            )
        except Exception as e:
            print(f"[controller] {side} emit joint_commands error: {e!r}")
        duration = _choose_action_duration()

        outcome = await _execute_goal(
            action, node_runner, last_position, desired_position, duration, arm_module
        )

        if outcome[0] == "completed":
            print(f"[controller] {side} arm completed at position: {outcome[1]}")
            last_position = outcome[1]
        elif outcome[0] == "cancelled":
            print(f"[controller] {side} arm cancelled at position: {outcome[1]}")
            last_position = outcome[1]
        elif outcome[0] == "closed":
            print(f"[controller] {side} arm action closed")
            break

        final_position = list(last_position)

        # Use timeout to avoid blocking forever if client doesn't request result
        try:
            await asyncio.wait_for(
                action.handle_result_next_request(
                    lambda _request, p=final_position: arm_module.ResultResponse(
                        final_position=p
                    )
                ),
                timeout=10.0,
            )
        except asyncio.TimeoutError:
            print(
                f"[controller] {side} arm result request timed out, "
                "continuing to next goal"
            )
        except Exception as e:
            print(f"[controller] {side} arm result request error: {e}")
            break


async def _wait_for_goal(action, arm_module):
    goal_holder = []

    def on_goal(request):
        goal_holder.append(request)
        return arm_module.GoalResponse(accepted=True)

    await action.handle_goal_next_request(on_goal)
    return goal_holder[0] if goal_holder else None


def _choose_action_duration():
    nanos = int(time.time() * 1_000_000_000) % 1_000_000_000
    millis = 1000 + (nanos % 2000)
    return millis / 1000.0


async def _execute_goal(action, node_runner, start, target, duration, arm_module):
    await action.emit_feedback(list(start))

    cancel = await _poll_cancel(action, arm_module)
    if cancel == "cancelled":
        return ("cancelled", list(start))
    if cancel == "closed":
        return ("closed", None)

    steps, step_duration = _feedback_plan(duration)
    current = list(start)

    for step in range(1, steps + 1):
        await asyncio.sleep(step_duration)

        cancel = await _poll_cancel(action, arm_module)
        if cancel == "cancelled":
            return ("cancelled", current)
        if cancel == "closed":
            return ("closed", None)

        ratio = step / steps
        current = _interpolate_position(start, target, ratio)
        cmd_positions = [float(v) for v in current]
        try:
            await joint_commands.emit(node_runner, cmd_positions, 1.0)
        except Exception:
            pass
        await action.emit_feedback(current)

        cancel = await _poll_cancel(action, arm_module)
        if cancel == "cancelled":
            return ("cancelled", current)
        if cancel == "closed":
            return ("closed", None)

    return ("completed", list(target))


async def _poll_cancel(action, arm_module):
    try:
        await asyncio.wait_for(
            action.handle_cancel_next_request(
                lambda _request: arm_module.CancelResponse(
                    accepted=True, error_message=None
                )
            ),
            timeout=0,
        )
        return "cancelled"
    except asyncio.TimeoutError:
        return "none"


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
