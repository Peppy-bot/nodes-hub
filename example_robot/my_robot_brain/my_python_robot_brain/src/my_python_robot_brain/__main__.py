import asyncio

from peppygen import NodeBuilder, NodeRunner, QoSProfile
from peppygen.parameters import Parameters
from peppygen.consumed_actions import (
    robot_controller_move_arm as arm,
)
from peppygen.consumed_topics import camera_video_stream as video_stream


def _log_arm_result(side: str, result) -> None:
    # get_result returns a typed terminal outcome rather than raising, so map
    # each status. Completed/Cancelled carry the result payload; Abandoned and
    # Expired do not.
    if result.status == arm.ResultStatus.COMPLETED:
        print(f"[brain] {side} arm completed at position: {result.data.final_position}")
    elif result.status == arm.ResultStatus.CANCELLED:
        print(f"[brain] {side} arm cancelled at position: {result.data.final_position}")
    elif result.status == arm.ResultStatus.ABANDONED:
        print(f"[brain] {side} arm abandoned the goal without a result")
    else:  # ResultStatus.EXPIRED
        print(f"[brain] {side} arm result expired before it was fetched")


async def ai_process(node_runner: NodeRunner):
    print("[brain] AI process started, waiting for video frames...")
    token = node_runner.cancellation_token()

    while not token.is_cancelled():
        # Subscribe to video frames from the camera, racing the wait against
        # shutdown so the loop returns cleanly instead of relying on the
        # runtime's post-hook task cancellation.
        receive = asyncio.ensure_future(
            video_stream.on_next_message_received(node_runner)
        )
        cancelled = asyncio.ensure_future(token.cancelled())
        done, _pending = await asyncio.wait(
            {receive, cancelled}, return_when=asyncio.FIRST_COMPLETED
        )
        cancelled.cancel()
        if receive not in done:
            receive.cancel()
            break
        try:
            _instance_id, frame = receive.result()
            print("[brain] Received video frame")
        except Exception as e:
            print(f"Failed to receive video frame: {e}")
            continue

        # Process the frame and generate fake arm positions
        fake_position = [
            frame.frame[0],
            frame.frame[1],
            frame.frame[2],
        ]
        print(f"[brain] Generated arm position: {fake_position}")

        # Fire action goals to both arms concurrently
        print("[brain] Firing goals to both arms...")
        left_goal = arm.GoalRequest(arm_id=0, desired_position=fake_position)
        right_goal = arm.GoalRequest(arm_id=1, desired_position=fake_position)

        goal_timeout = 5.0
        result_timeout = 10.0

        left_goal_result, right_goal_result = await asyncio.gather(
            arm.ActionHandle.fire_goal(
                node_runner, left_goal, goal_timeout, QoSProfile.Standard
            ),
            arm.ActionHandle.fire_goal(
                node_runner, right_goal, goal_timeout, QoSProfile.Standard
            ),
            return_exceptions=True,
        )

        # Get the action handles from accepted goals
        left_handle = None
        if isinstance(left_goal_result, Exception):
            print(f"Failed to fire left arm goal: {left_goal_result}")
        elif left_goal_result.data.accepted:
            print("[brain] Left arm goal accepted")
            left_handle = left_goal_result
        else:
            print("[brain] Left arm goal rejected")

        right_handle = None
        if isinstance(right_goal_result, Exception):
            print(f"Failed to fire right arm goal: {right_goal_result}")
        elif right_goal_result.data.accepted:
            print("[brain] Right arm goal accepted")
            right_handle = right_goal_result
        else:
            print("[brain] Right arm goal rejected")

        # Wait for results from both arms concurrently (only if goals were accepted)
        if left_handle and right_handle:
            left_result, right_result = await asyncio.gather(
                left_handle.get_result(result_timeout),
                right_handle.get_result(result_timeout),
                return_exceptions=True,
            )
            if isinstance(left_result, Exception):
                print(f"[brain] Failed to get left arm result: {left_result}")
            else:
                _log_arm_result("Left", left_result)
            if isinstance(right_result, Exception):
                print(f"[brain] Failed to get right arm result: {right_result}")
            else:
                _log_arm_result("Right", right_result)
        elif left_handle:
            try:
                _log_arm_result("Left", await left_handle.get_result(result_timeout))
            except Exception as e:
                print(f"[brain] Failed to get left arm result: {e}")
        elif right_handle:
            try:
                _log_arm_result("Right", await right_handle.get_result(result_timeout))
            except Exception as e:
                print(f"[brain] Failed to get right arm result: {e}")
        else:
            print("[brain] Both arm goals failed, skipping result wait")

    print("[brain] AI process stopped (shutdown requested)")


async def setup(params: Parameters, node_runner: NodeRunner) -> list[asyncio.Task]:
    return [asyncio.create_task(ai_process(node_runner))]


def main():
    NodeBuilder().run(setup)


if __name__ == "__main__":
    main()
