import asyncio

from peppygen import NodeBuilder, NodeRunner
from peppygen.parameters import Parameters
from peppygen.consumed_actions.robot_controller import move_arm as arm
from peppygen.consumed_topics.camera import video_stream as video_stream

from my_python_robot_brain.frames import LatestValueMailbox


def _accepted_handle(side: str, fired):
    """The handle of a goal the arm accepted. A goal that could not be fired
    or that the arm refused is logged, with the kind and message of the
    refusal, and gives None."""
    if isinstance(fired, Exception):
        print(f"[brain] Failed to fire the {side} arm goal: {fired}")
        return None
    if fired.refusal is not None:
        print(f"[brain] {side} arm goal refused: {fired.refusal}")
        return None
    print(f"[brain] {side} arm goal accepted")
    return fired


def _log_arm_result(side: str, result) -> None:
    # get_result returns a typed terminal outcome rather than raising, so map
    # each status. Completed/Cancelled carry the result payload, Failed the
    # failure kind and message; Abandoned and Expired carry nothing.
    if result.status == arm.ResultStatus.COMPLETED:
        print(f"[brain] {side} arm completed at position: {result.data.final_position}")
    elif result.status == arm.ResultStatus.FAILED:
        print(f"[brain] {side} arm failed: {result.failure}")
    elif result.status == arm.ResultStatus.CANCELLED:
        print(
            f"[brain] {side} arm cancelled ({result.cancel_reason}) "
            f"at position: {result.data.final_position}"
        )
    elif result.status == arm.ResultStatus.ABANDONED:
        print(f"[brain] {side} arm abandoned the goal without a result")
    else:  # ResultStatus.EXPIRED
        print(f"[brain] {side} arm result expired before it was fetched")


async def read_video_frames(
    node_runner: NodeRunner, latest_frame: LatestValueMailbox
) -> None:
    """Drain the camera continuously, retaining only its newest frame."""
    token = node_runner.cancellation_token()

    try:
        subscription = await video_stream.subscribe(node_runner)
        while not token.is_cancelled():
            try:
                received = await subscription.next()
            except Exception as e:
                print(f"Failed to receive video frame: {e}")
                continue
            if received is None:
                break  # subscription closed
            _producer, frame = received
            latest_frame.offer(frame)
    finally:
        # Wake the worker if the subscription closes or this task is cancelled.
        latest_frame.close()


async def ai_process(node_runner: NodeRunner, latest_frame: LatestValueMailbox):
    print("[brain] AI process started, waiting for video frames...")
    token = node_runner.cancellation_token()

    while not token.is_cancelled():
        frame = await latest_frame.get()
        if frame is None:
            break
        print("[brain] Received latest video frame")

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
                node_runner,
                arm.bound_producer(node_runner),
                left_goal,
                goal_timeout,
            ),
            arm.ActionHandle.fire_goal(
                node_runner,
                arm.bound_producer(node_runner),
                right_goal,
                goal_timeout,
            ),
            return_exceptions=True,
        )

        # Get the action handles from accepted goals
        left_handle = _accepted_handle("Left", left_goal_result)
        right_handle = _accepted_handle("Right", right_goal_result)

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
    # Log when the shutdown/cancel signal is received so it is visible in the
    # node's stdout.
    async def announce_shutdown():
        print("[brain] Shutdown signal received")

    node_runner.on_shutdown(announce_shutdown)
    latest_frame = LatestValueMailbox()
    return [
        # Return both tasks so the runtime supervises failures in either half of
        # the reader/worker pipeline.  The reader prevents transport backpressure;
        # the worker remains serial so frames cannot create overlapping goal sets.
        asyncio.create_task(read_video_frames(node_runner, latest_frame)),
        asyncio.create_task(ai_process(node_runner, latest_frame)),
    ]


def main():
    NodeBuilder().run(setup)


if __name__ == "__main__":
    main()
