"""Wire-level boundary tests: the brain's real `setup` booted through the
generated harness against an ephemeral, per-test Zenoh router.

The harness starts one mock per dependency slot: the `camera` contract link
(video_stream topic + video_stream_info service) and the `robot_controller`
node link (move_arm action on the real action engine). The camera mock feeds
frames; the robot_controller mock plays the producer side of the consumed
move_arm goals the brain fires. No daemon, no launcher, no sleeps.
"""

from peppygen.consumed_actions.robot_controller import move_arm
from peppygen.consumed_topics.camera import video_stream
from peppygen.fixtures import harness

from my_python_robot_brain.__main__ import setup

GOAL_TIMEOUT = 10.0
SUBSCRIBER_TIMEOUT = 10.0


def _frame_message(frame_id: int, rgb: bytes) -> video_stream.Message:
    """A tiny typed video_stream frame; the brain only reads frame[0:3]."""
    return video_stream.Message(
        header=video_stream.MessageHeader(timestamp=100.0 + frame_id, frame_id=frame_id),
        encoding="rgb8",
        width=2,
        height=1,
        frame=rgb,
    )


async def _goal_pair(controller):
    """The two concurrent goals (left arm 0, right arm 1) the brain fires per
    frame, keyed by arm_id — arrival order between the two is not defined."""
    first = await controller.next_goal(GOAL_TIMEOUT)
    second = await controller.next_goal(GOAL_TIMEOUT)
    goals = {pending.request.arm_id: pending for pending in (first, second)}
    assert set(goals) == {0, 1}
    return goals


async def _accept_and_complete(goals):
    for pending in goals.values():
        active = await pending.accept()
        await active.complete(
            move_arm.ResultResponseData(
                final_position=pending.request.desired_position
            )
        )


async def test_frame_drives_move_arm_goals_and_loops_to_next_frame():
    """One published frame makes the brain fire a move_arm goal per arm carrying
    the frame's first three bytes as desired_position; completing both results
    lets it loop and process a second frame the same way — the consumed-action
    fire_goal/get_result boundary end to end."""
    async with harness.start(setup) as h:
        camera = h.mocks.deps.camera.video_stream
        controller = h.mocks.deps.robot_controller.move_arm

        # First publish waits for the brain's subscription to match, so this
        # cannot race the boot.
        await camera.publish(_frame_message(0, bytes([10, 20, 30, 40, 50, 60])))

        goals = await _goal_pair(controller)
        for pending in goals.values():
            assert pending.request.desired_position == [10, 20, 30]
        await _accept_and_complete(goals)

        # Both results collected, the brain loops back to the mailbox: a second
        # frame must produce a second pair of goals with its own payload.
        await camera.publish(_frame_message(1, bytes([40, 50, 60, 70, 80, 90])))

        goals = await _goal_pair(controller)
        for pending in goals.values():
            assert pending.request.desired_position == [40, 50, 60]
        await _accept_and_complete(goals)


async def test_shutdown_is_clean_after_boot():
    """Exiting the harness context with the brain idle (subscribed, parked on
    the frame mailbox) cancels its reader/worker tasks and closes the mocks
    without raising."""
    async with harness.start(setup) as h:
        # Bounded readiness check, not a sleep: the brain's video_stream
        # subscription is visible on the router, so setup ran to completion.
        assert await h.mocks.deps.camera.video_stream.wait_for_subscriber(
            SUBSCRIBER_TIMEOUT
        )
        assert h.setup_finished()
    # Reaching this line without an exception is the assertion.
