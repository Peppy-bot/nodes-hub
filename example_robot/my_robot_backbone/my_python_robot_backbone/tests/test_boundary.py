"""Wire-level boundary tests: the backbone's real `setup` booted through the
generated harness against an ephemeral, per-test Zenoh router.

The harness starts a mock arm on each dependency slot (`left_robot_arm`,
`right_robot_arm`), on the real action engine. A fixture caller sends
move_arm goals to the backbone, and the test plays the arm the backbone
forwards each goal to. No daemon, no launcher, no sleeps.
"""

import asyncio

from peppygen.consumed_actions.left_robot_arm import move_arm as arm
from peppygen.fixtures import harness
from peppygen.fixtures.exposed_actions import move_arm

from my_python_robot_backbone.__main__ import ARM_ID_LEFT, setup

TIMEOUT = 10.0
TARGET = [10, 20, 30]


def _send(h, arm_id: int = ARM_ID_LEFT):
    """Send a goal to the backbone in a task: the backbone admits it only once
    the arm admits the goal it forwards, which the test plays meanwhile."""
    request = move_arm.GoalRequestData(arm_id=arm_id, desired_position=TARGET)
    return asyncio.create_task(move_arm.send_goal(h, request, TIMEOUT))


async def _forwarded_and_accepted(h):
    """Send a goal for the left arm, accept the goal the backbone forwards,
    and return the backbone's goal handle and the arm's active goal."""
    sent = _send(h)
    pending = await h.mocks.deps.left_robot_arm.move_arm.next_goal(TIMEOUT)
    assert pending.request.desired_position == TARGET
    active = await pending.accept()
    handle = await sent
    assert handle.accepted
    return handle, active


async def test_a_refusal_of_the_arm_reaches_the_caller_with_its_kind():
    async with harness.start(setup) as h:
        sent = _send(h)
        pending = await h.mocks.deps.left_robot_arm.move_arm.next_goal(TIMEOUT)
        await pending.refuse(
            arm.FailureKind.PRECONDITION_FAILED, "a leader streams the arm"
        )
        handle = await sent
        assert not handle.accepted
        assert handle.refusal == move_arm.Failure(
            move_arm.FailureKind.PRECONDITION_FAILED, "a leader streams the arm"
        )


async def test_an_unknown_arm_is_rejected_and_a_busy_arm_fails_a_precondition():
    async with harness.start(setup) as h:
        unknown = await _send(h, arm_id=7)
        assert unknown.refusal.kind is move_arm.FailureKind.REJECTED
        assert unknown.refusal.message == "unknown arm_id 7"

        handle, active = await _forwarded_and_accepted(h)
        busy = await _send(h)
        assert busy.refusal.kind is move_arm.FailureKind.PRECONDITION_FAILED
        assert busy.refusal.message == "arm 0 is already moving"

        await active.complete(arm.ResultResponseData(final_position=TARGET))
        result = await handle.get_result(TIMEOUT)
        assert result.status is move_arm.ResultStatus.COMPLETED
        assert result.data.final_position == TARGET


async def test_a_failure_of_the_arm_fails_the_goal_with_its_kind():
    async with harness.start(setup) as h:
        handle, active = await _forwarded_and_accepted(h)
        await active.fail(arm.FailureKind.ATTEMPT_FAILED, "the arm stalled")
        result = await handle.get_result(TIMEOUT)
        assert result.status is move_arm.ResultStatus.FAILED
        assert result.failure == move_arm.Failure(
            move_arm.FailureKind.ATTEMPT_FAILED, "the arm stalled"
        )
        assert result.data is None


async def test_a_cancel_of_the_arm_alone_ends_the_goal_with_the_reason_of_the_arm():
    async with harness.start(setup) as h:
        handle, active = await _forwarded_and_accepted(h)
        await active.complete_cancelled(
            "stopped by a stop request",
            arm.ResultResponseData(final_position=[1, 2, 3]),
        )
        result = await handle.get_result(TIMEOUT)
        assert result.status is move_arm.ResultStatus.CANCELLED
        assert result.cancel_reason == "stopped by a stop request"
        assert result.data.final_position == [1, 2, 3]


async def test_a_cancel_of_the_caller_reaches_the_arm_and_ends_the_goal_for_its_own_cause():
    async with harness.start(setup) as h:
        handle, active = await _forwarded_and_accepted(h)
        cancel = await handle.cancel_goal(TIMEOUT)
        assert cancel.state is move_arm.CancelState.SIGNALLED

        # The backbone forwards the cancel: the arm's goal receives it from
        # its own caller, the backbone.
        await asyncio.wait_for(active.cancel_signal(), TIMEOUT)
        assert active.cancel_cause() == "caller"
        await active.complete_cancelled(
            "the arm stopped", arm.ResultResponseData(final_position=[4, 5, 6])
        )

        result = await handle.get_result(TIMEOUT)
        assert result.status is move_arm.ResultStatus.CANCELLED
        assert result.cancel_reason == "cancelled by its caller"
        assert result.data.final_position == [4, 5, 6]
