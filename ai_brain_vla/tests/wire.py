"""The node's goals sent over the wire, and the robot's measured limbs and
moves as the harness's mocks give them: what the scripted grab needs from
the backbone."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from weakref import WeakKeyDictionary

from peppygen import QoSProfile
from peppygen.consumed_actions.limb_motion import move_arm, move_gripper
from peppygen.consumed_services.limb_motion import check_arm_move
from peppygen.consumed_topics.limb_state import limb_states
from peppygen.exposed_actions.item_manipulation import grab_item as grab_action
from peppygen.fixtures.exposed_actions.item_manipulation import grab_item as grab_fx

WAIT = 10.0
# Goals sent before the brain has its first limb_states snapshot; each one
# is a whole goal round trip.
ATTEMPTS = 1000
NO_SNAPSHOT = "no limb_states snapshot received from the robot yet"


async def result_of(fixture, h, request):
    goal = await fixture.send_goal(h, request, QoSProfile.Standard, 5.0)
    assert goal.accepted, goal.reason
    result = await goal.get_result(WAIT)
    assert result.data is not None
    return result.data


@dataclass
class FakeLimbs:
    """The limbs the harness's backbone publishes on limb_states: the left
    grasp point where the last landed move put it, the left gripper's
    opening, and the stamp of the last snapshot. The right arm stays at
    rest with its gripper open."""

    left_position: list[float] = field(default_factory=lambda: [0.4, 0.2, 0.9])
    left_orientation: list[float] = field(default_factory=lambda: [0.0, 1.0, 0.0, 0.0])
    left_opening: float = 1.0
    stamp: float = 1.0


_LIMBS: WeakKeyDictionary = WeakKeyDictionary()


def limbs_of(h) -> FakeLimbs:
    """The fake backbone's limbs under harness `h`."""
    return _LIMBS.setdefault(h, FakeLimbs())


def limbs_at(limbs: FakeLimbs) -> limb_states.Message:
    """A limb_states snapshot of the OpenArm, two arms of seven joints, as
    `limbs` stand."""
    return limb_states.Message(
        timestamp=limbs.stamp, arm_names=["left_arm", "right_arm"], joints_per_arm=[7, 7], joint_positions=[0.0] * 14,
        positions=[*limbs.left_position, 0.4, -0.2, 0.9], orientations=[*limbs.left_orientation, 0.0, 1.0, 0.0, 0.0],
        gripper_names=["left_gripper", "right_gripper"], gripper_openings=[limbs.left_opening, 1.0],
    )


async def publish_limbs(h) -> None:
    """One snapshot of the fake backbone's limbs as they stand."""
    topic = h.mocks.deps.limb_state.limb_states
    assert await topic.wait_for_subscriber(WAIT)
    await topic.publish(limbs_at(limbs_of(h)))


def grab_by_pose(position, orientation=None) -> grab_action.GoalRequestData:
    return grab_action.GoalRequestData(gripper_name="left_gripper", item_id="", position=list(position), orientation=orientation, max_effort=0.0)


def plan(h, *answers: bool) -> None:
    """Scripts the backbone's next check_arm_move answers: True plans."""
    for success in answers:
        h.mocks.deps.limb_motion.check_arm_move.enqueue_response(
            check_arm_move.ResponseData(success=success, message="" if success else "goal pose not planned", duration_s=1.0)
        )


async def once_the_brain_has_limbs(h) -> None:
    """Publishes a snapshot stamped 1 s until a grab is no longer refused
    for want of one: a grab by pose in one orientation, whose one check the
    backbone refuses, so it moves nothing."""
    plan(h, False)
    for _ in range(ATTEMPTS):
        await publish_limbs(h)
        data = await result_of(grab_fx, h, grab_by_pose([0.5, 0.0, 0.9], [0.0, 1.0, 0.0, 0.0]))
        if data.message != NO_SNAPSHOT:
            assert data.message.endswith("with gripper 'left_gripper': goal pose not planned"), data.message
            return
    raise AssertionError("the brain never received the limb_states snapshot")


async def accepted_move(h):
    """Accepts the next move_arm goal and returns its request and its
    handle."""
    pending = await h.mocks.deps.limb_motion.move_arm.next_goal(WAIT)
    return pending.request, await pending.accept()


async def land_move(h) -> move_arm.GoalRequest:
    """The next move_arm goal, landed on its target: limb_states shows the
    left grasp point there before the goal completes."""
    request, active = await accepted_move(h)
    assert request.arm_name == "left_arm", request.arm_name
    limbs = limbs_of(h)
    limbs.left_position, limbs.left_orientation = list(request.position), list(request.orientation)
    limbs.stamp += 1.0
    await publish_limbs(h)
    await active.complete(
        move_arm.ResultResponseData(
            success=True, message="move complete", final_position=request.position,
            final_orientation=request.orientation, action_time=1.0,
        )
    )
    return request


async def accepted_grip(h):
    """Accepts the next move_gripper goal and returns its request and its
    handle."""
    pending = await h.mocks.deps.limb_motion.move_gripper.next_goal(WAIT)
    return pending.request, await pending.accept()


async def stop_grip_at(h, final_opening: float) -> move_gripper.GoalRequest:
    """The next move_gripper goal, its gripper standing still at
    `final_opening`."""
    request, active = await accepted_grip(h)
    await active.complete(
        move_gripper.ResultResponseData(success=True, message="move complete", final_opening=final_opening, action_time=0.5)
    )
    return request


async def publish_limbs_until(h, done: asyncio.Event, left_opening: float) -> None:
    """The backbone's stream from the lift on, the left gripper at
    `left_opening`: a snapshot with a later stamp after each one, the arm
    where the last landed move put it, until `done`, so whatever the order
    the lift's result and a snapshot reach the brain in, a later snapshot
    follows."""
    limbs = limbs_of(h)
    limbs.left_opening = left_opening
    while not done.is_set():
        limbs.stamp += 1.0
        await publish_limbs(h)
        await asyncio.sleep(0.01)
