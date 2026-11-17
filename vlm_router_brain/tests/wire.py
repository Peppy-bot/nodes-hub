"""The node under the generated harness: the answers its camera and robot
mocks give the brain at its start, one capture of colour and depth, the
robot's measured limbs and its moves, and the goals and services sent over
the wire, each read by how it ended."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from weakref import WeakKeyDictionary

import numpy as np

from peppygen.consumed_actions.limb_motion import move_arm, move_gripper
from peppygen.consumed_services.camera import depth_stream_info
from peppygen.consumed_services.geometry import get_color_intrinsics, get_depth_intrinsics
from peppygen.consumed_services.limb_motion import check_arm_move
from peppygen.consumed_services.sensor_mounts import describe_sensors
from peppygen.consumed_services.transform_lookup import lookup_transform
from peppygen.consumed_topics.camera import depth_stream, video_stream
from peppygen.consumed_topics.limb_state import limb_states
from peppygen.consumed_topics.wrist_cameras import video_stream as wrist_video_stream
from peppygen.exposed_actions.item_manipulation import grab_item as grab_action
from peppygen.fixtures.exposed_services.item_manipulation import get_state as get_state_fx
from peppygen.parameters import Parameters

WAIT = 10.0
# Search attempts made before the brain has the first capture and the
# camera's answers; each one is a whole goal round trip.
ATTEMPTS = 1000
# What a search is refused with while the brain lacks the capture or the
# camera's answers.
NO_SOURCE = "no perception source: "

PARAMS = {
    "limbs": "left_arm:left_gripper,right_arm:right_gripper",
    "perception_model": "gemini-robotics-er-2-preview",
    "manipulation_model": "scripted",
    "camera_name": "chest",
    "lookup_timeout_s": 0.5,
    "wrist_cameras": "left_arm:wrist_left,right_arm:wrist_right",
}

# The robot's cameras, each with its frame and its place in the robot frame
# at every time: the chest at the robot's origin looking along +Z, and
# `wrist_left` 0.5 m up looking straight down with its picture's right along
# the robot's +Y.
ROBOT_FRAME = "base_link"
CAMERAS = {
    "chest": ("chest_color_optical_frame", [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
    "wrist_left": ("wrist_left_color_optical_frame", [0.2, 0.2, 0.5], [0.7071, 0.7071, 0.0, 0.0]),
}
# How long the answerer of the lookups waits for the next one: longer than a
# test runs, so it stops when the harness stops its mocks.
LOOKUP_WAIT_S = 3600.0
# The answerer of each harness's lookups, kept while it runs.
_ANSWERERS: WeakKeyDictionary = WeakKeyDictionary()


def params(**overrides) -> Parameters:
    return Parameters.from_dict({**PARAMS, **overrides})


def answer_the_camera(h) -> None:
    """The depth unit, the 16 x 12 camera's intrinsics with a 90 degree
    lens, the robot's cameras of CAMERAS, and the place of each in the
    robot frame for every lookup the brain makes."""
    h.mocks.deps.camera.depth_stream_info.enqueue_response(
        depth_stream_info.ResponseData(width=16, height=12, frames_per_second=15, encoding="z16", depth_unit=0.001)
    )
    h.mocks.deps.geometry.get_color_intrinsics.enqueue_response(
        get_color_intrinsics.ResponseData(width=16, height=12, fx=6.0, fy=6.0, cx=8.0, cy=6.0, distortion_model="none", distortion=[])
    )
    h.mocks.deps.geometry.get_depth_intrinsics.enqueue_response(
        get_depth_intrinsics.ResponseData(
            width=16, height=12, fx=6.0, fy=6.0, cx=8.0, cy=6.0, distortion_model="none", distortion=[],
            depth_model="z", min_depth_m=0.1, max_depth_m=10.0, align_mode="depth_to_color",
        )
    )
    h.mocks.deps.sensor_mounts.describe_sensors.enqueue_response(
        describe_sensors.ResponseData(
            robot_frame=ROBOT_FRAME,
            sensor_names=list(CAMERAS),
            sensor_kinds=["rgbd_camera", "rgb_camera"],
            sensor_frames=[frame for frame, _, _ in CAMERAS.values()],
            carried_by=["", "left_arm"],
        )
    )
    _ANSWERERS[h] = asyncio.get_running_loop().create_task(_answer_lookups(h.mocks.deps.transform_lookup.lookup_transform))


async def _answer_lookups(service) -> None:
    """Answers each lookup of a camera's frame in the robot frame with its
    place of CAMERAS, and any other with the failure `unknown_frame`, until
    the harness stops its mocks."""
    places = {frame: (position, orientation) for frame, position, orientation in CAMERAS.values()}
    while True:
        try:
            request, responder = await service.next_request(LOOKUP_WAIT_S)
        except Exception:  # pylint: disable=broad-exception-caught
            # The harness stopped its mocks.
            return
        if request.target_frame != ROBOT_FRAME or request.source_frame not in places:
            await responder.respond_failure(
                lookup_transform.FailureKind.UNKNOWN_FRAME, f"no transform names {request.source_frame}"
            )
            continue
        position, orientation = places[request.source_frame]
        await responder.respond(lookup_transform.ResponseData(position=position, orientation=orientation, timestamp=request.time))


async def publish_a_capture(h, frame_id: int = 1) -> None:
    """One grey 16 x 12 colour frame and the depth frame of its capture, one
    metre everywhere."""
    camera = h.mocks.deps.camera
    assert await camera.video_stream.wait_for_subscriber(WAIT)
    assert await camera.depth_stream.wait_for_subscriber(WAIT)
    header = dict(timestamp=1.0, frame_id=frame_id, align_mode="depth_to_color")
    await camera.video_stream.publish(
        video_stream.Message(header=video_stream.MessageHeader(**header), encoding="rgb8", width=16, height=12, frame=bytes([128]) * (16 * 12 * 3))
    )
    await camera.depth_stream.publish(
        depth_stream.Message(
            header=depth_stream.MessageHeader(**header), encoding="z16", width=16, height=12,
            frame=np.full((12, 16), 1000, dtype="<u2").tobytes(),
        )
    )


async def ended(fixture, h, request):
    """How a goal the node accepted ended: its result response, whose
    status says how."""
    goal = await fixture.send_goal(h, request, 5.0)
    assert goal.accepted, goal.refusal
    return await goal.get_result(WAIT)


async def result_of(fixture, h, request):
    """The result of a goal that completed."""
    result = await ended(fixture, h, request)
    assert result.status == fixture.ResultStatus.COMPLETED, result.failure or result.cancel_reason
    return result.data


async def failure_of(fixture, h, request):
    """The failure of a goal that the node accepted and that failed: its
    kind and its message, and no result."""
    result = await ended(fixture, h, request)
    assert result.status == fixture.ResultStatus.FAILED, result
    assert result.data is None
    return result.failure


async def refusal_of(fixture, h, request):
    """The failure a goal the node refused carries."""
    goal = await fixture.send_goal(h, request, 5.0)
    assert not goal.accepted
    return goal.refusal


async def once_the_brain_sees(fixture, h, request):
    """How the first search ended that the brain does not refuse as having
    no perception source: the capture and the camera's answers reach it
    after its goal loops are up. For a brain whose perception model can
    search."""
    for _ in range(ATTEMPTS):
        goal = await fixture.send_goal(h, request, 5.0)
        if goal.accepted:
            return await goal.get_result(WAIT)
        refusal = goal.refusal
        assert refusal.kind is fixture.FailureKind.RESOURCE_UNAVAILABLE and refusal.message.startswith(NO_SOURCE), refusal
    raise AssertionError(f"the brain never saw the capture: {refusal.message}")


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
    for reachable in answers:
        h.mocks.deps.limb_motion.check_arm_move.enqueue_response(
            check_arm_move.ResponseData(reachable=reachable, reason="" if reachable else "goal pose not planned", duration_s=1.0)
        )


async def once_the_brain_has_limbs(h) -> None:
    """Publishes a snapshot stamped 1 s until get_state reports the openings
    of the grippers: the brain holds a limb_states snapshot, which a grab
    reads first. Nothing moves."""
    for _ in range(ATTEMPTS):
        await publish_limbs(h)
        if (await get_state_fx.poll(h, 5.0)).gripper_openings:
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
            final_position=request.position, final_orientation=request.orientation, ended_on_contact=False, action_time=1.0
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
        move_gripper.ResultResponseData(settled=True, final_opening=final_opening, action_time=0.5)
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


async def publish_wrist_until(h, done: asyncio.Event) -> None:
    """The wrist camera's stream: one grey 16 x 12 frame with a later stamp
    after another until `done`, so a step that waits for a frame captured
    after its move always gets one."""
    [camera] = h.mocks.deps.wrist_cameras
    assert await camera.video_stream.wait_for_subscriber(WAIT)
    stamp = 0
    while not done.is_set():
        stamp += 1
        await camera.video_stream.publish(
            wrist_video_stream.Message(
                header=wrist_video_stream.MessageHeader(timestamp=float(stamp), frame_id=stamp), encoding="rgb8", width=16, height=12,
                frame=bytes([128]) * (16 * 12 * 3),
            )
        )
        await asyncio.sleep(0.01)
