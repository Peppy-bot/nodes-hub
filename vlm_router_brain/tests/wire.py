"""The node under the generated harness: the answers its camera and robot
mocks give the brain at its start, one capture of colour and depth, and the
goals and services sent over the wire, each read by how it ended."""

from __future__ import annotations

import numpy as np

from peppygen.consumed_services.camera import depth_stream_info
from peppygen.consumed_services.camera_mounts import get_camera_poses
from peppygen.consumed_services.geometry import get_color_intrinsics, get_depth_intrinsics
from peppygen.consumed_topics.camera import depth_stream, video_stream
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
    "manipulation_model": "none",
    "camera_name": "chest",
}


def params(**overrides) -> Parameters:
    return Parameters.from_dict({**PARAMS, **overrides})


def answer_the_camera(h) -> None:
    """The depth unit, the 16 x 12 camera's intrinsics with a 90 degree
    lens, and the chest camera at the robot's origin looking along +Z."""
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
    h.mocks.deps.camera_mounts.get_camera_poses.enqueue_response(
        get_camera_poses.ResponseData(
            timestamp=7.0, camera_names=["chest"], positions=[0.0, 0.0, 0.0], orientations=[0.0, 0.0, 0.0, 1.0], carried_by=[""],
        )
    )


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
