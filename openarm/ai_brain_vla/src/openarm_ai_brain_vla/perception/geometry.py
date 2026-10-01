"""Where the camera model comes from: its intrinsics from the camera
itself, through `camera_geometry:v1` on the `geometry` slot, and its pose
from the robot, through `camera_mounts:v1` on the `camera_mounts` slot.
get_color_intrinsics gives the pinhole the detections' pixels are in;
get_depth_intrinsics says what a depth sample measures; get_camera_poses
says where the camera named by the `camera_name` parameter stands in the
robot frame.

The slot is optional, like the camera's. Vacant, the intrinsics never come
and every search is refused naming the slot. Bound, both services are asked
again every `INTRINSICS_RETRY_S` until each has answered with success and
the answers are usable, and until then a search is refused with the last
reason they were not. The contract gives one answer, success false with a
message, both to a camera that has not heard from its simulation yet and
to a device that knows nothing of its lens, so the brain keeps asking in
both cases: the first answers within seconds, the second refuses for the
node's whole life and the log says so once. A depth model other than "z"
is a reason too: the brain reads a sample as the distance along the
optical axis, which is what the camera model deprojects. The intrinsics
are those of the colour stream, which the detections' pixels are in; the
depth is read at the same pixels, so the frame store refuses a pair of
frames that are not aligned.

The pose is asked the same way: until the robot answers with success and
names the camera, every search is refused with the reason, and the log
says it once. The pose is read once: the brain looks through a camera
fixed to the robot's base, and reports the values the robot's design
gives, exact in a simulation and nominal on hardware.
"""

from __future__ import annotations

import asyncio
import logging

from peppygen.consumed_services.camera_mounts import get_camera_poses
from peppygen.consumed_services.geometry import get_color_intrinsics, get_depth_intrinsics

from ..ports import Refusal
from ..waiting import unless_cancelled
from .camera import CameraModel, Intrinsics

logger = logging.getLogger(__name__)

INTRINSICS_TIMEOUT_S = 5.0
INTRINSICS_RETRY_S = 2.0
VACANT = "the geometry slot is vacant: link the camera's camera_geometry to it"
MOUNTS_VACANT = "the camera_mounts slot is vacant: link the robot's camera_mounts to it"
# The depth model the brain reads: a sample is the distance along the
# optical axis.
Z_DEPTH = "z"


def intrinsics_from(data) -> Intrinsics:
    """The camera model of a successful get_color_intrinsics answer; a
    ValueError names what is unusable."""
    return Intrinsics(
        int(data.width), int(data.height), float(data.fx), float(data.fy), float(data.cx), float(data.cy),
        str(data.distortion_model), tuple(float(c) for c in (data.distortion or ())),
    )


async def camera_intrinsics(node_runner, producer) -> Intrinsics:
    """One round of both questions: the colour intrinsics when the camera
    answers both with success, measures depth along the optical axis and
    reports a usable pinhole; a Refusal naming the reason otherwise."""
    try:
        color = (await get_color_intrinsics.poll(node_runner, producer, INTRINSICS_TIMEOUT_S)).data
        depth = (await get_depth_intrinsics.poll(node_runner, producer, INTRINSICS_TIMEOUT_S)).data
    except Exception as error:
        raise Refusal(f"the camera's geometry is not answered yet ({error!r})") from error
    if not color.success:
        raise Refusal(f"the camera does not know its colour intrinsics: {color.message}")
    if not depth.success:
        raise Refusal(f"the camera does not know its depth intrinsics: {depth.message}")
    if depth.depth_model != Z_DEPTH:
        raise Refusal(
            f"the camera's depth samples are '{depth.depth_model}' distances; the brain reads distances along the optical axis ('{Z_DEPTH}')"
        )
    try:
        return intrinsics_from(color)
    except ValueError as error:
        raise Refusal(f"the camera's intrinsics are unusable: {error}") from error


async def _learn(token, perceiver, reason_attribute: str, ask, learned) -> None:
    """Asks `ask` for the camera until it answers one, then gives it to the
    perceiver and logs it through `learned`; meanwhile the perceiver's
    `reason_attribute` carries the last reason the answer was refused."""
    reported = ""
    while not token.is_cancelled():
        try:
            camera = await ask()
        except Refusal as refusal:
            setattr(perceiver, reason_attribute, refusal.message)
            if refusal.message != reported:
                logger.info("%s; asking again every %g s", refusal.message, INTRINSICS_RETRY_S)
                reported = refusal.message
            await unless_cancelled(token, asyncio.sleep(INTRINSICS_RETRY_S))
            continue
        perceiver.set_camera(camera)
        learned(camera)
        return


async def learn_intrinsics(node_runner, token, perceiver) -> None:
    """Asks the geometry slot for the camera's intrinsics until it has
    them, and gives the perceiver its camera; meanwhile the perceiver
    carries the reason it has none."""
    producer = get_color_intrinsics.bound_producer(node_runner)
    if producer is None:
        perceiver.intrinsics_reason = VACANT
        return

    async def ask() -> CameraModel:
        # The perceiver's camera is read after the answer, so a pose the
        # other learner gave it meanwhile is kept.
        intrinsics = await camera_intrinsics(node_runner, producer)
        return perceiver.camera.with_intrinsics(intrinsics)

    def learned(camera: CameraModel) -> None:
        k = camera.intrinsics
        logger.info(
            "camera geometry: %dx%d fx %.1f fy %.1f cx %.1f cy %.1f %s",
            k.width, k.height, k.fx, k.fy, k.cx, k.cy, k.distortion_model,
        )

    await _learn(token, perceiver, "intrinsics_reason", ask, learned)


def camera_pose_from(data, camera_name: str):
    """The pose of `camera_name` in a get_camera_poses answer, as (position,
    orientation); a Refusal when the answer is no success or names no such
    camera."""
    if not data.success:
        raise Refusal(f"the robot does not know where its cameras stand: {data.message}")
    names = list(data.camera_names)
    if camera_name not in names:
        carried = ", ".join(names) if names else "none"
        raise Refusal(f"the robot carries no camera named '{camera_name}' (it carries: {carried})")
    index = names.index(camera_name)
    position = tuple(float(v) for v in data.positions[index * 3 : index * 3 + 3])
    orientation = tuple(float(v) for v in data.orientations[index * 4 : index * 4 + 4])
    return position, orientation


async def camera_pose(node_runner, producer, camera_name: str):
    """One question to the robot: where `camera_name` stands, or a Refusal
    naming why the answer is not usable."""
    try:
        answer = (await get_camera_poses.poll(node_runner, producer, INTRINSICS_TIMEOUT_S)).data
    except Exception as error:
        raise Refusal(f"the robot's camera mounts are not answered yet ({error!r})") from error
    return camera_pose_from(answer, camera_name)


async def learn_camera_pose(node_runner, token, perceiver, camera_name: str) -> None:
    """Asks the camera_mounts slot where the camera stands until the robot
    answers and names it, and gives the perceiver its camera; meanwhile the
    perceiver carries the reason it is not placed."""
    producer = get_camera_poses.bound_producer(node_runner)
    if producer is None:
        perceiver.pose_reason = MOUNTS_VACANT
        return

    async def ask() -> CameraModel:
        position, orientation = await camera_pose(node_runner, producer, camera_name)
        try:
            return perceiver.camera.with_pose(position, orientation)
        except ValueError as error:
            raise Refusal(str(error)) from error

    def learned(camera: CameraModel) -> None:
        logger.info(
            "camera '%s' stands at (%.3f, %.3f, %.3f) in the robot frame, looking along %s",
            camera_name, *camera.pose.position, tuple(round(v, 3) for v in camera.forward()),
        )

    await _learn(token, perceiver, "pose_reason", ask, learned)
