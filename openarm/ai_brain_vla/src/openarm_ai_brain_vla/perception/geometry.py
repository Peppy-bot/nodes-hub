"""Where the camera model's intrinsics come from: the camera itself, through
`camera_geometry:v1` on the `geometry` slot. get_color_intrinsics gives the
pinhole the detections' pixels are in; get_depth_intrinsics says what a
depth sample measures.

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
"""

from __future__ import annotations

import asyncio
import logging

from peppygen.consumed_services.geometry import get_color_intrinsics, get_depth_intrinsics

from ..ports import Refusal
from ..waiting import unless_cancelled
from .camera import Intrinsics

logger = logging.getLogger(__name__)

INTRINSICS_TIMEOUT_S = 5.0
INTRINSICS_RETRY_S = 2.0
VACANT = "the geometry slot is vacant: link the camera's camera_geometry to it"
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


async def learn_intrinsics(node_runner, token, perceiver) -> None:
    """Asks the geometry slot for the camera's intrinsics until it has
    them, and gives the perceiver its camera; meanwhile the perceiver
    carries the reason it has none."""
    producer = get_color_intrinsics.bound_producer(node_runner)
    if producer is None:
        perceiver.camera_reason = VACANT
        return
    reported = ""
    while not token.is_cancelled():
        try:
            intrinsics = await camera_intrinsics(node_runner, producer)
        except Refusal as refusal:
            perceiver.camera_reason = refusal.message
            if refusal.message != reported:
                logger.info("%s; asking again every %g s", refusal.message, INTRINSICS_RETRY_S)
                reported = refusal.message
            await unless_cancelled(token, asyncio.sleep(INTRINSICS_RETRY_S))
            continue
        perceiver.set_camera(perceiver.camera.with_intrinsics(intrinsics))
        logger.info(
            "camera geometry: %dx%d fx %.1f fy %.1f cx %.1f cy %.1f %s",
            intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy, intrinsics.distortion_model,
        )
        return
