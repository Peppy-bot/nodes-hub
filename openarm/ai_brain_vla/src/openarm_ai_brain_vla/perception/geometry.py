"""Where the camera model's intrinsics come from: the camera itself, through
`camera_geometry:v1`'s get_color_intrinsics on the `geometry` slot.

The slot is optional, like the camera's. Vacant, the intrinsics never come
and every search is refused naming the slot. Bound, the service is asked
until it answers; a camera that refuses (a UVC camera knows nothing of its
lens) is final and the refusal's message is the reason; an answer that is
not a usable pinhole is refused too. The intrinsics are those of the colour
stream, which the detections' pixels are in; the depth is read at the same
pixels, so the depth stream has to be aligned to colour, as the sim relays
and the ZED publish it and as realsense_d4xx does under depth_to_color.
"""

from __future__ import annotations

import asyncio

from peppygen.consumed_services.geometry import get_color_intrinsics

from .camera import Intrinsics

INTRINSICS_TIMEOUT_S = 5.0
INTRINSICS_RETRY_S = 2.0
VACANT = "the geometry slot is vacant: link the camera's camera_geometry to it"


def intrinsics_from(data) -> Intrinsics:
    """The camera model of a successful get_color_intrinsics answer; a
    ValueError names what is unusable."""
    return Intrinsics(
        int(data.width), int(data.height), float(data.fx), float(data.fy), float(data.cx), float(data.cy),
        str(data.distortion_model), tuple(float(c) for c in (data.distortion or ())),
    )


async def learn_intrinsics(node_runner, token, perceiver) -> None:
    """Asks the geometry slot for the colour intrinsics until it answers, and
    gives the perceiver its camera, or the reason it has none."""
    producer = get_color_intrinsics.bound_producer(node_runner)
    if producer is None:
        perceiver.camera_reason = VACANT
        return
    while not token.is_cancelled():
        try:
            answer = await get_color_intrinsics.poll(node_runner, producer, INTRINSICS_TIMEOUT_S)
        except Exception as error:
            perceiver.camera_reason = f"get_color_intrinsics not answered yet ({error!r})"
            await asyncio.sleep(INTRINSICS_RETRY_S)
            continue
        data = answer.data
        if not data.success:
            perceiver.camera_reason = f"the camera refuses its intrinsics: {data.message}"
            print(f"[brain] {perceiver.camera_reason}")
            return
        try:
            intrinsics = intrinsics_from(data)
        except ValueError as error:
            perceiver.camera_reason = f"the camera's intrinsics are unusable: {error}"
            print(f"[brain] {perceiver.camera_reason}")
            return
        perceiver.set_camera(perceiver.camera.with_intrinsics(intrinsics))
        print(
            f"[brain] camera geometry: {intrinsics.width}x{intrinsics.height} fx {intrinsics.fx:.1f} fy {intrinsics.fy:.1f} "
            f"cx {intrinsics.cx:.1f} cy {intrinsics.cy:.1f} {intrinsics.distortion_model}"
        )
        return
