"""Wiring for the ZED SDK rgbd_camera node.

A worker thread owns the grab loop (grab, retrieve left color and neural depth,
convert to wire bytes) and hands frames to the event loop through a drop-oldest
queue; the publish loop stamps each frame on the synchronized clock and emits
video_stream and depth_stream. The set_color_* services drive the SDK's camera
settings behind the shared camera lock. The heavy SDK calls run off the event
loop so the info and control services stay responsive.
"""

from __future__ import annotations

import asyncio
import queue
import threading

import peppygen.clock
from peppygen import NodeBuilder, NodeRunner
from peppygen.emitted_topics.rgbd_camera.v1 import depth_stream, video_stream
from peppygen.exposed_services.rgbd_camera.v1 import (
    depth_stream_info,
    set_color_brightness,
    set_color_contrast,
    set_color_exposure,
    set_color_gain,
    set_color_white_balance,
    video_stream_info,
)
from peppygen.parameters import Parameters

from . import conversions
from .zed import ZedCamera, ZedFrame

COLOR_ENCODING = "bgr8"
DEPTH_ENCODING = "z16"
# Depth is computed in the rectified-left (= published color) frame.
ALIGN_MODE = "depth_to_color"
# z16 depth is millimetres: 0.001 m per least-significant bit.
DEPTH_UNIT_M_PER_LSB = 0.001

# Bound the blocking queue get so the publish loop re-checks the cancellation
# token instead of parking forever on a non-daemon executor thread.
QUEUE_GET_TIMEOUT_SECONDS = 0.5
# How long shutdown waits for the grab worker to leave its current grab.
WORKER_STOP_TIMEOUT_SECONDS = 2.0

_SHUTDOWN = object()


def _offer_latest(frame_queue: "queue.Queue", item) -> None:
    """Drop-oldest offer: a lagging publish loop misses frames rather than
    applying backpressure to the grab worker (sensor_data QoS)."""
    try:
        frame_queue.put_nowait(item)
    except queue.Full:
        try:
            frame_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            frame_queue.put_nowait(item)
        except queue.Full:
            pass


async def _race_cancel(awaitable, token):
    """Await ``awaitable`` unless the node shuts down first; returns its result
    or None once the cancellation token fires."""
    task = asyncio.ensure_future(awaitable)
    cancelled = asyncio.ensure_future(token.cancelled())
    done, _pending = await asyncio.wait(
        {task, cancelled}, return_when=asyncio.FIRST_COMPLETED
    )
    cancelled.cancel()
    if task not in done:
        task.cancel()
        return None
    return task.result()


def _grab_forever(
    camera: ZedCamera, frame_queue: "queue.Queue", stop: threading.Event
) -> None:
    while not stop.is_set():
        frame = camera.grab()
        if frame is not None:
            _offer_latest(frame_queue, frame)


async def _publish_loop(
    node_runner: NodeRunner, frame_queue: "queue.Queue", geometry: tuple[int, int]
) -> None:
    width, height = geometry
    token = node_runner.cancellation_token()
    color_pub = await video_stream.declare_publisher(node_runner)
    depth_pub = await depth_stream.declare_publisher(node_runner)
    frame_id = 0
    while not token.is_cancelled():
        try:
            item = await asyncio.to_thread(
                frame_queue.get, True, QUEUE_GET_TIMEOUT_SECONDS
            )
        except queue.Empty:
            continue
        if item is _SHUTDOWN:
            break
        frame: ZedFrame = item
        stamp = peppygen.clock.now_ns() / 1e9
        color_header = video_stream.MessageHeader(
            stamp=stamp, frame_id=frame_id, align_mode=ALIGN_MODE
        )
        depth_header = depth_stream.MessageHeader(
            stamp=stamp, frame_id=frame_id, align_mode=ALIGN_MODE
        )
        await color_pub.publish(
            video_stream.build_message(
                color_header, COLOR_ENCODING, frame.width, frame.height, frame.bgr
            )
        )
        await depth_pub.publish(
            depth_stream.build_message(
                depth_header, DEPTH_ENCODING, frame.width, frame.height, frame.depth_z16
            )
        )
        frame_id = (frame_id + 1) % (2**32)


def _serve(service, handler):
    """One long-running task that answers a service until the node shuts down."""

    async def run(node_runner: NodeRunner) -> None:
        token = node_runner.cancellation_token()
        while not token.is_cancelled():
            try:
                await _race_cancel(
                    service.handle_next_request(node_runner, handler), token
                )
            except Exception as e:
                print(f"[zed_camera_sdk] {service.SERVICE_NAME} error: {e!r}")

    return run


def _stream_info_handlers(width: int, height: int, fps: int):
    def video_info(_request):
        return video_stream_info.Response(
            width=width, height=height, frames_per_second=fps, encoding=COLOR_ENCODING
        )

    def depth_info(_request):
        return depth_stream_info.Response(
            width=width,
            height=height,
            frames_per_second=fps,
            encoding=DEPTH_ENCODING,
            depth_unit=DEPTH_UNIT_M_PER_LSB,
        )

    return video_info, depth_info


def _color_service_handlers(camera: ZedCamera):
    async def exposure(request):
        mode = conversions.color_mode(request.data.mode)
        ok, message, current = await asyncio.to_thread(
            camera.set_exposure, mode, request.data.value
        )
        return set_color_exposure.Response(
            success=ok, message=message, current_value=current
        )

    async def white_balance(request):
        mode = conversions.color_mode(request.data.mode)
        ok, message, current = await asyncio.to_thread(
            camera.set_white_balance, mode, request.data.temperature
        )
        return set_color_white_balance.Response(
            success=ok, message=message, current_temperature=current
        )

    async def gain(request):
        ok, message, current = await asyncio.to_thread(
            camera.set_gain, request.data.value
        )
        return set_color_gain.Response(
            success=ok, message=message, current_value=current
        )

    async def brightness(request):
        ok, message, current = await asyncio.to_thread(
            camera.set_brightness, request.data.value
        )
        return set_color_brightness.Response(
            success=ok, message=message, current_value=current
        )

    async def contrast(request):
        ok, message, current = await asyncio.to_thread(
            camera.set_contrast, request.data.value
        )
        return set_color_contrast.Response(
            success=ok, message=message, current_value=current
        )

    return exposure, white_balance, gain, brightness, contrast


async def setup(params: Parameters, node_runner: NodeRunner) -> list[asyncio.Task]:
    resolution_member = conversions.resolution_member(params.resolution)
    fps = conversions.validate_frame_rate(params.resolution, params.frame_rate)
    depth_mode_member = conversions.depth_mode_member(params.depth_mode)
    min_depth_mm = conversions.min_depth_mm(params.min_depth_m)

    await peppygen.clock.init(node_runner)

    camera = await asyncio.to_thread(
        ZedCamera.open,
        resolution_member,
        fps,
        depth_mode_member,
        min_depth_mm,
        params.serial_number,
    )

    # One grab establishes the negotiated frame geometry the info services
    # report; the worker keeps grabbing fresh frames after this.
    first = await asyncio.to_thread(camera.grab)
    if first is None:
        camera.close()
        raise RuntimeError("ZED opened but the first grab returned no frame")
    width, height = first.width, first.height
    print(
        f"[zed_camera_sdk] {params.resolution} ({width}x{height}) @ {fps} fps, depth {params.depth_mode}"
    )

    frame_queue: queue.Queue = queue.Queue(maxsize=2)
    stop = threading.Event()
    worker = asyncio.create_task(
        asyncio.to_thread(_grab_forever, camera, frame_queue, stop)
    )

    async def shutdown():
        stop.set()
        _offer_latest(frame_queue, _SHUTDOWN)
        try:
            await asyncio.wait_for(asyncio.shield(worker), WORKER_STOP_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            print("[zed_camera_sdk] grab worker still stopping; leaving it to teardown")
        await asyncio.to_thread(camera.close)

    node_runner.on_shutdown(shutdown)

    video_info, depth_info = _stream_info_handlers(width, height, fps)
    exposure, white_balance, gain, brightness, contrast = _color_service_handlers(
        camera
    )
    services = [
        _serve(video_stream_info, video_info),
        _serve(depth_stream_info, depth_info),
        _serve(set_color_exposure, exposure),
        _serve(set_color_white_balance, white_balance),
        _serve(set_color_gain, gain),
        _serve(set_color_brightness, brightness),
        _serve(set_color_contrast, contrast),
    ]
    return [
        asyncio.create_task(_publish_loop(node_runner, frame_queue, (width, height))),
        worker,
    ] + [asyncio.create_task(run(node_runner)) for run in services]


def main():
    NodeBuilder().run(setup)


if __name__ == "__main__":
    main()
