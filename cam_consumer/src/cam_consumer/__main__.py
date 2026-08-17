import asyncio

from peppygen import NodeBuilder, NodeRunner
from peppygen.parameters import Parameters
from peppygen.consumed_services.camera import video_stream_info as camera_video_stream_info
from peppygen.consumed_topics.camera import video_stream as camera_video_stream

INFO_POLL_TIMEOUT_SECONDS = 5.0
INFO_RETRY_DELAY_SECONDS = 1.0


async def fetch_camera_info(node_runner: NodeRunner):
    """Poll `video_stream_info` until the camera answers or shutdown begins."""
    token = node_runner.cancellation_token()
    while not token.is_cancelled():
        try:
            response = await camera_video_stream_info.poll(
                node_runner,
                camera_video_stream_info.bound_producer(node_runner),
                timeout=INFO_POLL_TIMEOUT_SECONDS,
            )
            print(f"[cam_consumer] Locked onto camera instance_id: {response.instance_id}")
            return response.data
        except Exception as e:
            print(f"[cam_consumer] Failed to get camera info: {e}, retrying...")
            await asyncio.sleep(INFO_RETRY_DELAY_SECONDS)
    return None


async def consume_video_stream(node_runner: NodeRunner, camera_info) -> None:
    """Count and log frames as they arrive on `video_stream`."""
    token = node_runner.cancellation_token()
    print(
        f"[cam_consumer] Camera stream: {camera_info.width}x{camera_info.height} "
        f"@ {camera_info.frames_per_second} fps, encoding: {camera_info.encoding}"
    )

    # Subscribe once; the held subscription buffers frames in order, so the
    # loop never misses a frame published between iterations.
    subscription = await camera_video_stream.subscribe(node_runner)

    # One log line per second of video; the contract cannot express fps 0, but
    # a misbehaving camera must not turn the stride into a zero divide.
    log_stride = max(camera_info.frames_per_second, 1)
    frames_seen = 0
    while not token.is_cancelled():
        try:
            received = await subscription.next()
        except Exception as e:
            print(f"[cam_consumer] Failed to receive frame: {e}")
            continue
        if received is None:
            break  # subscription closed
        _producer, message = received
        frames_seen += 1
        if frames_seen % log_stride == 0:
            print(
                f"[cam_consumer] Received {frames_seen} frames "
                f"({message.width}x{message.height}, encoding: {message.encoding})"
            )

    print("[cam_consumer] Frame stream ended (shutdown requested)")


async def consume_camera(node_runner: NodeRunner) -> None:
    camera_info = await fetch_camera_info(node_runner)
    if camera_info is None:
        return
    await consume_video_stream(node_runner, camera_info)


async def setup(params: Parameters, node_runner: NodeRunner) -> list[asyncio.Task]:
    # Log when the shutdown/cancel signal is received so it is visible in the
    # node's stdout.
    async def announce_shutdown():
        print("[cam_consumer] Shutdown signal received")

    node_runner.on_shutdown(announce_shutdown)

    return [
        asyncio.create_task(consume_camera(node_runner)),
    ]


def main():
    NodeBuilder().run(setup)


if __name__ == "__main__":
    main()
