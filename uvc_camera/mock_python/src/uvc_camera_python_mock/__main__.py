import av
from av.codec.context import ThreadType
import asyncio
import json
import queue
import threading
import time
from importlib.resources import files
from pathlib import Path


from peppygen import NodeBuilder, NodeRunner, StandaloneConfig
from peppygen.exposed_services.uvc_camera.v1 import video_stream_info
from peppygen.emitted_topics.uvc_camera.v1 import video_stream
from peppygen.emitted_topics.uvc_camera.v1.video_stream import MessageHeader
from peppygen.parameters import Parameters

from uvc_camera_python_mock.frames import offer_latest_frame

ASSETS_DIR = Path(__file__).resolve().parent / "assets"

ENCODING_TO_AV_FORMAT = {
    "rgb8": "rgb24",
    "rgb": "rgb24",
}

# Sentinel pushed onto the frame queue at shutdown to wake the consumer if it is
# parked in `to_thread(frame_queue.get)` waiting on a decoder that has stopped.
_SHUTDOWN = object()


def get_source_video_fps(video_path: Path) -> int:
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    fps = stream.average_rate
    container.close()
    if fps and fps > 0:
        return round(float(fps))
    return 30  # Default fallback


async def wait_unless_cancelled(awaitable, token):
    """Await `awaitable`, racing it against node shutdown.

    Returns its result, or None once the cancellation token fires first, so
    long-running loops stop working instead of relying on the runtime's
    post-hook task cancellation.
    """
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


async def setup(params: Parameters, node_runner: NodeRunner) -> list[asyncio.Task]:
    video_params = params.video

    print(
        f"[uvc_camera] Video params: {video_params.resolution.width}x{video_params.resolution.height} "
        f"@ {video_params.frame_rate} fps, encoding: {video_params.topic_encoding}"
    )

    encoding = video_params.topic_encoding
    if encoding not in ENCODING_TO_AV_FORMAT:
        raise ValueError(
            f"Invalid encoding '{encoding}'. "
            f"Supported encodings: {', '.join(ENCODING_TO_AV_FORMAT)}"
        )

    # Probe source video to get its actual frame rate
    video_path = ASSETS_DIR / "robot.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    actual_fps = get_source_video_fps(video_path)
    print(f"[uvc_camera] Detected source video frame rate: {actual_fps} fps")

    width = video_params.resolution.width
    height = video_params.resolution.height
    av_format = ENCODING_TO_AV_FORMAT[encoding]
    frame_duration = 1.0 / video_params.frame_rate

    # Producer and consumer are decoupled through a small, drop-oldest queue
    # rather than a blocking one: a slow or briefly-stalled consumer (for
    # example, the first publish waiting on subscriber discovery during a cold
    # start) only causes stale frames to be dropped, it can never wedge the
    # decoder. This matches the stream's SensorData QoS, where lagging
    # subscribers are meant to miss frames rather than apply backpressure. The
    # handoff uses a thread-safe `queue.Queue` so the pipeline never depends on
    # a cross-thread event-loop wake-up to make progress.
    frame_queue: queue.Queue = queue.Queue(maxsize=2)
    # Stop signal for the decoder thread. Cancelling its asyncio wrapper task
    # cannot interrupt a thread that is already running, so the thread checks
    # this event between frames and the shutdown hook below sets it.
    stop = threading.Event()

    def decode_forever():
        # PyAV's decode and reformat calls are synchronous and hold the GIL for
        # tens of milliseconds at a time — long enough on the Lima VM used on
        # macOS to starve the asyncio event loop and make peppylib's native
        # health service miss heartbeats. Running the pipeline on a worker
        # thread keeps the event loop free. Decoding is paced to the target
        # frame rate so the worker does not peg a CPU core decoding far ahead of
        # what the consumer emits (which would only get dropped anyway).
        while not stop.is_set():
            print("[uvc_camera] Opening video file for playback...")
            with av.open(str(video_path)) as container:
                stream = container.streams.video[0]
                # Force single-threaded decoding. The asset is AV1, decoded by
                # libdav1d, whose default multithreaded path (thread_count =
                # logical cores, plus frame-level parallelism) can deadlock on
                # its internal worker pool when the CPU is saturated. That is
                # exactly what happens during a cold-start `stack launch`, where
                # every node spins up at once on the small build VM: the decoder
                # wedged on the very first frame, so the node never emitted and
                # was killed for being silent (this is the hang at "Opening
                # video file for playback...", not reproducible in isolation).
                # Single-threaded decode removes the worker pool entirely and so
                # the deadlock window with it; a 640x480 frame still decodes in
                # ~6 ms even with the 1080p reformat, far inside the frame
                # budget. Must be set before the first decode() call, while the
                # codec context is still closed.
                stream.codec_context.thread_type = ThreadType.NONE
                stream.codec_context.thread_count = 1
                # `container.decode(stream)` selects the decoder that matches the
                # file, so the mock keeps working if the asset is ever
                # re-encoded, and behaves the same on macOS and Linux.
                for frame in container.decode(stream):
                    if stop.is_set():
                        return
                    rgb_frame = frame.reformat(
                        width=width, height=height, format=av_format
                    )
                    # Read packed bytes directly from the plane to avoid a
                    # numpy dependency.
                    offer_latest_frame(frame_queue, bytes(rgb_frame.planes[0]))
                    time.sleep(frame_duration)
            print("[uvc_camera] Video ended, restarting from beginning...")

    decoder_task = asyncio.create_task(asyncio.to_thread(decode_forever))

    async def stop_decoder():
        # The decoder runs on asyncio's default executor, whose non-daemon
        # worker threads are joined at interpreter exit: the thread must be told
        # to stop and seen to finish (closing the av container) before the
        # runtime tears the node down.
        stop.set()
        await decoder_task
        # The decoder has stopped producing, so wake the consumer if it is
        # parked in `to_thread(frame_queue.get)`; otherwise its worker thread
        # would block until interpreter exit.
        try:
            frame_queue.put_nowait(_SHUTDOWN)
        except queue.Full:
            pass

    node_runner.on_shutdown(stop_decoder)

    # Log when the shutdown/cancel signal is received so it is visible in the
    # node's stdout.
    async def announce_shutdown():
        print("[uvc_camera] Shutdown signal received")

    node_runner.on_shutdown(announce_shutdown)

    return [
        # Service to expose camera info
        asyncio.create_task(
            listen_for_video_stream_info_requests(node_runner, video_params, actual_fps)
        ),
        # Video loop
        asyncio.create_task(run_video_loop(node_runner, video_params, frame_queue)),
    ]


async def run_video_loop(
    node_runner: NodeRunner, video_params, frame_queue: queue.Queue
):
    print("[uvc_camera] Starting video loop...")

    width = video_params.resolution.width
    height = video_params.resolution.height
    encoding = video_params.topic_encoding

    token = node_runner.cancellation_token()

    frame_id = 0
    last_print_time = time.monotonic()

    while not token.is_cancelled():
        # Wait for the next frame off the event loop so a blocking get never
        # stalls other coroutines (the health service, the info service). The
        # decoder paces production, so this returns at the target frame rate
        # without busy-polling, and the shutdown hook pushes _SHUTDOWN to wake
        # this get once the decoder has stopped.
        data = await asyncio.to_thread(frame_queue.get)
        if data is _SHUTDOWN:
            break

        header = MessageHeader(
            stamp=time.time(),
            frame_id=frame_id,
        )

        await video_stream.emit(node_runner, header, encoding, width, height, data)

        if time.monotonic() - last_print_time >= 3:
            print(f"[uvc_camera] Emitted frame {frame_id}")
            last_print_time = time.monotonic()

        frame_id = (frame_id + 1) % (2**32)


async def listen_for_video_stream_info_requests(
    node_runner: NodeRunner, video_params, actual_fps: int
):
    token = node_runner.cancellation_token()
    while not token.is_cancelled():
        try:
            await wait_unless_cancelled(
                video_stream_info.handle_next_request(
                    node_runner,
                    lambda _request: video_stream_info.Response(
                        width=video_params.resolution.width,
                        height=video_params.resolution.height,
                        frames_per_second=actual_fps,
                        encoding=video_params.topic_encoding,
                    ),
                ),
                token,
            )
        except Exception as e:
            print(f"get_camera_info service error: {e}")


def main():
    # Fallback configuration for standalone execution (e.g., `uv run`).
    # Ignored when the node is launched by the peppy daemon, which provides its own parameters.
    standalone_config = StandaloneConfig()

    mock_params_path = files("uvc_camera_python_mock") / "mock_parameters.json"
    if mock_params_path.is_file():
        mock_params = json.loads(mock_params_path.read_text())
        standalone_config = standalone_config.with_parameters(mock_params)

    NodeBuilder().standalone(standalone_config).run(setup)


if __name__ == "__main__":
    main()
