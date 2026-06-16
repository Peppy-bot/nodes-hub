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

from uvc_camera_python_mock.frames import offer_latest_frame, force_put

ASSETS_DIR = Path(__file__).resolve().parent / "assets"

ENCODING_TO_AV_FORMAT = {
    "rgb8": "rgb24",
    "rgb": "rgb24",
}

# Sentinel pushed onto the frame queue at shutdown to wake the consumer if it is
# parked in `to_thread(frame_queue.get)` waiting on a decoder that has stopped.
_SHUTDOWN = object()

# How long the shutdown hook waits for the decoder thread to finish its current
# frame and exit. The decoder checks `stop` between frames and decode is fast,
# so this is only approached if the decoder is briefly inside a native call; the
# worker is non-daemon, so interpreter finalization joins it regardless.
DECODER_STOP_TIMEOUT_SECONDS = 2.0
# Upper bound on a single blocking `frame_queue.get`, so the consumer re-checks
# the cancellation token periodically and its (non-daemon) worker can never park
# forever, which would otherwise stall interpreter finalization at shutdown.
CONSUMER_GET_TIMEOUT_SECONDS = 0.5


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

    # Declare the video_stream publisher once. This takes the central messenger
    # lock a single time; every per-frame publish below is then lock-free. The
    # previous per-call publish path re-took that lock on every frame, which
    # under a cold-start `stack launch` (the messenger busy exposing services
    # and discovering the subscriber) serialized the 30fps publish loop against
    # the rest of the node's messaging and was the prime suspect for the silent
    # cold-start stall.
    publisher = await video_stream.declare_publisher(node_runner)

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
        # tens of milliseconds at a time, long enough on the Lima VM used on
        # macOS to starve the asyncio event loop. Running the pipeline on a
        # worker thread keeps the loop free, and building the wire payload here
        # (not on the loop thread) keeps the multi-megabyte per-frame capnp
        # serialization off the loop's GIL time as well. Decoding is paced to
        # the target frame rate so the worker does not peg a CPU core decoding
        # far ahead of what the consumer emits (which would only get dropped).
        frame_id = 0
        while not stop.is_set():
            print("[uvc_camera] Opening video file for playback...")
            with av.open(str(video_path)) as container:
                stream = container.streams.video[0]
                # Force single-threaded decoding. The asset is AV1, decoded by
                # libdav1d, whose default multithreaded path (thread_count =
                # logical cores, plus frame-level parallelism) can deadlock on
                # its internal worker pool when the CPU is saturated during a
                # cold-start `stack launch`. Single-threaded decode removes that
                # worker pool entirely; a frame still decodes in a few ms even
                # with the 1080p reformat, far inside the frame budget. Must be
                # set before the first decode() call, while the codec context is
                # still closed.
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
                    # Serialize on this worker thread, off the event loop. Read
                    # packed bytes directly from the plane to avoid a numpy
                    # dependency.
                    header = MessageHeader(stamp=time.time(), frame_id=frame_id)
                    payload = video_stream.build_message(
                        header, encoding, width, height, bytes(rgb_frame.planes[0])
                    )
                    offer_latest_frame(frame_queue, payload)
                    frame_id = (frame_id + 1) % (2**32)
                    time.sleep(frame_duration)
            print("[uvc_camera] Video ended, restarting from beginning...")

    decoder_task = asyncio.create_task(asyncio.to_thread(decode_forever))

    async def stop_decoder():
        # The decoder and the consumer (run_video_loop's frame_queue.get) both
        # run on asyncio's default ThreadPoolExecutor, whose worker threads are
        # NOT daemons: CPython joins them at interpreter finalization, so either
        # one left blocked there stalls process exit until the daemon force-kills
        # the node. Make both exit promptly. A daemon thread is deliberately
        # avoided: one wedged inside a native libav call would be force-unwound
        # at finalization, which can crash the process.
        stop.set()
        # Wake the consumer first and unconditionally, before waiting on the
        # decoder, so a slow or stuck decoder can never leave it parked in
        # frame_queue.get. force_put empties the queue so the sentinel is never
        # lost to a queue.Full.
        force_put(frame_queue, _SHUTDOWN)
        # The decoder returns within one frame of seeing `stop`; bound the wait
        # so a decoder briefly inside a native call cannot consume the whole
        # grace window. shield() keeps wait_for from cancelling it mid native
        # call; if it does not return in time it is left to finalization.
        try:
            await asyncio.wait_for(
                asyncio.shield(decoder_task), DECODER_STOP_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            print("[uvc_camera] Decoder still stopping; leaving it to teardown")

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
        asyncio.create_task(run_video_loop(node_runner, frame_queue, publisher)),
    ]


async def run_video_loop(node_runner: NodeRunner, frame_queue: queue.Queue, publisher):
    print("[uvc_camera] Starting video loop...")

    token = node_runner.cancellation_token()

    emitted = 0
    last_print_time = time.monotonic()

    while not token.is_cancelled():
        # Wait for the next frame off the event loop so a blocking get never
        # stalls other coroutines (the health service, the info service). The
        # get is bounded so this consumer's worker re-checks the cancellation
        # token and can never park forever on a non-daemon executor thread; the
        # shutdown hook also pushes _SHUTDOWN to wake it immediately.
        try:
            data = await asyncio.to_thread(
                frame_queue.get, True, CONSUMER_GET_TIMEOUT_SECONDS
            )
        except queue.Empty:
            continue
        if data is _SHUTDOWN:
            break

        # `data` is the wire payload already serialized on the decoder worker;
        # the loop thread only performs the lock-free publish.
        await publisher.publish(data)

        emitted += 1
        if time.monotonic() - last_print_time >= 3:
            print(f"[uvc_camera] Emitted frame {emitted}")
            last_print_time = time.monotonic()


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
