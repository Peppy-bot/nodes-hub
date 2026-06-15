import av
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


def get_source_video_fps(video_path: Path) -> int:
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    fps = stream.average_rate
    container.close()
    if fps and fps > 0:
        return round(float(fps))
    return 30  # Default fallback


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

    return [
        # Service to expose camera info
        asyncio.create_task(
            listen_for_video_stream_info_requests(node_runner, video_params, actual_fps)
        ),
        # Video loop
        asyncio.create_task(run_video_loop(node_runner, video_params)),
    ]


# Sentinel pushed onto the frame queue to wake a blocked consumer on shutdown.
_SHUTDOWN = object()


async def run_video_loop(node_runner: NodeRunner, video_params):
    print("[uvc_camera] Starting video loop...")
    video_path = ASSETS_DIR / "robot.mp4"

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    print(f"[uvc_camera] Video file found: {video_path}")

    width = video_params.resolution.width
    height = video_params.resolution.height
    encoding = video_params.topic_encoding
    frame_duration = 1.0 / video_params.frame_rate

    av_format = ENCODING_TO_AV_FORMAT[encoding]

    # The decoder runs on a worker thread so PyAV's synchronous decode and
    # reformat calls never block the asyncio event loop. Producer and consumer
    # are decoupled through a small, drop-oldest queue rather than a blocking
    # one: a slow or briefly-stalled consumer (for example, the first publish
    # waiting on subscriber discovery during a cold start) only causes stale
    # frames to be dropped, it can never wedge the decoder. This matches the
    # stream's SensorData QoS, where lagging subscribers are meant to miss
    # frames rather than apply backpressure. The handoff uses a thread-safe
    # `queue.Queue` (not an `asyncio.Queue` driven from the worker thread via
    # `run_coroutine_threadsafe`), so the pipeline never depends on a
    # cross-thread event-loop wake-up to make progress.
    frame_queue: queue.Queue = queue.Queue(maxsize=2)
    stop = threading.Event()

    def decode_forever():
        # `container.decode` selects the decoder that matches the file, so the
        # mock keeps working if the asset is ever re-encoded, and behaves the
        # same on macOS and Linux. Decoding is paced to the target frame rate
        # so the worker does not peg a CPU core decoding far ahead of what the
        # consumer emits (which would only get dropped anyway).
        while not stop.is_set():
            print("[uvc_camera] Opening video file for playback...")
            with av.open(str(video_path)) as container:
                for frame in container.decode(video=0):
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

    decoder_thread = threading.Thread(
        target=decode_forever, name="uvc-camera-decoder", daemon=True
    )
    decoder_thread.start()

    frame_id = 0
    last_print_time = time.monotonic()

    try:
        while True:
            # Wait for the next frame off the event loop so a blocking get
            # never stalls other coroutines (the health service, the info
            # service). The decoder paces production, so this returns at the
            # target frame rate without busy-polling.
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
    finally:
        stop.set()
        # Unblock a consumer parked in `to_thread(frame_queue.get)` so its
        # worker thread can finish instead of leaking until process exit.
        try:
            frame_queue.put_nowait(_SHUTDOWN)
        except queue.Full:
            pass


async def listen_for_video_stream_info_requests(
    node_runner: NodeRunner, video_params, actual_fps: int
):
    while True:
        try:
            await video_stream_info.handle_next_request(
                node_runner,
                lambda _request: video_stream_info.Response(
                    width=video_params.resolution.width,
                    height=video_params.resolution.height,
                    frames_per_second=actual_fps,
                    encoding=video_params.topic_encoding,
                ),
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
