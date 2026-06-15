import asyncio
from pathlib import Path

import av
import numpy as np

from peppygen import NodeBuilder, NodeRunner
from peppygen.parameters import Parameters
from peppygen.consumed_services import camera_video_stream_info
from peppygen.consumed_topics import camera_video_stream


class VideoEncoder:
    """Encodes frames into the output mp4 incrementally, as they arrive.

    Encoding per frame (instead of one monolithic encode of the whole buffer
    at the end) keeps long synchronous work off the event loop, so closing
    the file at shutdown is just a fast flush that fits well within the
    shutdown grace window.
    """

    def __init__(self):
        self._container = None
        self._stream = None
        self._path: str | None = None
        self._frame_count = 0

    def open(self, width: int, height: int, fps: int):
        output_dir = Path("/tmp/video_reconstruction")
        output_dir.mkdir(parents=True, exist_ok=True)
        self._path = str(output_dir / "reconstructed_video.mp4")
        self._container = av.open(self._path, mode="w")
        self._stream = self._container.add_stream("h264", rate=fps)
        self._stream.width = width
        self._stream.height = height
        self._stream.pix_fmt = "yuv420p"

    def encode_frame(self, frame_data: bytes):
        rgb_array = np.frombuffer(frame_data, dtype=np.uint8).reshape(
            (self._stream.height, self._stream.width, 3)
        )
        video_frame = av.VideoFrame.from_ndarray(rgb_array, format="rgb24")
        video_frame.pts = self._frame_count
        self._frame_count += 1
        for packet in self._stream.encode(video_frame):
            self._container.mux(packet)

    def close(self) -> str | None:
        """Flush the encoder and close the container, producing a valid video.

        Idempotent; returns the output path when this call closed the file.
        """
        if self._container is None:
            return None
        for packet in self._stream.encode():
            self._container.mux(packet)
        self._container.close()
        self._container = None
        self._stream = None
        return self._path


async def setup(params: Parameters, node_runner: NodeRunner) -> list[asyncio.Task]:
    video_duration_seconds = params.video_duration_seconds
    encoder = VideoEncoder()

    # Awaited by the runtime before exit. Async so it runs on the node's event
    # loop, serialized with record_video (never mid-encode of a frame).
    async def finalize_video():
        path = encoder.close()
        if path is not None:
            print(f"Recording stopped by shutdown; partial video saved to: {path}")

    node_runner.on_shutdown(finalize_video)

    # Log when the shutdown/cancel signal is received so it is visible in the
    # node's stdout.
    async def announce_shutdown():
        print("[uvc_camera_video_reconstruction] Shutdown signal received")

    node_runner.on_shutdown(announce_shutdown)

    return [
        asyncio.create_task(record_video(node_runner, encoder, video_duration_seconds))
    ]


async def record_video(
    node_runner: NodeRunner, encoder: VideoEncoder, video_duration_seconds: int
):
    token = node_runner.cancellation_token()
    camera_info = None
    instance_id: str | None = None
    while camera_info is None:
        if token.is_cancelled():
            return
        try:
            response = await camera_video_stream_info.poll(node_runner, timeout=5.0)
            camera_info = response.data
            instance_id = response.instance_id
            print(
                f"Locked onto camera instance_id: {instance_id} — "
                f"{camera_info.width}x{camera_info.height} "
                f"@ {camera_info.frames_per_second} fps, encoding: {camera_info.encoding}"
            )
        except Exception as e:
            print(f"Failed to get camera info: {e}, retrying...")
            await asyncio.sleep(1)

    total_frames = video_duration_seconds * camera_info.frames_per_second
    print(
        f"Recording {total_frames} frames "
        f"({video_duration_seconds} seconds at {camera_info.frames_per_second} fps)..."
    )

    if token.is_cancelled():
        return
    encoder.open(camera_info.width, camera_info.height, camera_info.frames_per_second)

    for frame_num in range(total_frames):
        try:
            (
                _producer,
                message,
            ) = await camera_video_stream.on_next_message_received(node_runner)
            if token.is_cancelled():
                # Shutdown began; the finalize_video hook owns the flush/close.
                return
            encoder.encode_frame(message.frame)
            if (frame_num + 1) % camera_info.frames_per_second == 0:
                print(
                    f"Recorded {frame_num + 1}/{total_frames} frames "
                    f"({(frame_num + 1) // camera_info.frames_per_second} seconds)"
                )
        except Exception as e:
            print(f"Failed to record frame: {e}")

    print("Recording complete. Finalizing video...")

    try:
        path = encoder.close()
        print(f"Video saved to: {path}")
    except Exception as e:
        print(f"Failed to encode video: {e}")

    # One-shot job: the video is saved, so request node shutdown rather than
    # idling forever.
    token.cancel()


def main():
    NodeBuilder().run(setup)


if __name__ == "__main__":
    main()
