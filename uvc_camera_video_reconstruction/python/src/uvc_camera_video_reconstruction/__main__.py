import asyncio
from pathlib import Path

import av
import numpy as np

from peppygen import NodeBuilder, NodeRunner
from peppygen.parameters import Parameters
from peppygen.consumed_services import camera_stream_video_stream_info
from peppygen.consumed_topics import camera_stream_video_stream


async def setup(params: Parameters, node_runner: NodeRunner) -> list[asyncio.Task]:
    video_duration_seconds = params.video_duration_seconds
    return [asyncio.create_task(record_video(node_runner, video_duration_seconds))]


async def record_video(node_runner: NodeRunner, video_duration_seconds: int):
    camera_info = None
    instance_id: str | None = None
    while camera_info is None:
        try:
            response = await camera_stream_video_stream_info.poll(
                node_runner, timeout=5.0
            )
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

    frames: list[bytes] = []
    for frame_num in range(total_frames):
        try:
            (
                _instance_id,
                message,
            ) = await camera_stream_video_stream.on_next_message_received(
                node_runner, target_instance_id=instance_id
            )
            frames.append(message.frame)
            if (frame_num + 1) % camera_info.frames_per_second == 0:
                print(
                    f"Recorded {frame_num + 1}/{total_frames} frames "
                    f"({(frame_num + 1) // camera_info.frames_per_second} seconds)"
                )
        except Exception as e:
            print(f"Failed to receive frame: {e}")

    print("Recording complete. Encoding video...")

    try:
        path = encode_video(
            frames, camera_info.width, camera_info.height, camera_info.frames_per_second
        )
        print(f"Video saved to: {path}")
    except Exception as e:
        print(f"Failed to encode video: {e}")


def encode_video(frames: list[bytes], width: int, height: int, fps: int) -> str:
    output_dir = Path("/tmp/video_reconstruction")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / "reconstructed_video.mp4")

    container = av.open(output_path, mode="w")
    stream = container.add_stream("h264", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"

    for i, frame_data in enumerate(frames):
        rgb_array = np.frombuffer(frame_data, dtype=np.uint8).reshape(
            (height, width, 3)
        )
        video_frame = av.VideoFrame.from_ndarray(rgb_array, format="rgb24")
        video_frame.pts = i

        for packet in stream.encode(video_frame):
            container.mux(packet)

    # Flush encoder
    for packet in stream.encode():
        container.mux(packet)

    container.close()

    print(
        f"Video encoding complete: {width}x{height} @ {fps} fps, saved to {output_path}"
    )
    return output_path


def main():
    NodeBuilder().run(setup)


if __name__ == "__main__":
    main()
