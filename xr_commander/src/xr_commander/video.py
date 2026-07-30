"""Camera frames off the peppy bus and into the headset's WebRTC tracks.

One bound camera = one named track, keyed by instance id. Frames are handed to
teleop_xr as BGR: its track converts BGR-to-RGB before encoding.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from xr_commander.bus import CancellationToken, Latch, messages

# The colour encodings the rgb_camera / rgbd_camera contracts declare. A camera
# announcing anything else is refused rather than guessed at.
_BYTES_PER_PIXEL = {"rgb8": 3, "bgr8": 3, "yuyv": 2}


class FrameSink(Protocol):
    """Where decoded frames go (teleop_xr's ExternalVideoSource, in practice)."""

    def put_frame(self, frame: np.ndarray) -> None: ...


def decode_to_bgr(encoding: str, width: int, height: int, data: bytes) -> np.ndarray:
    """One contract frame as an HxWx3 BGR array; ValueError when undecodable.

    Raw encodings are checked against their declared geometry; mjpeg carries
    its own, so only the decode itself vouches for it.
    """
    if encoding == "mjpeg":
        frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("mjpeg frame failed to decode")
        return frame

    stride = _BYTES_PER_PIXEL.get(encoding)
    if stride is None:
        raise ValueError(f"unsupported encoding {encoding!r}")
    if width <= 0 or height <= 0:
        raise ValueError(f"degenerate frame geometry {width}x{height}")
    expected = width * height * stride
    if len(data) != expected:
        raise ValueError(
            f"{encoding} frame is {len(data)} bytes, expected {expected} "
            f"for {width}x{height}"
        )

    raw = np.frombuffer(data, dtype=np.uint8).reshape(height, width, stride)
    try:
        if encoding == "bgr8":
            # Copy: a frombuffer view aliases the wire buffer, which is
            # recycled while the encoder still holds the frame.
            return raw.copy()
        if encoding == "rgb8":
            return cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
        return cv2.cvtColor(raw, cv2.COLOR_YUV2BGR_YUY2)
    except cv2.error as e:
        raise ValueError(f"{encoding} frame failed to convert: {e}") from None


# One camera_views entry, matching teleop_xr's ViewConfig. It requires a
# capture device; the frame source is supplied directly, so it is never
# opened, and geometry travels with every frame.
CAMERA_VIEW = {"device": 0}


@dataclass(frozen=True)
class CameraTrack:
    """One camera's headset track.

    `instance_id` routes bus frames to the sink; `track_id` is the name the
    headset sees.
    """

    instance_id: str
    track_id: str
    sink: FrameSink


def camera_views(tracks: list[CameraTrack]) -> dict[str, dict]:
    """teleop_xr's camera panel config: one CAMERA_VIEW entry per track name."""
    return {t.track_id: dict(CAMERA_VIEW) for t in tracks}


def assert_unique_track_ids(tracks: list[CameraTrack]) -> None:
    """ValueError when two tracks share a headset name: the collision would
    silently drop one camera's frames."""
    track_ids = [t.track_id for t in tracks]
    if len(set(track_ids)) != len(track_ids):
        raise ValueError(f"camera track ids collide: {track_ids}")


def discover_tracks(
    node_runner,
    topic_module,
    make_sink: Callable[[], FrameSink],
    names: Mapping[str, str],
) -> list[CameraTrack]:
    """A track per camera bound to one slot, in binding order.

    Purely from the bindings: a camera that starts late streams into its
    waiting track whenever its first frame arrives.
    """
    return [
        CameraTrack(
            instance_id=producer.instance_id,
            track_id=names.get(producer.instance_id, producer.instance_id),
            sink=make_sink(),
        )
        for producer in topic_module.bound_producers(node_runner)
    ]


async def drain_frames(
    node_runner,
    topic_module,
    sinks: dict[str, FrameSink],
    token: CancellationToken,
    label: str,
) -> None:
    """Route every frame from one camera slot into its track's sink.

    Latest-value per producer, matching sensor_data QoS; an undecodable frame
    is logged once and dropped, never stopping the other producers.
    """
    subscription = await topic_module.subscribe(node_runner)
    # Latched per producer: a bad encoding repeats every frame.
    unusable: dict[str, Latch] = {}
    async for producer, message in messages(subscription, token, label):
        sink: FrameSink | None = sinks.get(producer.instance_id)
        if sink is None:
            continue  # a producer outside this slot's bound set
        latch = unusable.get(producer.instance_id)
        if latch is None:
            latch = unusable[producer.instance_id] = Latch()
        try:
            sink.put_frame(
                decode_to_bgr(
                    message.encoding, message.width, message.height, message.frame
                )
            )
            latch.clear()
        except Exception as e:
            latch.trip(f"{label} {producer.instance_id} frame unusable: {e!r}")
