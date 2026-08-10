"""Camera frames off the peppy bus and into the headset's WebRTC tracks.

One bound camera = one named track, keyed by instance id. Frames are handed to
teleop_xr as BGR: its track converts BGR-to-RGB before encoding.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from xr_commander.bus import CancellationToken, Latch, log, messages, ticks

# The colour encodings the rgb_camera / rgbd_camera contracts declare. A camera
# announcing anything else is refused rather than guessed at.
_BYTES_PER_PIXEL = {"rgb8": 3, "bgr8": 3, "yuyv": 2}

# A camera that has delivered nothing for this long gets its track blanked: in
# the headset a frozen frame is indistinguishable from live video.
_CAMERA_SILENT_S = 2.0
_SILENCE_POLL_S = 0.5
_NO_SIGNAL_HEIGHT = 180
_NO_SIGNAL_WIDTH = 320


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
    """One camera's headset track: the instance id is also the view name the
    headset sees."""

    instance_id: str
    sink: FrameSink


def camera_views(track_ids: Sequence[str]) -> dict[str, dict]:
    """teleop_xr's panel config: one CAMERA_VIEW entry per track name."""
    return {track_id: dict(CAMERA_VIEW) for track_id in track_ids}


def assert_unique_track_ids(track_ids: Sequence[str]) -> None:
    """ValueError when two tracks share a headset name: the collision would
    silently drop one track's frames."""
    duplicates = sorted({name for name in track_ids if track_ids.count(name) > 1})
    if duplicates:
        raise ValueError(
            f"headset view names collide: {duplicates}. Camera instance ids "
            "and the status panel's name must be unique"
        )


def discover_tracks(
    node_runner,
    topic_module,
    make_sink: Callable[[], FrameSink],
) -> list[CameraTrack]:
    """A track per camera bound to one slot, in binding order.

    Purely from the bindings: a camera that starts late streams into its
    waiting track whenever its first frame arrives.
    """
    return [
        CameraTrack(instance_id=producer.instance_id, sink=make_sink())
        for producer in topic_module.bound_producers(node_runner)
    ]


def shrink_to_width(frame: np.ndarray, max_width: int) -> np.ndarray:
    """The frame at most `max_width` wide, aspect kept; 0 disables.

    Headset panels render small at arm's length, so encoding sensor-native
    video wastes cores on the control path; consumers that need full
    resolution (the recorder) bind the cameras directly and never see this.
    """
    if max_width <= 0 or frame.shape[1] <= max_width:
        return frame
    scale = max_width / frame.shape[1]
    height = max(1, round(frame.shape[0] * scale))
    return cv2.resize(frame, (max_width, height), interpolation=cv2.INTER_AREA)


def _decode_and_shrink(message, view_max_width: int) -> np.ndarray:
    return shrink_to_width(
        decode_to_bgr(message.encoding, message.width, message.height, message.frame),
        view_max_width,
    )


async def drain_frames(
    node_runner,
    topic_module,
    sinks: dict[str, FrameSink],
    token: CancellationToken,
    label: str,
    view_max_width: int = 0,
    health: dict[str, float] | None = None,
) -> None:
    """Route every frame from one camera slot into its track's sink.

    Frames are handled in arrival order; the subscription's bounded queue is
    the only backpressure. Decode runs in a worker thread so a heavy frame
    never stalls the command streams sharing the loop. An undecodable frame is
    logged once and dropped, never stopping the other producers. `health`
    (instance id to monotonic seconds) is stamped per delivered frame for the
    silence watchdog.
    """
    try:
        subscription = await topic_module.subscribe(node_runner)
    except Exception as e:
        # Loud and fail-safe: the affected tracks simply never show video.
        log(f"{label} subscribe failed: {e!r}")
        return
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
            sink.put_frame(await asyncio.to_thread(_decode_and_shrink, message, view_max_width))
            if health is not None:
                health[producer.instance_id] = time.monotonic()
            latch.clear()
        except Exception as e:
            latch.trip(f"{label} {producer.instance_id} frame unusable: {e!r}")


def no_signal_frame(track_id: str) -> np.ndarray:
    """The frame a silent camera's track shows instead of its last image."""
    frame = np.zeros((_NO_SIGNAL_HEIGHT, _NO_SIGNAL_WIDTH, 3), dtype=np.uint8)
    cv2.putText(
        frame, "NO SIGNAL", (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2
    )
    cv2.putText(
        frame, track_id, (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (170, 170, 170), 1
    )
    return frame


async def watch_camera_silence(
    tracks: Sequence[CameraTrack],
    health: Mapping[str, float],
    token: CancellationToken,
    silent_after_s: float = _CAMERA_SILENT_S,
) -> None:
    """Blank a track whose camera stopped delivering, and say so once.

    Only a camera that has produced before is watched: a track with no frame
    yet shows nothing, which is already honest.
    """
    dark: set[str] = set()
    async for _ in ticks(_SILENCE_POLL_S, token):
        now = time.monotonic()
        for track in tracks:
            last = health.get(track.instance_id)
            if last is None:
                continue
            if now - last > silent_after_s:
                if track.instance_id not in dark:
                    dark.add(track.instance_id)
                    log(f"camera {track.instance_id} went silent; blanking its track")
                    try:
                        track.sink.put_frame(no_signal_frame(track.instance_id))
                    except Exception as e:
                        # One refusing sink must not end the watch for the rest.
                        log(f"camera {track.instance_id} blank failed: {e!r}")
            elif track.instance_id in dark:
                dark.discard(track.instance_id)
                log(f"camera {track.instance_id} recovered")
