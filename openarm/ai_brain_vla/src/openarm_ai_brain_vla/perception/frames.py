"""The frame store: the latest colour and depth frame from the camera
slot, kept by two small loops the node starts, and the decoding of those
frames into arrays the detector and the depth lookup take.

The camera slot is optional. When the launcher left it vacant the loops
return at once and the store stays empty, which the Perceiver reports as
"no perception source".
"""

from __future__ import annotations

import asyncio
import io
from dataclasses import dataclass
from typing import Optional

import numpy as np

from peppygen.consumed_services.camera import depth_stream_info
from peppygen.consumed_topics.camera import depth_stream, video_stream

from ..ports import Box, Refusal

DEPTH_INFO_TIMEOUT_S = 5.0
DEPTH_INFO_RETRY_S = 1.0
DEPTH_WINDOW_PX = 5


@dataclass(frozen=True)
class Frame:
    """A colour frame with the depth frame that goes with it and the depth
    unit, meters per depth value."""

    color: video_stream.Message
    depth: depth_stream.Message
    depth_unit: float


class FrameStore:
    def __init__(self) -> None:
        self.color: Optional[video_stream.Message] = None
        self.depth: Optional[depth_stream.Message] = None
        self.depth_unit: Optional[float] = None
        self.frames_seen = 0

    @property
    def available(self) -> bool:
        return self.color is not None and self.depth is not None and self.depth_unit is not None

    def latest(self) -> Optional[Frame]:
        if not self.available:
            return None
        return Frame(self.color, self.depth, self.depth_unit)

    async def run(self, node_runner, token) -> None:
        """Follows the camera slot until the node stops. Nothing to do when
        the slot is vacant."""
        producer = video_stream.bound_producer(node_runner)
        if producer is None:
            return
        await self._learn_depth_unit(node_runner, producer, token)
        if token.is_cancelled():
            return
        followers = [
            asyncio.create_task(self._follow_color(node_runner)),
            asyncio.create_task(self._follow_depth(node_runner)),
        ]
        cancelled = asyncio.ensure_future(token.cancelled())
        try:
            await asyncio.wait([cancelled, *followers], return_when=asyncio.FIRST_COMPLETED)
        finally:
            cancelled.cancel()
            for follower in followers:
                follower.cancel()

    async def _learn_depth_unit(self, node_runner, producer, token) -> None:
        while not token.is_cancelled():
            try:
                info = await depth_stream_info.poll(node_runner, producer, DEPTH_INFO_TIMEOUT_S)
                self.depth_unit = float(info.data.depth_unit)
                return
            except Exception as error:
                print(f"[brain] depth_stream_info not answered yet ({error!r}); retrying")
                await asyncio.sleep(DEPTH_INFO_RETRY_S)

    async def _follow_color(self, node_runner) -> None:
        subscription = await video_stream.subscribe(node_runner)
        async for _producer, message in subscription:
            self.color = message
            self.frames_seen += 1

    async def _follow_depth(self, node_runner) -> None:
        subscription = await depth_stream.subscribe(node_runner)
        async for _producer, message in subscription:
            self.depth = message


def decode_color(message: video_stream.Message) -> np.ndarray:
    """The colour frame as an H x W x 3 uint8 RGB array."""
    encoding = message.encoding.lower()
    width, height = message.width, message.height
    if encoding in ("rgb8", "bgr8"):
        expected = width * height * 3
        if len(message.frame) != expected:
            raise Refusal(f"colour frame holds {len(message.frame)} bytes, expected {expected} for {width}x{height} {encoding}")
        image = np.frombuffer(message.frame, dtype=np.uint8).reshape(height, width, 3)
        return image[:, :, ::-1] if encoding == "bgr8" else image
    if encoding in ("mjpeg", "jpeg", "jpg"):
        from PIL import Image

        with Image.open(io.BytesIO(message.frame)) as decoded:
            return np.asarray(decoded.convert("RGB"))
    raise Refusal(f"unsupported colour encoding '{message.encoding}'")


def decode_depth(message: depth_stream.Message, depth_unit: float) -> np.ndarray:
    """The depth frame in meters as an H x W float32 array; 0 where the
    camera had no reading."""
    encoding = message.encoding.lower()
    width, height = message.width, message.height
    if encoding != "z16":
        raise Refusal(f"unsupported depth encoding '{message.encoding}'")
    expected = width * height * 2
    if len(message.frame) != expected:
        raise Refusal(f"depth frame holds {len(message.frame)} bytes, expected {expected} for {width}x{height} z16")
    raw = np.frombuffer(message.frame, dtype="<u2").reshape(height, width)
    return raw.astype(np.float32) * float(depth_unit)


def depth_at(depth_m: np.ndarray, box: Box, color_width: int, color_height: int) -> Optional[float]:
    """The depth under a box's centre: the median of the valid readings in
    a small window, in the depth frame's own resolution. None when the
    window holds no reading."""
    height, width = depth_m.shape
    u, v = box.centre
    x = int(round(u * width / max(1, color_width)))
    y = int(round(v * height / max(1, color_height)))
    half = DEPTH_WINDOW_PX // 2
    x0, x1 = max(0, x - half), min(width, x + half + 1)
    y0, y1 = max(0, y - half), min(height, y + half + 1)
    if x0 >= x1 or y0 >= y1:
        return None
    window = depth_m[y0:y1, x0:x1]
    valid = window[window > 0.0]
    if valid.size == 0:
        return None
    return float(np.median(valid))
