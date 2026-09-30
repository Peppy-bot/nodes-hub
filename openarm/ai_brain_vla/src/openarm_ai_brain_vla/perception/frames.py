"""The frame store: the colour and depth frames the camera slot publishes,
kept by two small loops the node starts, paired by capture, and decoded
into the arrays the detector and the depth lookup take.

Colour and depth are paired by `header.frame_id`, the capture-pair counter
rgbd_camera:v1 gives both streams, so the depth under a box is the depth of
the instant the picture was taken. The last `FRAME_BUFFER` frames of each
stream are kept, so a stream a few frames ahead of the other still pairs.
The depth is read at the colour pixels, so the two streams must be
aligned: a pair whose frames name `align_mode` "none", or two different
alignments, is refused with the reason, never deprojected.

The camera slot is optional. When the launcher left it vacant the loops
return at once and the store stays empty, which every search reports as
"no perception source".
"""

from __future__ import annotations

import asyncio
import io
import logging
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from peppygen.consumed_services.camera import depth_stream_info
from peppygen.consumed_topics.camera import depth_stream, video_stream

from ..ports import Box, Refusal
from ..waiting import unless_cancelled

logger = logging.getLogger(__name__)

DEPTH_INFO_TIMEOUT_S = 5.0
DEPTH_INFO_RETRY_S = 1.0
DEPTH_WINDOW_PX = 5
# Frames of each stream kept for pairing.
FRAME_BUFFER = 8
# The alignments under which colour and depth share one viewpoint, as the
# camera contracts name them.
ALIGNED_MODES = ("depth_to_color", "color_to_depth")


@dataclass(frozen=True)
class Frame:
    """A colour frame with the depth frame of the same capture and the
    depth unit, meters per depth value."""

    color: video_stream.Message
    depth: depth_stream.Message
    depth_unit: float

    @classmethod
    def paired(cls, color: video_stream.Message, depth: depth_stream.Message, depth_unit: float) -> "Frame":
        """The two frames as one, or a Refusal when their streams are not
        aligned to each other."""
        mode = color.header.align_mode
        if depth.header.align_mode != mode:
            raise Refusal(f"the colour and depth frames name different alignments ('{mode}' and '{depth.header.align_mode}')")
        if mode not in ALIGNED_MODES:
            raise Refusal(f"the camera's depth is not aligned to its colour (align_mode '{mode}'); the brain reads depth at colour pixels")
        return cls(color, depth, depth_unit)


class FrameStore:
    def __init__(self) -> None:
        self.colors: deque[video_stream.Message] = deque(maxlen=FRAME_BUFFER)
        self.depths: deque[depth_stream.Message] = deque(maxlen=FRAME_BUFFER)
        self.depth_unit: Optional[float] = None
        self.frames_seen = 0

    def add_color(self, message: video_stream.Message) -> None:
        self.colors.append(message)
        self.frames_seen += 1

    def add_depth(self, message: depth_stream.Message) -> None:
        self.depths.append(message)

    def latest(self) -> Frame:
        """The newest colour frame with the depth frame of its capture, or
        a Refusal naming what is missing: a stream, the depth unit, a pair
        of one capture, or the alignment."""
        if not self.colors:
            raise Refusal("no camera frame received")
        if not self.depths:
            raise Refusal("no depth frame received")
        if self.depth_unit is None:
            raise Refusal("the camera has not answered depth_stream_info yet")
        depths = {message.header.frame_id: message for message in self.depths}
        for color in reversed(self.colors):
            depth = depths.get(color.header.frame_id)
            if depth is not None:
                return Frame.paired(color, depth, self.depth_unit)
        raise Refusal(
            f"no colour and depth frame of one capture received (latest colour frame_id {self.colors[-1].header.frame_id}, "
            f"depth {self.depths[-1].header.frame_id})"
        )

    def why_unavailable(self) -> str:
        """Why `latest` would refuse, or the empty string."""
        try:
            self.latest()
        except Refusal as refusal:
            return refusal.message
        return ""

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
        try:
            await unless_cancelled(token, asyncio.wait(followers, return_when=asyncio.FIRST_COMPLETED))
        finally:
            for follower in followers:
                follower.cancel()

    async def _learn_depth_unit(self, node_runner, producer, token) -> None:
        while not token.is_cancelled():
            try:
                info = await depth_stream_info.poll(node_runner, producer, DEPTH_INFO_TIMEOUT_S)
            except Exception as error:
                logger.info("depth_stream_info not answered yet (%r); asking again in %g s", error, DEPTH_INFO_RETRY_S)
                await unless_cancelled(token, asyncio.sleep(DEPTH_INFO_RETRY_S))
                continue
            self.depth_unit = float(info.data.depth_unit)
            return

    async def _follow_color(self, node_runner) -> None:
        subscription = await video_stream.subscribe(node_runner)
        async for _producer, message in subscription:
            self.add_color(message)

    async def _follow_depth(self, node_runner) -> None:
        subscription = await depth_stream.subscribe(node_runner)
        async for _producer, message in subscription:
            self.add_depth(message)


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
