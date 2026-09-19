"""The core Perceiver: frames in, world detections out. The detector
behind it is the only swappable piece; the frames, the depth lookup, the
camera model and the duplicate rule live here, once, so two detectors
cannot drift apart on geometry.

A scan asks the detector for its whole vocabulary. An identify search
asks for the description's phrase and lets the core pick the best match
afterwards (`state.best_match`), so no detector implements selection.
"""

from __future__ import annotations

import asyncio
from typing import Sequence

from ..ports import Box, CancelToken, Detection, Detector, Refusal
from .camera import CameraModel
from .frames import FrameStore, decode_color, decode_depth, depth_at

DEFAULT_TIMEOUT_S = 10.0
DUPLICATE_IOU = 0.5


class Perceiver:
    def __init__(self, detector: Detector, frames: FrameStore, camera: CameraModel) -> None:
        self.detector = detector
        self.frames = frames
        self.camera = camera

    @property
    def available(self) -> bool:
        return self.detector.available and self.frames.available

    def why_unavailable(self) -> str:
        if not self.detector.available:
            return f"no perception source: perception_backend is '{self.detector.name}'"
        if not self.frames.available:
            return "no perception source: no camera frame received"
        return ""

    async def load(self, model: str) -> None:
        await asyncio.to_thread(self.detector.load, model)

    async def scan(self, phrases: Sequence[str], cancel: CancelToken, timeout_s: float) -> list[Detection]:
        """Looks once at the latest frame for `phrases`, or for everything
        the detector knows when `phrases` is empty."""
        reason = self.why_unavailable()
        if reason:
            raise Refusal(reason)
        frame = self.frames.latest()
        assert frame is not None
        image = decode_color(frame.color)
        depth_m = decode_depth(frame.depth, frame.depth_unit)
        self.detector.set_vocabulary(list(phrases))
        timeout = timeout_s if timeout_s > 0.0 else DEFAULT_TIMEOUT_S
        try:
            boxes = await asyncio.wait_for(asyncio.to_thread(self.detector.detect, image), timeout)
        except asyncio.TimeoutError:
            raise Refusal(f"the search did not finish within {timeout:g} s") from None
        cancel.check()
        detections: list[Detection] = []
        for box in merge_duplicates(boxes):
            depth = depth_at(depth_m, box, frame.color.width, frame.color.height)
            if depth is None:
                continue
            u, v = box.centre
            position = self.camera.deproject(u, v, depth, frame.color.width, frame.color.height)
            detections.append(Detection(label=box.label, position=position, confidence=box.confidence))
        return detections


def merge_duplicates(boxes: Sequence[Box]) -> list[Box]:
    """Two boxes of one label that overlap by more than `DUPLICATE_IOU`
    are one item: the more confident box stays."""
    kept: list[Box] = []
    for box in sorted(boxes, key=lambda b: b.confidence, reverse=True):
        if any(other.label == box.label and iou(other, box) > DUPLICATE_IOU for other in kept):
            continue
        kept.append(box)
    return kept


def iou(a: Box, b: Box) -> float:
    x0, y0 = max(a.x0, b.x0), max(a.y0, b.y0)
    x1, y1 = min(a.x1, b.x1), min(a.y1, b.y1)
    overlap = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = a.area + b.area - overlap
    return overlap / union if union > 0.0 else 0.0
