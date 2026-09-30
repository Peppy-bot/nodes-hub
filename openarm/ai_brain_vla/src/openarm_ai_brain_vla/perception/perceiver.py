"""The core Perceiver: frames in, world detections out. The detector
behind it is the only swappable piece; the frames, the depth lookup, the
camera model and the duplicate rule live here, once, so two detectors
cannot drift apart on geometry.

A scan asks the detector for its whole vocabulary. An identify search
asks for the description's phrase and lets the core pick the best match
afterwards (`state.best_match`), so no detector implements selection.

Both blocking calls of a detector, the load and a search, run in a daemon
thread of their own rather than in the event loop's executor: the
executor's threads are joined when the process exits, and a load that
takes a minute would hold the node past its shutdown window, while a
daemon thread dies with the process. A search is awaited until its thread
returns, whatever its deadline: the detector keeps the deadline between
its stages, and the lane the search runs in stays busy until the thread
has returned, so no second search starts on the same models while the
first still runs.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Callable, Optional, Sequence

from ..ports import Box, CancelToken, Deadline, Detection, Detector, Refusal, SearchTimeout, iou
from .camera import CameraModel
from .frames import FrameStore, decode_color, decode_depth, depth_at

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 10.0
DUPLICATE_IOU = 0.5


async def in_daemon_thread(function: Callable, *args):
    """Runs `function(*args)` in a daemon thread and returns what it
    returned, or raises what it raised."""
    loop = asyncio.get_running_loop()
    done: asyncio.Future = loop.create_future()

    def settle(result, error: Optional[Exception]) -> None:
        if done.done():
            return
        if error is None:
            done.set_result(result)
        else:
            done.set_exception(error)

    def run() -> None:
        try:
            outcome = (function(*args), None)
        except Exception as error:
            outcome = (None, error)
        try:
            loop.call_soon_threadsafe(settle, *outcome)
        except RuntimeError:
            # The loop is closed: the node stopped while the call ran, and
            # nothing awaits the outcome any more.
            pass

    threading.Thread(target=run, name=f"brain-{function.__name__}", daemon=True).start()
    return await done


class Perceiver:
    def __init__(self, detector: Detector, frames: FrameStore, camera: CameraModel) -> None:
        self.detector = detector
        self.frames = frames
        self.camera = camera
        # Why the camera cannot place a pixel yet, until its intrinsics come.
        self.camera_reason = "the camera's intrinsics have not been received yet"
        self._loading: Optional[asyncio.Task] = None
        self._load_error = ""

    def set_camera(self, camera: CameraModel) -> None:
        self.camera = camera
        if camera.ready:
            self.camera_reason = ""

    @property
    def available(self) -> bool:
        return self.why_unavailable() == ""

    def why_unavailable(self) -> str:
        if not self.detector.available:
            if self._loading is not None and not self._loading.done():
                return f"no perception source: perception_backend '{self.detector.name}' is still loading"
            if self._load_error:
                return f"no perception source: {self._load_error}"
            return f"no perception source: perception_backend is '{self.detector.name}'"
        frames = self.frames.why_unavailable()
        if frames:
            return f"no perception source: {frames}"
        if not self.camera.ready:
            return f"no perception source: {self.camera_reason}"
        return ""

    async def load(self, model: str, gallery: str) -> None:
        """Loads the backend's model. A failure is raised, and kept as the
        reason every search is refused with from then on."""
        try:
            await in_daemon_thread(self.detector.load, model, gallery)
        except Exception as error:
            self._load_error = f"perception_backend '{self.detector.name}' could not load: {error}"
            raise

    def start_loading(self, model: str, gallery: str) -> asyncio.Task:
        """Begins the load in the background and returns its task. A backend
        that takes a minute to load must not hold the node's start: the node
        is healthy at once and refuses searches as still loading until the
        load ends, or with the failure if it fails."""

        async def run() -> None:
            try:
                await self.load(model, gallery)
            except Exception:
                logger.error("%s", self._load_error)

        self._loading = asyncio.create_task(run())
        return self._loading

    async def loaded(self) -> None:
        """Waits for a load begun with `start_loading` to end."""
        if self._loading is not None:
            await self._loading

    async def scan(self, phrases: Sequence[str], cancel: CancelToken, timeout_s: float) -> list[Detection]:
        """Looks once at the latest frame for `phrases`, or for everything
        the detector knows when `phrases` is empty, within `timeout_s`, the
        default budget when that is zero."""
        reason = self.why_unavailable()
        if reason:
            raise Refusal(reason)
        frame = self.frames.latest()
        image = decode_color(frame.color)
        depth_m = decode_depth(frame.depth, frame.depth_unit)
        self.detector.set_vocabulary(list(phrases))
        deadline = Deadline.after(timeout_s if timeout_s > 0.0 else DEFAULT_TIMEOUT_S)
        try:
            boxes = await in_daemon_thread(self.detector.detect, image, deadline)
        except SearchTimeout as timeout:
            raise Refusal(str(timeout)) from None
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
        if any(other.label == box.label and iou(other.xyxy, box.xyxy) > DUPLICATE_IOU for other in kept):
            continue
        kept.append(box)
    return kept
