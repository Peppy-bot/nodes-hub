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

from dataclasses import dataclass

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


@dataclass(frozen=True)
class Look:
    """What one look at the camera gave: the detections, each placed in the
    robot frame with its region in the picture, and the picture itself:
    its size in pixels, its capture time and its capture-pair counter, so a
    region can be read against the same frame. The detector's other boxes
    are kept for the log: each duplicate of a more confident box, and each
    box with no depth reading under its centre."""

    detections: list[Detection]
    image_width: int
    image_height: int
    frame_timestamp: float
    frame_id: int
    duplicates: list[Box]
    unplaced: list[Box]

    def describe(self, search_started_ns: int) -> str:
        """The look in a few words for the log: the frame, when the camera
        took it against the start of the search, and every box of the
        detector with its confidence, the placed ones first, then the
        duplicates and the boxes with no depth, each marked so."""
        age_s = search_started_ns / 1e9 - self.frame_timestamp
        if age_s >= 0.0:
            taken = f"{age_s:.2f} s before the search"
        else:
            taken = f"{-age_s:.2f} s after the search began"
        boxes = [f"{detection.label} {detection.confidence:.2f}" for detection in self.detections]
        boxes += [f"{box.label} {box.confidence:.2f} (duplicate)" for box in self.duplicates]
        boxes += [f"{box.label} {box.confidence:.2f} (no depth)" for box in self.unplaced]
        if not boxes:
            return f"frame {self.frame_id} taken {taken}, no box"
        count = "1 box" if len(boxes) == 1 else f"{len(boxes)} boxes"
        return f"frame {self.frame_id} taken {taken}, {count}: {', '.join(boxes)}"


class Perceiver:
    def __init__(self, detector: Detector, frames: FrameStore, camera: CameraModel) -> None:
        self.detector = detector
        self.frames = frames
        self.camera = camera
        # Why the camera cannot place a pixel yet: its intrinsics, until
        # the camera has answered, and its pose, until the robot has.
        self.intrinsics_reason = "the camera's intrinsics have not been received yet"
        self.pose_reason = "the camera's pose has not been received from the robot yet"
        self._loading: Optional[asyncio.Task] = None
        self._load_error = ""

    def set_camera(self, camera: CameraModel) -> None:
        self.camera = camera
        if camera.intrinsics is not None:
            self.intrinsics_reason = ""
        if camera.placed:
            self.pose_reason = ""

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
        if self.camera.intrinsics is None:
            return f"no perception source: {self.intrinsics_reason}"
        if not self.camera.placed:
            return f"no perception source: {self.pose_reason}"
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

    async def scan(self, phrases: Sequence[str], cancel: CancelToken, timeout_s: float) -> Look:
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
        kept, duplicates = merge_duplicates(boxes)
        detections: list[Detection] = []
        unplaced: list[Box] = []
        for box in kept:
            depth = depth_at(depth_m, box, frame.color.width, frame.color.height)
            if depth is None:
                unplaced.append(box)
                continue
            u, v = box.centre
            position = self.camera.deproject(u, v, depth, frame.color.width, frame.color.height)
            detections.append(
                Detection(label=box.label, position=position, confidence=box.confidence, region=box.xyxy)
            )
        return Look(
            detections=detections,
            image_width=int(frame.color.width),
            image_height=int(frame.color.height),
            frame_timestamp=float(frame.color.header.timestamp),
            frame_id=int(frame.color.header.frame_id),
            duplicates=duplicates,
            unplaced=unplaced,
        )


def merge_duplicates(boxes: Sequence[Box]) -> tuple[list[Box], list[Box]]:
    """Two boxes of one label that overlap by more than `DUPLICATE_IOU`
    are one item: the more confident box stays. The boxes that stay, then
    the duplicates, each list from the most confident box down."""
    kept: list[Box] = []
    duplicates: list[Box] = []
    for box in sorted(boxes, key=lambda b: b.confidence, reverse=True):
        if any(other.label == box.label and iou(other.xyxy, box.xyxy) > DUPLICATE_IOU for other in kept):
            duplicates.append(box)
            continue
        kept.append(box)
    return kept, duplicates
