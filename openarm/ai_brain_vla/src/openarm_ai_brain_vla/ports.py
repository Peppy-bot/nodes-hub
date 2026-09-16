"""The seams of the brain: the plain data that crosses them, the two
interfaces a backend implements, and the cancel token every long call
takes.

The core owns every contract rule. A backend only answers one question:
"what do you see in this image" (a `Detector`) or "do this with the arm"
(a `Manipulator`). Neither one mints item ids, touches the gripper table,
or completes a goal, so swapping a backend cannot break a contract rule.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]


class Refusal(Exception):
    """A goal the contracts say to refuse in the result, or a sequence that
    failed: `message` becomes the result's message and success is false."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class Cancelled(Exception):
    """A sequence stopped on request: by its own caller (a cancel on the
    goal handle) or by anyone (an abort). `by_caller` picks the completion
    the contract wants: cancelled for the first, a failed result for the
    second."""

    def __init__(self, reason: str, by_caller: bool) -> None:
        super().__init__(reason)
        self.reason = reason
        self.by_caller = by_caller


class CancelToken:
    """Set once, by whoever stops the sequence. Backends call `check()`
    between steps so a stop takes effect at the next step."""

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self.reason = ""
        self.by_caller = False

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self, reason: str, *, by_caller: bool) -> None:
        if not self._event.is_set():
            self.reason = reason
            self.by_caller = by_caller
            self._event.set()

    def check(self) -> None:
        if self.cancelled:
            raise Cancelled(self.reason, self.by_caller)

    async def wait(self) -> None:
        await self._event.wait()


@dataclass(frozen=True)
class Pose:
    """A grasp-point pose in the world frame limb_motion uses: meters, and
    a unit quaternion [x, y, z, w] or none when the caller left the choice
    to the brain."""

    position: Vec3
    orientation: Optional[Quat] = None


@dataclass(frozen=True)
class Box:
    """One detection in an image: a label from the detector's vocabulary,
    its confidence from 0 to 1, and the box in pixels, x to the right and
    y down, [x0, x1) by [y0, y1)."""

    label: str
    confidence: float
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def centre(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)

    @property
    def area(self) -> float:
        return max(0.0, self.x1 - self.x0) * max(0.0, self.y1 - self.y0)


@dataclass(frozen=True)
class Detection:
    """A box turned into the world: the label, where the item is, and how
    sure the detector was. Orientation is set only by a backend that
    resolves it; the contracts make it optional for that reason."""

    label: str
    position: Vec3
    confidence: float = 0.0
    orientation: Optional[Quat] = None


@dataclass
class Item:
    """A known item: the id the brain minted for it and what was last seen."""

    item_id: str
    label: str
    position: Vec3
    orientation: Optional[Quat]
    confidence: float
    seen_at_ns: int


@dataclass
class Gripper:
    """One gripper from the fixed set the launcher named, the arm that
    carries it, and the item it holds according to the results so far."""

    name: str
    arm: str
    held_item_id: Optional[str] = None

    @property
    def holding(self) -> bool:
        return self.held_item_id is not None


@dataclass(frozen=True)
class Outcome:
    """What a manipulation backend reports when a sequence ends. The
    measured grasp-point position when the jaws opened, for place_item."""

    success: bool
    message: str = ""
    final_position: Optional[Vec3] = None


class Detector(Protocol):
    """The swappable piece of perception: a model that finds boxes in one
    RGB image. The core does everything around it: frames, depth, the
    camera model, duplicate boxes, ids. `detect` and `load` may block; the
    core runs them off the event loop."""

    name: str

    @property
    def available(self) -> bool: ...

    def load(self, model: str) -> None: ...

    def set_vocabulary(self, phrases: Sequence[str]) -> None:
        """The labels to look for. Empty means the detector's own
        vocabulary, which is what a scan asks for."""
        ...

    def detect(self, image) -> list[Box]:
        """`image` is an H x W x 3 uint8 RGB array."""
        ...


class Manipulator(Protocol):
    """The swappable piece of manipulation. Every move it makes goes
    through the `Robot` it is started with, so a stop can cancel the move
    in flight whatever the backend is. A scripted sequence stands here
    first; a learned policy can stand here later."""

    name: str

    @property
    def available(self) -> bool: ...

    async def start(self, robot) -> None: ...

    async def grab(self, item: Item, gripper: Gripper, max_effort: float, cancel: CancelToken) -> Outcome: ...

    async def drop(self, gripper: Gripper, cancel: CancelToken) -> Outcome: ...

    async def place(self, gripper: Gripper, pose: Pose, cancel: CancelToken) -> Outcome: ...

    async def stop(self) -> None:
        """Halt whatever is in flight; the running call returns soon
        after, reporting it was stopped."""
        ...
