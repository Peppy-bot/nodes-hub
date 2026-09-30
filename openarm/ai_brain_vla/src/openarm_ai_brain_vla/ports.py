"""The seams of the brain: the plain data that crosses them, the two
interfaces a backend implements, the cancel token every long call takes,
and the deadline every search carries.

The core owns every contract rule. A backend only answers one question:
"what do you see in this image" (a `Detector`) or "do this with the arm"
(a `Manipulator`). Neither one mints item ids, touches the gripper table,
or completes a goal, so swapping a backend cannot break a contract rule.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]
# A region of a picture: x0, y0, x1, y1 in pixels, x to the right and y
# down, the corners of the box that holds an item.
Region = tuple[float, float, float, float]


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


class SearchTimeout(Exception):
    """A detector stopped at its deadline: the search has no answer, and
    the core refuses it naming the budget."""

    def __init__(self, budget_s: float) -> None:
        super().__init__(f"the search did not finish within {budget_s:g} s")
        self.budget_s = budget_s


@dataclass(frozen=True)
class Deadline:
    """When a search must have answered: an instant of the monotonic clock
    and the budget it was set from. A detector calls `check` between its
    stages, so a search ends at the next stage boundary after the deadline
    rather than running on."""

    at: float
    budget_s: float

    @classmethod
    def after(cls, budget_s: float) -> "Deadline":
        return cls(time.monotonic() + budget_s, budget_s)

    def remaining_s(self) -> float:
        return max(0.0, self.at - time.monotonic())

    def check(self) -> None:
        if time.monotonic() >= self.at:
            raise SearchTimeout(self.budget_s)


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
    def xyxy(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over union of two boxes given as x0, y0, x1, y1."""
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    overlap = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap
    return float(overlap / union) if union > 0.0 else 0.0


@dataclass(frozen=True)
class Detection:
    """A box turned into the world: the label, where the item is, how sure
    the detector was, and the box itself, where the item is in the picture
    the look used. Orientation is set only by a backend that resolves it;
    the contracts make it optional for that reason."""

    label: str
    position: Vec3
    confidence: float = 0.0
    orientation: Optional[Quat] = None
    region: Optional[Region] = None


@dataclass(frozen=True)
class Coverage:
    """The labels one look names with authority. A known item the look did
    not see is dropped only when the look covers its label, and an item the
    look saw takes the look's label only when the look covers it. So a scan
    that cannot name an item leaves it alone, and a search by description
    covers nothing: it never renames an item and never drops one.
    `every_label` is a look that can name anything, an open-vocabulary scan."""

    labels: frozenset[str] = frozenset()
    every_label: bool = False

    def covers(self, label: str) -> bool:
        return self.every_label or label in self.labels


@dataclass
class Item:
    """A known item: the id the brain minted for it and what was last seen.
    `region` is where the look that last saw it found it in the picture;
    none for an item the last look did not see, or one addressed by
    pose."""

    item_id: str
    label: str
    position: Vec3
    orientation: Optional[Quat]
    confidence: float
    seen_at_ns: int
    region: Optional[Region] = None


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
    core runs each in a thread of its own."""

    name: str
    # The confidence a detection is kept at, 0 to 1, in the backend's own
    # measure of it. The brain sets it from the perception_confidence
    # parameter before the load when that parameter is above zero.
    min_confidence: float

    @property
    def available(self) -> bool: ...

    def load(self, model: str, gallery: str) -> None:
        """`model` is the perception_model parameter, the identifier of a
        model the backend reads; `gallery` the perception_gallery parameter,
        a directory of pictures of particular items. Each backend says what
        it makes of them, and fails the load naming a value it does not
        take."""
        ...

    def set_vocabulary(self, phrases: Sequence[str]) -> None:
        """The labels to look for. Empty means the detector's own
        vocabulary, which is what a scan asks for."""
        ...

    def detect(self, image, deadline: Deadline) -> list[Box]:
        """`image` is an H x W x 3 uint8 RGB array. The detector checks
        `deadline` between its stages and raises `SearchTimeout` once it has
        passed."""
        ...

    def scan_coverage(self) -> Coverage:
        """The labels a scan, a search with the empty vocabulary, can name;
        nothing while the backend is not loaded."""
        ...


class Manipulator(Protocol):
    """The swappable piece of manipulation. Every move it makes goes
    through the `Robot` it is started with, so a stop can cancel the move
    in flight whatever the backend is."""

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
