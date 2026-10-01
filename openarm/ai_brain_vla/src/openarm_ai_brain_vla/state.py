"""The brain's memory and the contract rules about names: the grippers,
the known items, and how a goal's names resolve or get refused. Plain
data and pure functions, no robot calls, so every rule has a fast test.

Rules written down here because the code depends on them:

- Item ids. An id is `<label>_<n>-<run token>`. The run token is drawn
  once per brain process, so after a restart an id from the earlier run is
  refused as unknown instead of naming whatever item this run numbered the
  same. `n` counts per label stem and only grows, so a dropped id never
  comes back.
- An item is the thing at its place. A detection keeps the id of a known
  item within `MATCH_RADIUS_M` of it: one with the detection's label
  first, then the nearest. The label only breaks ties, because one object
  gets different labels from different searches (the vocabulary's name in
  a scan, the caller's words in an identify search). Any other detection
  mints a new id. An item a gripper holds is in the gripper, not where it
  was grabbed, so no detection matches it. The cost: an item swapped for
  another within the radius between two looks hands its id to the new one.
- What a look drops and renames is its `Coverage`. A scan drops a known
  item it did not see when it could have named it, unless a gripper holds
  it, so grab_item cannot be sent to an item that left the table; an item
  only a description finds survives it. An identify search covers nothing:
  it refreshes what it found and drops nothing. An item that moved further
  than the radius gets a new id, and a covering scan drops the old one.
- Released items. A place_item moves the item to the pose it was put at,
  so the next scan keeps its id. A drop_item leaves it at the position it
  was grabbed at, since nothing measured where it landed.
- Regions. An item carries the region of the look that last saw it, in
  that look's picture. A look that did not see a known item it keeps
  clears the region: the picture has moved on, and a region of an older
  picture would be reported against the new one.
- Holding follows results, not jaws. A gripper holds an item after a
  successful grab and stops holding after a successful drop or place.
  Refusals, failures, cancels and aborts leave the flag as it was. A grab
  of an item a gripper holds is refused naming that gripper: the item is
  in its jaws, not at the position it was grabbed at.
- Gripper to arm. The backbone names both by side, so `left_gripper`
  drives with `left_arm`. That is `arm_of`, the one function to change for
  a robot that names its limbs differently.
"""

from __future__ import annotations

import math
import secrets
from typing import Iterable, Optional, Sequence

from .ports import Coverage, Detection, Gripper, Item, Quat, Refusal, Vec3
from .words import named_by

MATCH_RADIUS_M = 0.05


def new_run_token() -> str:
    """The token every id of one brain process carries: six hex digits, so
    two runs share one with a chance of 1 in 16.7 million."""
    return secrets.token_hex(3)


def arm_of(gripper_name: str) -> str:
    """The limb_motion arm name for a gripper name, by the backbone's side
    naming: `<side>_gripper` drives with `<side>_arm`."""
    suffix = "_gripper"
    if gripper_name.endswith(suffix):
        return gripper_name[: -len(suffix)] + "_arm"
    return gripper_name + "_arm"


def distance(a: Vec3, b: Vec3) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def best_match(detections: Sequence[Detection], description: str) -> Optional[Detection]:
    """The one detection a description names, the rule identify_item uses.
    With a description, the highest confidence among the detections whose
    label it names (`words.named_by`: the label is the description, else
    holds every word of it, stopwords aside); without one, the highest
    confidence of all, the item the detector judged most prominent."""
    if not detections:
        return None
    if not description.strip():
        return max(detections, key=lambda d: d.confidence)
    named = named_by(description, [d.label for d in detections])
    if not named:
        return None
    return max((detections[i] for i in named), key=lambda d: d.confidence)


class State:
    def __init__(self, gripper_names: Iterable[str], run_token: str, match_radius_m: float = MATCH_RADIUS_M) -> None:
        names = [name.strip() for name in gripper_names if name.strip()]
        if not names:
            raise ValueError("gripper_names must name at least one gripper")
        if len(set(names)) != len(names):
            raise ValueError("gripper_names must be distinct")
        if not run_token.isalnum():
            raise ValueError(f"run_token must be letters and digits, got {run_token!r}")
        self.grippers: dict[str, Gripper] = {name: Gripper(name=name, arm=arm_of(name)) for name in names}
        self.items: dict[str, Item] = {}
        self.run_token = run_token
        self.match_radius_m = match_radius_m
        self._counts: dict[str, int] = {}

    # Grippers

    def gripper_names(self) -> list[str]:
        return list(self.grippers)

    def gripper_for_grab(self, name: str, near: Optional[Vec3]) -> Gripper:
        """The gripper grab_item closes: the named one, which must exist and
        be free, or a free one suited to the item's position when the name
        is empty. Suited means on the item's side: world +Y is the robot's
        left, so a `left` gripper takes items with y >= 0 and a `right` one
        the rest; with one free gripper, that one."""
        if name:
            gripper = self.grippers.get(name)
            if gripper is None:
                raise Refusal(f"unknown gripper '{name}'")
            if gripper.holding:
                raise Refusal(f"gripper '{name}' already holds item '{gripper.held_item_id}'")
            return gripper
        free = [gripper for gripper in self.grippers.values() if not gripper.holding]
        if not free:
            raise Refusal("no gripper is free")
        if near is None or len(free) == 1:
            return free[0]
        return min(free, key=lambda gripper: _side_cost(gripper.name, near))

    def holder(self, name: str) -> Gripper:
        """The gripper drop_item or place_item releases: the named one,
        which must hold something, or the one gripper holding an item when
        the name is empty."""
        if name:
            gripper = self.grippers.get(name)
            if gripper is None:
                raise Refusal(f"unknown gripper '{name}'")
            if not gripper.holding:
                raise Refusal(f"gripper '{name}' holds nothing")
            return gripper
        holders = [gripper for gripper in self.grippers.values() if gripper.holding]
        if len(holders) == 1:
            return holders[0]
        if not holders:
            raise Refusal("no gripper holds an item")
        raise Refusal("more than one gripper holds an item; name the gripper")

    def set_held(self, gripper: Gripper, item_id: str) -> None:
        gripper.held_item_id = item_id

    def clear_held(self, gripper: Gripper) -> str:
        """Releases what the gripper holds and returns its id. The item
        keeps the position it was grabbed at."""
        item_id = gripper.held_item_id or ""
        gripper.held_item_id = None
        return item_id

    def clear_held_placed(self, gripper: Gripper, position: Vec3, orientation: Optional[Quat]) -> str:
        """Releases what the gripper holds at the pose place_item put it at
        and returns its id. The item takes that pose, so the next scan finds
        it where it stands and keeps its id."""
        item_id = self.clear_held(gripper)
        item = self.items.get(item_id)
        if item is not None:
            item.position = position
            item.orientation = orientation
        return item_id

    def snapshot(self) -> tuple[list[str], list[bool], list[str]]:
        """The three index-aligned arrays get_state reports."""
        grippers = list(self.grippers.values())
        return (
            [gripper.name for gripper in grippers],
            [gripper.holding for gripper in grippers],
            [gripper.held_item_id or "" for gripper in grippers],
        )

    # Items

    def item(self, item_id: str) -> Item:
        item = self.items.get(item_id)
        if item is None:
            raise Refusal(f"unknown item '{item_id}'")
        return item

    def item_to_grab(self, item_id: str) -> Item:
        """The known item grab_item closes on: it must exist, and no gripper
        may hold it already."""
        item = self.item(item_id)
        holder = self.holder_of(item_id)
        if holder is not None:
            raise Refusal(f"item '{item_id}' is held by gripper '{holder.name}'")
        return item

    def holder_of(self, item_id: str) -> Optional[Gripper]:
        return next((gripper for gripper in self.grippers.values() if gripper.held_item_id == item_id), None)

    def remember(self, detections: Sequence[Detection], now_ns: int, *, coverage: Coverage) -> list[Item]:
        """Folds one look's detections into the known items and returns
        them in the detections' order. `coverage` is what the look names
        with authority: the unseen items it drops, except held ones, and
        the seen items whose label it replaces."""
        held = self._held_ids()
        seen: list[Item] = []
        claimed: set[str] = set()
        for detection in detections:
            item = self._match(detection, claimed, held)
            if item is None:
                item = self._mint(detection.label, detection.position, detection.orientation, detection.confidence, now_ns)
                item.region = detection.region
            else:
                _refresh(item, detection, now_ns, coverage)
            claimed.add(item.item_id)
            seen.append(item)
        for item_id, item in list(self.items.items()):
            if item_id in claimed:
                continue
            if item_id not in held and coverage.covers(item.label):
                del self.items[item_id]
            else:
                item.region = None
        return seen

    def mint_from_pose(self, position: Vec3, orientation: Optional[Quat], now_ns: int) -> Item:
        """An id for an item a goal addressed by pose rather than by id."""
        return self._mint("item", position, orientation, 0.0, now_ns)

    def _held_ids(self) -> set[str]:
        return {gripper.held_item_id for gripper in self.grippers.values() if gripper.held_item_id is not None}

    def _match(self, detection: Detection, claimed: set[str], held: set[str]) -> Optional[Item]:
        """The known item at the detection's place: among the items within
        the match radius that no earlier detection of this look claimed and
        no gripper holds, one with the detection's label first, then the
        nearest."""
        near = [
            item
            for item in self.items.values()
            if item.item_id not in claimed
            and item.item_id not in held
            and distance(item.position, detection.position) <= self.match_radius_m
        ]
        if not near:
            return None
        return min(near, key=lambda item: (item.label != detection.label, distance(item.position, detection.position)))

    def _mint(self, label: str, position: Vec3, orientation: Optional[Quat], confidence: float, now_ns: int) -> Item:
        stem = _stem(label)
        count = self._counts.get(stem, 0) + 1
        self._counts[stem] = count
        item = Item(
            item_id=f"{stem}_{count}-{self.run_token}",
            label=label,
            position=position,
            orientation=orientation,
            confidence=confidence,
            seen_at_ns=now_ns,
        )
        self.items[item.item_id] = item
        return item


def _refresh(item: Item, detection: Detection, now_ns: int, coverage: Coverage) -> None:
    """A known item as a look saw it again: where it is now, and the look's
    label when the look names that label with authority."""
    item.position = detection.position
    item.orientation = detection.orientation
    item.confidence = detection.confidence
    item.seen_at_ns = now_ns
    item.region = detection.region
    if coverage.covers(detection.label):
        item.label = detection.label


def _stem(label: str) -> str:
    """A label as an id stem: lower case, spaces and punctuation as one
    underscore, `item` when nothing is left."""
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in label.strip().lower())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    cleaned = cleaned.strip("_")
    return cleaned or "item"


def _side_cost(gripper_name: str, near: Vec3) -> int:
    name = gripper_name.lower()
    y = near[1]
    if "left" in name:
        return 0 if y >= 0.0 else 1
    if "right" in name:
        return 0 if y < 0.0 else 1
    return 1
