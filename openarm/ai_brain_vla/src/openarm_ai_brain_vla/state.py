"""The brain's memory and the contract rules about names: the grippers,
the known items, and how a goal's names resolve or get refused. Plain
data and pure functions, no robot calls, so every rule has a fast test.

Rules written down here because the code depends on them:

- Item ids. A scan's detection with the same label as a known item and
  within `MATCH_RADIUS_M` of it keeps that item's id; any other detection
  gets a new id, `<label>_<n>`. A known item that a full scan did not see
  is dropped, unless a gripper holds it, so grab_item cannot be sent to an
  item that left the table. An identify search is not a full scan: it
  refreshes what it found and drops nothing.
- Holding follows results, not jaws. A gripper holds an item after a
  successful grab and stops holding after a successful drop or place.
  Refusals, failures, cancels and aborts leave the flag as it was.
- Gripper to arm. The backbone names both by side, so `left_gripper`
  drives with `left_arm`. That is `arm_of`, the one function to change for
  a robot that names its limbs differently.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

from .ports import Detection, Gripper, Item, Quat, Refusal, Vec3

MATCH_RADIUS_M = 0.05


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
    With a description, the highest confidence among detections whose
    label is that description, then among those whose label contains one
    of its words; without one, the highest confidence of all, the item
    the detector judged most prominent."""
    if not detections:
        return None
    wanted = description.strip().lower()
    if not wanted:
        return max(detections, key=lambda d: d.confidence)
    exact = [d for d in detections if d.label.strip().lower() == wanted]
    if exact:
        return max(exact, key=lambda d: d.confidence)
    words = [w for w in wanted.split() if w]
    loose = [d for d in detections if any(w in d.label.lower() for w in words)]
    if loose:
        return max(loose, key=lambda d: d.confidence)
    return None


class State:
    def __init__(self, gripper_names: Iterable[str], match_radius_m: float = MATCH_RADIUS_M) -> None:
        names = [name.strip() for name in gripper_names if name.strip()]
        if not names:
            raise ValueError("gripper_names must name at least one gripper")
        if len(set(names)) != len(names):
            raise ValueError("gripper_names must be distinct")
        self.grippers: dict[str, Gripper] = {name: Gripper(name=name, arm=arm_of(name)) for name in names}
        self.items: dict[str, Item] = {}
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
        item_id = gripper.held_item_id or ""
        gripper.held_item_id = None
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

    def remember(self, detections: Sequence[Detection], now_ns: int, *, complete: bool) -> list[Item]:
        """Folds detections into the known items and returns them in the
        detections' order. `complete` says the detections are everything in
        view (a scan), so unseen items are dropped except held ones."""
        seen: list[Item] = []
        claimed: set[str] = set()
        for detection in detections:
            match = self._match(detection, claimed)
            if match is None:
                match = self._mint(detection.label, detection.position, detection.orientation, detection.confidence, now_ns)
            else:
                match.position = detection.position
                match.orientation = detection.orientation
                match.confidence = detection.confidence
                match.seen_at_ns = now_ns
            claimed.add(match.item_id)
            seen.append(match)
        if complete:
            held = {gripper.held_item_id for gripper in self.grippers.values() if gripper.holding}
            for item_id in list(self.items):
                if item_id not in claimed and item_id not in held:
                    del self.items[item_id]
        return seen

    def mint_from_pose(self, position: Vec3, orientation: Optional[Quat], now_ns: int) -> Item:
        """An id for an item a goal addressed by pose rather than by id."""
        return self._mint("item", position, orientation, 0.0, now_ns)

    def _match(self, detection: Detection, claimed: set[str]) -> Optional[Item]:
        best: Optional[Item] = None
        best_distance = self.match_radius_m
        for item in self.items.values():
            if item.item_id in claimed or item.label != detection.label:
                continue
            gap = distance(item.position, detection.position)
            if gap <= best_distance:
                best, best_distance = item, gap
        return best

    def _mint(self, label: str, position: Vec3, orientation: Optional[Quat], confidence: float, now_ns: int) -> Item:
        stem = _stem(label)
        count = self._counts.get(stem, 0) + 1
        self._counts[stem] = count
        item = Item(
            item_id=f"{stem}_{count}",
            label=label,
            position=position,
            orientation=orientation,
            confidence=confidence,
            seen_at_ns=now_ns,
        )
        self.items[item.item_id] = item
        return item


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
