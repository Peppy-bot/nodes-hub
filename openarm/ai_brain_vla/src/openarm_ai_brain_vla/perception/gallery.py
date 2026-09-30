"""An enrolment gallery: pictures of particular items, which sam3_siglip
names a box by before its vocabulary does, so an enrolled item is reported
under its own name ("coffee can") rather than a general one ("can"). The
node ships none; a launch names one through `perception_gallery`, a
directory the container can see holding a harvester dataset:

- `manifest.json`, the frames and every item's box in them: `images`, each
  with its `image` path and its `objects_on_table`, each with a `class`, a
  `bbox_xyxy_px` box and the `visible_fraction` of the item that shows;
- `classes.txt`, one class a line, in the order the items are numbered
  (else the manifest's `classes`);
- `prompts.txt`, the plain words each class is prompted and reported by,
  one a line in the same order; without it a class's words are its name
  with spaces.

A box in a frame is a reference crop of its item when at least
`MIN_VISIBLE` of the item shows. An item with no such crop fails the
load: nothing could name it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..words import named_by

MANIFEST_FILE = "manifest.json"
CLASSES_FILE = "classes.txt"
PROMPTS_FILE = "prompts.txt"
# A box in a frame is a reference crop when this much of its item shows.
MIN_VISIBLE = 0.8


@dataclass(frozen=True)
class Crop:
    """One reference crop of a gallery item: a frame and the box of the
    item in it."""

    image: str
    box: tuple[float, float, float, float]
    class_index: int


@dataclass(frozen=True)
class Gallery:
    """The enrolled items: their class keys, the plain phrases they are
    prompted and reported by, and their reference crops."""

    root: Path
    classes: tuple[str, ...]
    phrases: tuple[str, ...]
    crops: tuple[Crop, ...]

    def index_of(self, description: str) -> list[int]:
        """The enrolled items a description names, by the core's one rule
        (`words.named_by`) over their phrases. Empty when none does."""
        return named_by(description, self.phrases)

    def image_path(self, crop: Crop) -> Path:
        return self.root / crop.image


def load_gallery(gallery: str) -> Optional[Gallery]:
    """The gallery `perception_gallery` names: None when it is empty, else
    the harvester dataset in that directory. Raises ValueError naming what
    is wrong when the directory is not one."""
    name = gallery.strip()
    if not name:
        return None
    root = Path(name).expanduser()
    if not root.is_dir():
        raise ValueError(f"gallery {root} is not a directory the node can see")
    if not (root / MANIFEST_FILE).is_file():
        raise ValueError(f"gallery {root} holds no {MANIFEST_FILE}")
    return load_harvest(root)


def load_harvest(root: Path) -> Gallery:
    """A harvester dataset, refused with the reason when an item has no
    crop to be named from."""
    manifest = _json(root / MANIFEST_FILE)
    classes = _lines(root / CLASSES_FILE) or [str(c) for c in manifest.get("classes", [])]
    if not classes:
        raise ValueError(f"gallery {root} names no items: {CLASSES_FILE} is missing or empty")
    phrases = _lines(root / PROMPTS_FILE) or [c.replace("_", " ") for c in classes]
    if len(phrases) != len(classes):
        raise ValueError(f"gallery {root}: {PROMPTS_FILE} has {len(phrases)} lines for {len(classes)} classes")
    crops = _boxed_crops(manifest, {c: i for i, c in enumerate(classes)})
    covered = {crop.class_index for crop in crops}
    missing = [c for i, c in enumerate(classes) if i not in covered]
    if missing:
        raise ValueError(f"gallery {root} has no crop at or above {MIN_VISIBLE:g} visible for {missing}")
    return Gallery(root, tuple(classes), tuple(phrases), tuple(crops))


def _boxed_crops(manifest: dict, index_of: dict[str, int]) -> list[Crop]:
    """The boxes of known classes in a manifest's frames that show at least
    `MIN_VISIBLE` of their item."""
    crops: list[Crop] = []
    for record in manifest.get("images", []):
        image = str(record["image"])
        for obj in record.get("objects_on_table", []):
            box = obj.get("bbox_xyxy_px")
            if box and obj.get("visible_fraction", 0.0) >= MIN_VISIBLE and obj.get("class") in index_of:
                crops.append(Crop(image, tuple(float(v) for v in box), index_of[obj["class"]]))
    return crops


def _json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except ValueError as error:
        raise ValueError(f"gallery file {path} is not JSON: {error}") from error


def _lines(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]
