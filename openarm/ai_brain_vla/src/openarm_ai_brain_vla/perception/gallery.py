"""An enrolment gallery: pictures of particular items, which sam3_siglip
names a box by before its vocabulary does, so an enrolled item is reported
under its own name ("coffee can") rather than a general one ("can"). The
node ships none; a launch names one through `perception_model`, a directory
the container can see.

Two layouts are read:

- a release: `index.json` holding `classes`, the class names in order, and
  `objects`, per class its `label`, a `background` flag and the paths of its
  crops. Beside it may sit the SigLIP prototypes, the npz `index.json`'s
  `prototypes.file` names, one L2-normalised mean image embedding per class
  in class order; without them the crops are embedded at load. An index
  whose `frames.manifest` names a manifest in the directory takes its crops
  from there instead, as boxes in whole frames.
- a harvester dataset: `manifest.json` with the frames and every item's box
  in them, `classes.txt` and `prompts.txt`.

A class flagged `background` (the robot's own arm, the empty floor) is a
picture of what is not an item: a box nearest to it is dropped, and a
description never names it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

INDEX_FILE = "index.json"
MANIFEST_FILE = "manifest.json"
# The perception_model value that names no gallery, like the empty one.
NO_GALLERY = "none"
# A box in a frame is a reference crop when this much of its item shows.
MIN_VISIBLE = 0.8
# Words a description carries that name nothing.
STOPWORDS = frozenset({"a", "an", "the", "this", "that", "my", "some", "please", "me", "of", "it"})


@dataclass(frozen=True)
class Crop:
    """One reference crop of a gallery item: an image and, when the image
    is a whole frame, the box of the item in it; a release's own crops are
    the whole image, box None."""

    image: str
    box: Optional[tuple[float, float, float, float]]
    class_index: int


@dataclass(frozen=True)
class Gallery:
    """The enrolled items: their keys, their plain phrases, the reference
    crops, and the prototypes as the release stores them, when it does."""

    root: Path
    classes: tuple[str, ...]
    phrases: tuple[str, ...]
    crops: tuple[Crop, ...]
    prototypes: Optional[np.ndarray] = None
    # Per class, whether it is background: named to be dropped, never asked for.
    background: tuple[bool, ...] = ()

    def is_background(self, index: int) -> bool:
        return bool(self.background[index]) if index < len(self.background) else False

    def item_indices(self) -> list[int]:
        """The classes that are items: every class but the background ones."""
        return [i for i in range(len(self.classes)) if not self.is_background(i)]

    def index_of(self, description: str) -> list[int]:
        """The gallery items a description names: those whose key or phrase
        it is, else those whose phrase holds every word of it, whole words,
        articles aside. Empty when none does, and a background class is
        never one. Stricter than the core's own rule on purpose: with a
        hundred names, "blue ball" must not become the gallery's "blue pen"
        and lose the words route that would find it."""
        wanted = description.strip().lower().replace("_", " ")
        if not wanted:
            return []
        items = self.item_indices()
        exact = [i for i in items if wanted in (self.classes[i].lower().replace("_", " "), self.phrases[i].lower())]
        if exact:
            return exact
        words = {w for w in wanted.split() if w and w not in STOPWORDS}
        if not words:
            return []
        return [i for i in items if words <= set(self.phrases[i].lower().split())]

    def image_path(self, crop: Crop) -> Path:
        return self.root / crop.image


def load_gallery(model: str) -> Optional[Gallery]:
    """The gallery `perception_model` names: None for "" or "none", else
    the release or harvester dataset in that directory. Raises ValueError
    naming what is wrong when the directory is not one."""
    name = model.strip()
    if not name or name.lower() == NO_GALLERY:
        return None
    root = Path(name).expanduser()
    if not root.is_dir():
        raise ValueError(f"gallery {root} is not a directory the node can see")
    # A release holds both an index and a manifest; a harvester dataset only
    # the manifest.
    if (root / INDEX_FILE).is_file():
        return load_release(root)
    if (root / MANIFEST_FILE).is_file():
        return load_harvest(root)
    raise ValueError(f"gallery {root} holds no {INDEX_FILE} or {MANIFEST_FILE}")


def load_release(root: Path) -> Gallery:
    """A release: classes and their crops from `index.json`, phrases from
    the labels, prototypes from the npz it names."""
    index = _json(root / INDEX_FILE)
    classes = [str(c) for c in index.get("classes", [])]
    objects = index.get("objects", {})
    if not classes or not isinstance(objects, dict):
        raise ValueError(f"gallery {root}: {INDEX_FILE} names no classes")
    phrases = [phrase_for(name, objects.get(name, {})) for name in classes]
    background = [bool(objects.get(name, {}).get("background", False)) for name in classes]
    frames = index.get("frames") or {}
    manifest_path = str(frames.get("manifest", "")) if isinstance(frames, dict) else ""
    if manifest_path and (root / manifest_path).is_file():
        crops = _boxed_crops(_json(root / manifest_path), {c: i for i, c in enumerate(classes)})
    else:
        crops = [Crop(str(path), None, i) for i, name in enumerate(classes) for path in objects.get(name, {}).get("crops", [])]
    prototypes = None
    meta = index.get("prototypes") or {}
    if isinstance(meta, dict) and meta.get("file") and (root / str(meta["file"])).is_file():
        with np.load(root / str(meta["file"])) as z:
            table = np.asarray(z["prototypes"], dtype=np.float32)
        if table.shape[0] != len(classes):
            raise ValueError(f"gallery {root}: {table.shape[0]} prototypes for {len(classes)} classes")
        prototypes = table
    return Gallery(root, tuple(classes), tuple(phrases), tuple(crops), prototypes, tuple(background))


def load_harvest(root: Path) -> Gallery:
    """A harvester dataset, refused with the reason when an item has no
    crop to be named from."""
    manifest = _json(root / MANIFEST_FILE)
    classes = _lines(root / "classes.txt") or [str(c) for c in manifest.get("classes", [])]
    if not classes:
        raise ValueError(f"gallery {root} names no items: classes.txt is missing or empty")
    phrases = _lines(root / "prompts.txt") or [c.replace("_", " ") for c in classes]
    if len(phrases) != len(classes):
        raise ValueError(f"gallery {root}: prompts.txt has {len(phrases)} lines for {len(classes)} classes")
    crops = _boxed_crops(manifest, {c: i for i, c in enumerate(classes)})
    covered = {crop.class_index for crop in crops}
    missing = [c for i, c in enumerate(classes) if i not in covered]
    if missing:
        raise ValueError(f"gallery {root} has no crop at or above {MIN_VISIBLE:g} visible for {missing}")
    return Gallery(root, tuple(classes), tuple(phrases), tuple(crops), None, tuple(False for _ in classes))


def phrase_for(class_name: str, entry: dict) -> str:
    """The plain words a class is prompted and reported by: its label less a
    dataset prefix ("YCB apple" is "apple", "YCB rigid sponge" a "sponge"),
    else the class name with spaces."""
    label = str(entry.get("label", "")).strip()
    if not label:
        return class_name.replace("_", " ").strip().lower()
    label = re.sub(r"^ycb\s+", "", label, flags=re.I)
    label = re.sub(r"\brigid\s+", "", label, flags=re.I)
    return label.strip().lower()


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
