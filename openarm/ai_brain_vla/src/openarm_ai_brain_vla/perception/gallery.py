"""An enrolment gallery: pictures of particular items, which sam3_siglip
names a box by before its vocabulary does, so an enrolled item is reported
under its own name ("coffee can") rather than a general one ("can"). A
launch names one through `perception_gallery`, which is empty by default,
no gallery, and otherwise either the name of an enrolment the node ships
(`enrolments/<name>/`) or a directory the container can see. Two forms:

- **Prototypes**, the form the node ships: `prototypes.npz` with one SigLIP
  embedding per class, L2-normalised, and the model that made them in
  `gallery.json`, so the backend can refuse a file made by another
  checkpoint; `classes.txt`, one class a line; `prompts.txt`, the plain
  words each class is prompted and reported by, one a line in the same
  order (without it a class's words are its name with spaces).
  `gallery.json` also lists the `background` classes, pictures of the
  robot's own arm and gripper and of the empty table, which name nothing: a
  box nearest one of them is dropped. The node ships `waldo_catalogue`, the
  simulation's table objects, which the simulation launchers select; it is
  built by `tools/build_enrolment.py`, which records its source.
- **A harvester dataset**, pictures to embed at load: `manifest.json`, the
  frames and every item's box in them (`images`, each with its `image` path
  and its `objects_on_table`, each with a `class`, a `bbox_xyxy_px` box and
  the `visible_fraction` of the item that shows), with the same
  `classes.txt` and `prompts.txt`. A box in a frame is a reference crop of
  its item when at least `MIN_VISIBLE` of the item shows; an item with no
  such crop fails the load, since nothing could name it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..words import named_by

MANIFEST_FILE = "manifest.json"
CLASSES_FILE = "classes.txt"
PROMPTS_FILE = "prompts.txt"
PROTOTYPES_FILE = "prototypes.npz"
METADATA_FILE = "gallery.json"
# Where the enrolments the node ships live, one directory a name.
SHIPPED_DIRECTORY = Path(__file__).with_name("enrolments")
# A box in a frame is a reference crop when this much of its item shows.
MIN_VISIBLE = 0.8


@dataclass(frozen=True)
class Crop:
    """One reference crop of a gallery item: a frame and the box of the
    item in it."""

    image: str
    box: tuple[float, float, float, float]
    class_index: int


@dataclass(frozen=True, eq=False)
class Gallery:
    """The enrolled classes: their keys, the plain phrases they are prompted
    and reported by, and either their prototypes, ready to compare with, or
    their reference crops to embed. `background` are the classes that name
    nothing, the robot's own body and the empty table; `model` is the
    SigLIP checkpoint that made the prototypes, empty for crops."""

    root: Path
    classes: tuple[str, ...]
    phrases: tuple[str, ...]
    crops: tuple[Crop, ...] = ()
    prototypes: Optional[np.ndarray] = None
    background: frozenset[int] = field(default_factory=frozenset)
    model: str = ""
    source: str = ""

    def item_indices(self) -> list[int]:
        """The classes that are items, not background."""
        return [i for i in range(len(self.classes)) if i not in self.background]

    def is_background(self, index: int) -> bool:
        return index in self.background

    def index_of(self, description: str) -> list[int]:
        """The enrolled items a description names, by the core's one rule
        (`words.named_by`) over their phrases; a background class is never
        named. Empty when none is."""
        return [i for i in named_by(description, self.phrases) if i not in self.background]

    def image_path(self, crop: Crop) -> Path:
        return self.root / crop.image


def shipped() -> dict[str, Path]:
    """The enrolments the node ships, by name."""
    if not SHIPPED_DIRECTORY.is_dir():
        return {}
    return {p.name: p for p in sorted(SHIPPED_DIRECTORY.iterdir()) if p.is_dir()}


def load_gallery(gallery: str) -> Optional[Gallery]:
    """The gallery `perception_gallery` names: None when it is empty, the
    shipped enrolment of that name, else the enrolment or harvester dataset
    in that directory. Raises ValueError naming what is wrong when it is
    none of these."""
    name = gallery.strip()
    if not name:
        return None
    root = shipped().get(name) or Path(name).expanduser()
    if not root.is_dir():
        names = ", ".join(shipped()) or "none"
        raise ValueError(f"gallery {root} is not an enrolment the node ships ({names}) and is not a directory the node can see")
    if (root / PROTOTYPES_FILE).is_file():
        return load_prototypes(root)
    if (root / MANIFEST_FILE).is_file():
        return load_harvest(root)
    raise ValueError(f"gallery {root} holds no {MANIFEST_FILE} and no {PROTOTYPES_FILE}")


def load_prototypes(root: Path) -> Gallery:
    """An enrolment in the prototypes form, refused with the reason when
    its files disagree."""
    classes = _lines(root / CLASSES_FILE)
    metadata = _json(root / METADATA_FILE) if (root / METADATA_FILE).is_file() else {}
    try:
        with np.load(root / PROTOTYPES_FILE) as archive:
            files = list(archive.files)
            prototypes = np.asarray(archive["prototypes"], dtype=np.float32) if "prototypes" in files else None
            stored = [str(c) for c in archive["classes"]] if "classes" in files else []
    except (OSError, ValueError, AttributeError) as error:
        raise ValueError(f"gallery {root}: {PROTOTYPES_FILE} cannot be read: {error}") from error
    if prototypes is None:
        raise ValueError(f"gallery {root}: {PROTOTYPES_FILE} holds no 'prototypes' array")
    if not classes:
        classes = stored
    if not classes:
        raise ValueError(f"gallery {root} names no items: {CLASSES_FILE} is missing or empty")
    if stored and stored != classes:
        raise ValueError(f"gallery {root}: the classes in {PROTOTYPES_FILE} differ from {CLASSES_FILE}")
    if prototypes.ndim != 2 or prototypes.shape[0] != len(classes):
        count = prototypes.shape[0] if prototypes.ndim >= 1 else 0
        raise ValueError(f"gallery {root}: {PROTOTYPES_FILE} holds {count} prototypes for {len(classes)} classes")
    norms = np.linalg.norm(prototypes, axis=1)
    if not np.all(np.isfinite(prototypes)) or (norms == 0.0).any():
        raise ValueError(f"gallery {root}: {PROTOTYPES_FILE} holds an empty or non-finite prototype")
    prototypes = prototypes / norms[:, None]
    phrases = _phrases(root, classes)
    background_names = [str(b) for b in metadata.get("background", [])]
    unknown = [b for b in background_names if b not in classes]
    if unknown:
        raise ValueError(f"gallery {root}: {METADATA_FILE} names background classes that are not enrolled: {unknown}")
    background = frozenset(classes.index(b) for b in background_names)
    if len(background) == len(classes):
        raise ValueError(f"gallery {root} enrols no item: every class is background")
    return Gallery(
        root,
        tuple(classes),
        tuple(phrases),
        prototypes=prototypes,
        background=background,
        model=str(metadata.get("model", "")),
        source=str(metadata.get("source", "")),
    )


def load_harvest(root: Path) -> Gallery:
    """A harvester dataset, refused with the reason when an item has no
    crop to be named from."""
    manifest = _json(root / MANIFEST_FILE)
    classes = _lines(root / CLASSES_FILE) or [str(c) for c in manifest.get("classes", [])]
    if not classes:
        raise ValueError(f"gallery {root} names no items: {CLASSES_FILE} is missing or empty")
    phrases = _phrases(root, classes)
    crops = _boxed_crops(manifest, {c: i for i, c in enumerate(classes)})
    covered = {crop.class_index for crop in crops}
    missing = [c for i, c in enumerate(classes) if i not in covered]
    if missing:
        raise ValueError(f"gallery {root} has no crop at or above {MIN_VISIBLE:g} visible for {missing}")
    return Gallery(root, tuple(classes), tuple(phrases), tuple(crops))


def _phrases(root: Path, classes: list[str]) -> list[str]:
    phrases = _lines(root / PROMPTS_FILE) or [c.replace("_", " ") for c in classes]
    if len(phrases) != len(classes):
        raise ValueError(f"gallery {root}: {PROMPTS_FILE} has {len(phrases)} lines for {len(classes)} classes")
    return phrases


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
