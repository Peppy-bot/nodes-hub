"""YOLOE-11M with the gallery's crops as visual prompts: `perception_backend: "yoloe_vp"`.

The real-time pipeline of the September 2026 perception study
(yolo_world_eval, candidate 4): 81% of identify_item queries correct on the
Waldo cameras and 76% on Isaac at 21 ms a frame on a GPU, the wrong item for
3 to 6% and a phantom box for 8 to 9% of queries about absent items. Where
SAM 3 and SigLIP take two seconds, this answers before the next frame.

A YOLOE is a YOLO whose classes are embeddings instead of a fixed list.
Here each class is described by what it looks like: the gallery's
reference boxes are rasterised as visual prompts and YOLOE's prompt encoder
turns each image's boxes into one embedding per item present; the
embeddings are averaged per item over the gallery and L2-normalised, the
way YOLOE's own validator builds them, and replace the class list. The
model then runs over a frame like any YOLO, with per-class NMS at IoU 0.7
and 640 px input, the study's settings, and keeps boxes at confidence 0.25
and above, the confidence the study scored at (the study says 0.64 halves
the phantom boxes at the cost of misses; `MIN_CONFIDENCE` is that knob).

The gallery is the published pack `gallery_url` names, or the directory
`perception_model` overrides it with, the same as `sam3_siglip` reads
(gallery_store.py), with one condition: the prompt encoder was trained on
boxes inside frames. A pack that ships its frames and their manifest gives
those, as a harvester dataset does; a pack of cut crops alone does not, and
measured on the study's Waldo frames a class table built from crops names
almost nothing right (0% of identify queries with each crop as a whole-image
prompt, 5.5% with each crop placed on a plain canvas), so such a pack is
refused by name. A class with no boxed crop, such as a background class cut
from a segmentation render, gets no row: this model has no "none of these"
answer, and the table only holds what it can be shown.

The table is also kept to the items the study measured this backend on: the
YCB objects, those whose catalogue ids start with `TABLE_ITEM_PREFIX`, 28
rows. Every region is scored against every row and nothing says "none of
these", so accuracy falls as the table grows: built from the same frames,
28 rows found 80.1% of items and 117 rows 68.1%, and on the live table the
117-row table named lab glassware on the robot and refused the items in
front of it. A limitation of the model, recorded here for the day it is
worked around; a harvester dataset, which names no catalogue ids, is taken
whole as before. Unlike sam3_siglip this backend cannot work without a
gallery: one that cannot be had fails the load with the reason, and the node
keeps serving everything else. The weights, `yoloe-11m-seg.pt`, are fetched once from Ultralytics'
release assets into `WEIGHTS_DIR` under the daemon user's home, which the
container sees. Ultralytics is AGPL-3.0 unless licensed otherwise.

The vocabulary is the gallery: a scan finds every enrolled item, an identify
search whose description names one runs the same scan (the search the
study scored) and leaves the choice to the core, and a description the
gallery does not know finds nothing, since a visual prompt cannot be made
from words. The text-prompted YOLOE the study dropped is not offered.

Ultralytics reads a numpy image as BGR, OpenCV's order, so the RGB frame
is flipped before it goes in. Everything around the model is plain numpy
and tested without it; torch and ultralytics are imported by `load` alone.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ..ports import Box
from .gallery_store import GalleryUnavailable, resolve
from .sam3_siglip import Crop, Gallery, load_gallery, normalised

logger = logging.getLogger(__name__)

WEIGHTS = "yoloe-11m-seg.pt"
WEIGHTS_DIR = Path("~/.cache/openarm_ai_brain_vla")
# The study's settings.
IMGSZ = 640
NMS_IOU = 0.7
MAX_DET = 300
MIN_CONFIDENCE = 0.25
# The catalogue ids whose classes make the table when the gallery names
# them: the YCB objects of the study.
TABLE_ITEM_PREFIX = "ycb_"


def prototypes_from(per_image: Sequence[tuple[Sequence[int], np.ndarray]], n_classes: int) -> np.ndarray:
    """(classes, dim): the mean of each class's per-image embeddings over the
    gallery, L2-normalised. `per_image` pairs the sorted class indices
    present in one image with the (classes present, dim) embeddings the
    prompt encoder returned for it, in that order. A class no image shows
    is refused by name: it could never be found."""
    sums: Optional[np.ndarray] = None
    counts = np.zeros(n_classes, dtype=np.int64)
    for classes, embeddings in per_image:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if sums is None:
            sums = np.zeros((n_classes, embeddings.shape[1]), dtype=np.float32)
        for c, e in zip(classes, embeddings):
            sums[c] += e
            counts[c] += 1
    if sums is None or (counts == 0).any():
        missing = [i for i in range(n_classes) if counts[i] == 0]
        raise ValueError(f"the gallery has no visible crop for items {missing}")
    return normalised(sums)


def wanted_classes(gallery: Gallery, phrases: Sequence[str]) -> list[int]:
    """The gallery items a search is for: all of them for a scan, else the
    union of what each phrase names; empty when no phrase names one."""
    wanted = [p for p in phrases if p.strip()]
    if not wanted:
        return list(range(len(gallery.classes)))
    indices: set[int] = set()
    for phrase in wanted:
        indices.update(gallery.index_of(phrase))
    return sorted(indices)


def boxes_from(
    xyxy: np.ndarray,
    confidence: np.ndarray,
    classes: np.ndarray,
    labels: Sequence[str],
    wanted: Sequence[int],
    min_confidence: float = MIN_CONFIDENCE,
) -> list[Box]:
    """The model's boxes of the wanted items at `min_confidence` and above,
    labelled with the items' phrases."""
    keep = set(int(c) for c in wanted)
    out: list[Box] = []
    for box, conf, cls in zip(xyxy, confidence, classes):
        cls = int(cls)
        if cls not in keep or float(conf) < min_confidence:
            continue
        out.append(Box(label=labels[cls], confidence=float(conf), x0=float(box[0]), y0=float(box[1]), x1=float(box[2]), y1=float(box[3])))
    return out


def table_rows(gallery: Gallery) -> list[int]:
    """The gallery classes the model's table holds, in class order: those
    with at least one boxed crop, not flagged background, and, when the
    gallery names catalogue ids, standing for a `TABLE_ITEM_PREFIX` item."""
    boxed = {c.class_index for c in gallery.crops if c.box is not None}
    rows = [i for i in range(len(gallery.classes)) if i in boxed and not gallery.is_background(i)]
    if gallery.variants:
        rows = [i for i in rows if any(v.startswith(TABLE_ITEM_PREFIX) for v in gallery.variants[i])]
    return rows


def by_image(crops: Sequence[Crop]) -> dict[str, list[Crop]]:
    grouped: dict[str, list[Crop]] = {}
    for crop in crops:
        grouped.setdefault(crop.image, []).append(crop)
    return grouped


def prompt_boxes(crops: Sequence[Crop], width: int, height: int) -> np.ndarray:
    """The visual prompts of one image: each crop's box, or the whole image
    for a crop that is the image."""
    return np.array([c.box if c.box else (0.0, 0.0, float(width), float(height)) for c in crops], dtype=np.float32)


class Model:
    """YOLOE with the gallery's prototypes as its classes. Imports torch and
    ultralytics at construction and nowhere else."""

    def __init__(self, gallery: Gallery, weights_dir: Path = WEIGHTS_DIR, device: Optional[str] = None, min_confidence: float = MIN_CONFIDENCE) -> None:
        os.environ.setdefault("YOLO_VERBOSE", "False")
        import torch
        from ultralytics import YOLOE
        from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
        from ultralytics.utils.downloads import attempt_download_asset

        self.torch = torch
        self.device = device or ("0" if torch.cuda.is_available() else "cpu")
        weights_dir = weights_dir.expanduser()
        weights_dir.mkdir(parents=True, exist_ok=True)
        self.weights = attempt_download_asset(str(weights_dir / WEIGHTS))
        self.model = YOLOE(self.weights)
        predictor = YOLOEVPSegPredictor(overrides={
            "task": self.model.model.task, "mode": "predict", "save": False, "verbose": False,
            "batch": 1, "device": self.device, "imgsz": IMGSZ,
        })
        predictor.set_prompts({"bboxes": np.zeros((1, 4), dtype=np.float32), "cls": np.zeros(1, dtype=int)})
        predictor.setup_model(model=self.model.model, verbose=False)
        from PIL import Image

        # Only boxed crops of non-background classes make rows; a row's
        # position in the table maps back to its gallery class index.
        self.rows = table_rows(gallery)
        row_of = {c: r for r, c in enumerate(self.rows)}
        crops = [c for c in gallery.crops if c.box is not None and c.class_index in row_of]
        gallery.fetch_crops()
        per_image: list[tuple[list[int], np.ndarray]] = []
        for image_key, group in by_image(crops).items():
            image_path = gallery.image_path(group[0])
            classes = [row_of[c.class_index] for c in group]
            with Image.open(image_path) as image:
                boxes = prompt_boxes(group, image.width, image.height)
            predictor.set_prompts({"bboxes": boxes, "cls": np.array(classes)})
            embeddings = predictor.get_vpe(str(image_path))[0]  # (unique classes, dim), sorted by class
            per_image.append((sorted(set(classes)), embeddings.float().cpu().numpy()))
        prototypes = prototypes_from(per_image, len(self.rows))
        self.n_crops = len(crops)
        self.model.set_classes([gallery.phrases[i] for i in self.rows], torch.from_numpy(prototypes).unsqueeze(0).to(predictor.device))
        self.kwargs = dict(conf=min_confidence, iou=NMS_IOU, max_det=MAX_DET, imgsz=IMGSZ, device=self.device, verbose=False, agnostic_nms=False)
        # The first call pays for the kernels; take it here, not on the first search.
        self.predict(np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8))

    def predict(self, image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(xyxy pixels, confidence, class index) for one RGB frame."""
        bgr = np.ascontiguousarray(image_rgb[:, :, ::-1])
        result = self.model.predict(bgr, **self.kwargs)[0]
        boxes = result.boxes
        rows = boxes.cls.cpu().numpy().astype(int)
        classes = np.array([self.rows[r] for r in rows], dtype=int)
        return boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), classes


class YoloeVpDetector:
    name = "yoloe_vp"

    def __init__(self) -> None:
        self._gallery: Optional[Gallery] = None
        self._model: Optional[Model] = None
        self._vocabulary: list[str] = []
        # The confidence a detection is kept at; the brain sets it from the
        # perception_confidence parameter before the load.
        self.min_confidence = MIN_CONFIDENCE

    @property
    def available(self) -> bool:
        return self._gallery is not None and self._model is not None

    def load(self, model: str, gallery: str = "") -> None:
        """Reads the gallery `model` or `gallery` names, fetches the weights
        and builds the prototypes. A gallery that cannot be had, or a library
        that is missing, fails the load with the reason."""
        try:
            source = resolve(model, gallery)
        except GalleryUnavailable as error:
            raise RuntimeError(f"the yoloe_vp backend needs its gallery: {error}") from error
        if source is None:
            raise ValueError("the yoloe_vp backend needs a gallery: perception_model names none and gallery_url is empty")
        loaded = load_gallery(source)
        if not any(c.box is not None for c in loaded.crops):
            raise ValueError(
                f"the yoloe_vp backend needs boxes in frames and {loaded.name} ships crops only: "
                "name a pack with frames, or a harvester dataset, in perception_model"
            )
        try:
            weights = Model(loaded, min_confidence=self.min_confidence)
        except ImportError as error:
            raise RuntimeError(f"the yoloe_vp backend needs torch and ultralytics (the node's yoloe extra): {error}") from error
        self._gallery, self._model = loaded, weights
        logger.info(
            "yoloe_vp: %d items from %s, %d reference crops, %s on %s",
            len(loaded.classes), loaded.name, weights.n_crops, weights.weights, weights.device,
        )

    def set_vocabulary(self, phrases: Sequence[str]) -> None:
        self._vocabulary = list(phrases)

    def detect(self, image: np.ndarray) -> list[Box]:
        if self._gallery is None or self._model is None:
            return []
        wanted = wanted_classes(self._gallery, self._vocabulary)
        if not wanted:
            return []
        xyxy, confidence, classes = self._model.predict(image)
        return boxes_from(xyxy, confidence, classes, self._gallery.phrases, wanted, self.min_confidence)
