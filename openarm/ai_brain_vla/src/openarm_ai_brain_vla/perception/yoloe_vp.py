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

`perception_model` is the gallery directory, the same one `sam3_siglip`
reads. The weights, `yoloe-11m-seg.pt`, are fetched once from Ultralytics'
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
from .sam3_siglip import Crop, Gallery, load_gallery, normalised

logger = logging.getLogger(__name__)

WEIGHTS = "yoloe-11m-seg.pt"
WEIGHTS_DIR = Path("~/.cache/openarm_ai_brain_vla")
# The study's settings.
IMGSZ = 640
NMS_IOU = 0.7
MAX_DET = 300
MIN_CONFIDENCE = 0.25


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


def by_image(crops: Sequence[Crop]) -> dict[Path, list[Crop]]:
    grouped: dict[Path, list[Crop]] = {}
    for crop in crops:
        grouped.setdefault(crop.image, []).append(crop)
    return grouped


class Model:
    """YOLOE with the gallery's prototypes as its classes. Imports torch and
    ultralytics at construction and nowhere else."""

    def __init__(self, gallery: Gallery, weights_dir: Path = WEIGHTS_DIR, device: Optional[str] = None) -> None:
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
        per_image: list[tuple[list[int], np.ndarray]] = []
        for image_path, crops in by_image(gallery.crops).items():
            classes = [c.class_index for c in crops]
            boxes = np.array([c.box for c in crops], dtype=np.float32)
            predictor.set_prompts({"bboxes": boxes, "cls": np.array(classes)})
            embeddings = predictor.get_vpe(str(image_path))[0]  # (unique classes, dim), sorted by class
            per_image.append((sorted(set(classes)), embeddings.float().cpu().numpy()))
        prototypes = prototypes_from(per_image, len(gallery.classes))
        self.n_crops = sum(len(v) for v in by_image(gallery.crops).values())
        self.model.set_classes(list(gallery.phrases), torch.from_numpy(prototypes).unsqueeze(0).to(predictor.device))
        self.kwargs = dict(conf=MIN_CONFIDENCE, iou=NMS_IOU, max_det=MAX_DET, imgsz=IMGSZ, device=self.device, verbose=False, agnostic_nms=False)
        # The first call pays for the kernels; take it here, not on the first search.
        self.predict(np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8))

    def predict(self, image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(xyxy pixels, confidence, class index) for one RGB frame."""
        bgr = np.ascontiguousarray(image_rgb[:, :, ::-1])
        result = self.model.predict(bgr, **self.kwargs)[0]
        boxes = result.boxes
        return boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy().astype(int)


class YoloeVpDetector:
    name = "yoloe_vp"

    def __init__(self) -> None:
        self._gallery: Optional[Gallery] = None
        self._model: Optional[Model] = None
        self._vocabulary: list[str] = []

    @property
    def available(self) -> bool:
        return self._gallery is not None and self._model is not None

    def load(self, model: str) -> None:
        """Reads the gallery at `model`, fetches the weights and builds the
        prototypes. A gallery that is not one, or a library that is missing,
        fails the load with the reason."""
        if not model.strip():
            raise ValueError("perception_model must be the gallery directory for yoloe_vp, and it is empty")
        gallery = load_gallery(Path(model.strip()).expanduser())
        try:
            loaded = Model(gallery)
        except ImportError as error:
            raise RuntimeError(f"the yoloe_vp backend needs torch and ultralytics (the node's yoloe extra): {error}") from error
        self._gallery, self._model = gallery, loaded
        logger.info(
            "yoloe_vp: %d items from %s, %d reference crops, %s on %s",
            len(gallery.classes), gallery.root, loaded.n_crops, loaded.weights, loaded.device,
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
        return boxes_from(xyxy, confidence, classes, self._gallery.phrases, wanted, MIN_CONFIDENCE)
