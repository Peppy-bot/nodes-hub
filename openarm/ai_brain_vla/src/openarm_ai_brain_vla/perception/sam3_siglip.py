"""SAM 3 to find, SigLIP to name: `perception_backend: "sam3_siglip"`.

The most accurate pipeline of the September 2026 perception study
(yolo_world_eval, candidate 1): 87% of identify_item queries correct on
both simulators' cameras, 1 to 2% the wrong item, 1 to 2% phantom boxes,
about 2 s a frame on an A10. SAM 3 is prompted with each item's name and
returns every instance it finds; the boxes are pooled with the names SAM 3
gave them thrown away, and each crop is named by its SigLIP embedding
against prototypes, the mean embedding of the reference crops of each item
in a gallery. Identity comes from what the items look like rather than
from what they are called, which is what made this the most accurate.

`perception_model` is the gallery directory: `manifest.json` listing images
and the boxes of the items in them (the harvester's format), `classes.txt`
naming the items, and `prompts.txt` giving each a plain phrase in the same
order (the class name with spaces when absent). The directory and the
models' weights, the Hugging Face cache, must be visible inside the node's
container; paths under the daemon user's home are.

The settings are the study's: proposals at objectness 0.05, class-agnostic
NMS at IoU 0.6, at most 40 a frame, crops grown by a tenth, SigLIP so400m in
half precision, a softmax at temperature 100 over cosine similarities. A
detection is kept at objectness times class probability of 0.25 and above,
the confidence the study scored at.

A scan looks for every item of the gallery. An identify search whose
description names a gallery item runs that same scan and leaves the choice
to the core: this is the search the study scored, and SAM 3 prompted with
one name alone misses items it finds under another (a mug it boxes as a
bowl, say), while naming against the whole gallery keeps the wrong item
under the right name from being returned. A description the gallery does
not know is searched open-vocabulary: SAM 3 is prompted with the
description and each box is named by SigLIP between the description and a
few background phrases, the study's text naming.

Everything around the models is plain numpy and tested without them; torch
and transformers are imported by `load` alone, so selecting another backend
never imports them.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ..ports import Box

logger = logging.getLogger(__name__)

# SAM 3's official weights are gated behind a licence click-through on the
# Hub; this mirror carries the same files. SigLIP so400m is the study's namer.
SAM3_REPO = "jetjodh/sam3"
SIGLIP_REPO = "google/siglip-so400m-patch14-384"

# The study's settings, in the order the pipeline applies them.
PROPOSAL_CONF = 0.05
PROPOSAL_IOU = 0.6
MAX_PROPOSALS = 40
CROP_MARGIN = 0.1
TEMPERATURE = 100.0
GALLERY_MIN_VISIBLE = 0.8
MIN_CONFIDENCE = 0.25
# The narrowest crop sent to SigLIP. A crop one or three pixels on a side
# reads to the image processor as a channel axis; a box that thin holds no
# item anyway, so it is widened about its centre before cropping.
MIN_CROP_PX = 8
# What a crop is compared with beside a description the gallery does not
# know, so a crop that looks like none of them is dropped rather than named.
BACKGROUND_PHRASES = ("an empty table surface", "a white robot gripper", "a plain wooden board", "a blue wall")

GALLERY_NAMING = "gallery"
TEXT_NAMING = "text"


@dataclass(frozen=True)
class Crop:
    """One reference crop of a gallery item: the image it is in and its box."""

    image: Path
    box: tuple[float, float, float, float]
    class_index: int


@dataclass(frozen=True)
class Gallery:
    """The items the backend can name: their keys, their plain phrases, and
    the reference crops their prototypes are built from."""

    root: Path
    classes: tuple[str, ...]
    phrases: tuple[str, ...]
    crops: tuple[Crop, ...]

    def index_of(self, description: str) -> list[int]:
        """The gallery items a description names: the one whose key or
        phrase it is, else every one whose phrase contains a word of it,
        the rule identify_item's own matching uses. Empty when none does."""
        wanted = description.strip().lower().replace("_", " ")
        if not wanted:
            return []
        for i, (key, phrase) in enumerate(zip(self.classes, self.phrases)):
            if wanted in (key.lower().replace("_", " "), phrase.lower()):
                return [i]
        words = [w for w in wanted.split() if w]
        return [i for i, phrase in enumerate(self.phrases) if any(w in phrase.lower() for w in words)]


@dataclass(frozen=True)
class Plan:
    """What one search runs: the phrases SAM 3 is prompted with, and how the
    boxes are named, against the gallery's prototypes or against the text of
    the searched phrases."""

    prompts: tuple[str, ...]
    naming: str
    labels: tuple[str, ...]


def load_gallery(root: Path, min_visible: float = GALLERY_MIN_VISIBLE) -> Gallery:
    """The gallery under `root`, refused with the reason when it is not one."""
    if not root.is_dir():
        raise ValueError(f"perception_model must be the gallery directory for sam3_siglip, got {str(root)!r}")
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"gallery {root} has no manifest.json")
    manifest = json.loads(manifest_path.read_text())
    classes = _lines(root / "classes.txt") or [str(c) for c in manifest.get("classes", [])]
    if not classes:
        raise ValueError(f"gallery {root} names no items: classes.txt is missing or empty")
    phrases = _lines(root / "prompts.txt") or [c.replace("_", " ") for c in classes]
    if len(phrases) != len(classes):
        raise ValueError(f"gallery {root}: prompts.txt has {len(phrases)} lines for {len(classes)} classes")
    index_of = {c: i for i, c in enumerate(classes)}
    crops: list[Crop] = []
    for record in manifest.get("images", []):
        image = root / record["image"]
        for obj in record.get("objects_on_table", []):
            box = obj.get("bbox_xyxy_px")
            if not box or obj.get("visible_fraction", 0.0) < min_visible or obj.get("class") not in index_of:
                continue
            crops.append(Crop(image, tuple(float(v) for v in box), index_of[obj["class"]]))
    counts = [0] * len(classes)
    for crop in crops:
        counts[crop.class_index] += 1
    missing = [c for c, n in zip(classes, counts) if n == 0]
    if missing:
        raise ValueError(f"gallery {root} has no crop at or above {min_visible:g} visible for {missing}")
    return Gallery(root, tuple(classes), tuple(phrases), tuple(crops))


def _lines(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def plan_for(gallery: Gallery, phrases: Sequence[str]) -> Plan:
    """The search for `phrases`: the whole gallery when they are empty or
    name gallery items, the study's search, the core picking the item after;
    or, for phrases the gallery does not know, the phrases themselves named
    by their text."""
    wanted = [p.strip() for p in phrases if p.strip()]
    if not wanted or any(gallery.index_of(phrase) for phrase in wanted):
        return Plan(gallery.phrases, GALLERY_NAMING, gallery.phrases)
    return Plan(tuple(wanted), TEXT_NAMING, tuple(wanted))


def class_agnostic_nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float, max_keep: int) -> list[int]:
    """The indices to keep, most confident first: a box overlapping a kept
    one by more than `iou_threshold` goes, whatever either was called."""
    if len(boxes) == 0:
        return []
    order = np.argsort(-scores, kind="stable")
    kept: list[int] = []
    for i in order:
        if len(kept) >= max_keep:
            break
        if all(iou(boxes[i], boxes[k]) <= iou_threshold for k in kept):
            kept.append(int(i))
    return kept


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    overlap = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap
    return float(overlap / union) if union > 0.0 else 0.0


def grown(box: Sequence[float], margin: float, width: int, height: int) -> tuple[int, int, int, int]:
    """The crop of a box grown by `margin` of its size on each side, at
    least `MIN_CROP_PX` on each side about the box's centre, and kept inside
    the image, as the study cropped."""
    x0, y0, x1, y1 = box
    dx, dy = margin * (x1 - x0), margin * (y1 - y0)
    x0, y0 = max(0.0, x0 - dx), max(0.0, y0 - dy)
    x1, y1 = min(float(width), x1 + dx), min(float(height), y1 + dy)
    x0, x1 = _at_least(x0, x1, MIN_CROP_PX, width)
    y0, y1 = _at_least(y0, y1, MIN_CROP_PX, height)
    return (int(x0), int(y0), int(x1), int(y1))


def _at_least(lo: float, hi: float, size: int, limit: int) -> tuple[float, float]:
    """`[lo, hi)` as it is when at least `size` wide; otherwise widened about
    its centre to `size` and kept inside `[0, limit)`, shifted rather than
    shrunk when it runs over the edge."""
    if hi - lo >= size:
        return lo, hi
    centre = (lo + hi) / 2.0
    lo, hi = centre - size / 2.0, centre + size / 2.0
    if lo < 0.0:
        hi, lo = min(float(limit), hi - lo), 0.0
    if hi > limit:
        lo, hi = max(0.0, lo - (hi - limit)), float(limit)
    return lo, hi


def name_by_prototypes(embeddings: np.ndarray, prototypes: np.ndarray, temperature: float) -> tuple[np.ndarray, np.ndarray]:
    """For each embedding, the prototype it is nearest and the softmax
    probability of that choice over cosine similarities scaled by
    `temperature`, the study's naming."""
    logits = temperature * (embeddings @ prototypes.T)
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    best = probs.argmax(axis=1)
    return best, probs[np.arange(len(best)), best]


def normalised(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.where(norms == 0.0, 1.0, norms)


def detections_from(
    boxes: np.ndarray,
    objectness: np.ndarray,
    best: np.ndarray,
    probability: np.ndarray,
    labels: Sequence[str],
    min_confidence: float = MIN_CONFIDENCE,
) -> list[Box]:
    """The boxes named as one of `labels`, at objectness times class
    probability of `min_confidence` and above. A box named beyond the
    labels, as one of the background phrases, is dropped."""
    out: list[Box] = []
    for box, o, b, p in zip(boxes, objectness, best, probability):
        if b >= len(labels):
            continue
        confidence = float(o) * float(p)
        if confidence < min_confidence:
            continue
        out.append(Box(label=labels[int(b)], confidence=confidence, x0=float(box[0]), y0=float(box[1]), x1=float(box[2]), y1=float(box[3])))
    return out


class Models:
    """The two models on one device, and the three calls the backend makes
    on them. Imports torch and transformers at construction and nowhere
    else."""

    def __init__(self, sam3_repo: str = SAM3_REPO, siglip_repo: str = SIGLIP_REPO, device: Optional[str] = None) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor, Sam3Model, Sam3Processor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.half = self.device.startswith("cuda")
        self.sam3_processor = Sam3Processor.from_pretrained(sam3_repo)
        self.sam3 = Sam3Model.from_pretrained(sam3_repo).to(self.device).eval()
        self.siglip_processor = AutoProcessor.from_pretrained(siglip_repo)
        self.siglip = AutoModel.from_pretrained(siglip_repo, dtype=torch.float16 if self.half else torch.float32).to(self.device).eval()
        self._text_inputs: dict[str, object] = {}

    def propose(self, image, prompts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """Every instance SAM 3 finds for any of `prompts`: boxes in pixels
        and their objectness, the query logit times the presence logit, as
        SAM 3's own post-processing scores them. The frame is encoded once
        and the detector decoder runs once per prompt."""
        torch = self.torch
        width, height = image.size
        scale = np.array([width, height, width, height], dtype=np.float32)
        boxes, scores = [], []
        with torch.inference_mode():
            pixel_values = self.sam3_processor(images=image, return_tensors="pt").to(self.device)["pixel_values"]
            vision = self.sam3.get_vision_features(pixel_values=pixel_values)
            for prompt in prompts:
                text = self._text_inputs.get(prompt)
                if text is None:
                    text = self._text_inputs[prompt] = self.sam3_processor(text=prompt, return_tensors="pt").to(self.device)
                outputs = self.sam3(vision_embeds=vision, input_ids=text["input_ids"], attention_mask=text.get("attention_mask"))
                score = outputs.pred_logits.sigmoid()[0]
                if outputs.presence_logits is not None:
                    score = score * outputs.presence_logits.sigmoid()[0]
                boxes.append(outputs.pred_boxes[0].float().cpu().numpy() * scale)
                scores.append(score.float().cpu().numpy())
        if not boxes:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        return np.concatenate(boxes), np.concatenate(scores)

    def embed_images(self, crops: list) -> np.ndarray:
        torch = self.torch
        with torch.inference_mode():
            inputs = self.siglip_processor(images=crops, return_tensors="pt").to(self.device)
            pixel_values = inputs["pixel_values"].to(torch.float16 if self.half else torch.float32)
            features = self._pooled(self.siglip.get_image_features(pixel_values=pixel_values))
        return normalised(features.float().cpu().numpy())

    def embed_texts(self, phrases: Sequence[str]) -> np.ndarray:
        torch = self.torch
        with torch.inference_mode():
            inputs = self.siglip_processor(text=list(phrases), padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(self.device)
            features = self._pooled(self.siglip.get_text_features(input_ids=inputs["input_ids"]))
        return normalised(features.float().cpu().numpy())

    def _pooled(self, features):
        return features if self.torch.is_tensor(features) else features.pooler_output


class Sam3SiglipDetector:
    name = "sam3_siglip"

    def __init__(self) -> None:
        self._gallery: Optional[Gallery] = None
        self._models: Optional[Models] = None
        self._prototypes: Optional[np.ndarray] = None
        self._vocabulary: list[str] = []
        self._text_prototypes: dict[tuple[str, ...], np.ndarray] = {}

    @property
    def available(self) -> bool:
        return self._models is not None and self._prototypes is not None

    def load(self, model: str) -> None:
        """Reads the gallery at `model`, loads the two models and builds the
        gallery's prototypes. A gallery that is not one, or models that
        cannot be loaded, fail the node's start with the reason."""
        from PIL import Image

        if not model.strip():
            raise ValueError("perception_model must be the gallery directory for sam3_siglip, and it is empty")
        gallery = load_gallery(Path(model.strip()).expanduser())
        try:
            models = Models()
        except ImportError as error:
            raise RuntimeError(f"the sam3_siglip backend needs torch and transformers (the node's sam3-siglip extra): {error}") from error
        prototypes = np.zeros((len(gallery.classes), 0), dtype=np.float32)
        sums: Optional[np.ndarray] = None
        counts = np.zeros(len(gallery.classes), dtype=np.int64)
        by_image: dict[Path, list[Crop]] = {}
        for crop in gallery.crops:
            by_image.setdefault(crop.image, []).append(crop)
        for image_path, crops in by_image.items():
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                embeddings = models.embed_images([image.crop(grown(c.box, CROP_MARGIN, image.width, image.height)) for c in crops])
            if sums is None:
                sums = np.zeros((len(gallery.classes), embeddings.shape[1]), dtype=np.float32)
            for crop, embedding in zip(crops, embeddings):
                sums[crop.class_index] += embedding
                counts[crop.class_index] += 1
        assert sums is not None
        prototypes = normalised(sums)
        # The first CUDA call pays for the kernels; take it here, not on the
        # first search.
        models.propose(Image.new("RGB", (64, 64)), gallery.phrases[:1])
        self._gallery, self._models, self._prototypes = gallery, models, prototypes
        logger.info(
            "sam3_siglip: %d items from %s, %d reference crops, on %s",
            len(gallery.classes), gallery.root, int(counts.sum()), models.device,
        )

    def set_vocabulary(self, phrases: Sequence[str]) -> None:
        self._vocabulary = list(phrases)

    def detect(self, image: np.ndarray) -> list[Box]:
        from PIL import Image

        if self._gallery is None or self._models is None or self._prototypes is None:
            return []
        plan = plan_for(self._gallery, self._vocabulary)
        frame = Image.fromarray(np.ascontiguousarray(image))
        boxes, objectness = self._models.propose(frame, plan.prompts)
        above = objectness >= PROPOSAL_CONF
        boxes, objectness = boxes[above], objectness[above]
        keep = class_agnostic_nms(boxes, objectness, PROPOSAL_IOU, MAX_PROPOSALS)
        if not keep:
            return []
        boxes, objectness = boxes[keep], objectness[keep]
        crops = [frame.crop(grown(box, CROP_MARGIN, frame.width, frame.height)) for box in boxes]
        embeddings = self._models.embed_images(crops)
        if plan.naming == GALLERY_NAMING:
            prototypes = self._prototypes
        else:
            prototypes = self._text_prototypes.get(plan.labels)
            if prototypes is None:
                prototypes = self._models.embed_texts(
                    [f"a photo of a {p}" for p in plan.labels] + [f"a photo of {b}" for b in BACKGROUND_PHRASES]
                )
                self._text_prototypes[plan.labels] = prototypes
        best, probability = name_by_prototypes(embeddings, prototypes, TEMPERATURE)
        return detections_from(boxes, objectness, best, probability, plan.labels, MIN_CONFIDENCE)
