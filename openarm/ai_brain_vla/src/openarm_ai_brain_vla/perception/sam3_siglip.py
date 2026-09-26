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

The gallery is the published pack `gallery_url` names, fetched once and
cached (gallery_store.py): 126 items of the Waldo catalogue with a SigLIP
prototype each already computed, so loading reads one small file instead of
embedding 2,564 crops. `perception_model` can override it with a directory:
an unpacked release, or a harvester dataset (`manifest.json` with boxes,
`classes.txt`, `prompts.txt`) whose crops are embedded here. The models'
weights, the Hugging Face cache, must be visible inside the node's
container; paths under the daemon user's home are.

SAM 3 is prompted with the one word "object", not with every item's name.
Measured on the study's Waldo frames with its 28-item gallery, that scan
found 90.6% of items against 87.0% for the 28 name prompts, missed 5.9%
against 12.1%, named the wrong item for 3.6% against 1.0% and boxed an
absent one for 4.6% against 1.2%, in 0.94 s a frame against 3.4 s; and the
time no longer grows with the gallery, where 126 name prompts took 13 s. An
identify search adds the names its description matches to that prompt, so
the item asked for is proposed even where "object" would miss it.

The pack may carry background classes, pictures of the robot's own arm and
gripper and of the empty floor, flagged `background` in its index. They sit
in the prototype table so a box on the robot has something nearer than any
item, and a box named as one of them is dropped: that is how a scan says
"none of these". A description never matches a background class.

`MIN_CONFIDENCE` is the study's threshold; the node's `perception_confidence`
parameter overrides it per launch, since with background rows in the table
the pack's own re-check keeps 0.10 clean (91% of items found, 0.4% phantom).

The softmax names every box after the nearest prototype however far it is,
so a box on the robot that resembles no item still gets an item's name when
no background row is nearer. `SIMILARITY_FLOOR` is the second "none of
these": a box whose best cosine to any prototype is under it is dropped.
Measured on the study's Waldo frames against the pack's 120 rows, boxes on a
real item have a best cosine of 0.82 or more for 95% of them (median 0.94),
phantom boxes a median of 0.80 and at most 0.86. At 0.80 the floor drops 25
of 37 phantoms for 4 of 288 true items; at 0.83, 34 of 37 for 11 of 288. It
applies to naming by picture only: image-to-text cosines run lower and the
text route has its background phrases instead.

The gallery is optional. Empty names the node's default pack, "none" no
gallery; either that or a gallery that cannot be had (a pack the network
does not give and the cache lacks, a directory that is not there) leaves
the backend working by words alone: every search takes the text
route below, and a scan looks for `DEFAULT_VOCABULARY`, a handful of
generic table items, since without a gallery there is nothing else to look
for. The reason the gallery is missing is logged and kept, never fatal.

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
from typing import Optional, Sequence, Union

import numpy as np

from ..ports import Box
from .gallery_store import NO_GALLERY, GalleryUnavailable, HarvestDir, Pack, Source, phrase_for, resolve

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
# The least a crop may resemble its nearest prototype to be an item at all
# (the module docstring has the measurement behind the value).
SIMILARITY_FLOOR = 0.80
# The narrowest crop sent to SigLIP. A crop one or three pixels on a side
# reads to the image processor as a channel axis; a box that thin holds no
# item anyway, so it is widened about its centre before cropping.
MIN_CROP_PX = 8
# What a crop is compared with beside a description the gallery does not
# know, so a crop that looks like none of them is dropped rather than named.
BACKGROUND_PHRASES = ("an empty table surface", "a white robot gripper", "a plain wooden board", "a blue wall")
# What a scan looks for when there is no gallery to enumerate: generic
# names of what stands on a robot's table, named by words.
DEFAULT_VOCABULARY = ("cup", "mug", "bottle", "can", "box", "bowl", "plate", "fruit", "ball", "block", "tool", "toy")

# Words a description carries that name nothing.
STOPWORDS = frozenset({"a", "an", "the", "this", "that", "my", "some", "please", "me", "of", "it"})

GALLERY_NAMING = "gallery"
TEXT_NAMING = "text"
# What SAM 3 is asked for when the search is for gallery items: one generic
# prompt whose proposals are named against every prototype (see the module
# docstring for the measurement).
GENERIC_PROMPTS = ("object",)


@dataclass(frozen=True)
class Crop:
    """One reference crop of a gallery item: an image and, when the image
    is a whole frame, the box of the item in it; a pack's crops are the
    whole image, box None."""

    image: str
    box: Optional[tuple[float, float, float, float]]
    class_index: int


@dataclass(frozen=True)
class Gallery:
    """The items the backend can name: their keys, their plain phrases, the
    reference crops, and the prototypes when the source ships them."""

    name: str
    classes: tuple[str, ...]
    phrases: tuple[str, ...]
    crops: tuple[Crop, ...]
    source: Source
    prototypes: Optional[np.ndarray] = None
    # Per class, whether it is background: named to be dropped, never asked for.
    background: tuple[bool, ...] = ()
    # Per class, the catalogue ids it stands for (a pack's `variants` and
    # `catalogue_id`); empty for a harvester dataset.
    variants: tuple[tuple[str, ...], ...] = ()

    def is_background(self, index: int) -> bool:
        return bool(self.background[index]) if index < len(self.background) else False

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
        items = [i for i in range(len(self.classes)) if not self.is_background(i)]
        exact = [i for i in items if wanted in (self.classes[i].lower().replace("_", " "), self.phrases[i].lower())]
        if exact:
            return exact
        words = {w for w in wanted.split() if w and w not in STOPWORDS}
        if not words:
            return []
        return [i for i in items if words <= set(self.phrases[i].lower().split())]

    def image_path(self, crop: Crop) -> Path:
        """The local file of a crop's image, fetched first when the source
        is a pack the cache lacks."""
        if isinstance(self.source, Pack):
            return self.source.file(crop.image)
        return self.source.root / crop.image

    def fetch_crops(self) -> None:
        """Brings every crop image into the cache, in parallel, for a backend
        that builds its class table from pictures."""
        if isinstance(self.source, Pack):
            self.source.files(sorted({c.image for c in self.crops}))


@dataclass(frozen=True)
class Plan:
    """What one search runs: the phrases SAM 3 is prompted with, and how the
    boxes are named, against the gallery's prototypes or against the text of
    the searched phrases."""

    prompts: tuple[str, ...]
    naming: str
    labels: tuple[str, ...]


def load_gallery(source: Union[Source, Path], min_visible: float = GALLERY_MIN_VISIBLE) -> Gallery:
    """The gallery a source holds: a pack's index and prototypes, or a
    harvester dataset's manifest and frames (a bare path is one)."""
    if isinstance(source, Pack):
        return load_pack(source)
    root = source if isinstance(source, Path) else source.root
    return load_harvest(root, min_visible)


def load_pack(pack: Pack) -> Gallery:
    """A release of the published pack: classes and their crops from
    `index.json`, phrases from the labels, prototypes from the npz it names."""
    index = pack.index()
    classes = [str(c) for c in index.get("classes", [])]
    objects = index.get("objects", {})
    if not classes or not isinstance(objects, dict):
        raise ValueError(f"gallery {pack.prefix}: index.json names no classes")
    crops: list[Crop] = []
    phrases: list[str] = []
    background: list[bool] = []
    variants: list[tuple[str, ...]] = []
    for i, name in enumerate(classes):
        entry = objects.get(name, {})
        phrases.append(phrase_for(name, entry))
        background.append(bool(entry.get("background", False)))
        ids = [str(v) for v in entry.get("variants", []) if v] + ([str(entry["catalogue_id"])] if entry.get("catalogue_id") else [])
        variants.append(tuple(dict.fromkeys(ids)))
    frames = index.get("frames") or {}
    manifest_path = str(frames.get("manifest", "")) if isinstance(frames, dict) else ""
    if manifest_path and pack.has(manifest_path):
        # The frames the crops were cut from, with their boxes: what a
        # backend that builds its own table from pictures needs.
        manifest = json.loads(pack.file(manifest_path).read_text())
        index_of = {c: i for i, c in enumerate(classes)}
        for record in manifest.get("images", []):
            image = str(record["image"])
            for obj in record.get("objects_on_table", []):
                box = obj.get("bbox_xyxy_px")
                if box and obj.get("visible_fraction", 0.0) >= GALLERY_MIN_VISIBLE and obj.get("class") in index_of:
                    crops.append(Crop(image, tuple(float(v) for v in box), index_of[obj["class"]]))
    else:
        for i, name in enumerate(classes):
            for path in objects.get(name, {}).get("crops", []):
                crops.append(Crop(str(path), None, i))
    prototypes = None
    meta = index.get("prototypes") or {}
    if isinstance(meta, dict) and meta.get("file") and pack.has(str(meta["file"])):
        with np.load(pack.file(str(meta["file"]))) as z:
            table = np.asarray(z["prototypes"], dtype=np.float32)
        if table.shape[0] != len(classes):
            raise ValueError(f"gallery {pack.prefix}: {table.shape[0]} prototypes for {len(classes)} classes")
        prototypes = normalised(table)
    return Gallery(pack.prefix, tuple(classes), tuple(phrases), tuple(crops), pack, prototypes, tuple(background), tuple(variants))


def load_harvest(root: Path, min_visible: float = GALLERY_MIN_VISIBLE) -> Gallery:
    """A harvester dataset under `root`, refused with the reason when it is not one."""
    if not root.is_dir():
        raise ValueError(f"{root} must be the gallery directory of a harvester dataset, and it does not exist")
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
        image = str(record["image"])
        for obj in record.get("objects_on_table", []):
            box = obj.get("bbox_xyxy_px")
            if not box or obj.get("visible_fraction", 0.0) < min_visible or obj.get("class") not in index_of:
                continue
            crops.append(Crop(image, tuple(float(v) for v in box), index_of[obj["class"]]))
    missing = [c for c in classes if not any(cr.class_index == i for i, cr in ((index_of[c], cr) for cr in crops))]
    if missing:
        raise ValueError(f"gallery {root} has no crop at or above {min_visible:g} visible for {missing}")
    return Gallery(str(root), tuple(classes), tuple(phrases), tuple(crops), HarvestDir(root), None, tuple(False for _ in classes))


def _lines(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def plan_for(gallery: Optional[Gallery], phrases: Sequence[str]) -> Plan:
    """The search for `phrases`. With a gallery, a scan prompts SAM 3 with
    the generic prompt and names every box by prototype; an identify whose
    description names gallery items adds their phrases to that prompt and
    names the same way. Without a gallery, or for a description it does not
    know, the phrases themselves (or the default vocabulary for a scan) are
    prompted for and named by text."""
    wanted = [p.strip() for p in phrases if p.strip()]
    if gallery is not None:
        if not wanted:
            return Plan(GENERIC_PROMPTS, GALLERY_NAMING, gallery.phrases)
        matched: list[str] = []
        for phrase in wanted:
            for i in gallery.index_of(phrase):
                if gallery.phrases[i] not in matched:
                    matched.append(gallery.phrases[i])
        if matched:
            return Plan(GENERIC_PROMPTS + tuple(matched), GALLERY_NAMING, gallery.phrases)
    if not wanted:
        return Plan(DEFAULT_VOCABULARY, TEXT_NAMING, DEFAULT_VOCABULARY)
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


def under_floor(embeddings: np.ndarray, prototypes: np.ndarray, floor: float = SIMILARITY_FLOOR) -> np.ndarray:
    """Per embedding, whether its best cosine to any prototype is under
    `floor`: a box that resembles nothing enrolled."""
    return (embeddings @ prototypes.T).max(axis=1) < floor


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
    background: Sequence[bool] = (),
) -> list[Box]:
    """The boxes named as one of `labels`, at objectness times class
    probability of `min_confidence` and above. A box named beyond the
    labels, as one of the background phrases, or as a label flagged
    `background`, is dropped: that is the "none of these" answer."""
    out: list[Box] = []
    for box, o, b, p in zip(boxes, objectness, best, probability):
        if b >= len(labels) or (b < len(background) and background[b]):
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

    def __init__(self, models_factory=None) -> None:
        self._gallery: Optional[Gallery] = None
        self._models: Optional[Models] = None
        self._prototypes: Optional[np.ndarray] = None
        self._vocabulary: list[str] = []
        self._text_prototypes: dict[tuple[str, ...], np.ndarray] = {}
        self._models_factory = models_factory or Models
        # Why there is no gallery, when there is none: "" with one.
        self.gallery_reason = ""
        # The confidence a detection is kept at; the brain sets it from the
        # perception_confidence parameter before the load.
        self.min_confidence = MIN_CONFIDENCE

    @property
    def available(self) -> bool:
        return self._models is not None

    def load(self, model: str, gallery: str = "") -> None:
        """Loads the two models, then the gallery `model` or `gallery` names
        and its prototypes. Models that cannot be loaded fail the load with
        the reason; a gallery that cannot be had does not: the backend works
        by words alone and says why in the log."""
        from PIL import Image

        try:
            models = self._models_factory()
        except ImportError as error:
            raise RuntimeError(f"the sam3_siglip backend needs torch and transformers (the node's sam3-siglip extra): {error}") from error
        loaded: Optional[Gallery] = None
        try:
            source = resolve(model, gallery)
            loaded = load_gallery(source) if source is not None else None
            if loaded is not None:
                self.gallery_reason = ""
            elif model.strip().lower() == NO_GALLERY:
                self.gallery_reason = 'perception_model is "none"'
            else:
                self.gallery_reason = "perception_model names no gallery and gallery_url is empty"
        except (GalleryUnavailable, ValueError, OSError) as error:
            self.gallery_reason = str(error)
        if loaded is None:
            models.propose(Image.new("RGB", (64, 64)), GENERIC_PROMPTS)
            self._gallery, self._models, self._prototypes = None, models, None
            logger.warning("sam3_siglip: no gallery (%s): naming by words, a scan looks for %s", self.gallery_reason, ", ".join(DEFAULT_VOCABULARY))
            return
        if loaded.prototypes is not None:
            prototypes = loaded.prototypes
            how = "prototypes shipped with the pack"
        else:
            prototypes = self._embed_prototypes(loaded, models)
            how = f"{len(loaded.crops)} reference crops embedded"
        # The first CUDA call pays for the kernels; take it here, not on the
        # first search.
        models.propose(Image.new("RGB", (64, 64)), GENERIC_PROMPTS)
        self._gallery, self._models, self._prototypes = loaded, models, prototypes
        logger.info("sam3_siglip: %d items from %s, %s, on %s", len(loaded.classes), loaded.name, how, models.device)

    @staticmethod
    def _embed_prototypes(gallery: Gallery, models: "Models") -> np.ndarray:
        """One prototype per item: the mean SigLIP embedding of its crops,
        each grown by a tenth when it is a box in a frame."""
        from PIL import Image

        sums: Optional[np.ndarray] = None
        counts = np.zeros(len(gallery.classes), dtype=np.int64)
        by_image: dict[str, list[Crop]] = {}
        for crop in gallery.crops:
            by_image.setdefault(crop.image, []).append(crop)
        for image_key, crops in by_image.items():
            with Image.open(gallery.image_path(crops[0])) as image:
                image = image.convert("RGB")
                pieces = [image.crop(grown(c.box, CROP_MARGIN, image.width, image.height)) if c.box else image for c in crops]
                embeddings = models.embed_images(pieces)
            if sums is None:
                sums = np.zeros((len(gallery.classes), embeddings.shape[1]), dtype=np.float32)
            for crop, embedding in zip(crops, embeddings):
                sums[crop.class_index] += embedding
                counts[crop.class_index] += 1
        if sums is None or (counts == 0).any():
            missing = [gallery.classes[i] for i in range(len(gallery.classes)) if counts[i] == 0]
            raise ValueError(f"gallery {gallery.name} has no crop for {missing}")
        return normalised(sums)

    def set_vocabulary(self, phrases: Sequence[str]) -> None:
        self._vocabulary = list(phrases)

    def detect(self, image: np.ndarray) -> list[Box]:
        from PIL import Image

        if self._models is None:
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
        if plan.naming == GALLERY_NAMING and self._prototypes is not None:
            prototypes = self._prototypes
        else:
            prototypes = self._text_prototypes.get(plan.labels)
            if prototypes is None:
                prototypes = self._models.embed_texts(
                    [f"a photo of a {p}" for p in plan.labels] + [f"a photo of {b}" for b in BACKGROUND_PHRASES]
                )
                self._text_prototypes[plan.labels] = prototypes
        best, probability = name_by_prototypes(embeddings, prototypes, TEMPERATURE)
        background = ()
        if plan.naming == GALLERY_NAMING and self._gallery is not None:
            background = self._gallery.background
            # Naming by picture: a box that resembles no prototype enough is
            # nothing enrolled, whatever the softmax made of it.
            probability = np.where(under_floor(embeddings, prototypes, SIMILARITY_FLOOR), 0.0, probability)
        return detections_from(boxes, objectness, best, probability, plan.labels, self.min_confidence, background)
