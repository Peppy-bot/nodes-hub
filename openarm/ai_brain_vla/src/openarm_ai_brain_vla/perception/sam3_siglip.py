"""SAM 3 to find, SigLIP to name: `perception_backend: "sam3_siglip"`.

SAM 3 proposes the objects in a frame and SigLIP names each crop by words.
The node ships no pictures of items, so what a scan reports is whatever
SigLIP can name, not a list someone enrolled. The two models' weights are
staged in the node image at build (weights.py): a load reads them from
there, and the backend fetches nothing.

- A scan prompts SAM 3 with the one word "object" and names every crop by
  the nearest of the scan vocabulary (`vocabulary.txt`, the LVIS v1 category
  names) and the background phrases, by SigLIP's image-to-text similarity; a
  crop nearest a background phrase (an empty table, the robot's own arm) is
  dropped.
- A search for a description prompts SAM 3 with the description. A crop is
  the described item when it resembles the words more than the background
  phrases and at most `VOCABULARY_MARGIN` less than the vocabulary's nearest
  name, so a bowl is not returned for "mug".
- An enrolment gallery (gallery.py), named by `perception_gallery`, adds
  pictures of particular items: a crop whose embedding is at
  `SIMILARITY_FLOOR` or nearer an enrolled item's prototype takes that
  item's name, and a description that names an enrolled item runs the scan
  with the item's name added to SAM 3's prompt, the core picking the item
  by name afterwards.
- `perception_model` names a directory the container can see holding other
  SAM 3 weights, as transformers saves a model; empty is the staged ones.

A search keeps its deadline between its three stages: before SAM 3
proposes, before SigLIP embeds the crops, and before the crops are named.
Past the deadline the search stops there and the core refuses it.

Measured on the 300 chest-camera frames of a Waldo harvest of the
catalogue's table items (947 items with their boxes, a box right at IoU
0.5), at the default confidence, on an A10: a scan found 77.2% of the
items with 0.40 boxes a frame on nothing, most of them on the robot's own
gripper, in about 1 s a frame; a search for an item in view by its
catalogue name returned it for 77.9%, another item for 1.0%, and nothing
for 19.3%; a search for an item not in view returned something for 4.7%.

SAM 3's objectness is its query score alone. Its presence score, SAM 3's
own guess whether the prompt's concept is in the frame, refuses concepts
its queries box well: on those frames it put "sugar box" at 0.03 over a box
its query scored 0.89 at IoU 0.98, and multiplied in it left 37% of the
searches for items in view with nothing. The vocabulary check takes its
place against items that are not there: without it, a search for an item
not in view returned something for 36%.

SAM 3 is prompted with the one word "object" for a scan, not with names.
Measured on the perception study's Waldo frames with a 28-item gallery, on
the same A10, that scan found 90.6% of items against 87.0% for the 28 name
prompts, in 0.94 s a frame against 3.4 s, and its time does not grow with
what it looks for.

`SIMILARITY_FLOOR` guards an enrolled name: the softmax names every crop
after the nearest prototype however far it is. Measured on the study's
Waldo frames against a 120-row gallery, boxes on a real item have a best
cosine of 0.82 or more for 95% of them (median 0.94), phantom boxes a
median of 0.80 and at most 0.86. At 0.80 the floor drops 25 of 37 phantoms
for 4 of 288 true items. A crop under it is named by the vocabulary.

The other settings are the study's: proposals at objectness 0.05,
class-agnostic NMS at IoU 0.6, at most 40 a frame, crops grown by a tenth,
SigLIP so400m in half precision, a softmax at temperature 100 over cosine
similarities. A detection is kept at objectness times naming probability of
0.25 and above.

Everything around the models is plain numpy and tested without them; torch
and transformers are imported by `load` alone, so selecting another backend
never imports them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ..ports import Box, Coverage, Deadline, iou
from . import weights
from .gallery import Gallery, load_gallery

logger = logging.getLogger(__name__)

# The study's settings, in the order the pipeline applies them.
PROPOSAL_CONF = 0.05
PROPOSAL_IOU = 0.6
MAX_PROPOSALS = 40
CROP_MARGIN = 0.1
TEMPERATURE = 100.0
MIN_CONFIDENCE = 0.25
# The least a crop may resemble its nearest enrolled prototype to take that
# item's name (the module docstring has the measurement behind the value).
SIMILARITY_FLOOR = 0.80
# The narrowest crop sent to SigLIP. A crop one or three pixels on a side
# reads to the image processor as a channel axis; a box that thin holds no
# item anyway, so it is widened about its centre before cropping.
MIN_CROP_PX = 8
# What a crop is compared with beside the labels it may take, so a crop
# that looks like none of them is dropped rather than named: the study's
# four, and the robot's own arm and gripper, which stand in the chest
# camera's view.
BACKGROUND_PHRASES = (
    "an empty table surface",
    "a white robot gripper",
    "a plain wooden board",
    "a blue wall",
    "a grey robot arm",
    "a black robot gripper",
    "a robot arm",
    "part of a robot",
)
# How much less a crop may resemble the searched words than the nearest
# name of the vocabulary and still be what was asked for (cosine).
VOCABULARY_MARGIN = 0.03
# The scan's vocabulary: the names a crop is named among by words.
VOCABULARY_FILE = Path(__file__).with_name("vocabulary.txt")
# Phrases a text embedding call takes at once: the vocabulary is a thousand.
TEXT_BATCH = 256
# What SAM 3 is asked for when the search is for everything in view: one
# generic prompt (see the module docstring for the measurement).
GENERIC_PROMPTS = ("object",)


def load_vocabulary(path: Path = VOCABULARY_FILE) -> tuple[str, ...]:
    """The names in a vocabulary file: one a line, `#` lines are comments."""
    lines = (line.strip() for line in path.read_text().splitlines())
    return tuple(line for line in lines if line and not line.startswith("#"))


class Route(Enum):
    """The two ways a search names its boxes (see `plan_for`)."""

    SCAN = "scan"
    WORDS = "words"


@dataclass(frozen=True)
class Plan:
    """What one search runs: the phrases SAM 3 is prompted with, the labels
    a box is named among by words, and the route that names the boxes."""

    prompts: tuple[str, ...]
    labels: tuple[str, ...]
    route: Route


def plan_for(gallery: Optional[Gallery], phrases: Sequence[str], vocabulary: Sequence[str]) -> Plan:
    """The search for `phrases`. A scan, no phrases, takes the scan route:
    SAM 3 is prompted with the generic prompt and every box is named by the
    vocabulary, after an enrolled item's picture when there is a gallery. A
    description that names enrolled items takes the same route with their
    phrases added to the prompt, and the core picks the item by name
    afterwards. Any other description takes the words route: SAM 3 is
    prompted with the words, and a box is named by them when it resembles
    them more than the background phrases and nearly as much as the
    vocabulary's nearest name."""
    wanted = [p.strip() for p in phrases if p.strip()]
    if not wanted:
        return Plan(GENERIC_PROMPTS, tuple(vocabulary), Route.SCAN)
    enrolled: list[str] = []
    if gallery is not None:
        for phrase in wanted:
            for i in gallery.index_of(phrase):
                if gallery.phrases[i] not in enrolled:
                    enrolled.append(gallery.phrases[i])
    if enrolled:
        return Plan(GENERIC_PROMPTS + tuple(enrolled), tuple(vocabulary), Route.SCAN)
    return Plan(tuple(wanted), tuple(wanted), Route.WORDS)


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


def name_by_words(embeddings: np.ndarray, table: np.ndarray, labels: Sequence[str]) -> tuple[list[Optional[str]], np.ndarray]:
    """Per crop, the label it is nearest by words and the probability of
    that choice. `table` holds the text embeddings of `labels` then of the
    background phrases; a crop nearest a background phrase is named None."""
    best, probability = name_by_prototypes(embeddings, table, TEMPERATURE)
    return [labels[b] if b < len(labels) else None for b in best], probability


def unlike_the_words(embeddings: np.ndarray, words_table: np.ndarray, vocabulary_table: np.ndarray, margin: float = VOCABULARY_MARGIN) -> np.ndarray:
    """Per crop, whether it resembles the searched words (the first row of
    `words_table`) less than the vocabulary's nearest name by more than
    `margin`: a bowl found for "mug" looks more like a bowl than a mug."""
    return embeddings @ words_table[0] < (embeddings @ vocabulary_table.T).max(axis=1) - margin


def name_enrolled_first(
    embeddings: np.ndarray,
    prototypes: np.ndarray,
    gallery: Gallery,
    names: Sequence[Optional[str]],
    probability: np.ndarray,
) -> tuple[list[Optional[str]], np.ndarray]:
    """The names by words with every crop that resembles an enrolled item
    (its best cosine at `SIMILARITY_FLOOR` or above) renamed after that
    item."""
    best, enrolled_probability = name_by_prototypes(embeddings, prototypes, TEMPERATURE)
    near = ~under_floor(embeddings, prototypes)
    out_names, out_probability = list(names), probability.copy()
    for i in np.flatnonzero(near):
        out_names[i] = gallery.phrases[int(best[i])]
        out_probability[i] = enrolled_probability[i]
    return out_names, out_probability


def detections_from(
    boxes: np.ndarray,
    objectness: np.ndarray,
    names: Sequence[Optional[str]],
    probability: np.ndarray,
    min_confidence: float = MIN_CONFIDENCE,
) -> list[Box]:
    """The named boxes at objectness times naming probability of
    `min_confidence` and above. A box named None, as background, is dropped:
    that is the "none of these" answer."""
    out: list[Box] = []
    for box, o, name, p in zip(boxes, objectness, names, probability):
        if name is None:
            continue
        confidence = float(o) * float(p)
        if confidence < min_confidence:
            continue
        out.append(Box(label=name, confidence=confidence, x0=float(box[0]), y0=float(box[1]), x1=float(box[2]), y1=float(box[3])))
    return out


def embed_prototypes(gallery: Gallery, models: "Models") -> np.ndarray:
    """One prototype per enrolled item: the mean SigLIP embedding of its
    crops, each grown by a tenth of its box."""
    from PIL import Image

    sums: Optional[np.ndarray] = None
    counts = np.zeros(len(gallery.classes), dtype=np.int64)
    by_image: dict[str, list] = {}
    for crop in gallery.crops:
        by_image.setdefault(crop.image, []).append(crop)
    for crops in by_image.values():
        with Image.open(gallery.image_path(crops[0])) as image:
            image = image.convert("RGB")
            pieces = [image.crop(grown(c.box, CROP_MARGIN, image.width, image.height)) for c in crops]
            embeddings = models.embed_images(pieces)
        if sums is None:
            sums = np.zeros((len(gallery.classes), embeddings.shape[1]), dtype=np.float32)
        for crop, embedding in zip(crops, embeddings):
            sums[crop.class_index] += embedding
            counts[crop.class_index] += 1
    if sums is None or (counts == 0).any():
        missing = [gallery.classes[i] for i in range(len(gallery.classes)) if counts[i] == 0]
        raise ValueError(f"gallery {gallery.root} has no crop for {missing}")
    return normalised(sums)


def text_table(models: "Models", labels: Sequence[str]) -> np.ndarray:
    """The text embeddings a crop is named against by words: `labels`, then
    the background phrases."""
    phrases = [f"a photo of a {p}" for p in labels] + [f"a photo of {b}" for b in BACKGROUND_PHRASES]
    return np.concatenate([models.embed_texts(phrases[i : i + TEXT_BATCH]) for i in range(0, len(phrases), TEXT_BATCH)])


def sam3_weights(model: str) -> Path:
    """The directory SAM 3's weights are read from: the one
    `perception_model` names, else the staged one. Raises ValueError when
    the parameter names something that is not a directory: a repository on
    the Hub is not fetched."""
    name = model.strip()
    if not name:
        return weights.staged(weights.SAM3)
    directory = Path(name).expanduser()
    if not directory.is_dir():
        raise ValueError(f"perception_model {directory} is not a directory the node can see")
    return directory


class Models:
    """The two models on one device, each read from its directory with no
    network, and the three calls the backend makes on them. Imports torch
    and transformers at construction and nowhere else."""

    def __init__(self, sam3_directory: Path, siglip_directory: Path, device: Optional[str] = None) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor, Sam3Model, Sam3Processor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.half = self.device.startswith("cuda")
        self.sam3_processor = Sam3Processor.from_pretrained(sam3_directory, local_files_only=True)
        self.sam3 = Sam3Model.from_pretrained(sam3_directory, local_files_only=True).to(self.device).eval()
        self.siglip_processor = AutoProcessor.from_pretrained(siglip_directory, local_files_only=True)
        siglip_dtype = torch.float16 if self.half else torch.float32
        self.siglip = AutoModel.from_pretrained(siglip_directory, dtype=siglip_dtype, local_files_only=True).to(self.device).eval()

    def propose(self, image, prompts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """Every instance SAM 3 finds for any of `prompts`: boxes in pixels
        and their objectness, the query score alone. SAM 3's presence score,
        its own guess whether the prompt's concept is in the frame at all,
        is left out: it refuses concepts its queries box well (the module
        docstring has the measurement), and SigLIP decides what a box is.
        The frame is encoded once and the detector decoder runs once per
        prompt."""
        torch = self.torch
        width, height = image.size
        scale = np.array([width, height, width, height], dtype=np.float32)
        boxes, scores = [], []
        with torch.inference_mode():
            pixel_values = self.sam3_processor(images=image, return_tensors="pt").to(self.device)["pixel_values"]
            vision = self.sam3.get_vision_features(pixel_values=pixel_values)
            for prompt in prompts:
                text = self.sam3_processor(text=prompt, return_tensors="pt").to(self.device)
                outputs = self.sam3(vision_embeds=vision, input_ids=text["input_ids"], attention_mask=text.get("attention_mask"))
                score = outputs.pred_logits.sigmoid()[0]
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

    def __init__(self, models_factory=None, vocabulary: Optional[Sequence[str]] = None) -> None:
        # Builds the two models from SAM 3's directory and SigLIP's.
        self._models_factory = models_factory or Models
        # What a scan names a crop among by words.
        self.scan_vocabulary: tuple[str, ...] = tuple(vocabulary) if vocabulary is not None else load_vocabulary()
        self._models: Optional[Models] = None
        self._gallery: Optional[Gallery] = None
        self._prototypes: Optional[np.ndarray] = None
        self._search: list[str] = []
        # The text table of the scan vocabulary, made once at load: a
        # description's table is made for its search and dropped with it.
        self._vocabulary_table: Optional[np.ndarray] = None
        # The confidence a detection is kept at, objectness times naming
        # probability; the brain sets it from the perception_confidence
        # parameter before the load.
        self.min_confidence = MIN_CONFIDENCE

    @property
    def available(self) -> bool:
        return self._models is not None

    def load(self, model: str, gallery: str) -> None:
        """Reads the enrolment gallery `gallery` names, if any, then loads
        the two models from their staged weights, SAM 3 from the directory
        `model` names when that is set, and embeds the scan vocabulary and
        the enrolled items' crops. A named gallery that cannot be read and
        weights that are not where they are read from fail the load with
        the reason, before the models take their minute; so do models that
        cannot be loaded."""
        from PIL import Image

        enrolment = load_gallery(gallery)
        sam3_directory = sam3_weights(model)
        siglip_directory = weights.staged(weights.SIGLIP)
        try:
            models = self._models_factory(sam3_directory, siglip_directory)
        except ImportError as error:
            raise RuntimeError(f"the sam3_siglip backend needs torch and transformers (the node's sam3-siglip extra): {error}") from error
        prototypes = embed_prototypes(enrolment, models) if enrolment is not None else None
        self._vocabulary_table = text_table(models, self.scan_vocabulary)
        # The first CUDA call pays for the kernels; take it here, not on the
        # first search.
        models.propose(Image.new("RGB", (64, 64)), GENERIC_PROMPTS)
        self._gallery, self._models, self._prototypes = enrolment, models, prototypes
        enrolled = f"{len(enrolment.classes)} enrolled items from {enrolment.root}" if enrolment is not None else "no enrolment gallery"
        logger.info("sam3_siglip: a vocabulary of %d names, %s, on %s", len(self.scan_vocabulary), enrolled, models.device)

    def set_vocabulary(self, phrases: Sequence[str]) -> None:
        self._search = list(phrases)

    def scan_coverage(self) -> Coverage:
        """A scan names the vocabulary's names, and the enrolled items'
        phrases when there is a gallery."""
        if self._models is None:
            return Coverage()
        enrolled = self._gallery.phrases if self._gallery is not None else ()
        return Coverage(frozenset(self.scan_vocabulary) | frozenset(enrolled))

    def detect(self, image: np.ndarray, deadline: Deadline) -> list[Box]:
        from PIL import Image

        if self._models is None or self._vocabulary_table is None:
            return []
        plan = plan_for(self._gallery, self._search, self.scan_vocabulary)
        frame = Image.fromarray(np.ascontiguousarray(image))
        deadline.check()
        boxes, objectness = self._models.propose(frame, plan.prompts)
        above = objectness >= PROPOSAL_CONF
        boxes, objectness = boxes[above], objectness[above]
        keep = class_agnostic_nms(boxes, objectness, PROPOSAL_IOU, MAX_PROPOSALS)
        if not keep:
            return []
        boxes, objectness = boxes[keep], objectness[keep]
        crops = [frame.crop(grown(box, CROP_MARGIN, frame.width, frame.height)) for box in boxes]
        deadline.check()
        embeddings = self._models.embed_images(crops)
        deadline.check()
        if plan.route is Route.WORDS:
            table = text_table(self._models, plan.labels)
            names, probability = name_by_words(embeddings, table, plan.labels)
            unlike = unlike_the_words(embeddings, table, self._vocabulary_table[: len(self.scan_vocabulary)])
            names = [None if far else name for name, far in zip(names, unlike)]
        else:
            names, probability = name_by_words(embeddings, self._vocabulary_table, plan.labels)
            if self._gallery is not None and self._prototypes is not None:
                names, probability = name_enrolled_first(embeddings, self._prototypes, self._gallery, names, probability)
        return detections_from(boxes, objectness, names, probability, self.min_confidence)
