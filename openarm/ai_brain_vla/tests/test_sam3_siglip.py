"""The sam3_siglip backend around its models: the gallery, the search plan,
the proposal selection and the naming, all without torch. The models
themselves are exercised by test_sam3_siglip_models.py, which needs them
installed and skips otherwise."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from openarm_ai_brain_vla.perception import make_detector
from openarm_ai_brain_vla.perception.sam3_siglip import (
    BACKGROUND_PHRASES,
    GALLERY_NAMING,
    GENERIC_PROMPTS,
    MIN_CONFIDENCE,
    TEXT_NAMING,
    Sam3SiglipDetector,
    class_agnostic_nms,
    detections_from,
    grown,
    load_gallery,
    name_by_prototypes,
    normalised,
    plan_for,
)


def write_gallery(root: Path, *, prompts: bool = True, drop_class: str = "") -> Path:
    """A three-item gallery of tiny images, in the harvester's layout: one
    crop of each item at full visibility, one occluded crop that does not
    count, and one item that is on the table but boxless."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "images").mkdir(exist_ok=True)
    classes = ["coffee_can", "cracker_box", "banana"]
    (root / "classes.txt").write_text("\n".join(classes) + "\n")
    if prompts:
        (root / "prompts.txt").write_text("coffee can\ncracker box\nbanana\n")
    for name in ("000_chest.png", "001_chest.png"):
        Image.new("RGB", (64, 48), (120, 90, 60)).save(root / "images" / name)
    records = [
        {
            "image": "images/000_chest.png",
            "objects_on_table": [
                {"class": "coffee_can", "bbox_xyxy_px": [4, 4, 20, 30], "visible_fraction": 1.0},
                {"class": "cracker_box", "bbox_xyxy_px": [30, 6, 60, 40], "visible_fraction": 0.95},
                {"class": "banana", "bbox_xyxy_px": None, "visible_fraction": 0.0},
            ],
        },
        {
            "image": "images/001_chest.png",
            "objects_on_table": [
                {"class": "banana", "bbox_xyxy_px": [10, 10, 40, 22], "visible_fraction": 0.9},
                {"class": "coffee_can", "bbox_xyxy_px": [42, 20, 60, 44], "visible_fraction": 0.4},
                {"class": "mug", "bbox_xyxy_px": [1, 1, 9, 9], "visible_fraction": 1.0},
            ],
        },
    ]
    if drop_class:
        for record in records:
            record["objects_on_table"] = [o for o in record["objects_on_table"] if o["class"] != drop_class]
    (root / "manifest.json").write_text(json.dumps({"classes": classes, "images": records}))
    return root


def test_the_gallery_reads_the_items_and_their_visible_crops(tmp_path):
    gallery = load_gallery(write_gallery(tmp_path / "g"))
    assert gallery.classes == ("coffee_can", "cracker_box", "banana")
    assert gallery.phrases == ("coffee can", "cracker box", "banana")
    # The occluded coffee can, the boxless banana and the unknown mug are
    # left out; the rest are the crops the prototypes are built from.
    crops = sorted((c.class_index, Path(c.image).name, c.box) for c in gallery.crops)
    assert crops == [
        (0, "000_chest.png", (4.0, 4.0, 20.0, 30.0)),
        (1, "000_chest.png", (30.0, 6.0, 60.0, 40.0)),
        (2, "001_chest.png", (10.0, 10.0, 40.0, 22.0)),
    ]


def test_a_gallery_without_prompts_names_items_by_their_class(tmp_path):
    gallery = load_gallery(write_gallery(tmp_path / "g", prompts=False))
    assert gallery.phrases == ("coffee can", "cracker box", "banana")


def test_a_gallery_that_is_not_one_is_refused_with_the_reason(tmp_path):
    with pytest.raises(ValueError, match="must be the gallery directory"):
        load_gallery(tmp_path / "missing")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no manifest.json"):
        load_gallery(empty)
    # An item without a usable crop has no prototype, so it is refused
    # rather than named from nothing.
    with pytest.raises(ValueError, match=r"no crop .* \['banana'\]"):
        load_gallery(write_gallery(tmp_path / "g", drop_class="banana"))


def test_a_description_finds_the_gallery_item_it_names(tmp_path):
    gallery = load_gallery(write_gallery(tmp_path / "g"))
    assert gallery.index_of("cracker box") == [1]
    assert gallery.index_of("Cracker_Box") == [1]
    assert gallery.index_of("the coffee can please") == [0]
    # A word shared by several items names them all; the core picks after.
    assert gallery.index_of("can") == [0]
    # Two items' words in one description name neither: that is a words search.
    assert gallery.index_of("box can") == []
    assert gallery.index_of("red mug") == []
    assert gallery.index_of("   ") == []
    # Every word must be in the phrase: "coffee tin" is not the coffee can,
    # and a colour is not a match on its own.
    assert gallery.index_of("coffee tin") == []
    assert gallery.index_of("the coffee can please") == [0]
    assert gallery.index_of("red") == [] and gallery.index_of("the") == []


def test_the_plan_is_the_whole_gallery_a_known_item_or_the_text(tmp_path):
    gallery = load_gallery(write_gallery(tmp_path / "g"))
    everything = plan_for(gallery, [])
    assert everything.prompts == GENERIC_PROMPTS
    assert everything.naming == GALLERY_NAMING and everything.labels == gallery.phrases
    # A known item: the scan's generic prompt plus the item's own name, every
    # box named against the whole gallery, and the core picks the item by
    # name afterwards; a cracker box found for "coffee can" is called a
    # cracker box and never returned for it.
    known = plan_for(gallery, ["coffee can"])
    assert known.prompts == GENERIC_PROMPTS + ("coffee can",) and known.naming == GALLERY_NAMING and known.labels == gallery.phrases
    assert plan_for(gallery, ["the coffee can please"]) == known
    assert plan_for(gallery, ["can"]).prompts == GENERIC_PROMPTS + ("coffee can",)
    assert plan_for(gallery, ["blue can"]).naming == TEXT_NAMING
    # An unknown description: searched open-vocabulary and named by text.
    unknown = plan_for(gallery, ["red mug"])
    assert unknown == plan_for(gallery, ["  red mug "])
    assert unknown.prompts == ("red mug",) and unknown.naming == TEXT_NAMING and unknown.labels == ("red mug",)


def test_class_agnostic_nms_keeps_the_most_confident_of_overlapping_boxes():
    boxes = np.array([
        [0, 0, 10, 10],     # the coffee can, called a coffee can
        [1, 1, 11, 11],     # the same coffee can, called a cracker box: goes
        [50, 50, 60, 60],   # another item
        [52, 50, 62, 60],   # overlapping it, weaker: goes
        [0, 0, 5, 5],       # a quarter of the first box, IoU 0.25: stays
    ], dtype=np.float32)
    scores = np.array([0.6, 0.9, 0.5, 0.4, 0.3], dtype=np.float32)
    assert class_agnostic_nms(boxes, scores, 0.6, 40) == [1, 2, 4]
    assert class_agnostic_nms(boxes, scores, 0.6, 2) == [1, 2]
    assert class_agnostic_nms(boxes[:0], scores[:0], 0.6, 40) == []


def test_crops_grow_by_the_margin_and_stay_inside_the_image():
    assert grown((10, 10, 30, 50), 0.1, 100, 100) == (8, 6, 32, 54)
    assert grown((0, 0, 30, 50), 0.1, 100, 100) == (0, 0, 33, 55)
    assert grown((80, 70, 100, 100), 0.1, 100, 100) == (78, 67, 100, 100)
    # A box one pixel wide is widened to eight about its centre: 10.5 +- 4.
    assert grown((10, 10, 11, 40), 0.1, 100, 100) == (6, 7, 14, 43)
    # Widened at the image's edge, the crop shifts inwards rather than shrinks.
    assert grown((0, 50, 1, 60), 0.0, 100, 100) == (0, 50, 8, 60)
    assert grown((99, 50, 100, 60), 0.0, 100, 100) == (92, 50, 100, 60)


def test_naming_takes_the_nearest_prototype_with_a_sharp_softmax():
    prototypes = normalised(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
    embeddings = normalised(np.array([[0.9, 0.1, 0.0], [0.1, 0.0, 0.95], [0.6, 0.6, 0.0]]))
    best, probability = name_by_prototypes(embeddings, prototypes, 100.0)
    assert best.tolist() == [0, 2, 0]
    # At temperature 100 a clear match is nearly certain, a tie is not.
    assert probability[0] > 0.99 and probability[1] > 0.99
    assert abs(probability[2] - 0.5) < 0.01


def test_detections_keep_the_confident_named_boxes_and_drop_the_background():
    boxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50], [60, 60, 70, 70]], dtype=np.float32)
    objectness = np.array([0.9, 0.9, 0.3, 0.9])
    best = np.array([0, 1, 0, 2])            # the last is named as background
    probability = np.array([0.95, 0.5, 0.95, 0.99])
    labels = ("coffee can", "banana")
    out = detections_from(boxes, objectness, best, probability, labels)
    # 0.3 x 0.95 = 0.285 is kept at the study's threshold of 0.25; the
    # background box goes whatever its confidence.
    assert MIN_CONFIDENCE == 0.25
    assert [(d.label, round(d.confidence, 3)) for d in out] == [
        ("coffee can", 0.855),
        ("banana", 0.45),
        ("coffee can", 0.285),
    ]
    assert (out[0].x0, out[0].y0, out[0].x1, out[0].y1) == (0.0, 0.0, 10.0, 10.0)
    assert len(detections_from(boxes[2:3], objectness[2:3] * 0.5, best[2:3], probability[2:3], labels)) == 0


def test_the_backend_is_registered_and_unavailable_until_loaded():
    detector = make_detector("sam3_siglip")
    assert isinstance(detector, Sam3SiglipDetector)
    assert detector.name == "sam3_siglip"
    assert not detector.available
    detector.set_vocabulary(["banana"])
    assert detector.detect(np.zeros((4, 4, 3), dtype=np.uint8)) == []


def test_loading_without_the_models_says_which_extra(tmp_path):
    """The gallery is optional, the models are not: without torch the load
    fails naming the extra that installs them, whatever the gallery is."""
    pytest.importorskip("numpy")
    try:
        import torch  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("torch is installed here: the models would load")
    with pytest.raises(RuntimeError, match="sam3-siglip extra"):
        Sam3SiglipDetector().load("")
    with pytest.raises(RuntimeError, match="sam3-siglip extra"):
        Sam3SiglipDetector().load(str(write_gallery(tmp_path / "g")))

def test_the_background_phrases_are_the_studys():
    assert BACKGROUND_PHRASES[0] == "an empty table surface" and len(BACKGROUND_PHRASES) == 4


# ----------------------------------------------------------------- without a gallery

class FakeModels:
    """Stands in for the two models: one box per call, and embeddings that
    make every crop look like the first label."""

    device = "fake"

    def __init__(self) -> None:
        self.prompts: list[tuple[str, ...]] = []

    def propose(self, image, prompts):
        self.prompts.append(tuple(prompts))
        return np.array([[10.0, 10.0, 50.0, 40.0]], dtype=np.float32), np.array([0.9], dtype=np.float32)

    def embed_images(self, crops):
        return np.eye(16, dtype=np.float32)[: len(crops)]

    def embed_texts(self, phrases):
        return np.eye(16, dtype=np.float32)[: len(phrases)]


def test_without_a_gallery_the_plan_is_text_for_the_phrases_or_the_default_vocabulary():
    from openarm_ai_brain_vla.perception.sam3_siglip import DEFAULT_VOCABULARY

    assert plan_for(None, ["blue ball"]) == plan_for(None, [" blue ball "])
    assert plan_for(None, ["blue ball"]).naming == TEXT_NAMING and plan_for(None, ["blue ball"]).labels == ("blue ball",)
    scan = plan_for(None, [])
    assert scan.naming == TEXT_NAMING and scan.prompts == DEFAULT_VOCABULARY and scan.labels == DEFAULT_VOCABULARY


def test_no_gallery_leaves_the_backend_working_by_words(tmp_path):
    from openarm_ai_brain_vla.perception.sam3_siglip import DEFAULT_VOCABULARY

    detector = Sam3SiglipDetector(models_factory=FakeModels)
    detector.load("")
    assert detector.available and detector.gallery_reason == "perception_model names no gallery and gallery_url is empty"
    detector.set_vocabulary(["blue ball"])
    boxes = detector.detect(np.zeros((60, 80, 3), dtype=np.uint8))
    assert [(b.label, round(b.confidence, 2)) for b in boxes] == [("blue ball", 0.9)]
    detector.set_vocabulary([])
    boxes = detector.detect(np.zeros((60, 80, 3), dtype=np.uint8))
    assert [b.label for b in boxes] == [DEFAULT_VOCABULARY[0]]
    assert detector._models.prompts[-1] == DEFAULT_VOCABULARY


def test_a_gallery_that_cannot_be_had_is_a_reason_not_a_failure(tmp_path):
    detector = Sam3SiglipDetector(models_factory=FakeModels)
    detector.load(str(tmp_path / "missing"))
    assert detector.available and "does not exist" in detector.gallery_reason
    detector.load((tmp_path / "sources" / "openarm_item_gallery" / ("c" * 64)).as_uri())
    assert detector.available and "could not be fetched" in detector.gallery_reason


def test_with_a_gallery_the_prototypes_are_built_and_the_reason_is_empty(tmp_path):
    detector = Sam3SiglipDetector(models_factory=FakeModels)
    detector.load(str(write_gallery(tmp_path / "g")))
    assert detector.available and detector.gallery_reason == ""
    assert detector._prototypes is not None and detector._prototypes.shape[0] == 3
    detector.set_vocabulary([])
    boxes = detector.detect(np.zeros((60, 80, 3), dtype=np.uint8))
    assert [b.label for b in boxes] == ["coffee can"]


def test_a_pack_loads_its_prototypes_instead_of_embedding_crops(tmp_path):
    from openarm_ai_brain_vla.perception.gallery_store import open_pack
    from test_gallery_store import write_pack

    class CountingModels(FakeModels):
        embedded = 0

        def embed_images(self, crops):
            CountingModels.embedded += len(crops)
            return super().embed_images(crops)

    url, root, cache = write_pack(tmp_path / "store", dim=16)
    detector = Sam3SiglipDetector(models_factory=CountingModels)
    detector.load("", url)
    assert detector.available and detector.gallery_reason == ""
    assert detector._prototypes.shape == (4, 16) and CountingModels.embedded == 0
    assert detector._gallery.phrases == ("cube", "apple", "lemon", "robot arm")
    detector.set_vocabulary(["apple"])
    detector.detect(np.zeros((60, 80, 3), dtype=np.uint8))
    assert detector._models.prompts[-1] == GENERIC_PROMPTS + ("apple",)
    # perception_model "none" beats the gallery_url: words only.
    other = Sam3SiglipDetector(models_factory=FakeModels)
    other.load("none", url)
    assert other.available and other._gallery is None and other.gallery_reason == 'perception_model is "none"'


def test_a_box_named_as_background_is_dropped_and_the_threshold_is_the_detectors():
    boxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]], dtype=np.float32)
    objectness = np.array([0.9, 0.9, 0.9], dtype=np.float32)
    best = np.array([0, 2, 1]); probability = np.array([1.0, 1.0, 0.2], dtype=np.float32)
    labels = ("cube", "apple", "robot arm"); background = (False, False, True)
    kept = detections_from(boxes, objectness, best, probability, labels, 0.25, background)
    assert [b.label for b in kept] == ["cube"]
    kept = detections_from(boxes, objectness, best, probability, labels, 0.10, background)
    assert [b.label for b in kept] == ["cube", "apple"]


def test_the_pack_backgrounds_are_dropped_by_the_detector(tmp_path):
    from test_gallery_store import write_pack

    class ArmModels(FakeModels):
        """Every crop looks like the last prototype, the robot arm."""

        def embed_images(self, crops):
            e = np.zeros((len(crops), 16), dtype=np.float32); e[:, 3] = 1.0
            return e

    url, root, cache = write_pack(tmp_path / "store", dim=16)
    detector = Sam3SiglipDetector(models_factory=ArmModels)
    detector.load("", url)
    # A table where each row is its own axis, so the crop's axis 3 is the robot arm.
    detector._prototypes = np.eye(4, 16, dtype=np.float32)
    detector.set_vocabulary([])
    assert detector.detect(np.zeros((60, 80, 3), dtype=np.uint8)) == []
    detector.min_confidence = 0.10
    assert detector.detect(np.zeros((60, 80, 3), dtype=np.uint8)) == []


def test_a_box_that_resembles_no_prototype_enough_is_nothing():
    from openarm_ai_brain_vla.perception.sam3_siglip import SIMILARITY_FLOOR, under_floor

    prototypes = normalised(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32))
    near = normalised(np.array([[0.95, 0.3, 0.0]], dtype=np.float32))
    far = normalised(np.array([[0.5, 0.5, 0.7]], dtype=np.float32))
    assert under_floor(near, prototypes).tolist() == [False]
    assert under_floor(far, prototypes).tolist() == [True]
    assert 0.75 <= SIMILARITY_FLOOR <= 0.9


def test_the_floor_drops_a_confident_phantom_on_the_gallery_route(tmp_path):
    from test_gallery_store import write_pack

    class FarModels(FakeModels):
        """Every crop is equally unlike every prototype, so the softmax is
        flat and the best cosine is low."""

        def embed_images(self, crops):
            e = np.ones((len(crops), 16), dtype=np.float32) / 4.0
            return e

    url, root, cache = write_pack(tmp_path / "store", dim=16)
    detector = Sam3SiglipDetector(models_factory=FarModels)
    detector.load("", url)
    detector._prototypes = np.eye(4, 16, dtype=np.float32)
    detector.set_vocabulary([])
    detector.min_confidence = 0.10
    assert detector.detect(np.zeros((60, 80, 3), dtype=np.uint8)) == []
