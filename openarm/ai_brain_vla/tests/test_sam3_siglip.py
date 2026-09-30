"""The sam3_siglip backend around its models: the vocabulary, the search
plan, the proposal selection and the naming, all without torch. The models
themselves are exercised by test_sam3_siglip_models.py, which needs them
installed and skips otherwise."""

import numpy as np
import pytest

from conftest import NEVER, PASSED
from openarm_ai_brain_vla.perception import make_detector, weights
from openarm_ai_brain_vla.perception.gallery import load_gallery
from openarm_ai_brain_vla.ports import Coverage, SearchTimeout
from openarm_ai_brain_vla.perception.sam3_siglip import (
    BACKGROUND_PHRASES,
    GENERIC_PROMPTS,
    MIN_CONFIDENCE,
    SIMILARITY_FLOOR,
    Route,
    VOCABULARY_MARGIN,
    Sam3SiglipDetector,
    class_agnostic_nms,
    detections_from,
    grown,
    load_vocabulary,
    name_by_prototypes,
    normalised,
    plan_for,
    under_floor,
    unlike_the_words,
)
from test_gallery import write_harvest

# Every load finds the models' weights on the machine.
pytestmark = pytest.mark.usefixtures("staged_weights")

VOCABULARY = ("cup", "banana")


def test_the_vocabulary_is_a_general_one_each_name_once():
    vocabulary = load_vocabulary()
    assert len(vocabulary) == 1198 and len(set(vocabulary)) == len(vocabulary)
    assert all(name == name.lower() and "(" not in name and "_" not in name for name in vocabulary)
    assert {"cup", "mug", "bottle", "banana", "apple", "sponge", "wrench", "stapler"} <= set(vocabulary)


def test_a_vocabulary_file_skips_comments_and_blank_lines(tmp_path):
    path = tmp_path / "words.txt"
    path.write_text("# a comment\ncup\n\n  banana  \n")
    assert load_vocabulary(path) == ("cup", "banana")


def test_the_plan_is_the_vocabulary_an_enrolled_item_or_the_words(tmp_path):
    gallery = load_gallery(str(write_harvest(tmp_path / "g")))
    # A scan: the generic prompt, every box named by the vocabulary, after
    # an enrolled picture when there is a gallery.
    assert plan_for(None, [], VOCABULARY) == (scan := plan_for(None, ["  "], VOCABULARY)) == plan_for(gallery, [], VOCABULARY)
    assert scan.prompts == GENERIC_PROMPTS and scan.labels == VOCABULARY and scan.route is Route.SCAN
    # An enrolled item: the scan's generic prompt plus the item's own name,
    # named the scan's way, and the core picks the item by name afterwards;
    # a cracker box found for "coffee can" is called a cracker box and never
    # returned for it.
    known = plan_for(gallery, ["coffee can"], VOCABULARY)
    assert known.prompts == GENERIC_PROMPTS + ("coffee can",) and known.labels == VOCABULARY and known.route is Route.SCAN
    assert plan_for(gallery, ["the coffee can please"], VOCABULARY) == known
    assert plan_for(gallery, ["can"], VOCABULARY).prompts == GENERIC_PROMPTS + ("coffee can",)
    # Any other description: searched and named by its own words.
    for words in (plan_for(gallery, ["red mug"], VOCABULARY), plan_for(None, [" red mug "], VOCABULARY)):
        assert words.prompts == ("red mug",) and words.labels == ("red mug",) and words.route is Route.WORDS
    assert plan_for(gallery, ["blue can"], VOCABULARY).route is Route.WORDS


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
    names = ["coffee can", "banana", "coffee can", None]   # the last is named as background
    probability = np.array([0.95, 0.5, 0.95, 0.99])
    out = detections_from(boxes, objectness, names, probability)
    # 0.3 x 0.95 = 0.285 is kept at the study's threshold of 0.25; the
    # background box goes whatever its confidence.
    assert MIN_CONFIDENCE == 0.25
    assert [(d.label, round(d.confidence, 3)) for d in out] == [
        ("coffee can", 0.855),
        ("banana", 0.45),
        ("coffee can", 0.285),
    ]
    assert (out[0].x0, out[0].y0, out[0].x1, out[0].y1) == (0.0, 0.0, 10.0, 10.0)
    assert detections_from(boxes[2:3], objectness[2:3] * 0.5, names[2:3], probability[2:3]) == []
    assert [d.label for d in detections_from(boxes, objectness, names, probability, 0.5)] == ["coffee can"]


def test_a_box_that_resembles_no_prototype_enough_is_nothing_enrolled():
    prototypes = normalised(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32))
    near = normalised(np.array([[0.95, 0.3, 0.0]], dtype=np.float32))
    far = normalised(np.array([[0.5, 0.5, 0.7]], dtype=np.float32))
    assert under_floor(near, prototypes).tolist() == [False]
    assert under_floor(far, prototypes).tolist() == [True]
    assert 0.75 <= SIMILARITY_FLOOR <= 0.9


def test_a_search_refuses_a_box_that_looks_more_like_another_name():
    words = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)         # "mug"
    vocabulary = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    crops = normalised(np.array([
        [0.6, 0.8, 0.0],    # more a bowl than a mug: refused
        [0.8, 0.6, 0.0],    # more a mug than a bowl: kept
        [0.7, 0.71, 0.0],   # nearly as much a mug: kept within the margin
    ], dtype=np.float32))
    assert unlike_the_words(crops, words, vocabulary).tolist() == [True, False, False]
    assert 0.0 < VOCABULARY_MARGIN < 0.1


def test_the_backend_is_registered_and_unavailable_until_loaded():
    detector = make_detector("sam3_siglip")
    assert isinstance(detector, Sam3SiglipDetector)
    assert detector.name == "sam3_siglip"
    assert detector.scan_vocabulary == load_vocabulary()
    assert not detector.available
    detector.set_vocabulary(["banana"])
    assert detector.detect(np.zeros((4, 4, 3), dtype=np.uint8), NEVER) == []


def test_loading_without_the_models_says_which_extra(tmp_path):
    """The models are needed whatever the gallery: without torch the load
    fails naming the extra that installs them."""
    try:
        import torch  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("torch is installed here: the models would load")
    with pytest.raises(RuntimeError, match="sam3-siglip extra"):
        Sam3SiglipDetector().load("", "")
    with pytest.raises(RuntimeError, match="sam3-siglip extra"):
        Sam3SiglipDetector().load("", str(write_harvest(tmp_path / "g")))


def test_the_background_phrases_are_the_studys_and_the_robots_own_body():
    assert BACKGROUND_PHRASES[:4] == ("an empty table surface", "a white robot gripper", "a plain wooden board", "a blue wall")
    assert {"a robot arm", "a black robot gripper"} <= set(BACKGROUND_PHRASES)


# ----------------------------------------------------------------- with fake models

DIM = 64


class FakeModels:
    """Stands in for the two models: one box per call; every distinct text
    is a unit axis of its own, in the order first seen, and every crop is
    `looks_like`, a mix of texts by weight, so a test says what the crop
    looks like. Records the two directories it was built from."""

    device = "fake"
    looks_like = {"a photo of a cup": 1.0}

    def __init__(self, sam3_directory, siglip_directory) -> None:
        self.sam3_directory = sam3_directory
        self.siglip_directory = siglip_directory
        self.prompts: list[tuple[str, ...]] = []
        self.texts: list[str] = []
        self._axes: dict[str, int] = {}

    def axis(self, text: str) -> np.ndarray:
        return np.eye(DIM, dtype=np.float32)[self._axes.setdefault(text, len(self._axes))]

    def propose(self, image, prompts):
        self.prompts.append(tuple(prompts))
        return np.array([[10.0, 10.0, 50.0, 40.0]], dtype=np.float32), np.array([0.9], dtype=np.float32)

    def embed_images(self, crops):
        crop = normalised(sum(weight * self.axis(text) for text, weight in self.looks_like.items()))
        return np.stack([crop for _ in crops])

    def embed_texts(self, phrases):
        self.texts.extend(phrases)
        return np.stack([self.axis(p) for p in phrases])


def looking_like(**weights):
    """A FakeModels class whose crops look like the named texts; a keyword
    is the text's words with underscores, "a photo of a " before it."""
    return type("Fake", (FakeModels,), {"looks_like": {f"a photo of a {k.replace('_', ' ')}": w for k, w in weights.items()}})


def blank() -> np.ndarray:
    return np.zeros((60, 80, 3), dtype=np.uint8)


def detected(detector, phrases=(), deadline=NEVER):
    detector.set_vocabulary(list(phrases))
    return [(b.label, round(b.confidence, 2)) for b in detector.detect(blank(), deadline)]


def test_a_scan_names_what_it_finds_by_the_vocabulary():
    detector = Sam3SiglipDetector(models_factory=FakeModels, vocabulary=VOCABULARY)
    detector.load("", "")
    assert detector.available
    # The vocabulary is embedded once, at load, with the background phrases.
    texts = list(detector._models.texts)
    assert texts == ["a photo of a cup", "a photo of a banana"] + [f"a photo of {b}" for b in BACKGROUND_PHRASES]
    assert detected(detector) == [("cup", 0.9)]
    assert detector._models.prompts[-1] == GENERIC_PROMPTS
    assert detector._models.texts == texts


def test_the_models_are_built_from_the_weights_on_the_machine(staged_weights, no_network):
    detector = Sam3SiglipDetector(models_factory=FakeModels, vocabulary=VOCABULARY)
    detector.load("", "")
    assert detector._models.sam3_directory == staged_weights / weights.SAM3.directory_name
    assert detector._models.siglip_directory == staged_weights / weights.SIGLIP.directory_name


@pytest.fixture
def staging(monkeypatch):
    """Stands in for the download: records each model a load asks for and
    the directory it asks for it in, and names the model's directory."""
    asked = []

    def stage(source, directory):
        asked.append((source, directory))
        return directory / source.directory_name

    monkeypatch.setattr(weights, "stage", stage)
    return asked


def test_a_load_asks_for_both_models_in_the_directory_the_node_keeps_its_weights_in(staged_weights, staging):
    detector = Sam3SiglipDetector(models_factory=FakeModels, vocabulary=VOCABULARY)
    detector.load("", "")
    assert staging == [(weights.SAM3, staged_weights), (weights.SIGLIP, staged_weights)]
    assert detector._models.sam3_directory == staged_weights / weights.SAM3.directory_name
    assert detector._models.siglip_directory == staged_weights / weights.SIGLIP.directory_name


def test_other_sam3_weights_come_from_the_directory_the_model_parameter_names(tmp_path, staged_weights, staging):
    other = tmp_path / "other_sam3"
    other.mkdir()
    detector = Sam3SiglipDetector(models_factory=FakeModels, vocabulary=VOCABULARY)
    detector.load(f" {other} ", "")
    # The pinned SAM 3 is not asked for: a launch that names its own SAM 3
    # does not download the other.
    assert staging == [(weights.SIGLIP, staged_weights)]
    assert detector._models.sam3_directory == other
    assert detector._models.siglip_directory == staged_weights / weights.SIGLIP.directory_name


def test_a_repository_is_not_a_model_directory_and_nothing_is_downloaded(staging):
    made = []
    detector = Sam3SiglipDetector(models_factory=lambda *directories: made.append(directories), vocabulary=VOCABULARY)
    with pytest.raises(ValueError, match="perception_model facebook/sam3 is not a directory the node can see"):
        detector.load("facebook/sam3", "")
    assert not detector.available and made == [] and staging == []


def test_weights_that_cannot_be_downloaded_fail_the_load_before_the_models(monkeypatch):
    def stage(source, directory):
        raise weights.WeightsError(f"the download of {source.repository} stopped")

    monkeypatch.setattr(weights, "stage", stage)
    made = []
    detector = Sam3SiglipDetector(models_factory=lambda *directories: made.append(directories), vocabulary=VOCABULARY)
    with pytest.raises(weights.WeightsError, match="the download of jetjodh/sam3 stopped"):
        detector.load("", "")
    assert not detector.available and made == []


def test_a_scan_drops_a_box_on_the_robot_itself():
    class OnTheArm(FakeModels):
        looks_like = {"a photo of a robot arm": 1.0}

    detector = Sam3SiglipDetector(models_factory=OnTheArm, vocabulary=VOCABULARY)
    detector.load("", "")
    assert detected(detector) == []


def test_a_description_is_searched_by_its_own_words_and_its_table_is_not_kept():
    detector = Sam3SiglipDetector(models_factory=looking_like(blue_ball=1.0), vocabulary=VOCABULARY)
    detector.load("", "")
    at_load = len(detector._models.texts)
    assert detected(detector, ["blue ball"]) == [("blue ball", 0.9)]
    assert detector._models.prompts[-1] == ("blue ball",)
    # Each words search embeds its words and the background phrases again:
    # only the vocabulary's table lives for the backend's life.
    per_search = 1 + len(BACKGROUND_PHRASES)
    assert len(detector._models.texts) == at_load + per_search
    detected(detector, ["blue ball"])
    assert len(detector._models.texts) == at_load + 2 * per_search
    assert detector._vocabulary_table.shape[0] == len(VOCABULARY) + len(BACKGROUND_PHRASES)


class Ticking:
    """A deadline that passes at its n-th check, so a test says at which
    stage the search runs out of time."""

    def __init__(self, passes_at: int) -> None:
        self.checks = 0
        self.passes_at = passes_at

    def check(self) -> None:
        self.checks += 1
        if self.checks >= self.passes_at:
            raise SearchTimeout(1.0)

    def remaining_s(self) -> float:
        return 0.0 if self.checks >= self.passes_at else 1.0


def test_a_search_stops_at_its_deadline_between_its_stages():
    detector = Sam3SiglipDetector(models_factory=looking_like(blue_ball=1.0), vocabulary=VOCABULARY)
    detector.load("", "")
    proposals = len(detector._models.prompts)
    # Passed before anything: SAM 3 is not even asked.
    with pytest.raises(SearchTimeout, match="did not finish within 1 s"):
        detected(detector, ["blue ball"], PASSED)
    assert len(detector._models.prompts) == proposals
    # Passed after the proposals: SAM 3 ran, SigLIP did not embed a crop.
    embedded = len(detector._models.texts)
    with pytest.raises(SearchTimeout):
        detected(detector, ["blue ball"], Ticking(passes_at=2))
    assert len(detector._models.prompts) == proposals + 1 and len(detector._models.texts) == embedded
    # Passed after the crops are embedded: no names are made.
    with pytest.raises(SearchTimeout):
        detected(detector, ["blue ball"], Ticking(passes_at=3))
    assert len(detector._models.texts) == embedded
    # Never passed: the search answers.
    assert detected(detector, ["blue ball"], Ticking(passes_at=4)) == [("blue ball", 0.9)]


def test_a_search_does_not_return_what_looks_more_like_another_name():
    # A cup that looks a little like a mug is not returned for "mug"...
    detector = Sam3SiglipDetector(models_factory=looking_like(mug=0.6, cup=0.8), vocabulary=VOCABULARY)
    detector.load("", "")
    assert detected(detector, ["mug"]) == []
    # ...and a mug that looks a little like a cup is.
    detector = Sam3SiglipDetector(models_factory=looking_like(mug=0.8, cup=0.6), vocabulary=VOCABULARY)
    detector.load("", "")
    assert detected(detector, ["mug"]) == [("mug", 0.9)]


def test_the_backend_downloads_nothing_but_weights_the_machine_lacks(tmp_path, no_network):
    """A load and its searches reach for no network when the models'
    weights are on the machine: the vocabulary ships in the node, and a
    gallery is a directory the launch names."""
    assert len(load_vocabulary()) == 1198
    detector = Sam3SiglipDetector(models_factory=FakeModels, vocabulary=VOCABULARY)
    detector.load("", str(write_harvest(tmp_path / "g")))
    # Every crop of the harvest looks like the scan's box here, so the box
    # is an enrolled item, the first of three prototypes that are one.
    assert detected(detector) == [("coffee can", 0.3)]
    assert detected(detector, ["banana"]) == [("coffee can", 0.3)]
    assert detected(detector, ["blue ball"]) == []


def test_a_scan_covers_the_vocabulary_and_the_enrolled_items(tmp_path):
    detector = Sam3SiglipDetector(models_factory=FakeModels, vocabulary=VOCABULARY)
    assert detector.scan_coverage() == Coverage()
    detector.load("", "")
    assert detector.scan_coverage() == Coverage(frozenset(VOCABULARY))
    detector = Sam3SiglipDetector(models_factory=FakeModels, vocabulary=VOCABULARY)
    detector.load("", str(write_harvest(tmp_path / "g")))
    assert detector.scan_coverage() == Coverage(frozenset({"cup", "banana", "coffee can", "cracker box"}))


def test_a_named_gallery_that_cannot_be_read_fails_the_load_before_the_models(tmp_path):
    made = []

    def factory(sam3_directory, siglip_directory):
        made.append(sam3_directory)
        return FakeModels(sam3_directory, siglip_directory)

    detector = Sam3SiglipDetector(models_factory=factory, vocabulary=VOCABULARY)
    with pytest.raises(ValueError, match="is not a directory the node can see"):
        detector.load("", str(tmp_path / "missing"))
    assert not detector.available and made == []


def test_an_enrolled_item_is_named_by_its_pictures_built_at_load(tmp_path):
    class CountingModels(FakeModels):
        embedded = 0

        def embed_images(self, crops):
            CountingModels.embedded += len(crops)
            return super().embed_images(crops)

    detector = Sam3SiglipDetector(models_factory=CountingModels, vocabulary=VOCABULARY)
    detector.load("", str(write_harvest(tmp_path / "g")))
    # The three visible crops are embedded at load, one prototype per item;
    # every crop looks alike here, so the three prototypes are one and a
    # box near them takes the first enrolled name.
    assert CountingModels.embedded == 3
    assert detector._prototypes is not None and detector._prototypes.shape == (3, DIM)
    assert detector._gallery.phrases == ("coffee can", "cracker box", "banana")
    assert [label for label, _ in detected(detector)] == ["coffee can"]
    detected(detector, ["cracker box"])
    assert detector._models.prompts[-1] == GENERIC_PROMPTS + ("cracker box",)


def enrolled(detector, nearest: int) -> None:
    """Three prototypes on texts of their own, the one at `nearest` the
    crop itself, or none of them when `nearest` is -1."""
    models = detector._models
    rows = [models.axis(f"prototype {i}") for i in range(3)]
    if nearest >= 0:
        rows[nearest] = models.embed_images([None])[0]
    detector._prototypes = np.stack(rows)


def test_a_box_like_an_enrolled_item_takes_its_name_before_the_vocabulary(tmp_path):
    detector = Sam3SiglipDetector(models_factory=FakeModels, vocabulary=VOCABULARY)
    detector.load("", str(write_harvest(tmp_path / "g")))
    enrolled(detector, nearest=1)    # the harvest's "cracker box"
    assert detected(detector) == [("cracker box", 0.9)]


def test_a_box_unlike_every_enrolled_item_is_named_by_the_vocabulary(tmp_path):
    detector = Sam3SiglipDetector(models_factory=FakeModels, vocabulary=VOCABULARY)
    detector.load("", str(write_harvest(tmp_path / "g")))
    enrolled(detector, nearest=-1)
    assert detected(detector) == [("cup", 0.9)]
