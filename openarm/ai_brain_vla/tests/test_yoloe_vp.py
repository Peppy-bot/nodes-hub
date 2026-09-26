"""The yoloe_vp backend around its model: the prototype averaging, the
vocabulary mapping and the box selection, all without torch. The model is
exercised by test_yoloe_vp_models.py, which needs it installed and skips
otherwise."""

from pathlib import Path

import numpy as np
import pytest

from openarm_ai_brain_vla.perception import make_detector
from openarm_ai_brain_vla.perception.sam3_siglip import load_gallery
from openarm_ai_brain_vla.perception.yoloe_vp import (
    MIN_CONFIDENCE,
    YoloeVpDetector,
    boxes_from,
    by_image,
    prompt_boxes,
    table_rows,
    prototypes_from,
    wanted_classes,
)
from test_sam3_siglip import write_gallery


def test_prototypes_are_the_per_class_mean_normalised():
    per_image = [
        ([0, 1], np.array([[1.0, 0.0], [0.0, 2.0]])),
        ([0], np.array([[0.0, 1.0]])),
    ]
    prototypes = prototypes_from(per_image, 2)
    assert prototypes.shape == (2, 2)
    np.testing.assert_allclose(prototypes[0], [1.0, 1.0] / np.sqrt(2.0))
    np.testing.assert_allclose(prototypes[1], [0.0, 1.0])


def test_a_class_without_a_crop_is_refused_by_index():
    with pytest.raises(ValueError, match=r"\[2\]"):
        prototypes_from([([0, 1], np.ones((2, 3)))], 3)
    with pytest.raises(ValueError):
        prototypes_from([], 1)


def test_the_gallerys_crops_group_by_image(tmp_path):
    gallery = load_gallery(write_gallery(tmp_path / "g"))
    grouped = by_image(gallery.crops)
    assert sorted(Path(p).name for p in grouped) == ["000_chest.png", "001_chest.png"]
    assert sorted(c.class_index for c in grouped["images/000_chest.png"]) == [0, 1]


def test_the_vocabulary_maps_to_gallery_items(tmp_path):
    gallery = load_gallery(write_gallery(tmp_path / "g"))
    assert wanted_classes(gallery, []) == [0, 1, 2]
    assert wanted_classes(gallery, ["banana"]) == [2]
    assert wanted_classes(gallery, ["coffee_can", "cracker box"]) == [0, 1]
    assert wanted_classes(gallery, ["can"]) == [0]
    assert wanted_classes(gallery, ["wrench"]) == []


def test_boxes_keep_the_wanted_items_above_the_threshold():
    xyxy = np.array([[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50], [60, 60, 70, 70]], dtype=np.float32)
    conf = np.array([0.9, 0.3, MIN_CONFIDENCE - 0.01, 0.8], dtype=np.float32)
    cls = np.array([0, 1, 1, 2])
    boxes = boxes_from(xyxy, conf, cls, ["coffee can", "cracker box", "banana"], wanted=[0, 1])
    assert [(b.label, round(b.confidence, 2), b.x0) for b in boxes] == [("coffee can", 0.9, 0.0), ("cracker box", 0.3, 20.0)]


def test_the_registry_builds_the_backend_and_an_empty_model_is_refused():
    detector = make_detector("yoloe_vp")
    assert isinstance(detector, YoloeVpDetector) and detector.name == "yoloe_vp"
    assert not detector.available
    with pytest.raises(ValueError, match="needs a gallery"):
        detector.load("none")
    with pytest.raises(ValueError, match="needs a gallery"):
        detector.load("  ")  # no default gallery is set
    detector.set_vocabulary(["banana"])
    assert detector.detect(np.zeros((8, 8, 3), dtype=np.uint8)) == []


def test_a_gallery_that_cannot_be_had_fails_the_load_by_name(tmp_path):
    detector = YoloeVpDetector()
    with pytest.raises(RuntimeError, match="needs its gallery"):
        detector.load(str(tmp_path / "missing"))
    assert not detector.available


def test_a_crop_that_is_the_whole_image_prompts_with_the_whole_image(tmp_path):
    from openarm_ai_brain_vla.perception.sam3_siglip import Crop

    boxes = prompt_boxes([Crop("crops/a/0.jpg", None, 0), Crop("frame.png", (1.0, 2.0, 30.0, 40.0), 1)], 64, 48)
    assert boxes.tolist() == [[0.0, 0.0, 64.0, 48.0], [1.0, 2.0, 30.0, 40.0]]


def test_the_table_holds_boxed_non_background_classes_only(tmp_path):
    from openarm_ai_brain_vla.perception.gallery_store import open_pack
    from test_gallery_store import write_pack

    url, root, cache = write_pack(tmp_path / "store")
    gallery = load_gallery(open_pack(url, cache_dir=cache))
    # Only the apple stands for a YCB catalogue id; the robot arm has no frame box.
    assert table_rows(gallery) == [1]
    harvest = load_gallery(write_gallery(tmp_path / "g"))
    assert table_rows(harvest) == [0, 1, 2]


def test_a_pack_of_crops_alone_is_refused_by_name_and_a_missing_gallery_url_is_a_reason(tmp_path):
    from openarm_ai_brain_vla.perception.gallery_store import open_pack
    from test_gallery_store import write_pack

    url, root, cache = write_pack(tmp_path / "store")
    gallery = load_gallery(open_pack(url, cache_dir=cache))
    crops_only = gallery.__class__(gallery.name, gallery.classes, gallery.phrases,
                                   tuple(c.__class__(c.image, None, c.class_index) for c in gallery.crops),
                                   gallery.source, gallery.prototypes, gallery.background)
    import openarm_ai_brain_vla.perception.yoloe_vp as backend
    detector = YoloeVpDetector()
    original = backend.load_gallery
    backend.load_gallery = lambda source: crops_only
    try:
        with pytest.raises(ValueError, match="ships crops only"):
            detector.load("", url)
    finally:
        backend.load_gallery = original
    with pytest.raises(RuntimeError, match="needs its gallery"):
        detector.load("", (tmp_path / "missing.json").as_uri())
