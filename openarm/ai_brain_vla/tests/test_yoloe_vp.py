"""The yoloe_vp backend around its model: the prototype averaging, the
vocabulary mapping and the box selection, all without torch. The model is
exercised by test_yoloe_vp_models.py, which needs it installed and skips
otherwise."""

import numpy as np
import pytest

from openarm_ai_brain_vla.perception import make_detector
from openarm_ai_brain_vla.perception.sam3_siglip import load_gallery
from openarm_ai_brain_vla.perception.yoloe_vp import (
    MIN_CONFIDENCE,
    YoloeVpDetector,
    boxes_from,
    by_image,
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
    assert sorted(p.name for p in grouped) == ["000_chest.png", "001_chest.png"]
    assert sorted(c.class_index for c in grouped[gallery.root / "images" / "000_chest.png"]) == [0, 1]


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
    with pytest.raises(ValueError, match="gallery directory"):
        detector.load("  ")
    detector.set_vocabulary(["banana"])
    assert detector.detect(np.zeros((8, 8, 3), dtype=np.uint8)) == []
