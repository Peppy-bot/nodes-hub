"""The enrolment gallery a launch names by directory: a release, a
harvester dataset, or nothing."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from openarm_ai_brain_vla.perception.gallery import INDEX_FILE, load_gallery, phrase_for

CLASSES = ["cube", "apple", "lemon_polyhaven", "robot_arm"]
LABELS = {"apple": "YCB apple", "lemon_polyhaven": "Lemon", "robot_arm": "robot arm"}
BACKGROUND = {"robot_arm"}


def write_release(root: Path, *, dim: int = 8, frames: bool = True, prototypes: bool = True) -> Path:
    """A tiny release under `root`: two crops per class, and with `frames`
    two frames whose manifest boxes the item classes, none the background
    one; with `prototypes` the SigLIP table the index names."""
    objects = {}
    for i, name in enumerate(CLASSES):
        crops = []
        for n in range(2):
            path = f"crops/{name}/00{n}_chest_{n}.jpg"
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (24 + 4 * i, 20), (40 * i, 90, 60)).save(root / path, format="JPEG")
            crops.append(path)
        objects[name] = {"crops": crops, "background": name in BACKGROUND, **({"label": LABELS[name]} if name in LABELS else {})}
    index = {"classes": CLASSES, "objects": objects}
    if frames:
        (root / "frames").mkdir(exist_ok=True)
        records = []
        for n in range(2):
            Image.new("RGB", (96, 64), (70, 80, 90)).save(root / "frames" / f"00{n}_chest.jpg", format="JPEG")
            records.append({"image": f"frames/00{n}_chest.jpg", "objects_on_table": [
                {"class": "cube", "bbox_xyxy_px": [4 + n, 4, 30, 30], "visible_fraction": 1.0},
                {"class": "apple", "bbox_xyxy_px": [40, 8, 70, 40], "visible_fraction": 0.9 if n == 0 else 0.3},
                {"class": "lemon_polyhaven", "bbox_xyxy_px": [72, 30, 94, 60], "visible_fraction": 1.0},
            ]})
        (root / "manifest.json").write_text(json.dumps({"images": records}) + "\n")
        index["frames"] = {"dir": "frames", "manifest": "manifest.json", "count": 2}
    if prototypes:
        table = np.random.default_rng(0).normal(size=(len(CLASSES), dim)).astype(np.float32)
        table /= np.linalg.norm(table, axis=1, keepdims=True)
        np.savez(root / "prototypes.npz", prototypes=table.astype(np.float16))
        index["prototypes"] = {"file": "prototypes.npz", "model": "test", "normalised": True}
    (root / INDEX_FILE).write_text(json.dumps(index, indent=1) + "\n")
    return root


def write_harvest(root: Path, *, prompts: bool = True, drop_class: str = "") -> Path:
    """A three-item harvester dataset of tiny images: one crop of each item
    at full visibility, one occluded crop that does not count, and one item
    that is on the table but boxless."""
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


def test_no_gallery_is_named_by_an_empty_model_or_none():
    assert load_gallery("") is None
    assert load_gallery("  ") is None
    assert load_gallery("none") is None and load_gallery("None") is None


def test_a_url_is_not_a_gallery_and_nothing_is_fetched(no_network):
    with pytest.raises(ValueError, match="is not a directory the node can see"):
        load_gallery("https://assets.example.r2.dev/galleries/gallery.lock.json")


def test_a_named_gallery_that_is_not_one_is_refused_with_the_reason(tmp_path):
    with pytest.raises(ValueError, match="is not a directory the node can see"):
        load_gallery(str(tmp_path / "missing"))
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="holds no index.json or manifest.json"):
        load_gallery(str(tmp_path / "empty"))
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / INDEX_FILE).write_text("not json")
    with pytest.raises(ValueError, match="is not JSON"):
        load_gallery(str(broken))
    (broken / INDEX_FILE).write_text('{"classes": []}')
    with pytest.raises(ValueError, match="names no classes"):
        load_gallery(str(broken))


def test_phrases_come_from_labels_less_the_dataset_prefix():
    assert phrase_for("apple", {"label": "YCB apple"}) == "apple"
    assert phrase_for("sponge", {"label": "YCB rigid sponge"}) == "sponge"
    assert phrase_for("food_apple_01", {"label": "Red apple"}) == "red apple"
    assert phrase_for("cube", {}) == "cube"
    assert phrase_for("wood_block", {"label": ""}) == "wood block"


def test_a_release_reads_classes_phrases_crops_and_prototypes(tmp_path):
    gallery = load_gallery(str(write_release(tmp_path / "release")))
    assert gallery.classes == tuple(CLASSES)
    assert gallery.phrases == ("cube", "apple", "lemon", "robot arm")
    assert gallery.background == (False, False, False, True) and gallery.is_background(3)
    # With frames in the release the crops are boxes in frames: five kept,
    # the apple's second one under the visibility floor, none for the
    # background class.
    assert len(gallery.crops) == 5 and all(c.box is not None for c in gallery.crops)
    assert sorted({c.image for c in gallery.crops}) == ["frames/000_chest.jpg", "frames/001_chest.jpg"]
    assert all(gallery.image_path(c).is_file() for c in gallery.crops)
    assert gallery.prototypes is not None and gallery.prototypes.shape == (4, 8)
    np.testing.assert_allclose(np.linalg.norm(gallery.prototypes, axis=1), 1.0, atol=1e-3)
    # A description never names the background class.
    assert gallery.index_of("robot arm") == [] and gallery.index_of("arm") == []
    assert gallery.index_of("lemon") == [2] and gallery.index_of("a lemon") == [2]
    assert gallery.index_of("blue lemon") == []


def test_a_release_without_frames_or_prototypes_reads_its_own_crops(tmp_path):
    gallery = load_gallery(str(write_release(tmp_path / "release", frames=False, prototypes=False)))
    assert gallery.prototypes is None
    assert len(gallery.crops) == 8 and all(c.box is None for c in gallery.crops)
    assert all(gallery.image_path(c).is_file() for c in gallery.crops)


def test_prototypes_that_do_not_match_the_classes_are_refused(tmp_path):
    root = write_release(tmp_path / "release")
    np.savez(root / "prototypes.npz", prototypes=np.ones((3, 8), dtype=np.float16))
    with pytest.raises(ValueError, match="3 prototypes for 4 classes"):
        load_gallery(str(root))


def test_a_harvester_dataset_reads_the_items_and_their_visible_crops(tmp_path):
    gallery = load_gallery(str(write_harvest(tmp_path / "g")))
    assert gallery.classes == ("coffee_can", "cracker_box", "banana")
    assert gallery.phrases == ("coffee can", "cracker box", "banana")
    assert gallery.prototypes is None
    # The occluded coffee can, the boxless banana and the unknown mug are
    # left out; the rest are the crops the prototypes are built from.
    crops = sorted((c.class_index, Path(c.image).name, c.box) for c in gallery.crops)
    assert crops == [
        (0, "000_chest.png", (4.0, 4.0, 20.0, 30.0)),
        (1, "000_chest.png", (30.0, 6.0, 60.0, 40.0)),
        (2, "001_chest.png", (10.0, 10.0, 40.0, 22.0)),
    ]


def test_a_harvester_dataset_without_prompts_names_items_by_their_class(tmp_path):
    gallery = load_gallery(str(write_harvest(tmp_path / "g", prompts=False)))
    assert gallery.phrases == ("coffee can", "cracker box", "banana")


def test_an_item_without_a_usable_crop_is_refused_rather_than_named_from_nothing(tmp_path):
    with pytest.raises(ValueError, match=r"no crop .* \['banana'\]"):
        load_gallery(str(write_harvest(tmp_path / "g", drop_class="banana")))


def test_a_description_finds_the_gallery_item_it_names(tmp_path):
    gallery = load_gallery(str(write_harvest(tmp_path / "g")))
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
    assert gallery.index_of("red") == [] and gallery.index_of("the") == []
