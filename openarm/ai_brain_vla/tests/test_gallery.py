"""The enrolment gallery a launch names: an enrolment the node ships, a
directory of prototypes or a harvester dataset, or nothing."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from openarm_ai_brain_vla.perception import weights
from openarm_ai_brain_vla.perception.gallery import MANIFEST_FILE, METADATA_FILE, PROTOTYPES_FILE, Gallery, load_gallery, shipped

DIM = 8


def write_enrolment(root: Path, *, model: str = weights.SIGLIP.repository, background=("robot_arm",), prompts: bool = True, classes_file: bool = True, dim: int = DIM) -> Path:
    """A four-class enrolment in the prototypes form: three items and the
    robot's arm as background, each prototype a unit axis of its own."""
    root.mkdir(parents=True, exist_ok=True)
    classes = ["coffee_can", "cracker_box", "banana", "robot_arm"]
    prototypes = np.eye(dim, dtype=np.float32)[: len(classes)] * 3.0   # not yet normalised
    np.savez(root / PROTOTYPES_FILE, prototypes=prototypes.astype(np.float16), classes=np.array(classes))
    if classes_file:
        (root / "classes.txt").write_text("\n".join(classes) + "\n")
    if prompts:
        (root / "prompts.txt").write_text("coffee can\ncracker box\nbanana\nrobot arm\n")
    (root / METADATA_FILE).write_text(json.dumps({"format": "enrolment/v1", "model": model, "source": "a test", "background": list(background)}))
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
    (root / MANIFEST_FILE).write_text(json.dumps({"classes": classes, "images": records}))
    return root


def test_no_gallery_is_named_by_an_empty_string():
    assert load_gallery("") is None
    assert load_gallery("  ") is None


def test_a_url_is_not_a_gallery_and_nothing_is_fetched(no_network):
    with pytest.raises(ValueError, match="is not a directory the node can see"):
        load_gallery("https://assets.example.r2.dev/galleries/gallery.lock.json")


def test_the_node_ships_the_waldo_catalogue_as_prototypes_made_by_its_own_siglip(no_network):
    assert list(shipped()) == ["waldo_catalogue"]
    gallery = load_gallery("waldo_catalogue")
    assert gallery.prototypes is not None and gallery.prototypes.shape == (120, 1152)
    assert np.allclose(np.linalg.norm(gallery.prototypes, axis=1), 1.0, atol=1e-3)
    assert gallery.model == weights.SIGLIP.repository
    assert len(gallery.item_indices()) == 117 and len(gallery.background) == 3
    assert {gallery.classes[i] for i in gallery.background} == {"robot_arm", "robot_gripper", "empty_floor"}
    assert "Waldo catalogue" in gallery.source and gallery.crops == ()
    # The simulation's objects under their catalogue names, by the core's rule.
    assert [gallery.phrases[i] for i in gallery.index_of("mustard bottle")] == ["mustard bottle"]
    assert [gallery.phrases[i] for i in gallery.index_of("the cracker box")] == ["cracker box"]
    # A background row is never what a description names.
    assert gallery.index_of("robot arm") == [] and gallery.index_of("empty floor") == []
    assert all(p == p.strip() and p for p in gallery.phrases)


def test_a_missing_gallery_names_the_shipped_ones_in_its_refusal(tmp_path):
    with pytest.raises(ValueError, match=r"not an enrolment the node ships \(waldo_catalogue\) and is not a directory"):
        load_gallery("waldo_catalog")


def test_a_directory_of_prototypes_is_read_as_it_is(tmp_path):
    gallery = load_gallery(str(write_enrolment(tmp_path / "e")))
    assert isinstance(gallery, Gallery)
    assert gallery.classes == ("coffee_can", "cracker_box", "banana", "robot_arm")
    assert gallery.phrases == ("coffee can", "cracker box", "banana", "robot arm")
    assert gallery.prototypes.shape == (4, DIM) and gallery.prototypes.dtype == np.float32
    # Normalised on the way in, whatever the file holds.
    assert np.allclose(np.linalg.norm(gallery.prototypes, axis=1), 1.0)
    assert gallery.background == frozenset({3}) and gallery.is_background(3) and not gallery.is_background(0)
    assert gallery.item_indices() == [0, 1, 2]
    assert gallery.model == weights.SIGLIP.repository and gallery.crops == ()
    assert gallery.index_of("coffee can") == [0] and gallery.index_of("arm") == []


def test_a_directory_of_prototypes_takes_its_classes_from_the_file_when_there_is_no_list(tmp_path):
    gallery = load_gallery(str(write_enrolment(tmp_path / "e", classes_file=False, prompts=False)))
    assert gallery.classes == ("coffee_can", "cracker_box", "banana", "robot_arm")
    assert gallery.phrases == ("coffee can", "cracker box", "banana", "robot arm")


def test_prototypes_that_disagree_with_their_files_are_refused_with_the_reason(tmp_path):
    root = write_enrolment(tmp_path / "e")
    (root / "classes.txt").write_text("coffee_can\ncracker_box\nbanana\n")
    with pytest.raises(ValueError, match="classes in prototypes.npz differ from classes.txt"):
        load_gallery(str(root))
    root = write_enrolment(tmp_path / "f")
    np.savez(root / PROTOTYPES_FILE, prototypes=np.eye(DIM, dtype=np.float32)[:3])
    with pytest.raises(ValueError, match="holds 3 prototypes for 4 classes"):
        load_gallery(str(root))
    root = write_enrolment(tmp_path / "g")
    bad = np.eye(DIM, dtype=np.float32)[:4]
    bad[1] = 0.0
    np.savez(root / PROTOTYPES_FILE, prototypes=bad)
    with pytest.raises(ValueError, match="empty or non-finite prototype"):
        load_gallery(str(root))
    root = write_enrolment(tmp_path / "h", background=("robot_leg",))
    with pytest.raises(ValueError, match=r"background classes that are not enrolled: \['robot_leg'\]"):
        load_gallery(str(root))
    root = write_enrolment(tmp_path / "i", background=("coffee_can", "cracker_box", "banana", "robot_arm"))
    with pytest.raises(ValueError, match="enrols no item: every class is background"):
        load_gallery(str(root))
    root = write_enrolment(tmp_path / "j")
    (root / PROTOTYPES_FILE).write_bytes(b"not an archive")
    with pytest.raises(ValueError, match="prototypes.npz cannot be read"):
        load_gallery(str(root))


def test_a_named_gallery_that_is_not_one_is_refused_with_the_reason(tmp_path):
    with pytest.raises(ValueError, match="is not a directory the node can see"):
        load_gallery(str(tmp_path / "missing"))
    with pytest.raises(ValueError, match="is not a directory the node can see"):
        load_gallery("none")
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="holds no manifest.json and no prototypes.npz"):
        load_gallery(str(tmp_path / "empty"))
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / MANIFEST_FILE).write_text("not json")
    with pytest.raises(ValueError, match="is not JSON"):
        load_gallery(str(broken))
    (broken / MANIFEST_FILE).write_text('{"images": []}')
    with pytest.raises(ValueError, match="names no items"):
        load_gallery(str(broken))
    (broken / MANIFEST_FILE).write_text('{"classes": ["cup", "bowl"], "images": []}')
    (broken / "prompts.txt").write_text("cup\n")
    with pytest.raises(ValueError, match="prompts.txt has 1 lines for 2 classes"):
        load_gallery(str(broken))


def test_a_harvester_dataset_reads_the_items_and_their_visible_crops(tmp_path):
    gallery = load_gallery(str(write_harvest(tmp_path / "g")))
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
    assert all(gallery.image_path(c).is_file() for c in gallery.crops)


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
    assert gallery.index_of("a banana") == [2]
    # A word shared by several items names them all; the core picks after.
    assert gallery.index_of("can") == [0]
    # Two items' words in one description name neither: that is a words search.
    assert gallery.index_of("box can") == []
    assert gallery.index_of("red mug") == []
    assert gallery.index_of("   ") == []
    # Every word must be in the phrase, whole: "coffee tin" is not the
    # coffee can, "cand" is not a candle, and a colour or a stopword alone
    # names nothing.
    assert gallery.index_of("coffee tin") == []
    assert gallery.index_of("banan") == []
    assert gallery.index_of("red") == [] and gallery.index_of("the") == []
