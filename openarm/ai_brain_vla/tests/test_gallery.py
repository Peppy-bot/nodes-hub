"""The enrolment gallery a launch names by directory: a harvester
dataset, or nothing."""

import json
from pathlib import Path

import pytest
from PIL import Image

from openarm_ai_brain_vla.perception.gallery import MANIFEST_FILE, load_gallery


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


def test_a_named_gallery_that_is_not_one_is_refused_with_the_reason(tmp_path):
    with pytest.raises(ValueError, match="is not a directory the node can see"):
        load_gallery(str(tmp_path / "missing"))
    with pytest.raises(ValueError, match="is not a directory the node can see"):
        load_gallery("none")
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="holds no manifest.json"):
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
