"""Where a gallery comes from: the published pack through its lock file,
fetched into the cache and checked, a directory, or nothing. The store is a
file:// URL here, the same code path as https."""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from openarm_ai_brain_vla.perception.gallery_store import (
    INDEX_FILE,
    LOCK_FILE,
    GalleryUnavailable,
    HarvestDir,
    Pack,
    base_url_for,
    open_pack,
    phrase_for,
    request,
    resolve,
)
from openarm_ai_brain_vla.perception.sam3_siglip import load_gallery
from test_sam3_siglip import write_gallery

CLASSES = ["cube", "apple", "lemon_polyhaven", "robot_arm"]
LABELS = {"apple": "YCB apple", "lemon_polyhaven": "Lemon", "robot_arm": "robot arm"}
BACKGROUND = {"robot_arm"}


def write_pack(store: Path, *, digest: str = "a" * 64, dim: int = 8) -> tuple[str, Path, Path]:
    """A tiny release of the gallery pack under `store`, laid out as the
    bucket is: the lock at galleries/test/gallery.lock.json, the files under
    galleries/test/<digest>/. Returns the lock URL, the release directory and
    a cache directory beside them."""
    prefix = f"galleries/test/{digest}"
    root = store / prefix
    files: dict[str, bytes] = {}
    objects = {}
    for i, name in enumerate(CLASSES):
        crops = []
        for n in range(2):
            path = f"crops/{name}/00{n}_chest_{n}.jpg"
            buf = root / path
            buf.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (24 + 4 * i, 20), (40 * i, 90, 60)).save(buf, format="JPEG")
            files[path] = buf.read_bytes()
            crops.append(path)
        entry = {"catalogue_id": f"id_{name}", "crops": crops, "background": name in BACKGROUND}
        if name in LABELS:
            entry.update(label=LABELS[name], category="test", extent_m=[0.1, 0.1, 0.1],
                         variants=[f"ycb_{name}"] if name == "apple" else [f"polyhaven_{name}", f"id_{name}"])
        objects[name] = entry
    # Two frames with boxes for the item classes, none for the background one.
    (root / "frames").mkdir(exist_ok=True)
    records = []
    for n in range(2):
        Image.new("RGB", (96, 64), (70, 80, 90)).save(root / "frames" / f"00{n}_chest.jpg", format="JPEG")
        files[f"frames/00{n}_chest.jpg"] = (root / "frames" / f"00{n}_chest.jpg").read_bytes()
        records.append({"image": f"frames/00{n}_chest.jpg", "objects_on_table": [
            {"class": "cube", "bbox_xyxy_px": [4 + n, 4, 30, 30], "visible_fraction": 1.0},
            {"class": "apple", "bbox_xyxy_px": [40, 8, 70, 40], "visible_fraction": 0.9 if n == 0 else 0.3},
            {"class": "lemon_polyhaven", "bbox_xyxy_px": [72, 30, 94, 60], "visible_fraction": 1.0},
        ]})
    files["manifest.json"] = (json.dumps({"images": records}) + "\n").encode()
    (root / "manifest.json").write_bytes(files["manifest.json"])
    rng = np.random.default_rng(0)
    table = rng.normal(size=(len(CLASSES), dim)).astype(np.float32)
    table /= np.linalg.norm(table, axis=1, keepdims=True)
    np.savez(root / "prototypes.npz", prototypes=table.astype(np.float16), classes=np.array(CLASSES))
    files["prototypes.npz"] = (root / "prototypes.npz").read_bytes()
    index = {"classes": CLASSES, "objects": objects, "prototypes": {"file": "prototypes.npz", "model": "test", "normalised": True},
             "frames": {"dir": "frames", "manifest": "manifest.json", "count": 2}}
    files[INDEX_FILE] = (json.dumps(index, indent=1) + "\n").encode()
    (root / INDEX_FILE).write_bytes(files[INDEX_FILE])
    lock = {"version": 1, "prefix": prefix, "files": [{"path": p, "size": len(b), "sha256": hashlib.sha256(b).hexdigest()} for p, b in sorted(files.items())]}
    lock_path = store / "galleries" / "test" / LOCK_FILE
    lock_path.write_text(json.dumps(lock, indent=1) + "\n")
    return lock_path.as_uri(), root, store.parent / "cache"


def test_the_base_url_is_the_locks_origin():
    assert base_url_for("https://host/galleries/waldo_catalogue/gallery.lock.json", "galleries/waldo_catalogue/abc") == "https://host"
    assert base_url_for("file:///tmp/store/galleries/test/gallery.lock.json", "galleries/test/abc") == "file:///tmp/store"
    assert base_url_for("https://host/some/dir/gallery.lock.json", "packs/abc") == "https://host/some/dir"


def test_phrases_come_from_labels_less_the_dataset_prefix():
    assert phrase_for("apple", {"label": "YCB apple"}) == "apple"
    assert phrase_for("sponge", {"label": "YCB rigid sponge"}) == "sponge"
    assert phrase_for("food_apple_01", {"label": "Red apple"}) == "red apple"
    assert phrase_for("cube", {}) == "cube"
    assert phrase_for("wood_block", {"label": ""}) == "wood block"


def test_a_pack_is_opened_from_its_lock_and_fetches_only_what_is_asked(tmp_path):
    url, root, cache = write_pack(tmp_path / "store")
    pack = open_pack(url, cache_dir=cache)
    assert isinstance(pack, Pack) and pack.digest == "a" * 64 and len(pack.entries) == 13
    assert (pack.root / LOCK_FILE).is_file()
    index = pack.index()
    assert index["classes"] == CLASSES
    assert (pack.root / INDEX_FILE).is_file() and not (pack.root / "crops").exists()
    crop = pack.file("crops/apple/000_chest_0.jpg")
    assert crop.is_file() and crop == pack.root / "crops/apple/000_chest_0.jpg"
    assert len(pack.files(["crops/cube/000_chest_0.jpg", "crops/cube/001_chest_1.jpg"])) == 2


def test_the_gallery_reads_classes_phrases_crops_and_prototypes_from_the_pack(tmp_path):
    url, root, cache = write_pack(tmp_path / "store")
    gallery = load_gallery(open_pack(url, cache_dir=cache))
    assert gallery.classes == tuple(CLASSES)
    assert gallery.phrases == ("cube", "apple", "lemon", "robot arm")
    assert gallery.background == (False, False, False, True) and gallery.is_background(3)
    assert gallery.variants[0] == ("id_cube",) and gallery.variants[1] == ("ycb_apple", "id_apple")
    # With frames in the pack the crops are boxes in frames: five kept, the
    # apple's second one under the visibility floor, none for the background.
    assert len(gallery.crops) == 5 and all(c.box is not None for c in gallery.crops)
    assert sorted({c.image for c in gallery.crops}) == ["frames/000_chest.jpg", "frames/001_chest.jpg"]
    assert gallery.prototypes is not None and gallery.prototypes.shape == (4, 8)
    np.testing.assert_allclose(np.linalg.norm(gallery.prototypes, axis=1), 1.0, atol=1e-3)
    assert gallery.image_path(gallery.crops[0]).is_file()
    gallery.fetch_crops()
    assert sum(1 for _ in (gallery.source.root / "frames").rglob("*.jpg")) == 2
    # A description never names the background class.
    assert gallery.index_of("robot arm") == [] and gallery.index_of("arm") == []
    assert gallery.index_of("lemon") == [2] and gallery.index_of("a lemon") == [2]
    assert gallery.index_of("blue lemon") == []


def test_a_cached_release_answers_when_the_store_is_gone_and_a_damaged_file_is_refetched(tmp_path):
    url, root, cache = write_pack(tmp_path / "store")
    pack = open_pack(url, cache_dir=cache)
    pack.file("crops/apple/000_chest_0.jpg")
    pack.index()
    lock_copy = json.loads((pack.root / LOCK_FILE).read_text())
    # A pack opened again from the cache's own lock, with the bucket gone.
    shutil.rmtree(root)
    again = Pack(pack.root, pack.entries, pack.base_url, pack.prefix)
    assert again.index()["classes"] == CLASSES
    assert again.file("crops/apple/000_chest_0.jpg").is_file()
    (pack.root / INDEX_FILE).write_text("damaged")
    with pytest.raises(GalleryUnavailable, match="could not be fetched"):
        again.index()
    assert lock_copy["prefix"] == pack.prefix


def test_a_file_that_does_not_match_the_lock_is_refused(tmp_path):
    url, root, cache = write_pack(tmp_path / "store")
    (root / INDEX_FILE).write_text((root / INDEX_FILE).read_text() + " ")
    with pytest.raises(GalleryUnavailable, match="does not match its lock"):
        open_pack(url, cache_dir=cache).index()
    with pytest.raises(GalleryUnavailable, match="not in the lock"):
        open_pack(url, cache_dir=cache).file("crops/nowhere.jpg")


def test_a_broken_lock_is_refused(tmp_path):
    url, root, cache = write_pack(tmp_path / "store")
    lock = Path(url.replace("file://", ""))
    lock.write_text('{"version": 1, "prefix": "", "files": []}')
    with pytest.raises(GalleryUnavailable, match="no prefix or lists no files"):
        open_pack(url, cache_dir=cache)
    lock.write_text('{"version": 1, "prefix": "p", "files": [{"path": "../x", "size": 1, "sha256": "' + "0" * 64 + '"}]}')
    with pytest.raises(GalleryUnavailable, match="bad path"):
        open_pack(url, cache_dir=cache)
    lock.write_text("not json")
    with pytest.raises(GalleryUnavailable, match="not JSON"):
        open_pack(url, cache_dir=cache)
    with pytest.raises(GalleryUnavailable, match="could not be fetched"):
        open_pack((tmp_path / "missing.json").as_uri(), cache_dir=cache)


def test_resolve_prefers_the_model_over_the_gallery_url(tmp_path):
    url, root, cache = write_pack(tmp_path / "store")
    harvest = write_gallery(tmp_path / "harvest")
    assert resolve("", "", cache_dir=cache) is None
    assert resolve("none", url, cache_dir=cache) is None
    assert isinstance(resolve("", url, cache_dir=cache), Pack)
    assert resolve(str(harvest), url, cache_dir=cache) == HarvestDir(harvest)
    unpacked = resolve(str(root), url, cache_dir=cache)
    assert isinstance(unpacked, Pack) and unpacked.base_url is None and unpacked.index()["classes"] == CLASSES
    lock_dir = root.parent
    assert isinstance(resolve(str(lock_dir), "", cache_dir=cache), Pack)
    assert isinstance(resolve(url.replace("file://", ""), "", cache_dir=cache), Pack)
    with pytest.raises(GalleryUnavailable, match="does not exist"):
        resolve(str(tmp_path / "missing"), url, cache_dir=cache)
    (tmp_path / "empty").mkdir()
    with pytest.raises(GalleryUnavailable, match="holds no"):
        resolve(str(tmp_path / "empty"), url, cache_dir=cache)


def test_every_fetch_names_the_node_as_its_user_agent():
    req = request("https://host/galleries/x/gallery.lock.json")
    assert req.get_header("User-agent", "").startswith("openarm_ai_brain_vla/")
