"""The release of Waldo's catalogue the MuJoCo image bakes: what a staged
release holds and what the fetch refuses, against a store held in a dict,
with no network."""

from __future__ import annotations

import hashlib
import json

import pytest
from entries import waldo_catalogue


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _entry(path: str, data: bytes) -> dict:
    return {"path": path, "size": len(data), "sha256": _sha(data)}


def build_store(tree: dict[str, bytes], laid_over: dict[str, bytes], robot: str = "arm"):
    """A store of one release standing `robot`: its tree of `tree`, and the
    release's files of the robot `laid_over` it. Returns the store's keys
    and the release's digest."""
    tree_index = json.dumps({"files": [_entry(p, d) for p, d in sorted(tree.items())]}).encode()
    tree_digest = _sha(tree_index)
    robots = json.dumps([{"id": robot, "tree": f"robots/{robot}/{tree_digest}/index.json"}]).encode()
    release_files = {"robots.json": robots, **{f"robots/{robot}/{p}": d for p, d in laid_over.items()}}
    files = [waldo_catalogue.StoredFile(**_entry(p, d)) for p, d in release_files.items()]
    digest = waldo_catalogue.release_digest(files)
    index = json.dumps({
        "version": 1,
        "prefix": f"catalogues/{digest}",
        "files": [_entry(p, d) for p, d in sorted(release_files.items())],
    }).encode()
    keys = {f"catalogues/{digest}/index.json": index}
    keys.update({f"catalogues/{digest}/{p}": d for p, d in release_files.items()})
    keys[f"robots/{robot}/{tree_digest}/index.json"] = tree_index
    keys.update({f"robots/{robot}/{tree_digest}/{p}": d for p, d in tree.items()})
    return keys, digest


TREE = {"scene.xml": b"<scene/>", "arm.xml": b"<upstream arm/>", "meshes/link.stl": b"solid"}
LAID_OVER = {"mounted.xml": b"<wrapper/>", "arm.xml": b"<shadow arm/>"}


@pytest.fixture(name="store")
def store_fixture():
    return build_store(TREE, LAID_OVER)


def test_a_robot_is_its_tree_with_the_release_files_laid_over_it(tmp_path, store):
    keys, digest = store
    waldo_catalogue.fetch(tmp_path / "staged", digest, read=keys.__getitem__)
    root = tmp_path / "staged" / "arm"
    assert (root / "scene.xml").read_bytes() == b"<scene/>"
    assert (root / "meshes" / "link.stl").read_bytes() == b"solid"
    assert (root / "mounted.xml").read_bytes() == b"<wrapper/>"
    assert (root / "arm.xml").read_bytes() == b"<shadow arm/>"
    assert json.loads((tmp_path / "staged" / "robots.json").read_bytes())[0]["id"] == "arm"
    waldo_catalogue.verify(tmp_path / "staged", digest)


def test_a_staged_release_is_verified_and_kept(tmp_path, store):
    keys, digest = store
    waldo_catalogue.fetch(tmp_path / "staged", digest, read=keys.__getitem__)
    waldo_catalogue.fetch(tmp_path / "staged", digest, read=lambda key: pytest.fail(f"read {key}"))
    (tmp_path / "staged" / "arm" / "meshes" / "link.stl").write_bytes(b"other")
    with pytest.raises(ValueError, match="is not the meshes/link.stl its index pins"):
        waldo_catalogue.fetch(tmp_path / "staged", digest, read=keys.__getitem__)


def test_the_digest_rule_is_waldos(store):
    """The digest of an inventory, sorted by path, written with no spaces and
    a closing newline, as catalogue-files.ts writes it."""
    files = [waldo_catalogue.StoredFile("b", 1, "1" * 64), waldo_catalogue.StoredFile("a", 2, "2" * 64)]
    document = '{"version":1,"files":[{"path":"a","size":2,"sha256":"' + "2" * 64 + '"},{"path":"b","size":1,"sha256":"' + "1" * 64 + '"}]}\n'
    assert waldo_catalogue.release_digest(files) == _sha(document.encode())


@pytest.mark.parametrize(
    "tamper, refusal",
    [
        (lambda keys, digest: keys.__setitem__(f"catalogues/{digest}/robots.json", b"[]"), "is not the robots.json"),
        (lambda keys, digest: keys.__setitem__(f"catalogues/{digest}/robots/arm/mounted.xml", b"<x/>"), "is not the mounted.xml"),
        (lambda keys, digest: [keys.__setitem__(k, b"<x/>") for k in list(keys) if k.endswith("/scene.xml")], "is not the scene.xml"),
        (lambda keys, digest: [keys.__setitem__(k, b'{"files":[]}') for k in list(keys) if k.startswith("robots/") and k.endswith("/index.json")], "the tree index of arm hashes to"),
    ],
)
def test_a_file_that_is_not_the_pinned_one_stages_nothing(tmp_path, store, tamper, refusal):
    keys, digest = store
    tamper(keys, digest)
    with pytest.raises(ValueError, match=refusal):
        waldo_catalogue.fetch(tmp_path / "staged", digest, read=keys.__getitem__)
    assert list(tmp_path.iterdir()) == []


def test_an_index_that_is_not_the_pinned_release_is_refused(tmp_path, store):
    keys, digest = store
    other = "0" * 64
    keys[f"catalogues/{other}/index.json"] = keys[f"catalogues/{digest}/index.json"]
    with pytest.raises(ValueError, match=f"names 'catalogues/{digest}', not catalogues/{other}"):
        waldo_catalogue.fetch(tmp_path / "staged", other, read=keys.__getitem__)


def test_a_robot_the_release_does_not_stand_is_refused(tmp_path, store):
    keys, digest = store
    with pytest.raises(ValueError, match=r"stands no robot \['hand'\]; it stands \['arm'\]"):
        waldo_catalogue.fetch(tmp_path / "staged", digest, robots=["hand"], read=keys.__getitem__)


@pytest.mark.parametrize("path", ["../escape.xml", "/abs.xml", "a/../../b.xml"])
def test_a_tree_file_outside_its_tree_is_refused(tmp_path, path):
    keys, digest = build_store({**TREE, path: b"x"}, LAID_OVER)
    with pytest.raises(ValueError, match="which leaves its directory"):
        waldo_catalogue.fetch(tmp_path / "staged", digest, read=keys.__getitem__)


def test_a_tree_named_off_the_layout_is_refused():
    with pytest.raises(ValueError, match="robots/<id>/<sha256>/index.json"):
        waldo_catalogue.tree_key({"id": "arm", "tree": "robots/hand/" + "1" * 64 + "/index.json"})
