"""The release of Waldo's catalogue the MuJoCo image bakes: what a staged
release holds, what the fetch refuses, and which failed reads of the store
it makes again, against a store held in a dict, with no network."""

from __future__ import annotations

import hashlib
import http.client
import io
import json
import ssl
from urllib.error import HTTPError, URLError

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


@pytest.mark.parametrize("index", [b"[]", b'"catalogues"', b"null"], ids=["list", "string", "null"])
def test_an_index_that_is_not_a_json_object_is_refused(tmp_path, store, index):
    keys, digest = store
    keys[f"catalogues/{digest}/index.json"] = index
    with pytest.raises(ValueError, match=f"^the release index is not a JSON object naming catalogues/{digest}$"):
        waldo_catalogue.fetch(tmp_path / "staged", digest, read=keys.__getitem__)
    assert list(tmp_path.iterdir()) == []


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


class BrokenOff(io.BytesIO):
    """An answer that the store breaks off: urllib reads half of its body and
    raises IncompleteRead for the rest."""

    def read(self, *args):
        data = super().read(*args)
        raise http.client.IncompleteRead(data[: len(data) // 2], len(data) - len(data) // 2)


def http_error(code: int) -> HTTPError:
    return HTTPError(waldo_catalogue.STORE_URL, code, "status", hdrs=None, fp=None)


def store_opener(keys: dict[str, bytes], faults: dict[str, list], opened: list[str]):
    """An opener of the store's URLs that serves `keys`. For a key that
    `faults` lists, each attempt first takes the next fault of its list: the
    class BrokenOff, for an answer that breaks off, or an exception that the
    opening raises. Each key it opens is appended to `opened`."""

    def open_url(request, timeout):
        assert request.get_header("User-agent") == waldo_catalogue.USER_AGENT and timeout > 0
        key = request.full_url.removeprefix(waldo_catalogue.STORE_URL)
        opened.append(key)
        pending = faults.get(key, [])
        if not pending:
            return io.BytesIO(keys[key])
        fault = pending.pop(0)
        if fault is BrokenOff:
            return BrokenOff(keys[key])
        raise fault

    return open_url


def test_a_fetch_whose_every_key_breaks_off_once_stages_the_release(tmp_path, store):
    keys, digest = store
    opened, pauses = [], []
    open_url = store_opener(keys, {key: [BrokenOff] for key in keys}, opened)
    read = waldo_catalogue.store_reader(open_url=open_url, pause=pauses.append)
    waldo_catalogue.fetch(tmp_path / "staged", digest, read=read)
    waldo_catalogue.verify(tmp_path / "staged", digest)
    assert sorted(opened) == sorted([*keys, *keys])
    assert pauses == [waldo_catalogue.FIRST_PAUSE_S] * len(keys)


@pytest.mark.parametrize(
    "fault",
    [
        BrokenOff,
        http.client.RemoteDisconnected("Remote end closed connection without response"),
        ConnectionResetError(104, "Connection reset by peer"),
        TimeoutError("The read operation timed out"),
        URLError(ssl.SSLEOFError(8, "EOF occurred in violation of protocol")),
        URLError(OSError(-3, "Temporary failure in name resolution")),
        http_error(408),
        http_error(429),
        http_error(500),
        http_error(503),
    ],
    ids=lambda fault: getattr(fault, "__name__", None) or repr(fault),
)
def test_a_read_that_another_attempt_can_clear_is_made_again(store, fault):
    keys, digest = store
    key = f"catalogues/{digest}/robots.json"
    opened, pauses = [], []
    read = waldo_catalogue.store_reader(
        open_url=store_opener(keys, {key: [fault]}, opened), pause=pauses.append
    )
    assert read(key) == keys[key]
    assert opened == [key, key]
    assert pauses == [waldo_catalogue.FIRST_PAUSE_S]


def test_a_fault_that_stays_ends_the_read_with_the_fault_of_the_last_attempt(store):
    keys, digest = store
    key = f"catalogues/{digest}/robots.json"
    opened, pauses = [], []
    faults = {key: [BrokenOff] * (waldo_catalogue.READ_ATTEMPTS - 1) + [http_error(503)]}
    read = waldo_catalogue.store_reader(open_url=store_opener(keys, faults, opened), pause=pauses.append)
    with pytest.raises(HTTPError, match="status") as raised:
        read(key)
    assert raised.value.code == 503
    assert opened == [key] * waldo_catalogue.READ_ATTEMPTS
    assert pauses == [
        waldo_catalogue.FIRST_PAUSE_S * 2**attempt for attempt in range(waldo_catalogue.READ_ATTEMPTS - 1)
    ]


@pytest.mark.parametrize("code", [403, 404])
def test_a_refusal_of_the_store_is_not_read_again(tmp_path, store, code):
    keys, digest = store
    key = f"catalogues/{digest}/robots.json"
    opened, pauses = [], []
    read = waldo_catalogue.store_reader(
        open_url=store_opener(keys, {key: [http_error(code)]}, opened), pause=pauses.append
    )
    with pytest.raises(HTTPError) as raised:
        waldo_catalogue.fetch(tmp_path / "staged", digest, read=read)
    assert raised.value.code == code
    assert opened.count(key) == 1
    assert pauses == []
    assert list(tmp_path.iterdir()) == []
