#!/usr/bin/env python3
"""Each robot's MuJoCo model, baked into the MuJoCo base image: the files of
the release of Waldo's catalogue that RELEASE_DIGEST pins, from the public
waldo-assets store. Waldo loads its robots from the same release, so both
engines stand one model of each robot.

A release holds robots.json, which names each robot's tree, and the files the
release lays over each tree under robots/<id>/: the robot's wrapper and the
tree files it shadows. A release is keyed by the SHA-256 of its inventory
and a tree by the SHA-256 of its index, so the one digest pinned here names
every byte the image bakes: the release's index is held to the digest,
robots.json and each robot file to the index, each tree's index to the key
robots.json gives it, and each tree file to that index.

    python3 waldo_catalogue.py fetch <directory> [--robot <id>]... [--store <url>]

stages robots.json and the release's index at <directory>, and each robot's
model directory at <directory>/<id>/: its tree, with its index, and the
release's files of the robot over it. `--store` names another copy of the
store, such as `file:///path/` to a release staged before it is published.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Optional
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

STORE_URL = "https://waldo-assets.peppy.bot/"
# The store sits behind Cloudflare, which answers urllib's default agent with
# 403; a named one gets through.
USER_AGENT = "sim_base_images/waldo_catalogue.py"
# The release that private-nodes-hub's waldo/web/public/assets/catalogue.lock.json pins.
RELEASE_DIGEST = "9630b747e6e346bba20f8bb8a48bc930f20593feeaaa74f783005decc613b46f"
RELEASE_INDEX = "index.json"
ROBOTS = "robots.json"
TREE_INDEX = "index.json"
_DIGEST = re.compile(r"[0-9a-f]{64}")
_TREE_KEY = re.compile(r"robots/([A-Za-z0-9_-]+)/([0-9a-f]{64})/index\.json")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class StoredFile:
    """A file of an index: its path under the index's prefix, its size and its
    SHA-256."""

    path: str
    size: int
    sha256: str

    def check(self, data: bytes, where: str) -> bytes:
        if len(data) != self.size or sha256(data) != self.sha256:
            raise ValueError(f"{where} is not the {self.path} its index pins")
        return data


def release_digest(files: list[StoredFile]) -> str:
    """The digest a release is published under: the SHA-256 of its inventory,
    sorted by path, as Waldo's catalogue-files.ts writes it."""
    inventory = [
        {"path": f.path, "size": f.size, "sha256": f.sha256}
        for f in sorted(files, key=lambda f: f.path)
    ]
    document = json.dumps({"version": 1, "files": inventory}, separators=(",", ":")) + "\n"
    return sha256(document.encode())


def _stored_files(entries, what: str) -> list[StoredFile]:
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{what} lists no file")
    files = []
    for entry in entries:
        path = entry.get("path") if isinstance(entry, dict) else None
        size = entry.get("size") if isinstance(entry, dict) else None
        digest = entry.get("sha256") if isinstance(entry, dict) else None
        parts = PurePosixPath(path).parts if isinstance(path, str) else ()
        if not parts or path.startswith("/") or ".." in parts or "\\" in path:
            raise ValueError(f"{what} lists {path!r}, which leaves its directory")
        sized = isinstance(size, int) and size >= 0
        if not sized or not (isinstance(digest, str) and _DIGEST.fullmatch(digest)):
            raise ValueError(f"{what} gives {path} no size and SHA-256")
        files.append(StoredFile(path=path, size=size, sha256=digest))
    if len({f.path for f in files}) != len(files):
        raise ValueError(f"{what} lists a file twice")
    return files


def read_release_index(data: bytes, digest: str) -> list[StoredFile]:
    """The inventory of a release's index, refused unless the index is the
    release `digest` names."""
    index = json.loads(data)
    if index.get("prefix") != f"catalogues/{digest}":
        raise ValueError(
            f"the release index names {index.get('prefix')!r}, not catalogues/{digest}"
        )
    files = _stored_files(index.get("files"), "the release index")
    actual = release_digest(files)
    if actual != digest:
        raise ValueError(f"the release index hashes to {actual}, not the pinned {digest}")
    if ROBOTS not in {f.path for f in files}:
        raise ValueError(f"the release holds no {ROBOTS}")
    return files


def tree_key(robot: dict) -> tuple[str, str]:
    """The id of a robot of robots.json and the digest of the tree it names."""
    match = _TREE_KEY.fullmatch(str(robot.get("tree", "")))
    if match is None or match.group(1) != robot.get("id"):
        raise ValueError(
            f"robot {robot.get('id')!r} names its tree as robots/<id>/<sha256>/index.json, "
            f"not {robot.get('tree')!r}"
        )
    return match.group(1), match.group(2)


def read_tree_index(data: bytes, robot: str, digest: str) -> list[StoredFile]:
    """The inventory of a robot's tree, refused unless its index is the one
    `digest` names."""
    if sha256(data) != digest:
        raise ValueError(f"the tree index of {robot} hashes to {sha256(data)}, not {digest}")
    files = _stored_files(json.loads(data).get("files"), f"the tree index of {robot}")
    if TREE_INDEX in {f.path for f in files}:
        raise ValueError(f"the tree of {robot} lists its own {TREE_INDEX}")
    return files


def robot_files(release: list[StoredFile], robot: str) -> list[StoredFile]:
    """The files a release lays over the tree of `robot`, by their path in it."""
    prefix = f"robots/{robot}/"
    return [
        StoredFile(path=f.path[len(prefix):], size=f.size, sha256=f.sha256)
        for f in release
        if f.path.startswith(prefix)
    ]


Fetch = Callable[[str], bytes]


def store_reader(store: str = STORE_URL, open_url=urlopen) -> Fetch:
    """Reads one key of `store`."""
    base = store if store.endswith("/") else store + "/"

    def read(key: str) -> bytes:
        request = Request(base + key, headers={"User-Agent": USER_AGENT})
        with open_url(request, timeout=120) as response:
            return response.read()

    return read


def _robots_of(data: bytes, wanted: Optional[list[str]]) -> list[dict]:
    robots = json.loads(data)
    if not isinstance(robots, list):
        raise ValueError(f"{ROBOTS} lists no robot")
    ids = [robot.get("id") for robot in robots]
    unknown = sorted(set(wanted or []) - set(ids))
    if unknown:
        raise ValueError(f"the release stands no robot {unknown}; it stands {sorted(ids)}")
    return [robot for robot in robots if not wanted or robot.get("id") in wanted]


def stage(destination: Path, read: Fetch, digest: str, robots: Optional[list[str]]) -> None:
    """Writes the release `digest` names into `destination`, each file held to
    its index as it arrives."""
    index = read(f"catalogues/{digest}/{RELEASE_INDEX}")
    release = read_release_index(index, digest)
    by_path = {f.path: f for f in release}
    (destination / RELEASE_INDEX).write_bytes(index)
    listing = by_path[ROBOTS].check(read(f"catalogues/{digest}/{ROBOTS}"), ROBOTS)
    (destination / ROBOTS).write_bytes(listing)
    for robot in _robots_of(listing, robots):
        robot_id, tree = tree_key(robot)
        root = destination / robot_id
        root.mkdir()
        tree_index = read(f"robots/{robot_id}/{tree}/{TREE_INDEX}")
        (root / TREE_INDEX).write_bytes(tree_index)
        for f in read_tree_index(tree_index, robot_id, tree):
            _write(root / f.path, f.check(read(f"robots/{robot_id}/{tree}/{f.path}"), f.path))
        for f in robot_files(release, robot_id):
            key = f"catalogues/{digest}/robots/{robot_id}/{f.path}"
            _write(root / f.path, f.check(read(key), f.path))
        logger.info("%s staged from tree %s and release %s", robot_id, tree, digest)


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def verify(directory: Path, digest: str = RELEASE_DIGEST) -> None:
    """Checks a staged release file by file against the indexes it was staged
    from, so a staged directory is the pinned release or refused."""
    release = read_release_index((directory / RELEASE_INDEX).read_bytes(), digest)
    by_path = {f.path: f for f in release}
    listing = by_path[ROBOTS].check((directory / ROBOTS).read_bytes(), ROBOTS)
    for robot in _robots_of(listing, None):
        robot_id, tree = tree_key(robot)
        root = directory / robot_id
        if not root.is_dir():
            continue
        shadowed = {f.path: f for f in robot_files(release, robot_id)}
        for f in read_tree_index((root / TREE_INDEX).read_bytes(), robot_id, tree):
            if f.path not in shadowed:
                f.check((root / f.path).read_bytes(), str(root / f.path))
        for f in shadowed.values():
            f.check((root / f.path).read_bytes(), str(root / f.path))


def fetch(
    destination: Path,
    digest: str = RELEASE_DIGEST,
    robots: Optional[list[str]] = None,
    read: Optional[Fetch] = None,
) -> None:
    """Stages the release at `destination`. Files land in a scratch directory
    beside it and move into place together, so a failed fetch leaves nothing
    behind. A release already staged there is verified and kept."""
    if destination.exists():
        verify(destination, digest)
        logger.info("release %s already staged at %s", digest, destination)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        stage(scratch, read or store_reader(), digest, robots)
        scratch.rename(destination)
    except BaseException:
        shutil.rmtree(scratch, ignore_errors=True)
        raise


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    fetch_command = commands.add_parser("fetch", help="stage the pinned release at a directory")
    fetch_command.add_argument("directory", type=Path)
    fetch_command.add_argument("--robot", action="append", help="a robot to stage; repeatable")
    fetch_command.add_argument("--store", default=STORE_URL, help="the store to read")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    fetch(args.directory, robots=args.robot, read=store_reader(args.store))
    return 0


if __name__ == "__main__":
    sys.exit(main())
