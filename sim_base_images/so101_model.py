#!/usr/bin/env python3
"""The SO-101's MuJoCo model, baked into the MuJoCo base image: MuJoCo
Menagerie's robotstudio_so101 (Apache-2.0) at one commit, each file checked
against the digest so101_model.lock.json pins.

The lock names the upstream repository, the commit and every file the scene
reads, with its size and SHA-256, so the image carries one exact model for
good and a changed upstream file stops the build instead of reaching it.
Waldo's catalogue names the same commit for the same robot.

    python3 so101_model.py fetch <directory>
    python3 so101_model.py lock <checkout of the upstream directory>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

LOCK_PATH = Path(__file__).with_name("so101_model.lock.json")
USER_AGENT = "sim_base_images/so101_model.py"
LOCK_VERSION = 1


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_lock(path: Path = LOCK_PATH) -> dict:
    """The pinned model: its source and its inventory, refused unless it
    names a full commit and lists the scene it promises."""
    lock = json.loads(path.read_text())
    if lock.get("version") != LOCK_VERSION:
        raise ValueError(f"{path} is version {lock.get('version')!r}, not {LOCK_VERSION}")
    commit = lock.get("commit", "")
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError(f"{path} pins {commit!r}, which is not a full commit")
    files = lock.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"{path} lists no file")
    paths = [entry["path"] for entry in files]
    if len(set(paths)) != len(paths):
        raise ValueError(f"{path} lists a file twice")
    for name in paths:
        parts = Path(name).parts
        if Path(name).is_absolute() or ".." in parts or not parts:
            raise ValueError(f"{path} lists {name!r}, which leaves the model's directory")
    if lock.get("scene") not in paths:
        raise ValueError(f"{path} promises the scene {lock.get('scene')!r} and does not list it")
    return lock


def base_url(lock: dict) -> str:
    return (
        f"https://raw.githubusercontent.com/{lock['repository']}/{lock['commit']}/"
        f"{lock['directory']}/"
    )


def verify(directory: Path, lock: Optional[dict] = None) -> None:
    """Checks a staged model file by file against the lock."""
    lock = lock or read_lock()
    for entry in lock["files"]:
        path = directory / entry["path"]
        if not path.is_file():
            raise FileNotFoundError(f"{path} is missing from the staged SO-101 model")
        data = path.read_bytes()
        if len(data) != entry["size"] or sha256(data) != entry["sha256"]:
            raise ValueError(f"{path} does not match so101_model.lock.json")


def fetch(destination: Path, lock: Optional[dict] = None, open_url=urlopen) -> None:
    """Stages the pinned model at `destination`, each file checked against
    the lock as it arrives. Files land in a scratch directory beside
    `destination` and move into place together, so a failed fetch leaves
    nothing behind. A model already staged there is verified and kept."""
    lock = lock or read_lock()
    if destination.exists():
        verify(destination, lock)
        logger.info("SO-101 model already staged at %s", destination)
        return
    base = base_url(lock)
    destination.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        for entry in lock["files"]:
            request = Request(base + entry["path"], headers={"User-Agent": USER_AGENT})
            with open_url(request, timeout=120) as response:
                data = response.read()
            if len(data) != entry["size"] or sha256(data) != entry["sha256"]:
                raise ValueError(f"{base}{entry['path']} does not match so101_model.lock.json")
            target = scratch / entry["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        scratch.rename(destination)
    except BaseException:
        shutil.rmtree(scratch, ignore_errors=True)
        raise
    logger.info("SO-101 model staged at %s from %s", destination, base)


def inventory(checkout: Path, paths: list[str]) -> list[dict]:
    """The lock's inventory of `paths`, read from a checkout of the upstream
    directory at the commit being pinned."""
    files = []
    for name in sorted(paths):
        data = (checkout / name).read_bytes()
        files.append({"path": name, "size": len(data), "sha256": sha256(data)})
    return files


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    fetch_command = commands.add_parser("fetch", help="stage the pinned model at a directory")
    fetch_command.add_argument("directory", type=Path)
    lock_command = commands.add_parser(
        "lock", help="print the inventory of the files the lock lists, from a checkout"
    )
    lock_command.add_argument("checkout", type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.command == "fetch":
        fetch(args.directory)
        return 0
    lock = json.loads(LOCK_PATH.read_text())
    lock["files"] = inventory(args.checkout, [entry["path"] for entry in lock["files"]])
    print(json.dumps(lock, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
