"""Where a gallery comes from: the published gallery pack, fetched once and
kept, or a directory on disk.

The pack is one immutable release, laid out like Waldo's catalogue
releases. A lock file, `gallery.lock.json`, carries `version`, `prefix` and
`files`, each file with its `path`, `size` and `sha256`; every file lives at
`<base>/<prefix>/<path>`, where `<base>` is the origin the lock was fetched
from, and is checked against its size and hash after download. The prefix
ends in a digest of the contents, so a new release gets a new prefix and the
old one stays. Under the prefix:

- `index.json`: `classes`, the class names in order, and `objects`, per class
  its catalogue id, label, category, extent and the paths of its crops;
- the SigLIP prototypes as an npz named by `index.json`'s `prototypes.file`,
  one L2-normalised mean image embedding per class in class order;
- `crops/<class>/<frame>_<n>.jpg`, the reference crops, needed by a backend
  that builds its own class table from pictures (YOLOE) and not by one that
  reads the prototypes (SigLIP), so they are fetched only when asked for.

Files land in `CACHE_DIR/<digest>/` and a later load finds them there and
touches no network; a cached file is checked again before it is trusted. The
lock's URL is the node's `gallery_url` parameter, so a custom domain can
replace the development host without a code change.

A directory works too: an unpacked release (holding `index.json`) is read
in place, and a harvester dataset (holding `manifest.json`) is read by the
backends' own loader. `perception_model` "none" names no gallery.

Only the standard library and numpy: a gallery is sixteen megabytes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

logger = logging.getLogger(__name__)

LOCK_FILE = "gallery.lock.json"
INDEX_FILE = "index.json"
MANIFEST_FILE = "manifest.json"
CACHE_DIR = Path("~/.cache/openarm_ai_brain_vla/galleries")
NO_GALLERY = "none"
FETCH_TIMEOUT_S = 30.0
FETCH_WORKERS = 8
# Sent with every fetch. Cloudflare's development hostnames answer 403 to
# Python's default user agent and to nothing else.
USER_AGENT = "openarm_ai_brain_vla/1 (+https://peppy.bot)"

_SHA = re.compile(r"^[0-9a-f]{64}$")
_URL = ("http://", "https://", "file://")


class GalleryUnavailable(Exception):
    """The gallery named cannot be had: not on disk, not in the cache, and
    not fetched, or fetched and not matching its lock. The message says
    which and why."""


@dataclass(frozen=True)
class Entry:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class HarvestDir:
    """A harvester dataset on disk: manifest.json, classes.txt, prompts.txt
    and the frames the boxes refer to."""

    root: Path


class Pack:
    """One release of the gallery pack: its files, fetched on demand into
    `root` and checked against the lock. With no `base_url` the pack is an
    unpacked directory read as it is."""

    def __init__(self, root: Path, entries: dict[str, Entry], base_url: Optional[str], prefix: str) -> None:
        self.root = root
        self.entries = entries
        self.base_url = base_url
        self.prefix = prefix

    @property
    def digest(self) -> str:
        return self.prefix.rstrip("/").rsplit("/", 1)[-1]

    def has(self, path: str) -> bool:
        return path in self.entries or (self.base_url is None and (self.root / path).is_file())

    def file(self, path: str) -> Path:
        """The local path of one file of the pack, fetched and checked when
        the cache lacks it or holds a damaged copy."""
        return self.files([path])[0]

    def files(self, paths: Sequence[str]) -> list[Path]:
        """Several files at once, the missing ones fetched in parallel."""
        if self.base_url is None:
            out = []
            for path in paths:
                local = self.root / path
                if not local.is_file():
                    raise GalleryUnavailable(f"{local} is not in the gallery directory")
                out.append(local)
            return out
        missing = [p for p in paths if not self._cached_ok(p)]
        if missing:
            with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
                list(pool.map(self._fetch, missing))
            logger.info("gallery_store: fetched %d files of %s", len(missing), self.prefix)
        return [self.root / p for p in paths]

    def index(self) -> dict:
        try:
            return json.loads(self.file(INDEX_FILE).read_text())
        except ValueError as error:
            raise GalleryUnavailable(f"{self.prefix}/{INDEX_FILE} is not JSON: {error}") from error

    def _entry(self, path: str) -> Entry:
        try:
            return self.entries[path]
        except KeyError:
            raise GalleryUnavailable(f"{path} is not in the lock of {self.prefix}") from None

    def _cached_ok(self, path: str) -> bool:
        entry = self._entry(path)
        local = self.root / path
        return local.is_file() and local.stat().st_size == entry.size and sha256_of(local) == entry.sha256

    def _fetch(self, path: str) -> None:
        entry = self._entry(path)
        url = f"{self.base_url}/{self.prefix}/{path}"
        data = fetch(url)
        if len(data) != entry.size or hashlib.sha256(data).hexdigest() != entry.sha256:
            raise GalleryUnavailable(f"{url} does not match its lock entry (size {len(data)} vs {entry.size}): refusing the file")
        local = self.root / path
        local.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=local.parent, prefix=f".{local.name}.", delete=False) as tmp:
            tmp.write(data)
        Path(tmp.name).replace(local)


Source = Union[HarvestDir, Pack]


def resolve(model: str, gallery_url: str = "", cache_dir: Path = CACHE_DIR) -> Optional[Source]:
    """What `perception_model` and `gallery_url` name. `model` wins when set:
    "none" is no gallery, a directory is read in place, a lock file's path or
    URL is a pack. Empty falls back to `gallery_url`, the node's published
    pack; empty too is no gallery. Raises GalleryUnavailable when a gallery
    is named and cannot be had."""
    name = model.strip()
    if name.lower() == NO_GALLERY:
        return None
    if not name:
        name = gallery_url.strip()
        if not name:
            return None
    if not name.startswith(_URL):
        path = Path(name).expanduser()
        if path.is_dir():
            # An unpacked release holds both an index and a manifest; a
            # harvester dataset only the manifest.
            if (path / INDEX_FILE).is_file():
                return Pack(path, {}, None, path.name)
            if (path / MANIFEST_FILE).is_file():
                return HarvestDir(path)
            if (path / LOCK_FILE).is_file():
                return open_pack((path / LOCK_FILE).as_uri(), cache_dir)
            raise GalleryUnavailable(f"{path} holds no {MANIFEST_FILE}, {INDEX_FILE} or {LOCK_FILE}")
        if path.is_file() and path.name.endswith(".json"):
            return open_pack(path.resolve().as_uri(), cache_dir)
        raise GalleryUnavailable(f"gallery {path} does not exist")
    return open_pack(name, cache_dir)


def open_pack(lock_url: str, cache_dir: Path = CACHE_DIR) -> Pack:
    """Reads the lock at `lock_url` and returns the pack it describes, its
    files to be fetched under `cache_dir/<digest>/` as they are asked for."""
    lock = parse_lock(fetch(lock_url), lock_url)
    prefix = lock["prefix"]
    base = base_url_for(lock_url, prefix)
    root = cache_dir.expanduser() / prefix.rstrip("/").rsplit("/", 1)[-1]
    root.mkdir(parents=True, exist_ok=True)
    (root / LOCK_FILE).write_text(json.dumps(lock, indent=1) + "\n")
    entries = {e["path"]: Entry(e["path"], e["size"], e["sha256"]) for e in lock["files"]}
    return Pack(root, entries, base, prefix)


def parse_lock(data: bytes, where: str) -> dict:
    try:
        lock = json.loads(data)
    except ValueError as error:
        raise GalleryUnavailable(f"{where} is not JSON: {error}") from error
    prefix = lock.get("prefix") if isinstance(lock, dict) else None
    files = lock.get("files") if isinstance(lock, dict) else None
    if not isinstance(prefix, str) or not prefix.strip("/") or not isinstance(files, list) or not files:
        raise GalleryUnavailable(f"{where} has no prefix or lists no files")
    for entry in files:
        path = entry.get("path") if isinstance(entry, dict) else None
        if not isinstance(path, str) or not path or path.startswith("/") or ".." in path.split("/"):
            raise GalleryUnavailable(f"{where} lists a bad path {path!r}")
        if not isinstance(entry.get("size"), int) or entry["size"] < 0 or not _SHA.match(str(entry.get("sha256", ""))):
            raise GalleryUnavailable(f"{where} lists {path} without a size and sha256")
    lock["prefix"] = prefix.strip("/")
    return lock


def base_url_for(lock_url: str, prefix: str) -> str:
    """The origin files are served from: the lock's own directory, less the
    leading directories it shares with the prefix, so that
    `<base>/<prefix>/<path>` holds both for a bucket and for a directory."""
    lock_dir = lock_url.rsplit("/", 1)[0]
    prefix_dir = prefix.rstrip("/").rsplit("/", 1)[0] if "/" in prefix else ""
    if prefix_dir and lock_dir.endswith("/" + prefix_dir):
        return lock_dir[: -len(prefix_dir) - 1]
    return lock_dir


def request(url: str) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": USER_AGENT})


def fetch(url: str) -> bytes:
    try:
        with urllib.request.urlopen(request(url), timeout=FETCH_TIMEOUT_S) as response:
            return response.read()
    except (urllib.error.URLError, OSError, ValueError) as error:
        raise GalleryUnavailable(f"{url} could not be fetched: {error}") from error


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def phrase_for(class_name: str, entry: dict) -> str:
    """The plain words a class is prompted and reported by: its label less a
    dataset prefix ("YCB apple" is "apple", "YCB rigid sponge" a "sponge"),
    else the class name with spaces."""
    label = str(entry.get("label", "")).strip()
    if not label:
        return class_name.replace("_", " ").strip().lower()
    label = re.sub(r"^ycb\s+", "", label, flags=re.I)
    label = re.sub(r"\brigid\s+", "", label, flags=re.I)
    return label.strip().lower()
