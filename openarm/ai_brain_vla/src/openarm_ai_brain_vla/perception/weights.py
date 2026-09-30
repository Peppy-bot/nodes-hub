"""The weights of the two models the sam3_siglip backend loads, downloaded
to the machine by the first load that needs them.

A model is a `Source`: a repository of the Hugging Face Hub at one commit,
and the files of it that transformers reads. `weights.json` pins them: each
file by its size and by the SHA-256 of each of its pieces, `piece_bytes` of
the file in a row. `stage` names the directory that holds a model's files
and downloads them first when the machine does not have them. Only a node
whose launch selects the backend downloads them, once for the machine, and
every machine reads the same bytes.

SAM 3's official repository, facebook/sam3, is gated behind a licence
click-through on the Hub. The pinned one, jetjodh/sam3, is a mirror that
carries the same files, each with the same content hash at the pinned
revision. Its `sam3.pt`, the checkpoint of the original code base, is not a
file transformers reads. SigLIP so400m is the study's namer.

The weights are kept in the directory `WEIGHTS_DIRECTORY_VARIABLE` names,
else in `DEFAULT_DIRECTORY` of the daemon user's home, which the container
sees and which stays when the node image is built again:

    <directory>/lock                          held by the node that downloads
    <directory>/<name>-<revision>/            a whole model
    <directory>/<name>-<revision>.fetching/   a model that is being downloaded

A file is downloaded as `<file>.partial` in the `.fetching` directory. Its
bytes are written as they come, and each piece is compared with its pin
when its last byte is written: a piece with another hash is cut off the
partial file, so the pieces before the one a partial file ends in are
always checked ones. The file takes its name once its last piece is
checked, and the directory takes the model's name once it holds every file.
So a directory under a model's name always holds the whole model, checked,
and a load that finds it reads no network and hashes nothing.

A node can be stopped or killed at any point of a download. It leaves at
most one partial file for each file of the model, and the next start
continues each from its last byte with an HTTP range request, after it has
read again the one piece the partial file ends in. So a start that lasts
long enough to receive some bytes moves the download on, however many
starts it takes. The download runs on the standard library alone, in the
thread of the load: it starts no thread and no process that could hold the
node past its shutdown. Two nodes of one machine do not download the same
files twice: one holds the lock of the directory, and the other waits for
it and then finds the model.

    python -m openarm_ai_brain_vla.perception.weights fetch [directory]

downloads every model, for a machine that must have the weights before its
first launch or that a robot with no network copies the directory from.

    python -m openarm_ai_brain_vla.perception.weights pin <file>...

prints the pin of each file as `weights.json` holds it, for the files of a
revision whose content is known to be the right one.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import http.client
import json
import logging
import os
import re
import shutil
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, ContextManager, Iterator, Optional, Sequence

logger = logging.getLogger(__name__)

MODULE = "openarm_ai_brain_vla.perception.weights"
# Names the directory the weights are kept in. `peppy stack launch` forwards
# the launching shell's environment to the nodes it starts on that machine.
WEIGHTS_DIRECTORY_VARIABLE = "OPENARM_AI_BRAIN_VLA_WEIGHTS"
DEFAULT_DIRECTORY = Path("~/.cache/openarm_ai_brain_vla/weights")
PINS_FILE = Path(__file__).with_name("weights.json")
HUB = "https://huggingface.co"
LOCK_FILE = "lock"
FETCHING_SUFFIX = ".fetching"
PARTIAL_SUFFIX = ".partial"
# The most bytes read from the network and written to the disk at a time: a
# node that is killed loses what it was reading, and no more.
CHUNK_BYTES = 1 << 18
# The seconds a connection, or one read from it, can take before the attempt
# fails.
TIMEOUT_S = 60.0
# The attempts in a row that add no byte to a file before its download fails.
STALLED_ATTEMPTS = 3


class WeightsError(Exception):
    """A model's weights could not be put on the machine."""


@dataclass(frozen=True)
class File:
    """One file of a model: its path in the repository, its size in bytes,
    and the SHA-256 of each of its pieces, `piece_bytes` of the file in a
    row and the rest of it in the last one."""

    path: str
    size: int
    piece_bytes: int
    pieces: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.size <= 0 or self.piece_bytes <= 0:
            raise ValueError(f"{self.path} is pinned with {self.size} bytes in pieces of {self.piece_bytes}")
        if len(self.pieces) != -(-self.size // self.piece_bytes):
            raise ValueError(f"{self.path} is pinned with {len(self.pieces)} pieces, and {self.size} bytes make {-(-self.size // self.piece_bytes)} of {self.piece_bytes}")

    def piece_start(self, index: int) -> int:
        return index * self.piece_bytes

    def piece_end(self, index: int) -> int:
        return min(self.size, (index + 1) * self.piece_bytes)


@dataclass(frozen=True)
class Source:
    """One model on the Hub: its repository at one commit, named by the
    full hash, and the files of that commit that transformers reads."""

    name: str
    repository: str
    revision: str
    files: tuple[File, ...]

    @property
    def directory_name(self) -> str:
        return f"{self.name}-{self.revision}"

    def url(self, file: File) -> str:
        return f"{HUB}/{self.repository}/resolve/{self.revision}/{file.path}"


def pieces_of(content, piece_bytes: int) -> tuple[str, ...]:
    """The SHA-256 of each piece of a file open for reading."""
    return tuple(hashlib.sha256(piece).hexdigest() for piece in iter(lambda: content.read(piece_bytes), b""))


def pin_of(path: Path, piece_bytes: int) -> dict:
    """The pin of the file at `path`, as `weights.json` holds it."""
    with path.open("rb") as content:
        return {"path": path.name, "size": path.stat().st_size, "pieces": list(pieces_of(content, piece_bytes))}


@dataclass(frozen=True)
class Pins:
    """What a pins file holds: the size of a piece, and the models in the
    file's order."""

    piece_bytes: int
    sources: tuple[Source, ...]


def load_pins(path: Path = PINS_FILE) -> Pins:
    pins = json.loads(path.read_text())
    piece_bytes = pins["piece_bytes"]
    return Pins(
        piece_bytes,
        tuple(
            Source(
                model["name"],
                model["repository"],
                model["revision"],
                tuple(File(file["path"], file["size"], piece_bytes, tuple(file["pieces"])) for file in model["files"]),
            )
            for model in pins["models"]
        ),
    )


PINS = load_pins()
SOURCES = PINS.sources
SAM3, SIGLIP = SOURCES


@dataclass(frozen=True)
class Body:
    """A server's answer to the request for a file from an offset: the
    offset of the file its bytes start at, and the read that gives them, at
    most the number asked for and none at their end."""

    starts_at: int
    read: Callable[[int], bytes]


# Asks for the file at a URL from an offset, or raises OSError or
# http.client.HTTPException.
OpenAt = Callable[[str, int], ContextManager[Body]]


@contextmanager
def open_url(url: str, offset: int) -> Iterator[Body]:
    """The file at `url` from `offset`, over HTTP. The Hub answers with a
    redirect to the server that holds the bytes, which is followed with the
    same range."""
    headers = {"User-Agent": "openarm_ai_brain_vla"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=TIMEOUT_S) as response:
        yield Body(range_start(response.status, response.headers.get("Content-Range")), response.read)


def range_start(status: int, content_range: Optional[str]) -> int:
    """The offset a response's bytes start at: the one its Content-Range
    names when the server sends a part of the file (206), and 0 when it
    sends the whole file."""
    if status != http.client.PARTIAL_CONTENT:
        return 0
    named = re.fullmatch(r"bytes (\d+)-\d+/(?:\d+|\*)", (content_range or "").strip())
    if named is None:
        raise OSError(f"a partial answer with the Content-Range {content_range!r}")
    return int(named.group(1))


def node_directory() -> Path:
    """The directory the node keeps the weights in: the one
    `WEIGHTS_DIRECTORY_VARIABLE` names, else `DEFAULT_DIRECTORY`."""
    named = os.environ.get(WEIGHTS_DIRECTORY_VARIABLE, "").strip()
    return (Path(named) if named else DEFAULT_DIRECTORY).expanduser()


def stage(source: Source, directory: Path, open_at: OpenAt = open_url) -> Path:
    """The directory under `directory` that holds the files of `source`,
    downloaded first when it is not there. Raises WeightsError when the
    model cannot be downloaded."""
    staged_at = directory / source.directory_name
    if staged_at.is_dir():
        return staged_at
    directory.mkdir(parents=True, exist_ok=True)
    with locked(directory):
        # Another node can have downloaded the model while this one waited.
        if not staged_at.is_dir():
            fetch(source, directory, open_at).rename(staged_at)
    return staged_at


@contextmanager
def locked(directory: Path) -> Iterator[None]:
    """Holds the lock of `directory`, after the node that holds it now. The
    kernel releases the lock when its process ends, whatever ends it."""
    with (directory / LOCK_FILE).open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            logger.info("another node downloads into %s: waiting for it", directory)
            fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def fetch(source: Source, directory: Path, open_at: OpenAt) -> Path:
    """Downloads the files of `source` that its `.fetching` directory does
    not hold yet, and returns that directory, which then holds them all."""
    fetching = directory / f"{source.directory_name}{FETCHING_SUFFIX}"
    fetching.mkdir(exist_ok=True)
    missing = missing_bytes(source, fetching)
    free = free_bytes(fetching)
    if missing > free:
        raise WeightsError(f"the weights of {source.repository} need {missing} more bytes in {directory}, which has {free} free")
    logger.info("downloading the weights of %s into %s: %.0f MB to go", source.repository, directory, missing / 1e6)
    for file in source.files:
        fetch_file(source, file, fetching, open_at)
    return fetching


def fetch_file(source: Source, file: File, fetching: Path, open_at: OpenAt) -> None:
    """Puts `file` in `fetching` under its name: downloads what its partial
    file lacks, and names the file once its last piece is checked."""
    target = fetching / file.path
    if target.exists():
        return
    partial = partial_of(target)
    partial.touch()
    download(file, source.url(file), partial, open_at)
    with partial.open("rb") as content:
        # The bytes reach the disk before the name that says they are whole.
        os.fsync(content.fileno())
    partial.rename(target)


def download(file: File, url: str, partial: Path, open_at: OpenAt) -> None:
    """Appends to `partial` until it holds `file`, every piece checked. A
    failed attempt is followed by another from the byte the file ends at,
    until `STALLED_ATTEMPTS` of them in a row have added nothing. Raises
    WeightsError then, and when the server sends a piece that is not the
    pinned one."""
    stalled = 0
    held = keep_checked(file, partial)
    while held < file.size:
        try:
            receive(file, url, partial, held, open_at)
            failure = "the server ended its answer"
        except (OSError, http.client.HTTPException) as error:
            failure = str(error) or type(error).__name__
        held, held_before = partial.stat().st_size, held
        stalled = 0 if held > held_before else stalled + 1
        if stalled == STALLED_ATTEMPTS:
            raise WeightsError(f"the download of {url} stopped at {held} of {file.size} bytes: {failure}")


def keep_checked(file: File, partial: Path) -> int:
    """The bytes of `partial` that a download continues from. A node can
    be killed after it wrote the last byte of a piece and before it compared
    the piece, so a partial file that ends where a piece ends has that piece
    compared here, and cut off when it is another. A partial file larger
    than `file` is not one a download wrote, and is emptied."""
    held = partial.stat().st_size
    if held == 0:
        return 0
    if held > file.size:
        os.truncate(partial, 0)
        return 0
    index = (held - 1) // file.piece_bytes
    if held < file.piece_end(index):
        return held
    with partial.open("rb") as content:
        content.seek(file.piece_start(index))
        found = hashlib.sha256(content.read()).hexdigest()
    if found == file.pieces[index]:
        return held
    os.truncate(partial, file.piece_start(index))
    return file.piece_start(index)


def receive(file: File, url: str, partial: Path, held: int, open_at: OpenAt) -> None:
    """One attempt: asks for `file` from the byte `partial` ends at and
    appends what comes, at most up to the file's size, each chunk written
    through before the next is read. A server that sends the whole file
    makes the partial file start again. A piece is compared with its pin
    when its last byte is written, and cut off when it is another: the
    attempt then ends when the partial file held the start of that piece,
    and raises WeightsError when the server sent all of it."""
    with open_at(url, held) as body, partial.open("r+b") as out:
        if body.starts_at not in (0, held):
            raise OSError(f"the answer starts at byte {body.starts_at}, and byte {held} was asked for")
        out.truncate(body.starts_at)
        position = body.starts_at
        index = position // file.piece_bytes
        # What the partial file holds of the piece it ends in.
        out.seek(file.piece_start(index))
        piece = hashlib.sha256(out.read())
        while position < file.size and (chunk := body.read(min(CHUNK_BYTES, file.piece_end(index) - position))):
            out.write(chunk)
            out.flush()
            piece.update(chunk)
            position += len(chunk)
            if position < file.piece_end(index):
                continue
            if piece.hexdigest() != file.pieces[index]:
                out.truncate(file.piece_start(index))
                if body.starts_at > file.piece_start(index):
                    return
                raise WeightsError(f"{url} is not the pinned file: the SHA-256 of piece {index} is {piece.hexdigest()}, the pin is {file.pieces[index]}")
            if tenths(position, file) > tenths(file.piece_start(index), file):
                logger.info("%s: %d0%% of %.1f MB", url, tenths(position, file), file.size / 1e6)
            index, piece = index + 1, hashlib.sha256()


def tenths(position: int, file: File) -> int:
    """The whole tenths of `file` that come before `position`."""
    return 10 * position // file.size


def partial_of(target: Path) -> Path:
    return target.with_name(f"{target.name}{PARTIAL_SUFFIX}")


def size_of(path: Path) -> int:
    """The bytes `path` holds, 0 when it is not there."""
    return path.stat().st_size if path.exists() else 0


def missing_bytes(source: Source, fetching: Path) -> int:
    """The bytes of `source` that `fetching` does not hold yet."""
    lacking = (file for file in source.files if not (fetching / file.path).exists())
    return sum(max(0, file.size - size_of(partial_of(fetching / file.path))) for file in lacking)


def free_bytes(directory: Path) -> int:
    return shutil.disk_usage(directory).free


def main(arguments: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog=f"python -m {MODULE}", description="Downloads and pins the weights of the sam3_siglip backend's models.")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch_command = commands.add_parser("fetch", help="download every model that a directory does not hold")
    fetch_command.add_argument("directory", type=Path, nargs="?", help=f"where the weights are kept; the node's own directory when left out ({WEIGHTS_DIRECTORY_VARIABLE}, else {DEFAULT_DIRECTORY})")
    pin_command = commands.add_parser("pin", help=f"print the pin of each file, as {PINS_FILE.name} holds it")
    pin_command.add_argument("files", type=Path, nargs="+", help="a file of a model, whose content is the right one")
    asked = parser.parse_args(arguments)
    if asked.command == "pin":
        print(json.dumps([pin_of(path, PINS.piece_bytes) for path in asked.files], indent=2))
        return
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    into = asked.directory or node_directory()
    for source in SOURCES:
        logger.info("%s is in %s", source.repository, stage(source, into))


if __name__ == "__main__":
    main()
