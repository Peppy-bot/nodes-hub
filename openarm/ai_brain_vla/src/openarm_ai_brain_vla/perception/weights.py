"""The weights of the two models the sam3_siglip backend loads, downloaded
to the machine by the first load that needs them.

A model is a `Source`: a repository of the Hugging Face Hub at one commit,
and the files of it that transformers reads, each pinned by its size and
its SHA-256. `stage` names the directory that holds a model's files and
downloads them first when the machine does not have them. Only a node whose
launch selects the backend downloads them, once for the machine, and every
machine reads the same bytes.

The weights are kept in the directory `WEIGHTS_DIRECTORY_VARIABLE` names,
else in `DEFAULT_DIRECTORY` of the daemon user's home, which the container
sees and which stays when the node image is built again:

    <directory>/lock                          held by the node that downloads
    <directory>/<name>-<revision>/            a whole model
    <directory>/<name>-<revision>.fetching/   a model that is being downloaded

A file is downloaded as `<file>.partial` in the `.fetching` directory and
takes its name once it holds the pinned bytes. The directory takes the
model's name once it holds every file. So a directory under a model's name
always holds the whole model, checked, and a load that finds it reads no
network and hashes nothing.

A node can be stopped or killed at any point of a download. It leaves at
most one partial file for each file of the model, and the next start
continues each partial file from its last byte with an HTTP range request.
The download runs on the standard library alone, in the thread of the load:
it starts no thread and no process that could hold the node past its
shutdown. Two nodes of one machine do not download the same files twice:
one holds the lock of the directory, and the other waits for it and then
finds the model.

    python -m openarm_ai_brain_vla.perception.weights fetch [directory]

downloads every model, for a machine that must have the weights before its
first launch or that a robot with no network copies the directory from.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import http.client
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
HUB = "https://huggingface.co"
LOCK_FILE = "lock"
FETCHING_SUFFIX = ".fetching"
PARTIAL_SUFFIX = ".partial"
CHUNK_BYTES = 1 << 20
# The seconds a connection, or one read from it, can take before the attempt
# fails.
TIMEOUT_S = 60.0
# The attempts in a row that add no byte to a file before its download fails.
STALLED_ATTEMPTS = 3


class WeightsError(Exception):
    """A model's weights could not be put on the machine."""


@dataclass(frozen=True)
class File:
    """One file of a model: its path in the repository, its size in bytes
    and the SHA-256 of its content."""

    path: str
    size: int
    sha256: str


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


# SAM 3's official repository, facebook/sam3, is gated behind a licence
# click-through on the Hub. This mirror carries the same files, each with the
# same content hash at this revision. Its `sam3.pt`, the checkpoint of the
# original code base, is not a file transformers reads.
SAM3 = Source(
    "sam3",
    "jetjodh/sam3",
    "1aa50ce07302cb375f85d8084b68a0fb378b8d85",
    (
        File("config.json", 25843, "4616385e4b21f2e5e22c875b65679185cbccfa95de42542b9166f7dc3d57160f"),
        File("merges.txt", 524619, "9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a"),
        File("model.safetensors", 3439938512, "6d06f0a5f84e435071fe6603e61d0b4cc7b40e0d39d487cfd4d67d8cc11cc14a"),
        File("processor_config.json", 1712, "6420cf2671fa9309ea95bc0144a8b9861666d1c5f43c8db09e410dacda974fce"),
        File("special_tokens_map.json", 588, "2cdb3b8331a60c92fc1e55a13e9fd61fd2293c5a51275fdcccd62b780052530e"),
        File("tokenizer.json", 3642073, "6d9109cc838977f3ca94a379eec36aecc7c807e1785cd729660ca2fc0171fb35"),
        File("tokenizer_config.json", 799, "39670ad98457fe8f14ca59f6bb74591e9fc850974380a63993e5b8ffc865baa2"),
        File("vocab.json", 862328, "5047b556ce86ccaf6aa22b3ffccfc52d391ea4accdab9c2f2407da5b742d4363"),
    ),
)
# SigLIP so400m is the study's namer.
SIGLIP = Source(
    "siglip",
    "google/siglip-so400m-patch14-384",
    "9fdffc58afc957d1a03a25b10dba0329ab15c2a3",
    (
        File("config.json", 576, "adc04928d8fd19a61822584fe0cf2e813e5ebac17f3e49fb1ea096860ae6457b"),
        File("model.safetensors", 3511950624, "ea2abad2b7f8a9c1aa5e49a244d5d57ffa71c56f720c94bc5d240ef4d6e1d94a"),
        File("preprocessor_config.json", 368, "f59da2f87c3cd079bd4f8f3037e81b277c60c498e279a8020331f67a5a3157e8"),
        File("special_tokens_map.json", 409, "2b6a1ff67a27e0df9ac0c7d93250fc0d87431c7b366b3d5669217104f9088a26"),
        File("spiece.model", 798330, "1e5036bed065526c3c212dfbe288752391797c4bb1a284aa18c9a0b23fcaf8ec"),
        File("tokenizer.json", 2399357, "c6e405cb7c670d56636a9402c81023a55bc6c3c53d89cf02b92f5c5005bfe920"),
        File("tokenizer_config.json", 711, "d6423dae508cc3a129d22ea443841c111832a1a73125b8f25ea8736951698bcb"),
    ),
)
SOURCES = (SAM3, SIGLIP)


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
    file lacks, and names the file once its content is the pinned one. A
    partial file with other content is removed and the fetch fails, so the
    next one downloads the file from its start."""
    target = fetching / file.path
    if target.exists():
        return
    partial = partial_of(target)
    url = source.url(file)
    download(file, url, partial, open_at)
    with partial.open("rb") as content:
        found = hashlib.file_digest(content, "sha256").hexdigest()
        # The bytes reach the disk before the name that says they are whole.
        os.fsync(content.fileno())
    if found != file.sha256:
        partial.unlink()
        raise WeightsError(f"{url} is not the pinned file: its SHA-256 is {found}, the pin is {file.sha256}")
    partial.rename(target)


def download(file: File, url: str, partial: Path, open_at: OpenAt) -> None:
    """Appends to `partial` until it holds as many bytes as `file`. A
    failed attempt is followed by another from the byte the file ends at,
    until `STALLED_ATTEMPTS` of them in a row have added nothing."""
    stalled = 0
    while (held := size_of(partial)) < file.size:
        try:
            receive(file, url, partial, held, open_at)
            failure = "the server ended its answer"
        except (OSError, http.client.HTTPException) as error:
            failure = str(error) or type(error).__name__
        now_held = size_of(partial)
        stalled = 0 if now_held > held else stalled + 1
        if stalled == STALLED_ATTEMPTS:
            raise WeightsError(f"the download of {url} stopped at {now_held} of {file.size} bytes: {failure}")


def receive(file: File, url: str, partial: Path, held: int, open_at: OpenAt) -> None:
    """One attempt: asks for `file` from the byte `partial` ends at and
    appends what comes, at most up to the file's size, each chunk written
    through before the next is read, so a node that is killed loses the
    chunk it was reading and no more. A server that sends the whole file
    makes the partial file start again."""
    with open_at(url, held) as body, partial.open("ab") as out:
        if body.starts_at > held:
            raise OSError(f"the answer starts at byte {body.starts_at}, past byte {held}")
        out.truncate(body.starts_at)
        position = body.starts_at
        while position < file.size and (chunk := body.read(min(CHUNK_BYTES, file.size - position))):
            out.write(chunk)
            out.flush()
            before, position = position, position + len(chunk)
            if tenths(position, file) > tenths(before, file):
                logger.info("%s: %d0%% of %.1f MB", url, tenths(position, file), file.size / 1e6)


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
    parser = argparse.ArgumentParser(prog=f"python -m {MODULE}", description="Downloads the weights of the sam3_siglip backend's models.")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch_command = commands.add_parser("fetch", help="download every model that a directory does not hold")
    fetch_command.add_argument("directory", type=Path, nargs="?", help=f"where the weights are kept; the node's own directory when left out ({WEIGHTS_DIRECTORY_VARIABLE}, else {DEFAULT_DIRECTORY})")
    into = parser.parse_args(arguments).directory or node_directory()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for source in SOURCES:
        logger.info("%s is in %s", source.repository, stage(source, into))


if __name__ == "__main__":
    main()
