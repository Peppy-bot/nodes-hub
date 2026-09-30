"""The weights of the two models the sam3_siglip backend loads, staged into
the node image at build.

`fetch` downloads each model from the Hugging Face Hub, at the revision
pinned here, into a directory (apptainer.def runs it at build), and `staged`
names a staged model's directory for the backend to load. A start of the
node reads its weights from the image and reaches for no network: a node
that is stopped while it starts leaves nothing behind, and every build of
one commit of the node carries the same files.

A model is staged in `<directory>/<name>-<revision>`. `fetch` downloads it
into `<name>-<revision>.fetching` beside that and renames the directory
once the download has ended, so a directory under the staged name always
holds the whole model, and the next `fetch` starts an interrupted one again
from nothing.

    python -m openarm_ai_brain_vla.perception.weights fetch <directory>
"""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

MODULE = "openarm_ai_brain_vla.perception.weights"
# Names the directory the models are staged in. The node image exports it
# (apptainer.def); a native run sets it to the directory it fetched into.
WEIGHTS_DIRECTORY_VARIABLE = "OPENARM_AI_BRAIN_VLA_WEIGHTS"
FETCHING_SUFFIX = ".fetching"


@dataclass(frozen=True)
class Source:
    """One model on the Hub: its repository at one commit, named by the
    full hash so that every fetch gets the same files, less the files of
    the repository that transformers does not read."""

    name: str
    repository: str
    revision: str
    left_out: tuple[str, ...] = ()

    @property
    def directory_name(self) -> str:
        return f"{self.name}-{self.revision}"


# SAM 3's official repository, facebook/sam3, is gated behind a licence
# click-through on the Hub; this mirror carries the same files, each with
# the same content hash at this revision. `sam3.pt` is the checkpoint of the
# original code base: transformers reads `model.safetensors`.
SAM3 = Source("sam3", "jetjodh/sam3", "1aa50ce07302cb375f85d8084b68a0fb378b8d85", left_out=("sam3.pt",))
# SigLIP so400m is the study's namer.
SIGLIP = Source("siglip", "google/siglip-so400m-patch14-384", "9fdffc58afc957d1a03a25b10dba0329ab15c2a3")
SOURCES = (SAM3, SIGLIP)

# Downloads every file of a source into a directory, or raises.
Download = Callable[[Source, Path], None]


def download_from_hub(source: Source, directory: Path) -> None:
    """The files of `source` at its revision, from the Hugging Face Hub.
    huggingface_hub comes with the sam3-siglip extra and is imported here
    alone, so the backend's other modules never need it."""
    from huggingface_hub import snapshot_download

    snapshot_download(
        source.repository,
        revision=source.revision,
        local_dir=directory,
        ignore_patterns=list(source.left_out),
    )


def fetch(directory: Path, download: Download = download_from_hub) -> None:
    """Stages every model of `SOURCES` under `directory`."""
    for source in SOURCES:
        stage(source, directory, download)


def stage(source: Source, directory: Path, download: Download) -> Path:
    """Stages `source` under `directory` and returns where. A model already
    staged is left as it is; what an interrupted fetch left is removed, and
    the model is downloaded again whole."""
    staged_at = directory / source.directory_name
    if staged_at.is_dir():
        return staged_at
    fetching = directory / f"{source.directory_name}{FETCHING_SUFFIX}"
    if fetching.exists():
        shutil.rmtree(fetching)
    fetching.mkdir(parents=True)
    download(source, fetching)
    fetching.rename(staged_at)
    return staged_at


def staged(source: Source) -> Path:
    """The directory `source` is staged in, under the one
    `WEIGHTS_DIRECTORY_VARIABLE` names. Raises FileNotFoundError saying how
    the weights get there when they are not."""
    how = (
        "the node image stages them at build (apptainer.def); a native run needs "
        f"`python -m {MODULE} fetch <directory>` and {WEIGHTS_DIRECTORY_VARIABLE} naming that directory"
    )
    directory = os.environ.get(WEIGHTS_DIRECTORY_VARIABLE, "").strip()
    if not directory:
        raise FileNotFoundError(f"the weights of {source.repository} are not staged, {WEIGHTS_DIRECTORY_VARIABLE} is not set: {how}")
    staged_at = Path(directory).expanduser() / source.directory_name
    if not staged_at.is_dir():
        raise FileNotFoundError(f"the weights of {source.repository} are not staged at {staged_at}: {how}")
    return staged_at


def main(arguments: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog=f"python -m {MODULE}", description="Stages the weights of the sam3_siglip backend's models.")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch_command = commands.add_parser("fetch", help="download every model into a directory")
    fetch_command.add_argument("directory", type=Path, help="where the models are staged")
    fetch(parser.parse_args(arguments).directory)


if __name__ == "__main__":
    main()
