"""The launch parameters this node checks before it starts, parsed once into
an immutable config."""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import storage


@dataclass(frozen=True)
class Config:
    fps: int
    image_writer_threads: int
    max_staleness_s: float
    storage: storage.StorageTarget


def parse(params) -> Config:
    """The config, or ValueError naming the first parameter it refuses."""
    if params.fps <= 0:
        raise ValueError("fps must be positive")
    # add_frame runs on the event loop, so the library must encode images on
    # its own threads: at zero it writes every frame's PNG inline and stalls
    # the sampler and the drains behind it.
    if params.image_writer_threads <= 0:
        raise ValueError("image_writer_threads must be positive")
    target = storage.parse_storage(params.storage_root, params.s3_uri)
    if not math.isfinite(params.max_staleness_s) or params.max_staleness_s <= 0:
        raise ValueError(f"max_staleness_s must be positive, got {params.max_staleness_s}")
    return Config(
        fps=params.fps,
        image_writer_threads=params.image_writer_threads,
        max_staleness_s=params.max_staleness_s,
        storage=target,
    )
