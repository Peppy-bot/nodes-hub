#!/usr/bin/env python3
"""Expose the build cache path to subsequent CI steps."""

import os
from pathlib import Path
import sys


def main() -> int:
    # The directory the runs-on/action step of tests.yml bind-mounts onto the
    # job's sticky disk, so what a run leaves here is snapshotted after the
    # job and restored for the next one. RunsOn keeps a sticky disk per
    # architecture, so the name needs none.
    cache_dir = Path(os.environ["HOME"]) / ".cache" / "nodes-hub-ci"
    with Path(os.environ["GITHUB_ENV"]).open("a") as environment:
        environment.write(f"CI_CACHE_DIR={cache_dir}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
