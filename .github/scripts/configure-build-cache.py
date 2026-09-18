#!/usr/bin/env python3
"""Expose the persistent build cache path to subsequent CI steps."""

import os
from pathlib import Path
import sys


def main() -> int:
    # The directory outlives the job on the reused self-hosted runner. Its name
    # carries the architecture because the cached .sif images and crate build
    # outputs are built for the host that built them, so moving this repository
    # onto an arm runner starts from an empty cache rather than a wrong-arch
    # one. RUNNER_ARCH is set by Actions ("X64", "ARM64"); the fallback keeps
    # the script runnable outside CI.
    arch = os.environ.get("RUNNER_ARCH", "unknown")
    cache_dir = Path(os.environ["HOME"]) / ".cache" / f"nodes-hub-ci-{arch}"
    with Path(os.environ["GITHUB_ENV"]).open("a") as environment:
        environment.write(f"CI_CACHE_DIR={cache_dir}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
