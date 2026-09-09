#!/usr/bin/env python3
"""Expose the sticky disk cache path to subsequent CI steps."""

import os
from pathlib import Path
import sys


def main() -> int:
    cache_dir = Path(os.environ["HOME"]) / ".cache" / "nodes-hub-ci"
    with Path(os.environ["GITHUB_ENV"]).open("a") as environment:
        environment.write(f"CI_CACHE_DIR={cache_dir}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
