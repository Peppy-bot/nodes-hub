#!/usr/bin/env python3
"""Verify tests consumed the committed manifests and lockfiles unchanged."""

import subprocess
import sys


MANIFEST_PATTERNS = [
    ":(glob)**/Cargo.toml",
    ":(glob)**/Cargo.lock",
    ":(glob)**/pyproject.toml",
    ":(glob)**/uv.lock",
]


def main() -> int:
    # Run after the tests. Lockfile changes indicate dependency resolution
    # drift; manifest changes indicate peppy node sync updated scaffolding
    # that needs to be generated and committed locally.
    return subprocess.run(["git", "diff", "--exit-code", "--", *MANIFEST_PATTERNS]).returncode


if __name__ == "__main__":
    sys.exit(main())
