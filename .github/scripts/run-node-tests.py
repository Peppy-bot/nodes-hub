#!/usr/bin/env python3
"""Run every tracked native Node JavaScript and TypeScript test suite."""

import os
from pathlib import Path
import subprocess
import sys


EXCLUDED_DIRECTORIES = {
    "node_modules", ".peppy", "target", "dist", "build", ".venv", "venv",
}
TEST_PATTERNS = [
    ":(glob)**/*.test.[jt]s",
    ":(glob)**/*.test.[cm][jt]s",
    ":(glob)**/*.spec.[jt]s",
    ":(glob)**/*.spec.[cm][jt]s",
]


def main() -> int:
    # Trust only this workspace for the read-only discovery command.
    # NUL separators preserve filenames containing spaces or shell metacharacters.
    inventory = subprocess.run(
        [
            "git", "-c", f"safe.directory={os.environ['GITHUB_WORKSPACE']}",
            "ls-files", "-z", "--", *TEST_PATTERNS,
        ],
        stdout=subprocess.PIPE,
    )
    (Path(os.environ["RUNNER_TEMP"]) / "node-test-files").write_bytes(inventory.stdout)
    if inventory.returncode:
        return inventory.returncode

    tests = []
    for filename in inventory.stdout.split(b"\0"):
        if not filename:
            continue
        path = Path(os.fsdecode(filename))
        if EXCLUDED_DIRECTORIES.intersection(path.parts[:-1]):
            continue
        tests.append(f"./{path.as_posix()}")

    # Bare node --test performs its own discovery, so never invoke it with an
    # empty inventory. Native TypeScript tests use Node's type stripping.
    if not tests:
        print("No JavaScript or TypeScript Node test suites found.")
        return 0
    for test in tests:
        print(f"Running {test}", flush=True)
    return subprocess.run(["node", "--test", *tests]).returncode


if __name__ == "__main__":
    sys.exit(main())
