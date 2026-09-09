#!/usr/bin/env python3
"""Compile every tracked Python file without importing runtime dependencies."""

import os
import subprocess
import sys
import traceback
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    # A job container and checkout can have different owners. Trust only the
    # checkout for this read-only inventory, as the Node test job does.
    inventory = subprocess.check_output(
        [
            "git", "-c", f"safe.directory={root}", "-C", str(root),
            "ls-files", "-z", "--", "*.py",
        ]
    )
    files = [os.fsdecode(path) for path in inventory.split(b"\0") if path]
    failures = 0
    for filename in files:
        try:
            # Bytes preserve encoding declarations. Compilation catches syntax
            # errors even in launchers that tests never import, without running
            # the code or writing __pycache__ into the checkout.
            compile((root / filename).read_bytes(), filename, "exec", dont_inherit=True)
        except SyntaxError as error:
            failures += 1
            print("".join(traceback.format_exception_only(error)), file=sys.stderr, end="")

    print(f"Checked {len(files)} Python files; {failures} syntax errors.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
