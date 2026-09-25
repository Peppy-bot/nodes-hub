#!/usr/bin/env python3
"""Refresh the registered repositories and generate interfaces for every node with tests."""

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    workspace = Path(os.environ["GITHUB_WORKSPACE"])
    # The registry the hub-ci-peppy action wrote holds this checkout as the
    # nodes-hub entry and every sibling hub, which supply the contracts under
    # test, pinned by commit. With --strict, a hub that CI cannot read fails
    # the refresh instead of leaving its contracts out of the generated code.
    subprocess.run(["peppy", "repo", "refresh", "--strict"], check=True)

    inventory = Path(os.environ["RUNNER_TEMP"]) / "sync-dirs.txt"
    for node_dir in inventory.read_text().splitlines():
        print(f"::group::peppy node sync {node_dir}", flush=True)
        try:
            subprocess.run(
                ["peppy", "node", "sync", node_dir, "--include-repositories"],
                check=True, cwd=workspace,
            )
        finally:
            print("::endgroup::", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
