#!/usr/bin/env python3
"""Register the checkout and generate interfaces for every node with tests."""

import os
import re
import subprocess
import sys
from pathlib import Path


def github_repository_id(registry: Path) -> str | None:
    # Match the existing registry layout without introducing a JSON5 dependency:
    # the repository ID appears within three lines before its GitHub URL.
    lines = registry.read_text().splitlines()
    for index, line in enumerate(lines):
        if "Peppy-bot/nodes-hub" in line:
            for context in lines[max(0, index - 3):index + 1]:
                match = re.search(r"id: *([0-9]*),", context)
                if match:
                    return match[1]
    return None


def main() -> int:
    workspace = Path(os.environ["GITHUB_WORKSPACE"])
    registry = Path(os.environ["PEPPY_HOME"]) / "conf/repositories.json5"
    repo_id = github_repository_id(registry)
    # Replace the bundled GitHub copy so dependencies resolve to this checkout.
    # Keep other repositories, which supply the contracts under test.
    if repo_id:
        subprocess.run(["peppy", "repo", "remove", repo_id], check=True)
    subprocess.run(["peppy", "repo", "add", str(workspace)], check=True)
    subprocess.run(["peppy", "repo", "refresh"], check=True)

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
