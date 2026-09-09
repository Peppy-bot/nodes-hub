#!/usr/bin/env python3
"""Run every Rust workspace's tests in its node's build image."""

import os
from pathlib import Path
import subprocess
import sys

from ci_containers import container_command, derive_test_image, log_group


def main() -> int:
    cache = Path(os.environ["CI_CACHE_DIR"])
    temporary = Path(os.environ["RUNNER_TEMP"])
    projects = (temporary / "rust-test-dirs.txt").read_text().splitlines()
    failed = []
    for project in projects:
        image = derive_test_image(project, "peppybot/rust-cargo-base:latest")
        # Each node generates a different peppygen with the same package version.
        # Sharing a target directory could reuse another node's generated code.
        target = cache / "target" / project if project != "." else cache / "target-root"
        target.mkdir(parents=True, exist_ok=True)
        (cache / "cargo-home").mkdir(parents=True, exist_ok=True)
        command = container_command(
            project, image,
            binds=[
                f"{cache / 'cargo-home'}:/cargo",
                f"{target}:/target",
                f"{temporary / 'peppy-dist'}:/peppy-dist",
                os.environ["PEPPY_HOME"],
            ],
            variables={
                "CARGO_HOME": "/cargo",
                "RUSTUP_HOME": "/root/.rustup",
                "CARGO_TARGET_DIR": "/target",
                "CARGO_INCREMENTAL": "0",
                "PEPPY_ZENOHD_PATH": "/peppy-dist/bin/zenohd",
                "CARGO_TERM_COLOR": "always",
            },
        )
        with log_group(f"cargo test: {project}"):
            result = subprocess.run([*command, "cargo", "test", "--locked", "--workspace"])
        if result.returncode:
            print(f"::error::cargo test failed in {project}", flush=True)
            failed.append(project)

    if failed:
        print(f"failing crates: {', '.join(failed)}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
