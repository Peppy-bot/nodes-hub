#!/usr/bin/env python3
"""Run every Rust workspace's tests in its node's build image.

Cargo never removes an artifact that a changed dependency, feature or profile
superseded, so a target dir on the sticky disk would grow by a full set of test
binaries at each such change, until the disk is full. So each workspace builds
alone first, as `cargo test --no-run`, which lists on stdout in JSON the
outputs of every unit it used, while rustc's diagnostics still reach the log as
text. The script of peppy's cargo-target-sweep action records those units in
the target dir and removes every other one; before the build, it removes what a
run that stopped before its record left, and clears a target dir that has no
record. `cargo test` then runs the tests out of that build and compiles nothing.
"""

import os
from pathlib import Path
import shutil
import subprocess
import sys

from ci_containers import CARGO_HOME, cargo_home_bind, container_command, derive_test_image, log_group


def target_dir(cache: Path, project: str) -> Path:
    # Each node generates a different peppygen with the same package version.
    # Sharing a target directory could reuse another node's generated code.
    return cache / "target" / project if project != "." else cache / "target-root"


def run_sweep(*arguments: str | Path) -> None:
    script = os.environ["SWEEP_CARGO_TARGET_SCRIPT"]
    subprocess.run([sys.executable, script, *map(str, arguments)], check=True)


def prune_target_dirs(cache: Path, projects: list[str]) -> None:
    """Remove the target dir of each project the checkout no longer holds.

    Only the target dir of the project at the root is outside `target/`,
    because `target/` holds the target dirs of all the others.
    """
    run_sweep("prune", cache / "target", *(project for project in projects if project != "."))
    root_target = target_dir(cache, ".")
    if "." not in projects and root_target.exists():
        shutil.rmtree(root_target)


def build_tests(command: list[str], listing: Path) -> bool:
    with listing.open("w") as build_output:
        build = subprocess.run(
            [*command, "cargo", "test", "--locked", "--workspace", "--no-run",
             "--message-format=json-render-diagnostics"],
            stdout=build_output,
        )
    return build.returncode == 0


def main() -> int:
    cache = Path(os.environ["CI_CACHE_DIR"])
    temporary = Path(os.environ["RUNNER_TEMP"])
    projects = (temporary / "rust-test-dirs.txt").read_text().splitlines()
    listing = temporary / "cargo-build-listing.jsonl"
    prune_target_dirs(cache, (temporary / "all-rust-projects.txt").read_text().splitlines())
    failed = []
    for project in projects:
        image = derive_test_image(project, "peppybot/rust-cargo-base:latest")
        target = target_dir(cache, project)
        target.mkdir(parents=True, exist_ok=True)
        command = container_command(
            project, image,
            binds=[
                cargo_home_bind(cache),
                f"{target}:/target",
                f"{temporary / 'peppy-dist'}:/peppy-dist",
                os.environ["PEPPY_HOME"],
            ],
            variables={
                "CARGO_HOME": CARGO_HOME,
                "RUSTUP_HOME": "/root/.rustup",
                "CARGO_TARGET_DIR": "/target",
                "CARGO_INCREMENTAL": "0",
                "PEPPY_ZENOHD_PATH": "/peppy-dist/bin/zenohd",
                "CARGO_TERM_COLOR": "always",
            },
        )
        with log_group(f"cargo test: {project}"):
            run_sweep("sweep", target)
            built = build_tests(command, listing)
            # A build that stopped still lists what it built, and the record
            # keeps that beside the units of the previous build.
            run_sweep("record", target, listing, "/target")
            passed = built and subprocess.run(
                [*command, "cargo", "test", "--locked", "--workspace"]
            ).returncode == 0
        if not passed:
            print(f"::error::cargo test failed in {project}", flush=True)
            failed.append(project)

    if failed:
        print(f"failing crates: {', '.join(failed)}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
