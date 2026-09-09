#!/usr/bin/env python3
"""Report every Rust lockfile that no longer resolves after interface generation."""

import os
from pathlib import Path
import subprocess
import sys

from ci_containers import container_command, derive_test_image, log_group


def main() -> int:
    cache = Path(os.environ["CI_CACHE_DIR"])
    temporary = Path(os.environ["RUNNER_TEMP"])
    (cache / "cargo-home").mkdir(parents=True, exist_ok=True)
    projects = (temporary / "rust-test-dirs.txt").read_text().splitlines()
    stale = []
    reports = []
    for project in projects:
        # Resolve with the same toolchain and system libraries as the test run.
        image = derive_test_image(project, "peppybot/rust-cargo-base:latest")
        command = container_command(
            project, image,
            binds=[f"{cache / 'cargo-home'}:/cargo", os.environ["PEPPY_HOME"]],
            variables={"CARGO_HOME": "/cargo", "RUSTUP_HOME": "/root/.rustup"},
        )
        result = subprocess.run(
            [*command, "cargo", "metadata", "--locked", "--format-version", "1"],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
        )
        if result.returncode:
            stale.append(project)
            diagnostic = result.stderr.rstrip("\n")
            reports.append(f"{project}:\n{diagnostic}\n\n")

    report = "".join(reports)
    (temporary / "stale-lockfiles.txt").write_text(report)
    if not stale:
        return 0
    with log_group("what cargo reported"):
        print(report, end="", flush=True)
    installed = os.environ["PEPPY_INSTALLED"]
    summary = (
        "\n### Lockfiles do not resolve against the generated interfaces\n\n"
        f"`{installed}` generates a `.peppy/libs` these lockfiles cannot resolve against:\n"
        + "".join(f"- {project}\n" for project in stale)
        + "\nRun `scripts/relock.sh` against that release and commit what it rewrites.\n"
    )
    with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as stream:
        stream.write(summary)
    print(
        f"::error::{len(stale)} committed lockfile(s) do not resolve against "
        f"{installed}'s generated interfaces: run scripts/relock.sh and commit what it rewrites"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
