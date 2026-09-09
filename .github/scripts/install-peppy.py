#!/usr/bin/env python3
"""Install the complete Peppy release and record it for subsequent CI steps."""

import os
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    version = os.environ.get("PEPPY_VERSION") or "latest"
    if version == "latest":
        channel = "latest"
    elif re.fullmatch(r"v?[0-9]+\.[0-9]+\.[0-9]+", version):
        channel = f"v{version.removeprefix('v')}"
    else:
        print(
            "PEPPY_VERSION must be 'latest' or a release such as 0.25.0, "
            f"got: {version}",
            file=sys.stderr,
        )
        return 1

    machine = platform.machine()
    architecture = {
        "x86_64": "x86_64", "amd64": "x86_64",
        "aarch64": "aarch64", "arm64": "aarch64",
    }.get(machine)
    if architecture is None:
        print(f"unsupported runner architecture: {machine}", file=sys.stderr)
        return 1

    runner_temp = Path(os.environ["RUNNER_TEMP"])
    destination = runner_temp / "peppy-dist"
    destination.mkdir(parents=True, exist_ok=True)
    url = f"https://peppy.bot/{channel}/peppy-{architecture}-unknown-linux-gnu.tgz"
    # Keep the entire bundled layout: the daemon needs zenohd and apptainer.
    with tempfile.TemporaryDirectory(prefix="peppy-release-", dir=runner_temp) as temporary:
        archive = Path(temporary) / "release.tgz"
        subprocess.run(
            ["curl", "-fsSL", "--connect-timeout", "10", "--max-time", "300",
             "--output", str(archive), url],
            check=True,
        )
        subprocess.run(["tar", "-xzf", str(archive), "-C", str(destination)], check=True)

    binary = destination / "bin" / "peppy"
    binary.chmod(binary.stat().st_mode | 0o111)
    with Path(os.environ["GITHUB_PATH"]).open("a") as output:
        output.write(f"{binary.parent}\n")

    # Record the actual version, since the latest release can change between runs.
    installed = subprocess.run(
        [str(binary), "--version"], check=True, text=True, stdout=subprocess.PIPE,
    ).stdout.rstrip("\n")
    with Path(os.environ["GITHUB_ENV"]).open("a") as output:
        output.write(f"PEPPY_INSTALLED={installed}\n")
    summary = f"\nInstalled {installed} (channel: {channel})\n"
    print(summary, end="")
    with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as output:
        output.write(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
