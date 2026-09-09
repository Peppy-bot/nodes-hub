#!/usr/bin/env python3
"""Prepare the bundled Apptainer for unprivileged CI builds."""

import os
import pwd
import subprocess
import sys
from pathlib import Path


def subordinate_range(path: Path, user: str) -> int | None:
    """Return the next free range, or None if this user already has one."""
    try:
        lines = path.read_text().splitlines()
    except FileNotFoundError:
        lines = []
    entries = [line.split(":") for line in lines if line.strip()]
    if any(entry[0] == user for entry in entries):
        return None
    return max([100000, *(int(entry[1]) + int(entry[2]) for entry in entries)])


def main() -> int:
    apptainer = Path(os.environ["RUNNER_TEMP"]) / "peppy-dist/bin/apptainer"
    with Path(os.environ["GITHUB_ENV"]).open("a") as output:
        output.write(f"PEPPY_APPTAINER_DIR={apptainer}\n")
    environment = dict(os.environ, PEPPY_APPTAINER_DIR=str(apptainer))
    # Peppy applies its required namespace/AppArmor setup through sudo and rechecks it.
    subprocess.run(
        ["peppy", "container", "setup"], check=True,
        stdin=subprocess.DEVNULL, env=environment,
    )

    # Fakeroot additionally needs a subordinate UID and GID range for the runner.
    user = pwd.getpwuid(os.geteuid()).pw_name
    for kind in ("subuid", "subgid"):
        start = subordinate_range(Path("/etc") / kind, user)
        if start is not None:
            subprocess.run(
                ["sudo", "usermod", f"--add-{kind}s", f"{start}-{start + 65535}", user],
                check=True, env=environment,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
