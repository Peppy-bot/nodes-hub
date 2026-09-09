#!/usr/bin/env python3
"""Run every discovered Python suite in its owning project's build image."""

import hashlib
import os
from pathlib import Path
import subprocess
import sys

from ci_containers import container_command, derive_test_image, log_group


def main() -> int:
    cache = Path(os.environ["CI_CACHE_DIR"])
    temporary = Path(os.environ["RUNNER_TEMP"])
    projects = (temporary / "python-test-projects.txt").read_text().splitlines()
    tests_by_project = {}
    for line in (temporary / "python-test-project-files.tsv").read_text().splitlines():
        project, test = line.split("\t", 1)
        tests_by_project.setdefault(project, []).append(test)

    failed = []
    for project in projects:
        image = derive_test_image(project, "peppybot/python-uv-base:latest")
        for path in (cache / "uv/cache", cache / "uv/python", cache / "uv/home", temporary / "uv-environments"):
            path.mkdir(parents=True, exist_ok=True)
        # Downloads can be cached, but environments must be private to this run
        # and project. Hashing keeps root and nested project names distinct.
        environment = "/environments/" + hashlib.sha256(project.encode()).hexdigest()
        command = container_command(
            project, image,
            binds=[
                f"{cache / 'uv'}:/uv",
                f"{temporary / 'uv-environments'}:/environments",
                f"{temporary / 'peppy-dist'}:/peppy-dist",
            ],
            variables={
                "HOME": "/uv/home",
                "UV_CACHE_DIR": "/uv/cache",
                "UV_PYTHON_INSTALL_DIR": "/uv/python",
                "UV_PROJECT_ENVIRONMENT": environment,
                "PEPPY_ZENOHD_PATH": "/peppy-dist/bin/zenohd",
            },
        )
        with log_group(f"pytest: {project}"):
            result = subprocess.run([
                *command, "uv", "run", "--locked", "--with", "pytest", "pytest",
                *tests_by_project[project],
            ])
        if result.returncode:
            print(f"::error::pytest failed in {project}", flush=True)
            failed.append(project)

    if failed:
        print(f"failing projects: {', '.join(failed)}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
