#!/usr/bin/env python3
"""Discover every Rust and Python test and its project environment for CI."""

import fnmatch
import os
from pathlib import Path
import sys


EXCLUDED_DIRECTORIES = {
    ".git", ".peppy", "target", "node_modules", ".venv", "venv", "dist", "build",
    "__pycache__",
}


def fail_walk(error: OSError) -> None:
    raise error


def write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("".join(f"{line}\n" for line in lines))


def main() -> int:
    output_dir = Path(os.environ["RUNNER_TEMP"])
    rust_projects = []
    python_tests = []

    # Cargo discovers all unit, integration and documentation tests in each
    # workspace. Python files follow pytest's standard naming conventions.
    for directory, subdirectories, filenames in os.walk(".", onerror=fail_walk):
        subdirectories[:] = [
            name for name in subdirectories if name not in EXCLUDED_DIRECTORIES
        ]
        project = Path(directory)
        for filename in filenames:
            path = project / filename
            if path.is_symlink() or not path.is_file():
                continue
            if filename == "Cargo.toml":
                rust_projects.append(project.as_posix())
            if fnmatch.fnmatchcase(filename, "test_*.py") or fnmatch.fnmatchcase(
                filename, "*_test.py"
            ):
                python_tests.append(path.as_posix())

    # Match the workflow's C-locale byte ordering independently of locale.
    rust_projects.sort(key=os.fsencode)
    python_tests.sort(key=os.fsencode)
    write_lines(output_dir / "rust-test-dirs.txt", rust_projects)
    write_lines(output_dir / "python-test-files.txt", python_tests)

    # A nested project owns its own tests, and a root pyproject.toml may own
    # tests too. Keeping each file's exact owner avoids running it twice in
    # different environments.
    python_projects = set()
    with (
        (output_dir / "python-test-projects.txt").open("w") as projects_file,
        (output_dir / "python-test-project-files.tsv").open("w") as ownership_file,
    ):
        for test_file in python_tests:
            path = Path(test_file)
            project = path.parent
            while not (project / "pyproject.toml").is_file():
                if project == Path("."):
                    print(
                        f"::error::{test_file} has no pyproject.toml above it "
                        "to run pytest from"
                    )
                    return 1
                project = project.parent
            project_name = project.as_posix()
            python_projects.add(project_name)
            projects_file.write(f"{project_name}\n")
            relative_file = path.relative_to(project).as_posix()
            ownership_file.write(f"{project_name}\t{relative_file}\n")
    write_lines(
        output_dir / "python-test-projects.txt",
        sorted(python_projects, key=os.fsencode),
    )

    # Only tested nodes need generated interfaces. Their node dependencies
    # resolve through the daemon's registry directly from this checkout.
    sync_projects = [
        project
        for project in sorted(set(rust_projects) | python_projects, key=os.fsencode)
        if (Path(project) / "peppy.json5").is_file()
    ]
    write_lines(output_dir / "sync-dirs.txt", sync_projects)

    any_tests = "true" if rust_projects or python_tests else "false"
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
        output.write(f"any={any_tests}\n")
    with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as summary:
        summary.write("Running all discovered Rust and Python tests.\n")
        for title, paths in (
            ("Rust test crates", rust_projects),
            ("Python test files", python_tests),
        ):
            summary.write(f"\n### {title}\n")
            summary.write(
                "".join(f"- {path}\n" for path in paths) if paths else "_none found_\n"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
