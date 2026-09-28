#!/usr/bin/env python3
"""Discover the Rust and Python tests a change reaches, and their environments."""

from dataclasses import dataclass
import fnmatch
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys


EXCLUDED_DIRECTORIES = {
    ".git", ".github", ".peppy", "target", "node_modules", ".venv", "venv", "dist", "build",
    "__pycache__",
}


@dataclass(frozen=True)
class Inventory:
    """Every suite the repository holds, and the projects that own them.

    A project is a directory this repository tests as a unit: a cargo workspace
    or a node. Nothing finer, because a suite sits beside the project it covers
    rather than inside it (`sim_isaac/tests/`), and the project is what owns the
    image and the lockfile that suite runs against.
    """

    rust_projects: list[str]
    python_tests: list[str]
    projects: list[str]


@dataclass(frozen=True)
class Selection:
    """What this run tests, and the sentence its summary opens with."""

    scope: str
    rust_projects: list[str]
    python_tests: list[str]


def fail_walk(error: OSError) -> None:
    raise error


def write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("".join(f"{line}\n" for line in lines))


def take_inventory(root: Path = Path(".")) -> Inventory:
    """Walk the checkout for suites and for the projects that own them.

    Cargo discovers all unit, integration and documentation tests in each
    workspace. Python files follow pytest's standard naming conventions.
    """
    rust_projects = []
    python_tests = []
    projects = set()

    for directory, subdirectories, filenames in os.walk(root, onerror=fail_walk):
        subdirectories[:] = [
            name for name in subdirectories if name not in EXCLUDED_DIRECTORIES
        ]
        project = Path(directory).relative_to(root)
        for filename in filenames:
            path = project / filename
            if (root / path).is_symlink() or not (root / path).is_file():
                continue
            if filename == "Cargo.toml":
                rust_projects.append(project.as_posix())
            if filename in ("Cargo.toml", "peppy.json5"):
                projects.add(project.as_posix())
            if fnmatch.fnmatchcase(filename, "test_*.py") or fnmatch.fnmatchcase(
                filename, "*_test.py"
            ):
                python_tests.append(path.as_posix())

    # Match the workflow's C-locale byte ordering independently of locale.
    return Inventory(
        rust_projects=sorted(rust_projects, key=os.fsencode),
        python_tests=sorted(python_tests, key=os.fsencode),
        projects=sorted(projects, key=os.fsencode),
    )


def owning_project(path: str, projects: set[str]) -> str | None:
    """The project a path belongs to: the longest of its ancestors, or itself,
    that is one.

    `None` for a path no project holds, which is what sends a root-level change
    to every project.
    """
    candidate = PurePosixPath(path)
    while True:
        name = candidate.as_posix()
        if name in projects:
            return name
        if name == ".":
            return None
        candidate = candidate.parent


def changed_files() -> list[str] | None:
    """The paths a pull request touches, or `None` when this run covers
    everything.

    Only a pull request has a base to compare against; a push to main has none.
    A diff that fails and a diff that comes back empty read the same way, since
    a run that cannot tell what changed is a run that has to cover all of it.
    """
    if os.environ.get("GITHUB_EVENT_NAME") != "pull_request":
        return None
    base = os.environ.get("BASE_SHA", "")
    if not base:
        return None
    diff = subprocess.run(
        ["git", "diff", "--name-only", f"{base}...HEAD"],
        capture_output=True,
        text=True,
    )
    if diff.returncode:
        print(
            f"::warning::diffing against {base} failed, so this run covers every "
            f"project:\n{diff.stderr.strip()}",
            flush=True,
        )
        return None
    return [line for line in diff.stdout.splitlines() if line] or None


def everything(inventory: Inventory, scope: str) -> Selection:
    return Selection(scope, inventory.rust_projects, inventory.python_tests)


def select(inventory: Inventory, changed: list[str] | None) -> Selection:
    """Narrow the inventory to the projects a change reaches.

    A path no project holds reaches all of them: the workflow itself, the
    scripts it runs, anything at the root. Markdown is the exception, since
    nothing builds or tests it, so documentation outside every project is
    passed over rather than escalated.
    """
    if changed is None:
        return everything(inventory, "every project")

    reached: set[str] = set()
    projects = set(inventory.projects)
    for path in changed:
        project = owning_project(path, projects)
        if project:
            reached.add(project)
            continue
        if path.lower().endswith(".md"):
            continue
        return everything(inventory, f"every project ({path} belongs to none of them)")

    if not reached:
        return Selection(
            "nothing: this pull request only changes documentation outside every project",
            [],
            [],
        )
    return Selection(
        scope="the projects this pull request touches",
        rust_projects=[
            project
            for project in inventory.rust_projects
            if owning_project(project, reached)
        ],
        python_tests=[
            test for test in inventory.python_tests if owning_project(test, reached)
        ],
    )


def main() -> int:
    output_dir = Path(os.environ["RUNNER_TEMP"])
    inventory = take_inventory()
    selection = select(inventory, changed_files())

    write_lines(output_dir / "rust-test-dirs.txt", selection.rust_projects)
    write_lines(output_dir / "python-test-files.txt", selection.python_tests)

    # A nested project owns its own tests, and a root pyproject.toml may own
    # tests too. Keeping each file's exact owner avoids running it twice in
    # different environments.
    python_projects = set()
    with (
        (output_dir / "python-test-projects.txt").open("w") as projects_file,
        (output_dir / "python-test-project-files.tsv").open("w") as ownership_file,
    ):
        for test_file in selection.python_tests:
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
        for project in sorted(
            set(selection.rust_projects) | python_projects, key=os.fsencode
        )
        if (Path(project) / "peppy.json5").is_file()
    ]
    write_lines(output_dir / "sync-dirs.txt", sync_projects)

    any_tests = "true" if selection.rust_projects or selection.python_tests else "false"
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
        output.write(f"any={any_tests}\n")
    with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as summary:
        summary.write(f"Running {selection.scope}.\n")
        for title, paths in (
            ("Rust test crates", selection.rust_projects),
            ("Python test files", selection.python_tests),
        ):
            summary.write(f"\n### {title}\n")
            summary.write(
                "".join(f"- {path}\n" for path in paths) if paths else "_none selected_\n"
            )
        # A scoped run never reads as full coverage: what it left out is named
        # beside what it ran.
        selected_crates = set(selection.rust_projects)
        skipped = [
            project for project in inventory.rust_projects if project not in selected_crates
        ]
        if skipped:
            summary.write("\n### Skipped crates (untouched by this change)\n")
            summary.write("".join(f"- {project}\n" for project in skipped))
    return 0


if __name__ == "__main__":
    sys.exit(main())
