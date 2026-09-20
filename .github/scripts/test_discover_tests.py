#!/usr/bin/env python3
"""What `discover-tests.py` selects, and what it refuses to leave out."""

import importlib.util
from pathlib import Path
import tempfile
import unittest


def load_discovery():
    """Import the entry point beside this file.

    Scripts the workflow runs are named with hyphens, which no import statement
    can spell, so the module is loaded by path.
    """
    path = Path(__file__).with_name("discover-tests.py")
    spec = importlib.util.spec_from_file_location("discover_tests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


discovery = load_discovery()


def inventory(rust_projects=(), python_tests=(), projects=()):
    """An inventory with no filesystem behind it, so a case reads as the tree
    it describes."""
    return discovery.Inventory(
        rust_projects=list(rust_projects),
        python_tests=list(python_tests),
        projects=list(projects),
    )


REPOSITORY = inventory(
    rust_projects=["openarm/arm", "openarm/ker", "zed_camera"],
    python_tests=[
        "openarm/ai_brain/tests/test_smoke.py",
        "so101/leader/tests/test_params.py",
    ],
    projects=[
        "openarm/ai_brain",
        "openarm/arm",
        "openarm/ker",
        "so101/leader",
        "zed_camera",
    ],
)


class OwningProject(unittest.TestCase):
    def test_a_file_belongs_to_the_project_holding_it(self):
        self.assertEqual(
            discovery.owning_project("openarm/arm/src/main.rs", {"openarm/arm"}),
            "openarm/arm",
        )

    def test_a_project_owns_itself(self):
        self.assertEqual(
            discovery.owning_project("zed_camera", {"zed_camera"}), "zed_camera"
        )

    def test_the_longest_ancestor_wins(self):
        projects = {"openarm", "openarm/arm"}
        self.assertEqual(
            discovery.owning_project("openarm/arm/Cargo.toml", projects), "openarm/arm"
        )

    def test_a_path_no_project_holds_belongs_to_none(self):
        self.assertIsNone(
            discovery.owning_project(".github/workflows/tests.yml", {"openarm/arm"})
        )

    def test_a_project_at_the_root_holds_every_path(self):
        self.assertEqual(discovery.owning_project("scripts/run.sh", {"."}), ".")


class Select(unittest.TestCase):
    def test_a_run_with_no_diff_covers_every_project(self):
        selection = discovery.select(REPOSITORY, None)
        self.assertEqual(selection.rust_projects, REPOSITORY.rust_projects)
        self.assertEqual(selection.python_tests, REPOSITORY.python_tests)
        self.assertEqual(selection.scope, "every project")

    def test_a_change_inside_one_project_runs_that_project_alone(self):
        selection = discovery.select(REPOSITORY, ["openarm/arm/src/main.rs"])
        self.assertEqual(selection.rust_projects, ["openarm/arm"])
        self.assertEqual(selection.python_tests, [])

    def test_a_change_to_a_python_node_runs_its_tests_alone(self):
        selection = discovery.select(REPOSITORY, ["so101/leader/pyproject.toml"])
        self.assertEqual(selection.rust_projects, [])
        self.assertEqual(selection.python_tests, ["so101/leader/tests/test_params.py"])

    def test_a_test_file_selects_the_project_that_runs_it(self):
        selection = discovery.select(
            REPOSITORY, ["openarm/ai_brain/tests/test_smoke.py"]
        )
        self.assertEqual(
            selection.python_tests, ["openarm/ai_brain/tests/test_smoke.py"]
        )

    def test_several_changed_projects_all_run(self):
        selection = discovery.select(
            REPOSITORY, ["openarm/arm/src/main.rs", "zed_camera/Cargo.toml"]
        )
        self.assertEqual(selection.rust_projects, ["openarm/arm", "zed_camera"])

    def test_a_file_no_project_holds_runs_everything(self):
        selection = discovery.select(REPOSITORY, [".github/scripts/discover-tests.py"])
        self.assertEqual(selection.rust_projects, REPOSITORY.rust_projects)
        self.assertEqual(selection.python_tests, REPOSITORY.python_tests)
        self.assertIn(".github/scripts/discover-tests.py", selection.scope)

    def test_one_escalating_file_carries_the_whole_change(self):
        """A change that touches a project and the workflow runs everything,
        not just the project."""
        selection = discovery.select(
            REPOSITORY, ["openarm/arm/src/main.rs", "scripts/deploy.sh"]
        )
        self.assertEqual(selection.rust_projects, REPOSITORY.rust_projects)

    def test_documentation_outside_every_project_runs_nothing(self):
        selection = discovery.select(REPOSITORY, ["README.md", "docs/Guide.MD"])
        self.assertEqual(selection.rust_projects, [])
        self.assertEqual(selection.python_tests, [])
        self.assertIn("documentation", selection.scope)

    def test_documentation_inside_a_project_runs_that_project(self):
        selection = discovery.select(REPOSITORY, ["openarm/arm/README.md"])
        self.assertEqual(selection.rust_projects, ["openarm/arm"])

    def test_documentation_beside_an_escalating_file_still_escalates(self):
        selection = discovery.select(REPOSITORY, ["README.md", "Makefile"])
        self.assertEqual(selection.rust_projects, REPOSITORY.rust_projects)


class TakeInventory(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def write(self, relative: str, content: str = "") -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def test_it_finds_crates_nodes_and_suites(self):
        self.write("openarm/arm/Cargo.toml")
        self.write("openarm/arm/peppy.json5")
        self.write("openarm/ai_brain/peppy.json5")
        self.write("openarm/ai_brain/tests/test_smoke.py")
        self.write("so101/leader/tests/leader_test.py")
        found = discovery.take_inventory(self.root)
        self.assertEqual(found.rust_projects, ["openarm/arm"])
        self.assertEqual(found.projects, ["openarm/ai_brain", "openarm/arm"])
        self.assertEqual(
            found.python_tests,
            ["openarm/ai_brain/tests/test_smoke.py", "so101/leader/tests/leader_test.py"],
        )

    def test_it_looks_past_build_output_and_the_workflow_directory(self):
        self.write("zed_camera/Cargo.toml")
        self.write("zed_camera/target/debug/Cargo.toml")
        self.write("zed_camera/node_modules/pkg/test_vendored.py")
        self.write(".github/scripts/test_discover_tests.py")
        found = discovery.take_inventory(self.root)
        self.assertEqual(found.rust_projects, ["zed_camera"])
        self.assertEqual(found.python_tests, [])

    def test_a_python_file_that_only_looks_like_a_suite_is_not_one(self):
        self.write("zed_camera/testing_helpers.py")
        self.write("zed_camera/contest_data.py")
        found = discovery.take_inventory(self.root)
        self.assertEqual(found.python_tests, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
