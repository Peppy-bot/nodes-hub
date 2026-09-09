#!/usr/bin/env bash

# Cargo discovers unit, integration and documentation tests for every
# workspace member. Python tests follow pytest's file naming conventions
# and run in the nearest pyproject.toml's environment. Generated code,
# dependencies, virtual environments and build outputs are not ours to test.

set -euo pipefail
find . -type d \
  \( -name .git -o -name .peppy -o -name target -o -name node_modules \
     -o -name .venv -o -name venv -o -name dist -o -name build \
     -o -name __pycache__ \) -prune -o \
  -type f -name Cargo.toml -printf '%h\n' \
  | sed 's|^\./||' | sort -u \
  > "$RUNNER_TEMP/rust-test-dirs.txt"
find . -type d \
  \( -name .git -o -name .peppy -o -name target -o -name node_modules \
     -o -name .venv -o -name venv -o -name dist -o -name build \
     -o -name __pycache__ \) -prune -o \
  -type f \( -name 'test_*.py' -o -name '*_test.py' \) -print \
  | sed 's|^\./||' | sort \
  > "$RUNNER_TEMP/python-test-files.txt"

# Keep the exact owner with each file, so a nested Python project
# never also runs in its parent's environment. A root pyproject.toml
# can own tests too; a test without any project is a CI error.
: > "$RUNNER_TEMP/python-test-projects.txt"
: > "$RUNNER_TEMP/python-test-project-files.tsv"
while IFS= read -r test_file; do
  project=$(dirname "$test_file")
  while [ ! -f "$project/pyproject.toml" ]; do
    if [ "$project" = "." ]; then
      echo "::error::$test_file has no pyproject.toml above it to run pytest from"
      exit 1
    fi
    project=$(dirname "$project")
  done
  echo "$project" >> "$RUNNER_TEMP/python-test-projects.txt"
  relative_file=$test_file
  if [ "$project" != "." ]; then
    relative_file=${test_file#"$project/"}
  fi
  printf '%s\t%s\n' "$project" "$relative_file" \
    >> "$RUNNER_TEMP/python-test-project-files.tsv"
done < "$RUNNER_TEMP/python-test-files.txt"
sort -u -o "$RUNNER_TEMP/python-test-projects.txt" "$RUNNER_TEMP/python-test-projects.txt"

# Only a project that is a node and actually has tests needs
# its interface code generated. A tested node's dependency on another
# node resolves through the daemon's repository registry, which reads
# that node's manifest straight from the checkout, so the dependency
# itself needs no sync.
: > "$RUNNER_TEMP/sync-dirs.txt"
while IFS= read -r project; do
  if [ -f "$project/peppy.json5" ]; then
    echo "$project" >> "$RUNNER_TEMP/sync-dirs.txt"
  fi
done < <(sort -u "$RUNNER_TEMP/rust-test-dirs.txt" "$RUNNER_TEMP/python-test-projects.txt")

any=false
if [ -s "$RUNNER_TEMP/rust-test-dirs.txt" ] || [ -s "$RUNNER_TEMP/python-test-files.txt" ]; then
  any=true
fi
echo "any=$any" >> "$GITHUB_OUTPUT"

{
  echo "Running all discovered Rust and Python tests."
  echo
  echo "### Rust test crates"
  if [ -s "$RUNNER_TEMP/rust-test-dirs.txt" ]; then
    sed 's/^/- /' "$RUNNER_TEMP/rust-test-dirs.txt"
  else
    echo "_none found_"
  fi
  echo
  echo "### Python test files"
  if [ -s "$RUNNER_TEMP/python-test-files.txt" ]; then
    sed 's/^/- /' "$RUNNER_TEMP/python-test-files.txt"
  else
    echo "_none found_"
  fi
} >> "$GITHUB_STEP_SUMMARY"
