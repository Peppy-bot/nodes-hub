#!/usr/bin/env bash
#
# Regenerates every node's interface code and refreshes the lockfiles that
# resolve against it.
#
# A node builds against a `peppylib` and a `peppygen` generated into its
# gitignored `.peppy/libs` by the peppy release installed on this machine, and
# both are path dependencies, so what they require is an input to resolution.
# A release that changes those requirements leaves every committed lockfile in
# the repository stale at once, whichever node the release touched, which is
# what CI reports when it says the lockfiles no longer resolve against the
# generated interfaces. This is the fix: run it against the release CI
# installs and commit what it rewrites.
#
# It needs a running daemon, because `peppy node sync` reaches the
# node_generate service through it, and this checkout registered as a
# repository (`peppy repo add .`), so a node's dependency on another node here
# resolves to the working tree rather than to the published copy.
set -euo pipefail

for tool in peppy cargo uv; do
  if ! command -v "$tool" > /dev/null; then
    echo "$tool must be on PATH to relock" >&2
    exit 1
  fi
done

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

# `-printf` is GNU find and this runs on the maintainers' machines as much as
# on Linux, so directories come from dirname.
crate_dirs=()
while IFS= read -r dir; do crate_dirs+=("$dir"); done < <(
  find . -name Cargo.toml -not -path '*/target/*' -not -path '*/.peppy/*' \
    -exec dirname {} \; | sed 's|^\./||' | sort
)
uv_dirs=()
while IFS= read -r dir; do uv_dirs+=("$dir"); done < <(
  find . -name uv.lock -not -path '*/.peppy/*' -not -path '*/.venv/*' \
    -exec dirname {} \; | sed 's|^\./||' | sort
)

echo "relocking against $(peppy --version)"

# Interface code is generated per node, so only a project that is a node has
# any to generate: `openarm/sim_isaac/tests` is a uv project under a node and
# depends on nothing generated, so nothing there needs a sync.
for dir in $(printf '%s\n' "${crate_dirs[@]}" "${uv_dirs[@]}" | sort -u); do
  if [ -f "$dir/peppy.json5" ]; then
    echo "syncing $dir"
    peppy node sync "$dir" --include-repositories
  fi
done

# `cargo metadata` resolves and writes the lockfile without compiling
# anything, and rewrites only what the manifests now demand: every version the
# committed lockfile still satisfies stays where it is, so the diff is the
# release's doing and nothing else.
for dir in "${crate_dirs[@]}"; do
  echo "relocking $dir"
  (cd "$dir" && cargo metadata --format-version 1 > /dev/null)
done

for dir in "${uv_dirs[@]}"; do
  echo "relocking $dir"
  (cd "$dir" && uv lock)
done

echo
refreshed=$(git status --porcelain -- \
  ':(glob)**/Cargo.toml' ':(glob)**/Cargo.lock' \
  ':(glob)**/pyproject.toml' ':(glob)**/uv.lock')
if [ -z "$refreshed" ]; then
  echo "already current: nothing to commit"
else
  printf 'refreshed, commit these:\n%s\n' "$refreshed"
fi
