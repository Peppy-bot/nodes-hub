#!/usr/bin/env bash

# Each crate's tests run inside the container image its node's
# apptainer.def declares, so the runner host needs none of the crates'
# system build dependencies: realsense_d4xx builds against a
# librealsense2-dev that only its own def knows the apt repository for.
# A crate whose node ships no def, like the example_robot ones, gets the
# base image peppy scaffolds for a Rust node. Build and exec both run
# unprivileged through the same user-namespace apptainer install readied
# above.
#
# Inside the container the workspace is /work and the daemon's data root
# is bound at its host path, because each node's .peppy/libs entries are
# absolute symlinks into it. The base image's rustup lives under /root
# (its docker build ran as root) with no env set, hence the explicit
# RUSTUP_HOME; CARGO_HOME and the per-crate target dirs are on the sticky
# disk so registries and builds warm across runs. The target dirs
# are per crate, never shared: every crate vendors a generated
# `peppygen 0.1.0` with different contents, and in a shared target dir
# cargo reuses one crate's peppygen artifact for another. Incremental
# compilation is off: its caches are much of a debug target dir and the
# sticky disk is billed per gigabyte, while all it would save is the
# seconds a warm rebuild of a crate's own code takes.
#
# Every crate runs even after one fails, so a red run names
# every failing crate instead of the first. --locked proves the build
# consumed each committed lockfile rather than re-resolving it; the diff
# check at the end proves nothing rewrote them.

set -euo pipefail
failed=()
# The lists are read on fd 3, so a test that reads stdin cannot eat
# the crates queued behind it.
while IFS= read -r crate_dir <&3; do
  image=$("$GITHUB_WORKSPACE/.github/scripts/derive-test-image.sh" \
    "$crate_dir" peppybot/rust-cargo-base:latest)
  target_dir="$CI_CACHE_DIR/target/$crate_dir"
  if [ "$crate_dir" = "." ]; then
    target_dir="$CI_CACHE_DIR/target-root"
  fi
  mkdir -p "$CI_CACHE_DIR/cargo-home" "$target_dir"
  echo "::group::cargo test: $crate_dir"
  if "$PEPPY_APPTAINER_DIR/bin/apptainer" exec --cleanenv --no-home \
      --pwd "/work/$crate_dir" \
      --bind "$GITHUB_WORKSPACE:/work" \
      --bind "$CI_CACHE_DIR/cargo-home:/cargo" \
      --bind "$target_dir:/target" \
      --bind "$RUNNER_TEMP/peppy-dist:/peppy-dist" \
      --bind "$PEPPY_HOME" \
      --env CARGO_HOME=/cargo \
      --env RUSTUP_HOME=/root/.rustup \
      --env CARGO_TARGET_DIR=/target \
      --env CARGO_INCREMENTAL=0 \
      --env PEPPY_ZENOHD_PATH=/peppy-dist/bin/zenohd \
      --env CARGO_TERM_COLOR=always \
      "$image" cargo test --locked --workspace; then
    echo "::endgroup::"
  else
    echo "::endgroup::"
    echo "::error::cargo test failed in $crate_dir"
    failed+=("$crate_dir")
  fi
done 3< "$RUNNER_TEMP/rust-test-dirs.txt"
if [ "${#failed[@]}" -gt 0 ]; then
  echo "failing crates: ${failed[*]}" >&2
  exit 1
fi
