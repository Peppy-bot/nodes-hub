#!/usr/bin/env bash

# pytest runs in the same container the node ships in, for the same
# reason cargo does: xr_commander imports an opencv that links a libGL
# only its def installs, and lerobot_recorder's dataset writer shells out
# to an ffmpeg only its own def installs. A node shipping no def gets the
# base image peppy scaffolds for a Python node.
#
# uv's download cache and its managed interpreters are on the sticky
# disk, so a warm run installs a torch it already has instead of
# fetching lerobot's tree again. The environments built out of them are
# per run, because uv locks its cache but not a project environment, and
# two runs sharing one would corrupt it. --locked proves the environment
# came from the committed uv.lock rather than a fresh resolution. pytest
# is supplied here rather than required of each node, so a node that adds
# its first test file is run by CI whether or not it also declared a test
# dependency group.
#
# The peppy distribution is bound in and its zenohd named outright, the
# same way the Rust step names it: a suite that boots its node through
# the generated harness starts an ephemeral router of its own, and
# nothing in the container can find the binary to start it with.
# --cleanenv keeps the runner's PATH out, so the `zenohd` beside the
# `peppy` on it — the fallback the resolver would otherwise reach — is
# not visible here, and a node's own image has no reason to ship one.
#
# Every project runs even after one fails, so a red run names
# every failing node instead of the first.

set -euo pipefail
failed=()
# The list is read on fd 3, so a test that reads stdin cannot eat the
# projects queued behind it.
while IFS= read -r project <&3; do
  image=$("$GITHUB_WORKSPACE/.github/scripts/derive-test-image.sh" \
    "$project" peppybot/python-uv-base:latest)
  mkdir -p "$CI_CACHE_DIR/uv/cache" "$CI_CACHE_DIR/uv/python" \
    "$CI_CACHE_DIR/uv/home" "$RUNNER_TEMP/uv-environments"
  mapfile -t tests < <(awk -F '\t' -v project="$project" \
    '$1 == project { print $2 }' \
    "$RUNNER_TEMP/python-test-project-files.tsv")
  environment="/environments/$(printf '%s' "$project" | sha256sum | cut -d ' ' -f 1)"
  echo "::group::pytest: $project"
  if "$PEPPY_APPTAINER_DIR/bin/apptainer" exec --cleanenv --no-home \
      --pwd "/work/$project" \
      --bind "$GITHUB_WORKSPACE:/work" \
      --bind "$CI_CACHE_DIR/uv:/uv" \
      --bind "$RUNNER_TEMP/uv-environments:/environments" \
      --bind "$RUNNER_TEMP/peppy-dist:/peppy-dist" \
      --env HOME=/uv/home \
      --env UV_CACHE_DIR=/uv/cache \
      --env UV_PYTHON_INSTALL_DIR=/uv/python \
      --env UV_PROJECT_ENVIRONMENT="$environment" \
      --env PEPPY_ZENOHD_PATH=/peppy-dist/bin/zenohd \
      "$image" uv run --locked --with pytest pytest "${tests[@]}"; then
    echo "::endgroup::"
  else
    echo "::endgroup::"
    echo "::error::pytest failed in $project"
    failed+=("$project")
  fi
done 3< "$RUNNER_TEMP/python-test-projects.txt"
if [ "${#failed[@]}" -gt 0 ]; then
  echo "failing projects: ${failed[*]}" >&2
  exit 1
fi
