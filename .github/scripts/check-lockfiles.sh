#!/usr/bin/env bash

# `cargo test --locked --workspace` reports a stale lockfile by naming
# the lockfile and nothing else, once per crate, so a peppy release that changes what
# the generated `.peppy/libs` require reads as every crate in the
# repository failing for no stated reason: the release moves peppylib's
# own dependencies, and every committed lockfile pins what the release
# before it asked for. This resolves every discovered crate before any of
# them is compiled, and names the release and the fix when one cannot.
#
# Resolution runs in the crate's own image, the one the step below is
# about to build it in, because which versions satisfy a manifest is a
# question about the toolchain resolving it: a def is free to prepare
# one of its own, and a verdict reached under a rustc nothing here runs
# is a verdict about a build nothing here runs. Every image is cached by
# the time that step reaches it, so none is built twice.
#
# uv reports this same drift by naming the lockfile and the command that
# fixes it, so nothing here translates for it, and one
# `scripts/relock.sh` refreshes both languages anyway.

set -euo pipefail
mkdir -p "$CI_CACHE_DIR/cargo-home"
stale=()
reports="$RUNNER_TEMP/stale-lockfiles.txt"
: > "$reports"
# cargo's own account of each refusal is captured rather than logged
# as it happens: read in a row they are the noise this step exists to
# replace, and they belong under the error that explains them.
while IFS= read -r crate_dir <&3; do
  image=$("$GITHUB_WORKSPACE/.github/scripts/derive-test-image.sh" \
    "$crate_dir" peppybot/rust-cargo-base:latest)
  if ! report=$("$PEPPY_APPTAINER_DIR/bin/apptainer" exec --cleanenv --no-home \
      --pwd "/work/$crate_dir" \
      --bind "$GITHUB_WORKSPACE:/work" \
      --bind "$CI_CACHE_DIR/cargo-home:/cargo" \
      --bind "$PEPPY_HOME" \
      --env CARGO_HOME=/cargo \
      --env RUSTUP_HOME=/root/.rustup \
      "$image" cargo metadata --locked --format-version 1 2>&1 > /dev/null); then
    stale+=("$crate_dir")
    printf '%s:\n%s\n\n' "$crate_dir" "$report" >> "$reports"
  fi
done 3< "$RUNNER_TEMP/rust-test-dirs.txt"
if [ "${#stale[@]}" -eq 0 ]; then
  exit 0
fi
echo "::group::what cargo reported"
cat "$reports"
echo "::endgroup::"
{
  echo
  echo "### Lockfiles do not resolve against the generated interfaces"
  echo
  echo "\`$PEPPY_INSTALLED\` generates a \`.peppy/libs\` these lockfiles cannot resolve against:"
  printf -- '- %s\n' "${stale[@]}"
  echo
  echo "Run \`scripts/relock.sh\` against that release and commit what it rewrites."
} >> "$GITHUB_STEP_SUMMARY"
echo "::error::${#stale[@]} committed lockfile(s) do not resolve against $PEPPY_INSTALLED's generated interfaces: run scripts/relock.sh and commit what it rewrites"
exit 1
