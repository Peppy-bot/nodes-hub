#!/usr/bin/env bash

# Installed by extracting the release under RUNNER_TEMP rather than
# through scripts/install.sh: nothing here wants a ~/.peppy, and the
# whole job lives in the checkout and its own temp dir. Unlike the
# repository-index workflow this extracts the whole archive, not just
# bin/peppy, because `service serve` needs the bundled layout around the
# binary (zenohd, and the apptainer install under bin/apptainer that the
# next step readies). Set the PEPPY_VERSION repository variable to pin a
# release; it defaults to the latest. The generator inside peppy must
# match what the crates' committed Cargo.tomls expect of peppygen, so
# bump the pin together with peppy releases that change generated code.
#
# PEPPY_VERSION reaches the script through the environment rather than a
# `${{ }}` expansion, so its value is data the shell reads instead of
# source the shell parses, and the shape check turns a typo into a named
# failure rather than a 404 from a URL built out of it.
set -euo pipefail
version="${PEPPY_VERSION:-latest}"
if [[ "$version" == "latest" ]]; then
  channel="latest"
elif [[ "$version" =~ ^v?[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  channel="v${version#v}"
else
  echo "PEPPY_VERSION must be 'latest' or a release such as 0.25.0, got: $version" >&2
  exit 1
fi

case "$(uname -m)" in
  x86_64 | amd64) arch="x86_64" ;;
  aarch64 | arm64) arch="aarch64" ;;
  *) echo "unsupported runner architecture: $(uname -m)" >&2; exit 1 ;;
esac

dest="$RUNNER_TEMP/peppy-dist"
mkdir -p "$dest"
curl -fsSL --connect-timeout 10 --max-time 300 \
  "https://peppy.bot/$channel/peppy-$arch-unknown-linux-gnu.tgz" \
  | tar -xzf - -C "$dest"
chmod +x "$dest/bin/peppy"
echo "$dest/bin" >> "$GITHUB_PATH"

# The binary answers `--version` before building any app context,
# so no PEPPY_HOME is needed here. `latest` resolves to a different
# release over time, so the log and the run summary both record
# which release this run actually executed, and the lockfile check
# below names it as the cause when the interfaces it generates have
# moved out from under the committed lockfiles.
installed=$("$dest/bin/peppy" --version)
echo "PEPPY_INSTALLED=$installed" >> "$GITHUB_ENV"
printf '\nInstalled %s (channel: %s)\n' "$installed" "$channel" \
  | tee -a "$GITHUB_STEP_SUMMARY"
