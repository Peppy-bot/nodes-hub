#!/usr/bin/env bash
#
# Prints the path of a container image the given node's tests run in, building
# and caching it under $CI_CACHE_DIR first when it is not there yet.
#
# A node's `apptainer.def` builds the image the node ships in: it prepares
# whatever the node needs to compile, copies the source tree in through
# `%files`, and builds a release binary out of it. A test run brings its own
# sources, bound in from the checkout, so a test image needs the preparation
# half only. `%post` is therefore replayed up to the line that enters the
# directory `%files` copies the source to, and stops there: every line above
# it prepares the image (apt repositories and keyrings, exported build
# variables, system packages) and every line from it on builds a copy of the
# source the test run does not use.
#
# A node with no def is not containerized and so has no image of its own to
# derive from. It gets the base image peppy scaffolds for its language, which
# the caller names.
#
# Images live under $CI_CACHE_DIR keyed by the sha256 of the derived
# definition, so one is rebuilt only when the def it comes from changes.
# Everything but the resulting path goes to stderr, so the caller can read
# stdout as the path alone.
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: ${0##*/} <node directory> <base image for a node with no def>" >&2
  exit 2
fi
node_dir=$1
fallback_base=$2
: "${CI_CACHE_DIR:?must name the directory cached images live in}"
: "${PEPPY_APPTAINER_DIR:?must name the apptainer install to build with}"

definition=$(mktemp)
# The half-built image too, so a failed build leaves nothing behind for the
# next run to trip over.
partial=""
trap 'rm -f "$definition" ${partial:+"$partial"}' EXIT

def="$node_dir/apptainer.def"
if [ -f "$def" ]; then
  base=$(sed -n 's/^From:[[:space:]]*//p' "$def" | head -1)
  if [ -z "$base" ]; then
    echo "$def names no image on a From: line to derive a test image from" >&2
    exit 1
  fi
  # The destination `%files` copies the node to, which is the directory
  # `%post` enters right before it builds.
  source_dir=$(awk '
    /^%files/ { in_files = 1; next }
    /^%[a-z]/ { in_files = 0 }
    in_files && NF == 2 { print $2; exit }
  ' "$def")
  {
    echo "Bootstrap: docker"
    echo "From: $base"
    echo
    echo "%post"
    awk -v source_dir="$source_dir" '
      /^%post/ { in_post = 1; next }
      /^%[a-z]/ { in_post = 0 }
      in_post && source_dir != "" \
        && $0 ~ "^[[:space:]]*cd[[:space:]]+" source_dir "[[:space:]]*$" { exit }
      in_post { print }
    ' "$def"
  } > "$definition"
else
  {
    echo "Bootstrap: docker"
    echo "From: $fallback_base"
  } > "$definition"
fi

images="$CI_CACHE_DIR/test-images"
mkdir -p "$images"
image="$images/$(sha256sum < "$definition" | cut -d ' ' -f 1).sif"
if [ ! -f "$image" ]; then
  partial="$image.building.$$"
  {
    echo "::group::build the test image for $node_dir"
    cat "$definition"
    APPTAINER_CACHEDIR="$CI_CACHE_DIR/apptainer" \
      "$PEPPY_APPTAINER_DIR/bin/apptainer" build "$partial" "$definition"
    echo "::endgroup::"
  } >&2
  # Named only once it is whole, so a concurrent run never execs a partial
  # image out of the cache.
  mv "$partial" "$image"
  partial=""
fi
printf '%s\n' "$image"
