#!/usr/bin/env bash

# The path the disk is mounted at: the later steps and
# derive-test-image.sh reach the caches through this variable.

set -euo pipefail

echo "CI_CACHE_DIR=$HOME/.cache/nodes-hub-ci" >> "$GITHUB_ENV"
