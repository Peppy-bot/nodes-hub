#!/usr/bin/env bash
# Download each robot's Isaac Sim bundle from R2 into a staging directory: one
# pinned bundle per robot, from the isaac-sim-assets bucket, read with
# WALDO_R2_ACCESS_KEY_ID and WALDO_R2_SECRET_ACCESS_KEY.
# Usage: download_assets.sh [--output-dir <path>]
# Called by Dockerfile.isaac during the base image build.
# To rebuild base images: bash sim_base_images/build_base_images.sh
set -euo pipefail

OUTPUT_DIR=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        *) echo "ERROR: unknown argument: $1" >&2; exit 1 ;;
    esac
done

KEY_ID="${WALDO_R2_ACCESS_KEY_ID:?WALDO_R2_ACCESS_KEY_ID must be set}"
SECRET="${WALDO_R2_SECRET_ACCESS_KEY:?WALDO_R2_SECRET_ACCESS_KEY must be set}"
source "$(dirname "${BASH_SOURCE[0]}")/isaac_assets.env"
BUCKET="${ISAAC_ASSETS_BUCKET:?ISAAC_ASSETS_BUCKET must be set}"
ENDPOINT="${ISAAC_ASSETS_ENDPOINT:?ISAAC_ASSETS_ENDPOINT must be set}"

if ! command -v rclone &>/dev/null; then
    echo "ERROR: rclone not found. Install with: sudo apt-get install rclone" >&2
    exit 1
fi

_rclone() {
    RCLONE_CONFIG_R2_TYPE=s3 \
    RCLONE_CONFIG_R2_PROVIDER=Cloudflare \
    RCLONE_CONFIG_R2_ACCESS_KEY_ID="${KEY_ID}" \
    RCLONE_CONFIG_R2_SECRET_ACCESS_KEY="${SECRET}" \
    RCLONE_CONFIG_R2_ENDPOINT="${ENDPOINT}" \
    rclone "$@"
}

OUTPUT_DIR="${OUTPUT_DIR:-/tmp/.peppy_openarm_assets_isaac}"
echo "==> Downloading the Isaac Sim bundles..."
# One pinned bundle per robot, extracted into a directory of the robot's
# name under the output: what the image bakes as /opt/robot_assets/<robot>/,
# where the node's model entries look their stages up. Every pin is
# checked and every archive downloaded and verified before the output is
# touched, so a failure leaves nothing half staged. The SO-101 bundle is
# pinned a second time, by the checksum of the stage it holds, which
# sim_isaac/tests/test_so101_stage.py holds to the stage the tree builds.
ROBOTS=(openarm so101)
KEYS=(
    "${ISAAC_ASSETS_KEY:?ISAAC_ASSETS_KEY must be set}"
    "${SO101_ASSETS_KEY:?SO101_ASSETS_KEY must be set}"
)
CHECKSUMS=(
    "${ISAAC_ASSETS_SHA256:?ISAAC_ASSETS_SHA256 must be set}"
    "${SO101_ASSETS_SHA256:?SO101_ASSETS_SHA256 must be set}"
)
for i in "${!ROBOTS[@]}"; do
    if [[ "${KEYS[$i]}" != *.tar.gz || ! "${CHECKSUMS[$i]}" =~ ^[0-9a-f]{64}$ ]]; then
        echo "ERROR: the ${ROBOTS[$i]} bundle needs a pinned .tar.gz key and SHA-256 checksum" >&2
        exit 1
    fi
done
SO101_STAGE_CHECKSUM="${SO101_STAGE_SHA256:?SO101_STAGE_SHA256 must be set}"
if [[ ! "$SO101_STAGE_CHECKSUM" =~ ^[0-9a-f]{64}$ ]]; then
    echo "ERROR: the so101 bundle needs the pinned SHA-256 checksum of its stage" >&2
    exit 1
fi
SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT
for i in "${!ROBOTS[@]}"; do
    ARCHIVE="${SCRATCH}/${ROBOTS[$i]}.tar.gz"
    _rclone copyto "r2:${BUCKET}/${KEYS[$i]}" "$ARCHIVE" --progress
    printf '%s  %s\n' "${CHECKSUMS[$i]}" "$ARCHIVE" | sha256sum -c -
done
SO101_STAGE="$(tar -xzOf "${SCRATCH}/so101.tar.gz" so101.usd | sha256sum | cut -d' ' -f1)"
if [[ "$SO101_STAGE" != "$SO101_STAGE_CHECKSUM" ]]; then
    echo "ERROR: the so101 bundle holds the stage ${SO101_STAGE}, and the pinned stage is ${SO101_STAGE_CHECKSUM}" >&2
    exit 1
fi

rm -rf "$OUTPUT_DIR"
for i in "${!ROBOTS[@]}"; do
    mkdir -p "${OUTPUT_DIR}/${ROBOTS[$i]}"
    tar -xzf "${SCRATCH}/${ROBOTS[$i]}.tar.gz" -C "${OUTPUT_DIR}/${ROBOTS[$i]}"
done

echo "==> Done."
