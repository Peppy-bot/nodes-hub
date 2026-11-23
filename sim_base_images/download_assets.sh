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

OUTPUT_DIR="${OUTPUT_DIR:-/tmp/.peppy_isaac_bundles}"
echo "==> Downloading the Isaac Sim bundles..."
# One pinned bundle per robot, extracted into a directory of the robot's
# name under the output: what the image bakes as /opt/robot_assets/<robot>/,
# where the node's model entries look their stages up. Every pin is
# checked, every archive downloaded and verified, and every stage checked
# before the output is touched, so a failure leaves nothing half staged.
ROBOTS=(openarm so101 botshare_base)
# The prefix of each robot's pins in isaac_assets.env, <prefix>_KEY and
# <prefix>_SHA256.
PINS=(ISAAC_ASSETS SO101_ASSETS BOTSHARE_BASE_ASSETS)
KEYS=(
    "${ISAAC_ASSETS_KEY:?ISAAC_ASSETS_KEY must be set}"
    "${SO101_ASSETS_KEY:?SO101_ASSETS_KEY must be set}"
    "${BOTSHARE_BASE_ASSETS_KEY:?BOTSHARE_BASE_ASSETS_KEY must be set}"
)
CHECKSUMS=(
    "${ISAAC_ASSETS_SHA256:?ISAAC_ASSETS_SHA256 must be set}"
    "${SO101_ASSETS_SHA256:?SO101_ASSETS_SHA256 must be set}"
    "${BOTSHARE_BASE_ASSETS_SHA256:?BOTSHARE_BASE_ASSETS_SHA256 must be set}"
)
# The SO-101 and botshare base bundles are pinned a second time, by the
# checksum of the stage each holds as <robot>.usd, which
# sim_isaac/tests/test_<robot>_stage.py holds to the stage the tree builds,
# each with the prefix of its bundle's pins.
STAGED_ROBOTS=(so101 botshare_base)
STAGED_PINS=(SO101 BOTSHARE_BASE)
STAGE_CHECKSUMS=(
    "${SO101_STAGE_SHA256:?SO101_STAGE_SHA256 must be set}"
    "${BOTSHARE_BASE_STAGE_SHA256:?BOTSHARE_BASE_STAGE_SHA256 must be set}"
)
for i in "${!ROBOTS[@]}"; do
    if [[ "${KEYS[$i]}" != *.tar.gz || ! "${CHECKSUMS[$i]}" =~ ^[0-9a-f]{64}$ ]]; then
        echo "ERROR: the ${ROBOTS[$i]} bundle needs a pinned .tar.gz key and SHA-256 checksum: set ${PINS[$i]}_KEY to its <robot>/<version>/<sha256>.tar.gz and ${PINS[$i]}_SHA256 to its SHA-256 in isaac_assets.env" >&2
        exit 1
    fi
done
for i in "${!STAGED_ROBOTS[@]}"; do
    if [[ ! "${STAGE_CHECKSUMS[$i]}" =~ ^[0-9a-f]{64}$ ]]; then
        echo "ERROR: the ${STAGED_ROBOTS[$i]} bundle needs the pinned SHA-256 checksum of its stage: set ${STAGED_PINS[$i]}_STAGE_SHA256 to the SHA-256 of its ${STAGED_ROBOTS[$i]}.usd in isaac_assets.env" >&2
        exit 1
    fi
done
SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT
for i in "${!ROBOTS[@]}"; do
    ARCHIVE="${SCRATCH}/${ROBOTS[$i]}.tar.gz"
    _rclone copyto "r2:${BUCKET}/${KEYS[$i]}" "$ARCHIVE" --progress
    printf '%s  %s\n' "${CHECKSUMS[$i]}" "$ARCHIVE" | sha256sum -c -
done
for i in "${!STAGED_ROBOTS[@]}"; do
    ROBOT="${STAGED_ROBOTS[$i]}"
    STAGE="$(tar -xzOf "${SCRATCH}/${ROBOT}.tar.gz" "${ROBOT}.usd" | sha256sum | cut -d' ' -f1)"
    if [[ "$STAGE" != "${STAGE_CHECKSUMS[$i]}" ]]; then
        PIN="${STAGED_PINS[$i]}"
        echo "ERROR: the ${ROBOT} bundle holds the stage ${STAGE}, and the pinned stage is ${STAGE_CHECKSUMS[$i]}: pin ${PIN}_ASSETS_KEY, ${PIN}_ASSETS_SHA256 and ${PIN}_STAGE_SHA256 in isaac_assets.env from one build of the bundle" >&2
        exit 1
    fi
done

rm -rf "$OUTPUT_DIR"
for i in "${!ROBOTS[@]}"; do
    mkdir -p "${OUTPUT_DIR}/${ROBOTS[$i]}"
    tar -xzf "${SCRATCH}/${ROBOTS[$i]}.tar.gz" -C "${OUTPUT_DIR}/${ROBOTS[$i]}"
done

echo "==> Done."
