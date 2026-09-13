#!/usr/bin/env bash
# Download robot assets from R2 into staging directories.
# Usage: download_assets.sh --variant <isaac|mujoco> [--output-dir <path>]
# Called by Dockerfile.isaac and Dockerfile.mujoco during base image builds.
# To rebuild base images: RCLONE_S3_ACCESS_KEY_ID=<key> RCLONE_S3_SECRET_ACCESS_KEY=<secret> bash scripts/build_base_images.sh
set -euo pipefail

VARIANT=""
OUTPUT_DIR=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --variant) VARIANT="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        *) echo "ERROR: unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ "$VARIANT" != "isaac" && "$VARIANT" != "mujoco" ]]; then
    echo "ERROR: --variant must be 'isaac' or 'mujoco'" >&2
    exit 1
fi

KEY_ID="${RCLONE_S3_ACCESS_KEY_ID:?RCLONE_S3_ACCESS_KEY_ID must be set}"
SECRET="${RCLONE_S3_SECRET_ACCESS_KEY:?RCLONE_S3_SECRET_ACCESS_KEY must be set}"
ENDPOINT="https://b9abcee11c090aef5279f874ff078826.r2.cloudflarestorage.com"
BUCKET="peppy-data01"

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

OUTPUT_DIR="${OUTPUT_DIR:-/tmp/.peppy_robot_initializer_${VARIANT}}"
echo "==> Downloading ${VARIANT} assets..."
if [[ "$VARIANT" == "isaac" ]]; then
    source "$(dirname "${BASH_SOURCE[0]}")/isaac_assets.env"
    BUCKET="${ISAAC_ASSETS_BUCKET:?ISAAC_ASSETS_BUCKET must be set}"
    ENDPOINT="${ISAAC_ASSETS_ENDPOINT:?ISAAC_ASSETS_ENDPOINT must be set}"
    if [[ "${ISAAC_ASSETS_KEY:?ISAAC_ASSETS_KEY must be set}" != *.tar.gz \
        || ! "${ISAAC_ASSETS_SHA256:?ISAAC_ASSETS_SHA256 must be set}" =~ ^[0-9a-f]{64}$ ]]; then
        echo "ERROR: Isaac assets require a pinned .tar.gz key and SHA-256 checksum" >&2
        exit 1
    fi

    SCRATCH="$(mktemp -d)"
    trap 'rm -rf "$SCRATCH"' EXIT
    ARCHIVE="${SCRATCH}/isaac-assets.tar.gz"
    _rclone copyto "r2:${BUCKET}/${ISAAC_ASSETS_KEY}" "$ARCHIVE" --progress
    printf '%s  %s\n' "$ISAAC_ASSETS_SHA256" "$ARCHIVE" | sha256sum -c -

    rm -rf "$OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
    tar -xzf "$ARCHIVE" -C "$OUTPUT_DIR"
else
    rm -rf "$OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
    _rclone copy "r2:${BUCKET}/openarm01/${VARIANT}/assets/" "${OUTPUT_DIR}/" --progress
fi

echo "==> Done."
