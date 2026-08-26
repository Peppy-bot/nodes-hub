#!/usr/bin/env bash
# Download robot assets from R2 into /tmp staging dirs.
# Usage: download_assets.sh --variant <isaac|mujoco>
# Called by Dockerfile.isaac and Dockerfile.mujoco during base image builds — not for contributors.
# To rebuild base images: RCLONE_S3_ACCESS_KEY_ID=<key> RCLONE_S3_SECRET_ACCESS_KEY=<secret> bash scripts/build_base_images.sh
set -euo pipefail

VARIANT=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --variant) VARIANT="$2"; shift 2 ;;
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

echo "==> Downloading ${VARIANT} assets..."
rm -rf "/tmp/.peppy_robot_initializer_${VARIANT}"
mkdir -p "/tmp/.peppy_robot_initializer_${VARIANT}"
_rclone copy "r2:${BUCKET}/openarm01/${VARIANT}/assets/" "/tmp/.peppy_robot_initializer_${VARIANT}/" --progress

echo "==> Done."
