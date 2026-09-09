#!/usr/bin/env bash

# The daemon refuses to boot unless the apptainer install it resolves
# is ready for unprivileged use: `newuidmap` on PATH, and on a kernel
# that restricts unprivileged user namespaces through AppArmor (Ubuntu
# 24.04's default) a root-installed profile keyed to the canonical path
# of that install's `starter` binary. The VM is fresh, so every run
# readies the install it just extracted: `peppy container setup` applies
# whatever its fix script needs through sudo when stdin is not a
# terminal, and rechecks before exiting 0. Today's Blacksmith image
# mounts no AppArmor securityfs, so it needs nothing; the step is what
# keeps that an observation rather than an assumption. The daemon would
# find bin/apptainer beside bin/peppy by itself; it is exported because
# the test steps and derive-test-image.sh run the same install directly.
#
# `apptainer build` runs a def's %post as fakeroot, which needs a
# subordinate id range for the runner user besides newuidmap. The
# runner image ships one; should it ever not, this allocates one past
# the highest range in use rather than failing in a def's apt-get a
# cold build later.
set -euo pipefail
PEPPY_APPTAINER_DIR="$RUNNER_TEMP/peppy-dist/bin/apptainer"
echo "PEPPY_APPTAINER_DIR=$PEPPY_APPTAINER_DIR" >> "$GITHUB_ENV"
export PEPPY_APPTAINER_DIR
peppy container setup < /dev/null

user=$(id -un)
for kind in subuid subgid; do
  if ! grep -qs "^$user:" "/etc/$kind"; then
    start=$(awk -F: 'BEGIN { m = 100000 } { e = $2 + $3; if (e > m) m = e } END { print m }' \
      "/etc/$kind" 2>/dev/null || echo 100000)
    sudo usermod "--add-${kind}s" "$start-$((start + 65535))" "$user"
  fi
done
