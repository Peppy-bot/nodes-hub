#!/usr/bin/env bash

# The daemon gets its own data root and port under RUNNER_TEMP: the CLI
# finds it through PEPPY_HOME/daemon_state.json5, so pointing PEPPY_HOME
# at the run's directory is discovery and isolation from any other
# daemon in one. The name carries the run id because core-node names
# must be unique among daemons that can reach each other.
set -euo pipefail
PEPPY_HOME="$RUNNER_TEMP/peppy-home"
mkdir -p "$PEPPY_HOME"
PEPPY_MESSAGING_PORT=$(python3 "$GITHUB_WORKSPACE/.github/scripts/find-free-port.py")
{
  echo "PEPPY_HOME=$PEPPY_HOME"
  echo "PEPPY_MESSAGING_PORT=$PEPPY_MESSAGING_PORT"
} >> "$GITHUB_ENV"
export PEPPY_HOME PEPPY_MESSAGING_PORT

nohup peppy service serve --core-node-name "ci-tests-$GITHUB_RUN_ID" \
  > "$RUNNER_TEMP/peppy-serve.log" 2>&1 &
for _ in $(seq 1 30); do
  if [ -f "$PEPPY_HOME/daemon_state.json5" ]; then exit 0; fi
  sleep 1
done
echo "daemon did not come up; serve log:" >&2
cat "$RUNNER_TEMP/peppy-serve.log" >&2
exit 1
