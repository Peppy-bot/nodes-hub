#!/usr/bin/env bash

# The bundled defaults register the GitHub copy of this repository. With
# it present, one node's dependency on another node changed by this pull
# request could resolve to the stale main-branch copy instead of the
# checkout, so it is removed and the checkout registered in its place.
# The other hub repositories stay: the contracts these nodes implement
# genuinely live there.
set -euo pipefail
repo_id=$(grep -B 3 'Peppy-bot/nodes-hub' "$PEPPY_HOME/conf/repositories.json5" \
  | sed -n 's/.*id: *\([0-9]*\),.*/\1/p' | head -1)
if [ -n "$repo_id" ]; then
  peppy repo remove "$repo_id"
fi
peppy repo add "$GITHUB_WORKSPACE"
peppy repo refresh

# Only nodes with tests need syncing; the generated code is
# gitignored and per-node. --include-repositories lets a node's
# dependency on a contract or another node resolve through the
# registered repositories; nothing is in the node stack here.
while IFS= read -r node_dir; do
  echo "::group::peppy node sync $node_dir"
  peppy node sync "$node_dir" --include-repositories
  echo "::endgroup::"
done < "$RUNNER_TEMP/sync-dirs.txt"
