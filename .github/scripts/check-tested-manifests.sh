#!/usr/bin/env bash

# Proves what was tested is what is committed. The lock half proves the
# builds consumed the committed lockfiles rather than re-resolving them;
# the manifest half catches `peppy node sync` rewriting a Cargo.toml or a
# pyproject.toml, which happens whenever the repository lags the
# installed generator's scaffolding, and the fix is to run the sync
# locally and commit the result. Must stay after the test steps: placed
# before them, it proves nothing.

set -euo pipefail

git diff --exit-code -- \
  ':(glob)**/Cargo.toml' ':(glob)**/Cargo.lock' \
  ':(glob)**/pyproject.toml' ':(glob)**/uv.lock'
