#!/usr/bin/env bash
set -euo pipefail
# Native Node tests can use JavaScript or erasable TypeScript. Match
# test/spec files at any depth, including the repository root.
# The job container and checkout can have different owners. Trust only
# this workspace, for this read-only discovery command.
git -c safe.directory="$GITHUB_WORKSPACE" ls-files -z -- \
  ':(glob)**/*.test.[jt]s' ':(glob)**/*.test.[cm][jt]s' \
  ':(glob)**/*.spec.[jt]s' ':(glob)**/*.spec.[cm][jt]s' \
  > "$RUNNER_TEMP/node-test-files"
tests=()
while IFS= read -r -d '' file; do
  case "/$file" in
    */node_modules/*|*/.peppy/*|*/target/*|*/dist/*|*/build/*|*/.venv/*|*/venv/*) continue ;;
  esac
  tests+=("./$file")
done < "$RUNNER_TEMP/node-test-files"
# Bare node --test performs its own discovery; never invoke it when
# this inventory is empty.
if [ "${#tests[@]}" -eq 0 ]; then
  echo 'No JavaScript or TypeScript Node test suites found.'
  exit 0
fi
printf 'Running %s\n' "${tests[@]}"
node --test "${tests[@]}"
