#!/usr/bin/env bash
# Resolve TheRock origin/main tip and the rocm-systems / rocm-libraries
# submodule pointers recorded at that commit. No CI status is consulted.
set -euo pipefail

API="https://api.github.com/repos/ROCm/TheRock"
THEROCK_REMOTE="https://github.com/ROCm/TheRock.git"

if ! command -v jq >/dev/null 2>&1; then
  echo "::error::jq is required to parse TheRock API responses."
  exit 1
fi

curl_github() {
  local url="$1"
  local args=(-fsS -H "Accept: application/vnd.github+json")
  if [[ -n "${GH_API_TOKEN:-}" ]]; then
    args+=(-H "Authorization: Bearer ${GH_API_TOKEN}")
  fi
  curl "${args[@]}" "$url"
}

# The submodule pointer as recorded in TheRock tree at $2, not the tip of the
# component's own branch. That keeps the trio to one TheRock commit.
gitlink_at() {
  local path="$1" ref="$2"
  curl_github "${API}/contents/${path}?ref=${ref}" | jq -r 'select(.type == "submodule") | .sha // empty'
}

SHA="$(git ls-remote "$THEROCK_REMOTE" refs/heads/main | awk '{print $1}')"

if [[ ! "$SHA" =~ ^[0-9a-f]{40}$ ]]; then
  echo "::error::Could not resolve TheRock main tip from ${THEROCK_REMOTE}."
  exit 1
fi

SYSTEMS_SHA="$(gitlink_at rocm-systems "$SHA")"
LIBRARIES_SHA="$(gitlink_at rocm-libraries "$SHA")"

for pair in "rocm-systems:$SYSTEMS_SHA" "rocm-libraries:$LIBRARIES_SHA"; do
  if [[ ! "${pair#*:}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "::error::Could not read the ${pair%%:*} submodule pointer at ${SHA}."
    exit 1
  fi
done

echo "TheRock main tip: $SHA"
echo "  rocm-systems:   $SYSTEMS_SHA"
echo "  rocm-libraries: $LIBRARIES_SHA"

if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  {
    echo "sha=${SHA}"
    echo "systems_sha=${SYSTEMS_SHA}"
    echo "libraries_sha=${LIBRARIES_SHA}"
  } >> "$GITHUB_OUTPUT"
fi
