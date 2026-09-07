#!/usr/bin/env bash
# Resolve TheRock origin/main tip and the rocm-systems / rocm-libraries
# submodule pointers recorded at that commit. No CI status is consulted.
#
# All GitHub calls must be authenticated. Anonymous REST from GitHub-hosted
# runner IPs is limited to 60 requests/hour and fails with
# "API rate limit exceeded for <ip>".
set -euo pipefail

API="https://api.github.com/repos/ROCm/TheRock"

if [[ -z "${GH_API_TOKEN:-}" ]]; then
  echo "::error::GH_API_TOKEN is required; unauthenticated GitHub API calls will rate-limit."
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "::error::jq is required to parse TheRock API responses."
  exit 1
fi

curl_github() {
  local url="$1"
  curl -fsS \
    -H "Accept: application/vnd.github+json" \
    -H "Authorization: Bearer ${GH_API_TOKEN}" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "$url"
}

SHA="$(curl_github "${API}/git/ref/heads/main" | jq -r '.object.sha // empty')"

if [[ ! "$SHA" =~ ^[0-9a-f]{40}$ ]]; then
  echo "::error::Could not resolve TheRock main tip from ${API}."
  exit 1
fi

# The submodule pointer as recorded in TheRock tree at $SHA, not the tip of the
# component's own branch. That keeps the trio to one TheRock commit.
gitlink_at() {
  local path="$1"
  curl_github "${API}/contents/${path}?ref=${SHA}" | jq -r 'select(.type == "submodule") | .sha // empty'
}

SYSTEMS_SHA="$(gitlink_at rocm-systems)"
LIBRARIES_SHA="$(gitlink_at rocm-libraries)"

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
