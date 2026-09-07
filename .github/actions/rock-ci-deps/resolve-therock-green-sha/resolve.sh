#!/usr/bin/env bash
# Resolve TheRock origin/main tip and the rocm-systems / rocm-libraries
# submodule pointers recorded at that commit. No CI status is consulted.
set -euo pipefail

API="https://api.github.com/repos/ROCm/TheRock"
THEROCK_REMOTE="https://github.com/ROCm/TheRock.git"

# The tip comes over the git protocol rather than REST: it needs no token and
# is not charged against the 60 requests/hour anonymous REST budget that the
# shared runner IPs burn through.
SHA="$(git ls-remote "$THEROCK_REMOTE" refs/heads/main | awk '{print $1}')"

if [[ ! "$SHA" =~ ^[0-9a-f]{40}$ ]]; then
  echo "::error::Could not resolve TheRock main tip from ${THEROCK_REMOTE}."
  exit 1
fi

# Submodule pointers are reported for traceability only; checking TheRock out
# at $SHA already brings the matching gitlinks. A token that cannot see
# ROCm/TheRock (404) or an exhausted rate limit must not fail the promotion.
SYSTEMS_SHA=""
LIBRARIES_SHA=""

if [[ -n "${GH_API_TOKEN:-}" ]] && command -v jq >/dev/null 2>&1; then
  gitlink_at() {
    local path="$1" body
    body="$(curl -fsS \
      -H "Accept: application/vnd.github+json" \
      -H "Authorization: Bearer ${GH_API_TOKEN}" \
      -H "X-GitHub-Api-Version: 2022-11-28" \
      "${API}/contents/${path}?ref=${SHA}" 2>/dev/null)" || return 0
    printf '%s' "$body" | jq -r 'select(.type == "submodule") | .sha // empty'
  }

  SYSTEMS_SHA="$(gitlink_at rocm-systems)"
  LIBRARIES_SHA="$(gitlink_at rocm-libraries)"

  for pair in "rocm-systems:$SYSTEMS_SHA" "rocm-libraries:$LIBRARIES_SHA"; do
    if [[ ! "${pair#*:}" =~ ^[0-9a-f]{40}$ ]]; then
      echo "::warning::Could not read the ${pair%%:*} pointer at ${SHA}; reporting it as unknown."
    fi
  done
fi

echo "TheRock main tip: $SHA"
echo "  rocm-systems:   ${SYSTEMS_SHA:-<unknown>}"
echo "  rocm-libraries: ${LIBRARIES_SHA:-<unknown>}"

if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  {
    echo "sha=${SHA}"
    echo "systems_sha=${SYSTEMS_SHA}"
    echo "libraries_sha=${LIBRARIES_SHA}"
  } >> "$GITHUB_OUTPUT"
fi

