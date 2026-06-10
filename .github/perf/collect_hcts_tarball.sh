#!/usr/bin/env bash
# Deprecated: use collect_component_hcts.sh. Kept as a thin wrapper for older callers.
set -euo pipefail

BUILD_DIR="${1:?build directory}"
STAGE_NAME="${2:?stage name}"
FAMILY="${3:-generic}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/collect_component_hcts.sh" "${BUILD_DIR}" "${BUILD_DIR}/logs" "${FAMILY}"
