#!/usr/bin/env bash
# Collect .hcts / *.device.hcts files under BUILD_DIR into a tarball.
set -euo pipefail

BUILD_DIR="${1:?build directory}"
STAGE_NAME="${2:?stage name}"
FAMILY="${3:-generic}"

mkdir -p "${BUILD_DIR}/logs"
hcts_tar="${BUILD_DIR}/logs/hcts-${STAGE_NAME}-${FAMILY}.tar.gz"

# Avoid `find | grep -q` under pipefail: grep -q closes early and find gets SIGPIPE,
# which makes the pipeline look failed even when matches exist (flatbuffers .device.hcts).
mapfile -d '' hcts_files < <(find "${BUILD_DIR}" \( -name '*.hcts' -o -name '*.device.hcts' \) -print0 2>/dev/null || true)

if ((${#hcts_files[@]} == 0)); then
  echo "No .hcts files found under ${BUILD_DIR}" >&2
  echo "Hint: verify perf_launcher wraps HIP compiles (rocPRIM toolchain) and perf stat works." >&2
  exit 1
fi

printf '%s\0' "${hcts_files[@]}" | tar -czf "${hcts_tar}" --null -T -
ls -lh "${hcts_tar}"
echo "HCTS file count: ${#hcts_files[@]}"
find "${BUILD_DIR}" \( -name '*.hcts' -o -name '*.device.hcts' \) | head -10
