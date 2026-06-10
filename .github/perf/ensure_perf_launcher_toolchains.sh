#!/usr/bin/env bash
# Ensure TheRock ExternalProject toolchains use perf_launcher for HIP/math-libs builds.
# Super-project host targets (e.g. third-party flatbuffers) pick up the launcher from
# -DCMAKE_*_COMPILER_LAUNCHER, but generated *_toolchain.cmake files for amd-hip
# subprojects may still have empty launchers depending on configure order.
set -euo pipefail

BUILD_DIR="${1:?build directory}"
PERF_LAUNCHER_BIN="${2:?absolute path to perf_launcher.sh}"

test -x "${PERF_LAUNCHER_BIN}"

echo "=== Injecting perf_launcher into TheRock subproject toolchains ==="
echo "Launcher: ${PERF_LAUNCHER_BIN}"

patched=0
while IFS= read -r -d '' toolchain; do
  changed=0
  for var in C CXX HIP; do
    key="CMAKE_${var}_COMPILER_LAUNCHER"
    if grep -q "^set(${key} " "${toolchain}"; then
      if ! grep -q "${PERF_LAUNCHER_BIN}" "${toolchain}"; then
        sed -i "s|^set(${key} \"[^\"]*\")|set(${key} \"${PERF_LAUNCHER_BIN}\")|" "${toolchain}"
        changed=1
      fi
    else
      printf 'set(%s "%s")\n' "${key}" "${PERF_LAUNCHER_BIN}" >> "${toolchain}"
      changed=1
    fi
  done
  if (( changed )); then
    echo "Patched ${toolchain}"
    patched=$((patched + 1))
  fi
done < <(find "${BUILD_DIR}" -name '*_toolchain.cmake' -print0)

echo "Patched ${patched} toolchain file(s)"

# Subprojects configure on first build; force reconfigure after toolchain edits.
if (( patched > 0 )); then
  find "${BUILD_DIR}" -path '*/math-libs/*' -name 'configure.stamp' -print -delete
fi

# Sanity check: rocPRIM toolchain must reference perf_launcher when present.
rocprim_toolchain="$(find "${BUILD_DIR}" -path '*/rocPRIM/*_toolchain.cmake' | head -1 || true)"
if [[ -n "${rocprim_toolchain}" ]]; then
  grep "perf_launcher.sh" "${rocprim_toolchain}" >/dev/null \
    || { echo "ERROR: ${rocprim_toolchain} missing perf_launcher" >&2; exit 1; }
  echo "Verified rocPRIM toolchain: ${rocprim_toolchain}"
fi
