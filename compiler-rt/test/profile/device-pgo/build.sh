#!/usr/bin/env bash
# Standalone (non-TheRock) build of the toolchain + host/device runtimes used by
# the HIP device-PGO / code-coverage tests. See toolchain-cache.cmake and
# README.md for details.
#
#   ./build.sh [BUILD_DIR]
#
# Env knobs:
#   LLVM_SRC   path to the llvm-project checkout (default: repo root inferred
#              from this script's location)
#   JOBS       parallelism for ninja (default: nproc)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# .../compiler-rt/test/profile/device-pgo -> repo root is four levels up.
LLVM_SRC="${LLVM_SRC:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"
BUILD_DIR="${1:-${LLVM_SRC}/build/device-pgo}"
JOBS="${JOBS:-$(nproc)}"

echo "llvm-project source : ${LLVM_SRC}"
echo "build directory     : ${BUILD_DIR}"
echo "parallel jobs       : ${JOBS}"

cmake -G Ninja \
  -S "${LLVM_SRC}/llvm" \
  -B "${BUILD_DIR}" \
  -C "${SCRIPT_DIR}/toolchain-cache.cmake"

# 'runtimes' builds both the host (default) and amdgcn device runtime targets.
ninja -C "${BUILD_DIR}" -j "${JOBS}" \
  clang clang++ lld llvm-profdata llvm-cov FileCheck not runtimes

cat <<EOF

Build complete.

Toolchain bin : ${BUILD_DIR}/bin
Run the GPU tests with, e.g.:

  python3 ${SCRIPT_DIR}/../run_gpu_tests.py \\
      --toolchain-bin ${BUILD_DIR}/bin \\
      --hip-path \$ROCM_PATH \\
      ${SCRIPT_DIR}/../GPU ${SCRIPT_DIR}/../AMDGPU

(See README.md for the exact runner flags on your system.)
EOF
