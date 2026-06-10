#!/usr/bin/env bash
# Split TheRock build-tree HCTS into per-component tarballs (Jenkins naming).
# Also writes all-math-libs-{gpu}_hcts.tar.gz with every *.hcts / *.device.hcts file.
set -euo pipefail

BUILD_DIR="${1:?build directory}"
OUTPUT_DIR="${2:-${BUILD_DIR}/logs}"
GPU_TARGET="${3:-generic}"

mkdir -p "${OUTPUT_DIR}"

declare -A COMPONENT_PATTERNS=(
  [hipBLASLt]=hipblaslt
  [hipBLAS]=hipblas
  [hipBLAS-common]=hipblas-common
  [rocBLAS]=rocblas
  [rocFFT]=rocfft
  [hipFFT]=hipfft
  [rocSPARSE]=rocsparse
  [rocSOLVER]=rocsolver
  [rocRAND]=rocrand
  [hipRAND]=hiprand
  [rocPRIM]=rocprim
  [rocThrust]=rocthrust
  [hipCUB]=hipcub
  [MIOpen]=miopen
  [composable_kernel]=composablekernel
  [rccl]=rccl
  [llvm]=amd-llvm
  [comgr]=amd-comgr
  [hip]=hip-clr
)

mapfile -d '' all_hcts < <(
  find "${BUILD_DIR}" \( -name '*.hcts' -o -name '*.device.hcts' \) -print0 2>/dev/null || true
)

if ((${#all_hcts[@]} == 0)); then
  echo "No .hcts files found under ${BUILD_DIR}" >&2
  echo "Hint: verify perf_launcher wraps HIP compiles and perf stat works." >&2
  exit 1
fi

collect_for_pattern() {
  local name="$1"
  local pattern="$2"
  local files=()
  mapfile -d '' files < <(
    find "${BUILD_DIR}" -ipath "*${pattern}*" \
      \( -name '*.hcts' -o -name '*.device.hcts' \) \
      -print0 2>/dev/null || true
  )

  if ((${#files[@]} == 0)); then
    return 0
  fi

  local tar_path="${OUTPUT_DIR}/${name}_hcts.tar.gz"
  printf '%s\0' "${files[@]}" | tar -czf "${tar_path}" --null -T -
  echo "Created ${tar_path} (${#files[@]} files)"
}

for component in "${!COMPONENT_PATTERNS[@]}"; do
  collect_for_pattern "${component}" "${COMPONENT_PATTERNS[$component]}"
done

stage_family="${GPU_TARGET}"
if [[ "${stage_family}" == "" ]]; then
  stage_family="generic"
fi

all_tar="${OUTPUT_DIR}/all-math-libs-${stage_family}_hcts.tar.gz"
printf '%s\0' "${all_hcts[@]}" | tar -czf "${all_tar}" --null -T -
echo "Created ${all_tar} (${#all_hcts[@]} files)"

mapfile -d '' created < <(
  find "${OUTPUT_DIR}" -maxdepth 1 -name '*_hcts.tar.gz' -print0 2>/dev/null || true
)
if ((${#created[@]} == 0)); then
  echo "No HCTS tarballs created under ${OUTPUT_DIR}" >&2
  exit 1
fi

echo "HCTS tarball count: ${#created[@]}"
ls -lh "${OUTPUT_DIR}"/*_hcts.tar.gz
