// clang-format off
// RUN: %libomptarget-compileopt-generic
// RUN: %not --crash env -u LLVM_DISABLE_SYMBOLIZATION OFFLOAD_TRACK_ALLOCATION_TRACES=1 %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefixes=CHECK,NDEBG
// RUN: %libomptarget-compileopt-generic -g
// RUN: %not --crash env -u LLVM_DISABLE_SYMBOLIZATION OFFLOAD_TRACK_ALLOCATION_TRACES=1 %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefixes=CHECK,DEBUG
// clang-format on
//
// REQUIRES: gpu
// REQUIRES: multi_device
// UNSUPPORTED: nvidiagpu
// UNSUPPORTED: intelgpu

#include <omp.h>

int main(void) {
  void *Ptr = omp_target_alloc(8, /*DeviceNum=*/0);
  // Intentionally deallocate on a different device.
  omp_target_free(Ptr, /*DeviceNum=*/1);
  return 0;
}

// CHECK: OFFLOAD ERROR: deallocation of non-allocated device memory
// CHECK: dataDelete
// CHECK: omp_target_free
// NDEBG: main
// DEBUG: main {{.*}}target_alloc_wrong_device_free.c:[[@LINE-8]]
