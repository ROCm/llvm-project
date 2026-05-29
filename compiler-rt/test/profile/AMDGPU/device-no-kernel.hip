// Independence / robustness: an instrumented HIP program that never launches a
// kernel still writes its host .profraw, and the device drain is a clean no-op
// (no crash, no spurious device .profraw) because no device code is loaded.
//
// REQUIRES: amdgpu

// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: %clang -x hip --offload-arch=native \
// RUN:   -fprofile-instr-generate -fcoverage-mapping %s -o %t.dir/a.out \
// RUN:   -lamdhip64
// RUN: env LLVM_PROFILE_FILE=%t.dir/host.%%p.profraw %t.dir/a.out
// RUN: ls %t.dir/host.*.profraw
// No arch-prefixed device profraw should have been produced.
// RUN: not ls %t.dir/gfx*.profraw

#include <hip/hip_runtime.h>

// Defined but never launched.
__global__ void unused(int *p) { *p += 1; }

int main() {
  int n = 0;
  (void)hipGetDeviceCount(&n);
  return 0;
}
