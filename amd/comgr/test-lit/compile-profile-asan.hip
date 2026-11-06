// REQUIRES: comgr-has-amdgpu-profile-runtime
// REQUIRES: comgr-has-embedded-amdgpu-asan-runtime
// RUN: compile-hip-minimal --profile-asan %s %t.bin
// RUN: %llvm-objdump -t %t.bin | %FileCheck %s

// CHECK: __llvm_profile_instrument_gpu
// CHECK: __asan_memcpy

__attribute__((global))
void add(float *A, float *B, float *C) {
  *C = *A + *B;
}
