// REQUIRES: amdgpu-registered-target
// REQUIRES: x86-registered-target

// After decoupling the device profile drain (HSA introspection), CGCUDANV no
// longer emits any PGO-specific machinery for HIP. The device-side bounds
// table is provided solely by the linked device profile runtime
// (compiler-rt/lib/profile/InstrProfilingPlatformGPU.c) as the canonical
// __llvm_profile_sections symbol, so clang must NOT emit:
//   * a per-TU __llvm_profile_sections_<CUID> struct (device or host),
//   * a host-side void* shadow,
//   * __hipRegisterVar / __llvm_profile_offload_register_shadow_variable for it,
//   * a __llvm_profile_register_shadow.<CUID> constructor.
// Regular counter/data instrumentation must still be emitted on the device.

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -cuid=abc \
// RUN:   -fprofile-instrument=clang -emit-llvm -o - -x hip %s \
// RUN:   | FileCheck -check-prefixes=DEV,NOSHADOW %s

// RUN: %clang_cc1 -triple x86_64-linux-gnu -cuid=abc \
// RUN:   -fprofile-instrument=clang -emit-llvm -o - -x hip %s \
// RUN:   | FileCheck -check-prefixes=HOST,NOSHADOW %s

#define __device__ __attribute__((device))
#define __global__ __attribute__((global))

__device__ int helper(int x) { return x + 1; }
__global__ void kernel(int *p) { *p = helper(*p); }

// Device instrumentation is still emitted (counters/data), proving device PGO
// codegen is unaffected by removing the shadow machinery.
// DEV-DAG: @__profc_{{.*}}kernel
// DEV-DAG: @__profd_{{.*}}kernel

// The host still registers the real kernel, just not any profile shadow.
// HOST: define internal void @__hip_register_globals

// None of the removed per-CUID / shadow machinery may appear in either compile.
// NOSHADOW-NOT: __llvm_profile_sections_
// NOSHADOW-NOT: __llvm_profile_offload_register_shadow_variable
// NOSHADOW-NOT: __llvm_profile_offload_register_section_shadow_variable
// NOSHADOW-NOT: __llvm_profile_register_shadow.
