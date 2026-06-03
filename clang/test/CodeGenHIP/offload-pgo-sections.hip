// REQUIRES: amdgpu-registered-target
// REQUIRES: x86-registered-target

// The device profile drain has two host-dependent paths:
//
//   * Linux/Unix host: collection is decoupled via HSA introspection, so
//     CGCUDANV emits NO PGO-specific machinery for HIP. The device-side bounds
//     table is provided solely by the linked device profile runtime
//     (compiler-rt/lib/profile/InstrProfilingPlatformGPU.c) as the canonical
//     __llvm_profile_sections symbol. clang must NOT emit a per-TU
//     __llvm_profile_sections_<CUID> struct, a host-side void* shadow, the
//     __hipRegisterVar / __llvm_profile_offload_register_shadow_variable
//     registration, or a __llvm_profile_register_shadow.<CUID> constructor.
//
//   * Windows host (no HSA runtime): collection falls back to the legacy
//     host-shadow mechanism, so CGCUDANV DOES emit the per-TU device-side
//     bounds struct and the matching host-side shadow + registration.
//
// Regular counter/data instrumentation must be emitted on the device either way.

// --- Linux host: device side (no aux triple) ------------------------------
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -cuid=abc \
// RUN:   -fprofile-instrument=clang -emit-llvm -o - -x hip %s \
// RUN:   | FileCheck -check-prefixes=DEV,NOSHADOW %s

// --- Linux host: host side ------------------------------------------------
// RUN: %clang_cc1 -triple x86_64-linux-gnu -aux-triple amdgcn-amd-amdhsa \
// RUN:   -cuid=abc -fprofile-instrument=clang -emit-llvm -o - -x hip %s \
// RUN:   | FileCheck -check-prefixes=HOST,NOSHADOW %s

// --- Windows host: device side (Windows aux triple) -----------------------
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device \
// RUN:   -aux-triple x86_64-pc-windows-msvc -cuid=abc \
// RUN:   -fprofile-instrument=clang -emit-llvm -o - -x hip %s \
// RUN:   | FileCheck -check-prefixes=DEV,WINDEV %s

// --- Windows host: host side ----------------------------------------------
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -aux-triple amdgcn-amd-amdhsa \
// RUN:   -cuid=abc -fprofile-instrument=clang -emit-llvm -o - -x hip %s \
// RUN:   | FileCheck -check-prefixes=HOST,WINHOST %s

#define __device__ __attribute__((device))
#define __global__ __attribute__((global))

__device__ int helper(int x) { return x + 1; }
__global__ void kernel(int *p) { *p = helper(*p); }

// Device instrumentation is always emitted (counters/data), proving device PGO
// codegen is unaffected by the host-shadow machinery.
// DEV-DAG: @__profc_{{.*}}kernel
// DEV-DAG: @__profd_{{.*}}kernel

// The host always registers the real kernel.
// HOST-DAG: define internal void @__hip_register_globals

// --- Linux: none of the host-shadow machinery may appear ------------------
// NOSHADOW-NOT: __llvm_profile_sections_
// NOSHADOW-NOT: __llvm_profile_offload_register_shadow_variable
// NOSHADOW-NOT: __llvm_profile_offload_register_section_shadow_variable
// NOSHADOW-NOT: __llvm_profile_register_shadow.

// --- Windows device: the per-TU bounds struct is emitted, populated with the
// section start/stop bounds (here: the counters-start symbol). -------------
// WINDEV-DAG: @__llvm_profile_sections_{{[0-9a-f]+}} = {{.*}}constant {{.*}}@__start___llvm_prf_cnts

// --- Windows host: the shadow var is emitted and registered ---------------
// WINHOST-DAG: @__llvm_profile_sections_{{[0-9a-f]+}} = {{.*}}global ptr null
// WINHOST-DAG: call void @__hipRegisterVar({{.*}}@__llvm_profile_sections_
// WINHOST-DAG: call void @__llvm_profile_offload_register_shadow_variable(ptr @__llvm_profile_sections_
