// REQUIRES: amdgpu-registered-target

// HIP device side: the explicit-template-instantiation suppression path
// covers any implicit __host__ __device__ function reached via an explicit
// instantiation, not just virtual destructors. With a safe body, CodeGen
// emits the real function body and the explicit instantiation is usable.
//
// If the deferred-diag walker classifies the body as device-invalid
// (Sema flags the FunctionDecl in ASTContext::CUDADeviceInvalidFuncs),
// CodeGen replaces the body with `call @llvm.trap()` followed by
// `ret void` or `ret <T> poison`. That branch is exercised by real
// libstdc++/libc++ patterns; the cross-TU vtable consistency it
// preserves is covered by the existing dtor tests.

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -x hip \
// RUN:   -emit-llvm -o - %s | FileCheck %s

#include "Inputs/cuda.h"

// Constexpr member of an explicit-template-instantiation: implicit H+D
// via CUDAHostDeviceConstexpr inference. The new
// tryEmitCUDADeviceInvalidFunctionBody hook runs on this function and
// returns false (safe body), so CodeGen emits the real body.
template <typename T>
struct ETI_Constexpr {
  constexpr T add_one(T x) { return x + (T)1; }
};
template class ETI_Constexpr<float>;

// CHECK: define {{.*}} float @_ZN13ETI_ConstexprIfE7add_oneEf(
// CHECK-NOT: call void @llvm.trap()
// CHECK: ret float

// No leftover trap-stub symbol from the pre-redesign PR.
// CHECK-NOT: __clang_hip_unreachable_dtor
// CHECK-NOT: __clang_cuda_unreachable_dtor
