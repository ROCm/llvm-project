// REQUIRES: amdgpu-registered-target

// HIP device side: implicit __host__ __device__ dtor of an explicit
// template instantiation with a safe (trivial) body. Vtable slot
// references the real dtor mangled name; no separately-named trap stub.

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -x hip \
// RUN:   -emit-llvm -o - %s | FileCheck %s

#include "Inputs/cuda.h"

template <typename T>
struct ETI {
  virtual ~ETI() = default;
};
template class ETI<float>;

// CHECK: @_ZTV3ETIIfE =
// CHECK-SAME: @_ZN3ETIIfED1Ev
// CHECK-SAME: @_ZN3ETIIfED0Ev
// CHECK-NOT: __clang_hip_unreachable_dtor
