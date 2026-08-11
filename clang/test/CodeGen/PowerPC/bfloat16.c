// RUN: %clang_cc1 -triple powerpc64le-linux-gnu -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefix=PPC64LE
// RUN: %clang_cc1 -triple powerpc-linux-gnu -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefix=PPC32
// RUN: %clang_cc1 -triple powerpc64le-linux-gnu -target-cpu pwr10 -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefix=PPC64LE
//
// Test that __bf16 is accepted on PowerPC targets and that the Clang
// frontend emits the expected 'bfloat' IR type.  Arithmetic is soft-promoted
// to float.  Actual instruction selection is tested separately in
// llvm/test/CodeGen/PowerPC/bfloat16-soft-promote.ll.

// __bf16 must be accepted (no "not supported on this target" error).
__bf16 global_bf = 1.0;

// PPC64LE: @global_bf = {{.*}} bfloat {{.*}}, align 2
// PPC32:   @global_bf = {{.*}} bfloat {{.*}}, align 2

// Return type and parameters use 'bfloat'; arithmetic is promoted to float.
__bf16 add(__bf16 a, __bf16 b) {
  return a + b;
// PPC64LE-LABEL: define {{.*}} bfloat @add(bfloat noundef %a, bfloat noundef %b)
// PPC32-LABEL:   define {{.*}} bfloat @add(bfloat noundef %a, bfloat noundef %b)
// PPC64LE: fadd float
// PPC32:   fadd float
}

__bf16 mul(__bf16 a, __bf16 b) {
  return a * b;
// PPC64LE: fmul float
// PPC32:   fmul float
}

// Extend/truncate round-trips.
float to_float(__bf16 a) {
  return (float)a;
// PPC64LE: fpext bfloat {{.*}} to float
// PPC32:   fpext bfloat {{.*}} to float
}

__bf16 from_float(float a) {
  return (__bf16)a;
// PPC64LE: fptrunc float {{.*}} to bfloat
// PPC32:   fptrunc float {{.*}} to bfloat
}

// sizeof and alignof must both be 2.
_Static_assert(sizeof(__bf16) == 2, "sizeof(__bf16) != 2");
_Static_assert(_Alignof(__bf16) == 2, "_Alignof(__bf16) != 2");
