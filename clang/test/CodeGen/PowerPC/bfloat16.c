// RUN: %clang_cc1 -triple powerpc64le-linux-gnu -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefix=PPC64LE
// RUN: %clang_cc1 -triple powerpc-linux-gnu -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefix=PPC32
// RUN: %clang_cc1 -triple powerpc64le-linux-gnu -mcpu=pwr10 -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefix=PPC64LE
//
// Test that __bf16 is accepted on PowerPC targets and that the Clang
// frontend emits the expected 'bfloat' IR type with soft-promote libcall
// references.  Actual instruction selection is tested separately in
// llvm/test/CodeGen/PowerPC/bfloat16-soft-promote.ll.

// __bf16 must be accepted (no "not supported on this target" error).
__bf16 global_bf = 1.0bf16;

// PPC64LE: @global_bf = {{.*}} bfloat 0xR3F80
// PPC32:   @global_bf = {{.*}} bfloat 0xR3F80

// Basic arithmetic produces 'bfloat' typed IR; the backend soft-promotes.
__bf16 add(__bf16 a, __bf16 b) {
  return a + b;
// PPC64LE-LABEL: define {{.*}} bfloat @add(bfloat noundef %a, bfloat noundef %b)
// PPC32-LABEL:   define {{.*}} bfloat @add(bfloat noundef %a, bfloat noundef %b)
// PPC64LE: fadd bfloat
// PPC32:   fadd bfloat
}

__bf16 mul(__bf16 a, __bf16 b) {
  return a * b;
// PPC64LE: fmul bfloat
// PPC32:   fmul bfloat
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
