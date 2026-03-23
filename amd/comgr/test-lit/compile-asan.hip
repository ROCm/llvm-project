// COM: Run Comgr binary to compile HIP source with AddressSanitizer enabled
// RUN: compile-hip-asan %s %t.bin

// COM: Disassemble and check for ASAN instrumentation
// RUN: %llvm-objdump -d %t.bin | %FileCheck %s
// CHECK: <_Z3addPfS_S_>:
// CHECK: s_endpgm

__attribute__((global))
void add(float *A, float *B, float *C) {
    *C = *A + *B;
}
