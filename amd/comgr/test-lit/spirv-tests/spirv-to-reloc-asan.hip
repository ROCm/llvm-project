// REQUIRES: comgr-has-spirv-translator, comgr-has-amdgpu-target

// COM: Mark the kernel for ASan at the frontend; instrumentation runs when
// COM: Comgr compiles the SPIR-V to AMDGPU code for an xnack+ target.
// RUN: %clang -x hip --offload-arch=amdgcnspirv -nogpulib -nogpuinc \
// RUN:   --no-gpu-bundle-output --offload-device-only -O1 -c \
// RUN:   -Xclang -fsanitize=address %s -o %t.spv
// RUN: spirv-to-reloc --asan %t.spv %t.o

// COM: Verify the kernel and ASan instrumentation for its loads and store.
// RUN: %llvm-objdump -d %t.o | %FileCheck %s --check-prefix=KERNEL
// KERNEL-LABEL: <_Z3addPfS_S_>:
// KERNEL: s_endpgm
// RUN: %llvm-objdump -r %t.o | %FileCheck %s --check-prefix=ASAN
// ASAN-DAG: __asan_report_load4
// ASAN-DAG: __asan_report_store4

__attribute__((global)) void add(float *A, float *B, float *C) {
  *C = *A + *B;
}
