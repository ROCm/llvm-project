// REQUIRES: comgr-has-spirv

// COM: Create a heterogeneous offload bundle with one ELF entry (gfx906)
// COM: and one SPIRV entry (amdgcnspirv), then again with --offload-compress.
// RUN: %clang -c -x hip \
// RUN:    --offload-arch=gfx906 --offload-arch=amdgcnspirv \
// RUN:    -nogpulib -nogpuinc \
// RUN:    --gpu-bundle-output --offload-device-only \
// RUN:    %s -o %t.mixed.bundle
//
// RUN: %clang -c -x hip \
// RUN:    --offload-arch=gfx906 --offload-arch=amdgcnspirv \
// RUN:    -nogpulib -nogpuinc \
// RUN:    --gpu-bundle-output --offload-device-only \
// RUN:    --offload-compress \
// RUN:    %s -o %t.mixed.compressed.bundle

// COM: Extract both entries through Comgr's generic BUNDLE unbundle.
// COM: Detection picks the data kind per entry: ELF -> .o, SPIRV -> .spv.
// RUN: unbundle %t.mixed.bundle %t \
// RUN:    hipv4-amdgcn-amd-amdhsa--gfx906 \
// RUN:    hipv4-spirv64-amd-amdhsa--amdgcnspirv

// COM: ELF entry should be a valid AMDGPU object for gfx906.
// RUN: llvm-readobj --file-headers %t-hipv4-amdgcn-amd-amdhsa--gfx906.o \
// RUN:    | %FileCheck --check-prefix=ELF %s
// ELF: Format: elf64-amdgpu
// ELF: Machine: EM_AMDGPU
// ELF: Flags [
// ELF: EF_AMDGPU_MACH_AMDGCN_GFX906

// COM: SPIRV entry should round-trip through the SPIRV translator.
// RUN: spirv-translator %t-hipv4-spirv64-amd-amdhsa--amdgcnspirv.spv -o %t.spv.bc
// RUN: %llvm-dis %t.spv.bc -o - | %FileCheck --check-prefix=SPV %s
// SPV: target triple = "amdgcn-amd-amdhsa"
// SPV: define amdgpu_kernel void @_Z9add_valuePfS_S_

// COM: Same checks for the compressed bundle.
// RUN: unbundle %t.mixed.compressed.bundle %t.compressed \
// RUN:    hipv4-amdgcn-amd-amdhsa--gfx906 \
// RUN:    hipv4-spirv64-amd-amdhsa--amdgcnspirv
// RUN: llvm-readobj --file-headers %t.compressed-hipv4-amdgcn-amd-amdhsa--gfx906.o \
// RUN:    | %FileCheck --check-prefix=ELF %s
// RUN: spirv-translator %t.compressed-hipv4-spirv64-amd-amdhsa--amdgcnspirv.spv -o %t.compressed.spv.bc
// RUN: %llvm-dis %t.compressed.spv.bc -o - | %FileCheck --check-prefix=SPV %s

__attribute__((device))
void clean_value(float* ptr) { *ptr = 0; }

__attribute__((global))
void add_value(float* a, float* b, float* res) {
    *res = *a + *b;

    clean_value(a);
}
