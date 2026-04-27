// REQUIRES: comgr-has-spirv

// COM: Create a SPIRV bundle for amdgcnspirv
// RUN: %clang -c -x hip --offload-arch=amdgcnspirv \
// RUN:    -nogpulib -nogpuinc \
// RUN:    --gpu-bundle-output --offload-device-only \
// RUN:    %s -o %t.bundle.spv
//
// COM: Create a compressed SPIRV bundle (add --offload-compress)
// RUN: %clang -c -x hip --offload-arch=amdgcnspirv \
// RUN:    -nogpulib -nogpuinc \
// RUN:    --gpu-bundle-output --offload-device-only \
// RUN:    --offload-compress \
// RUN:    %s -o %t.compressed-bundle.spv

// COM: Extract via Comgr's generic BUNDLE unbundle. Detection picks
// COM: AMD_COMGR_DATA_KIND_SPIRV from the bytes, so the driver writes
// COM: the entry to a .spv file. If detection misclassified the payload
// COM: the spirv-translator step would fail to find its input.
// RUN: unbundle %t.bundle.spv %t hip-spirv64-amd-amdhsa--amdgcnspirv
// RUN: spirv-translator %t-hip-spirv64-amd-amdhsa--amdgcnspirv.spv -o %t.bc
// RUN: %llvm-dis %t.bc -o - | %FileCheck %s

// RUN: unbundle %t.compressed-bundle.spv %t.compressed hip-spirv64-amd-amdhsa--amdgcnspirv
// RUN: spirv-translator %t.compressed-hip-spirv64-amd-amdhsa--amdgcnspirv.spv -o %t.compressed.bc
// RUN: %llvm-dis %t.compressed.bc -o - | %FileCheck %s

// CHECK: target triple = "amdgcn-amd-amdhsa"
// CHECK: define void @_Z11clean_valuePf
// CHECK: define amdgpu_kernel void @_Z9add_valuePfS_S_

__attribute__((device))
void clean_value(float* ptr) { *ptr = 0; }

__attribute__((global))
void add_value(float* a, float* b, float* res) {
    *res = *a + *b;

    clean_value(a);
}
