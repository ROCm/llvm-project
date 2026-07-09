// Create bitcode bundle
// RUN: %clang -c -x hip --offload-arch=gfx900 --offload-arch=gfx1030 \
// RUN:    -nogpulib -nogpuinc -emit-llvm \
// RUN:    --gpu-bundle-output --offload-device-only \
// RUN:    %s -o %t.bundle.bc
//
// Create compressed bitcode bundle (add --offload-compress flag)
// RUN: %clang -c -x hip --offload-arch=gfx900 --offload-arch=gfx1030 \
// RUN:    -nogpulib -nogpuinc -emit-llvm \
// RUN:    --gpu-bundle-output --offload-device-only \
// RUN:    --offload-compress \
// RUN:    %s -o %t.compressed-bundle.bc
//
// Extract using Comgr's generic BUNDLE unbundle (kind detected from bytes)
// RUN: unbundle %t.bundle.bc %t \
// RUN:    hip-amdgcn-amd-amdhsa-unknown-gfx900 \
// RUN:    hip-amdgcn-amd-amdhsa-unknown-gfx1030
// RUN: %llvm-dis %t-hip-amdgcn-amd-amdhsa-unknown-gfx900.bc -o - | %FileCheck --check-prefixes=BOTH,GFX9 %s
// RUN: %llvm-dis %t-hip-amdgcn-amd-amdhsa-unknown-gfx1030.bc -o - | %FileCheck --check-prefixes=BOTH,GFX10 %s
//
// RUN: unbundle %t.compressed-bundle.bc %t.compressed \
// RUN:    hip-amdgcn-amd-amdhsa-unknown-gfx900 \
// RUN:    hip-amdgcn-amd-amdhsa-unknown-gfx1030
// RUN: %llvm-dis %t.compressed-hip-amdgcn-amd-amdhsa-unknown-gfx900.bc -o - | %FileCheck --check-prefixes=BOTH,GFX9 %s
// RUN: %llvm-dis %t.compressed-hip-amdgcn-amd-amdhsa-unknown-gfx1030.bc -o - | %FileCheck --check-prefixes=BOTH,GFX10 %s
//
// BOTH: target triple = "amdgcn-amd-amdhsa"
// GFX9: "target-cpu"="gfx900"
// GFX10: "target-cpu"="gfx1030"

__attribute__((device))
void add_value(float* a, float* b, float* res) {
    *res = *a + *b;
}
