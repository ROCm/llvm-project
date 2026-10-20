// A cache key must cover every concatenated compressed bundle, not only the
// first one. Populate the cache with a bundle that does not contain gfx906,
// then request gfx906 from A+B and compare with the uncached result.
//
// RUN: %clang -c -x hip --offload-arch=gfx900 \
// RUN:    -nogpulib -nogpuinc \
// RUN:    --gpu-bundle-output --offload-device-only \
// RUN:    -emit-llvm --offload-compress \
// RUN:    %s -o %t.first.bc
// RUN: %clang -c -x hip --offload-arch=gfx906 \
// RUN:    -nogpulib -nogpuinc \
// RUN:    --gpu-bundle-output --offload-device-only \
// RUN:    -emit-llvm --offload-compress \
// RUN:    %s -o %t.second.bc
// RUN: cat %t.first.bc %t.second.bc > %t.concat.bc
// RUN: rm -fr %t.cache
//
// RUN: AMD_COMGR_CACHE=1 AMD_COMGR_CACHE_DIR=%t.cache \
// RUN:    unbundle %t.first.bc \
// RUN:    hip-amdgcn-amd-amdhsa-unknown-gfx906 %t.populate.bc
// RUN: AMD_COMGR_CACHE=1 AMD_COMGR_CACHE_DIR=%t.cache \
// RUN:    unbundle %t.concat.bc \
// RUN:    hip-amdgcn-amd-amdhsa-unknown-gfx906 %t.cached.bc
// RUN: AMD_COMGR_CACHE=0 \
// RUN:    unbundle %t.concat.bc \
// RUN:    hip-amdgcn-amd-amdhsa-unknown-gfx906 %t.expected.bc
// RUN: cmp %t.expected.bc %t.cached.bc
// RUN: %llvm-dis %t.cached.bc -o - | %FileCheck %s
// CHECK: "target-cpu"="gfx906"

__attribute__((device))
void add_value(float* a, float* b, float* res) {
    *res = *a + *b;
}
