// A damaged unbundle cache entry is ignored and the input is unbundled again.
//
// RUN: %clang -c -x hip --offload-arch=gfx900 \
// RUN:    -nogpulib -nogpuinc \
// RUN:    --gpu-bundle-output --offload-device-only \
// RUN:    -emit-llvm \
// RUN:    --offload-compress \
// RUN:    %s -o %t.compressed-bundle.bc
//
// RUN: rm -fr %t.cache
// RUN: export AMD_COMGR_CACHE=1
// RUN: export AMD_COMGR_CACHE_DIR=%t.cache
//
// Fill the cache.
// RUN: unbundle %t.compressed-bundle.bc hip-amdgcn-amd-amdhsa-unknown-gfx900 \
// RUN:    %t.miss.bc
//
// Make the size of the first unbundled output larger than the entry. The
// entry still deserializes, but its output does not split into the outputs.
// RUN: printf '\377\377\377\377' | dd of=$(ls %t.cache/llvmcache-*) bs=1 \
// RUN:    seek=8 conv=notrunc 2>/dev/null
// RUN: unbundle %t.compressed-bundle.bc hip-amdgcn-amd-amdhsa-unknown-gfx900 \
// RUN:    %t.bad-size.bc
// RUN: cmp %t.miss.bc %t.bad-size.bc
//
// Truncate the entry so that it does not deserialize.
// RUN: truncate -s 4 $(ls %t.cache/llvmcache-*)
// RUN: unbundle %t.compressed-bundle.bc hip-amdgcn-amd-amdhsa-unknown-gfx900 \
// RUN:    %t.truncated.bc
// RUN: cmp %t.miss.bc %t.truncated.bc
//
// RUN: %llvm-dis %t.miss.bc -o - | %FileCheck %s
// CHECK: "target-cpu"="gfx900"

__attribute__((device))
void add_value(float* a, float* b, float* res) {
    *res = *a + *b;
}
