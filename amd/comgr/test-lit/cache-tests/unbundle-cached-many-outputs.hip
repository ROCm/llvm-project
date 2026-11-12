// A cache entry with more outputs than one unbundles to the same outputs, in
// the order of the bundle entry IDs, as a cache miss.
//
// RUN: %clang -c -x hip --offload-arch=gfx900 --offload-arch=gfx906 \
// RUN:    --offload-arch=gfx908 --offload-arch=gfx90a --offload-arch=gfx1030 \
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
// RUN: unbundle %t.compressed-bundle.bc \
// RUN:    hip-amdgcn-amd-amdhsa-unknown-gfx900 %t.miss.0.bc \
// RUN:    hip-amdgcn-amd-amdhsa-unknown-gfx906 %t.miss.1.bc \
// RUN:    hip-amdgcn-amd-amdhsa-unknown-gfx908 %t.miss.2.bc \
// RUN:    hip-amdgcn-amd-amdhsa-unknown-gfx90a %t.miss.3.bc \
// RUN:    hip-amdgcn-amd-amdhsa-unknown-gfx1030 %t.miss.4.bc
// RUN: COUNT=$(ls "%t.cache" | wc -l)
// RUN: [ 2 -eq $COUNT ]
//
// A cache hit gives the same outputs.
// RUN: unbundle %t.compressed-bundle.bc \
// RUN:    hip-amdgcn-amd-amdhsa-unknown-gfx900 %t.hit.0.bc \
// RUN:    hip-amdgcn-amd-amdhsa-unknown-gfx906 %t.hit.1.bc \
// RUN:    hip-amdgcn-amd-amdhsa-unknown-gfx908 %t.hit.2.bc \
// RUN:    hip-amdgcn-amd-amdhsa-unknown-gfx90a %t.hit.3.bc \
// RUN:    hip-amdgcn-amd-amdhsa-unknown-gfx1030 %t.hit.4.bc
// RUN: COUNT=$(ls "%t.cache" | wc -l)
// RUN: [ 2 -eq $COUNT ]
// RUN: cmp %t.miss.0.bc %t.hit.0.bc
// RUN: cmp %t.miss.1.bc %t.hit.1.bc
// RUN: cmp %t.miss.2.bc %t.hit.2.bc
// RUN: cmp %t.miss.3.bc %t.hit.3.bc
// RUN: cmp %t.miss.4.bc %t.hit.4.bc
// RUN: %llvm-dis %t.hit.0.bc -o - | %FileCheck --check-prefix=GFX900 %s
// RUN: %llvm-dis %t.hit.1.bc -o - | %FileCheck --check-prefix=GFX906 %s
// RUN: %llvm-dis %t.hit.2.bc -o - | %FileCheck --check-prefix=GFX908 %s
// RUN: %llvm-dis %t.hit.3.bc -o - | %FileCheck --check-prefix=GFX90A %s
// RUN: %llvm-dis %t.hit.4.bc -o - | %FileCheck --check-prefix=GFX1030 %s
//
// Make the size of the first output larger than the entry. The entry is
// ignored and the input is unbundled again.
// RUN: printf '\377\377\377\377' | dd of=$(ls %t.cache/llvmcache-*) bs=1 \
// RUN:    seek=8 conv=notrunc 2>/dev/null
// RUN: unbundle %t.compressed-bundle.bc \
// RUN:    hip-amdgcn-amd-amdhsa-unknown-gfx900 %t.bad.0.bc \
// RUN:    hip-amdgcn-amd-amdhsa-unknown-gfx906 %t.bad.1.bc \
// RUN:    hip-amdgcn-amd-amdhsa-unknown-gfx908 %t.bad.2.bc \
// RUN:    hip-amdgcn-amd-amdhsa-unknown-gfx90a %t.bad.3.bc \
// RUN:    hip-amdgcn-amd-amdhsa-unknown-gfx1030 %t.bad.4.bc
// RUN: cmp %t.miss.0.bc %t.bad.0.bc
// RUN: cmp %t.miss.1.bc %t.bad.1.bc
// RUN: cmp %t.miss.2.bc %t.bad.2.bc
// RUN: cmp %t.miss.3.bc %t.bad.3.bc
// RUN: cmp %t.miss.4.bc %t.bad.4.bc

// GFX900: "target-cpu"="gfx900"
// GFX906: "target-cpu"="gfx906"
// GFX908: "target-cpu"="gfx908"
// GFX90A: "target-cpu"="gfx90a"
// GFX1030: "target-cpu"="gfx1030"

__attribute__((device))
void add_value(float* a, float* b, float* res) {
    *res = *a + *b;
}
