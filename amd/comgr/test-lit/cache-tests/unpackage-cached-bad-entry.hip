// A damaged unpackage cache entry is ignored and the input is unpackaged
// again, falling back from the in-place path to re-executing.
//
// RUN: %clang -c -x hip --offload-arch=gfx900 \
// RUN: -nogpulib -nogpuinc -emit-llvm \
// RUN: --offload-device-only %s -o %t.gfx900.bc
//
// Let llvm-offload-binary pick zstd or zlib (whichever this build supports).
// RUN: llvm-offload-binary --compress \
// RUN: --image=file=%t.gfx900.bc,triple=amdgcn-amd-amdhsa,arch=gfx900,kind=hip \
// RUN: -o %t.compressed-package.bc
//
// RUN: rm -fr %t.cache
// RUN: export AMD_COMGR_CACHE=1
// RUN: export AMD_COMGR_CACHE_DIR=%t.cache
//
// Fill the cache.
// RUN: unpackage %t.compressed-package.bc amdgcn-amd-amdhsa gfx900 %t.miss.bc
//
// Make the size of the unpackaged output's name larger than the entry. The
// entry still deserializes (deserializeCacheEntry's own length-prefix frame
// is unaffected), but splitCachedOutput's inner length-prefix parse fails.
// RUN: printf '\377\377\377\377' | dd of=$(ls %t.cache/llvmcache-*) bs=1 \
// RUN:    seek=8 conv=notrunc 2>/dev/null
// RUN: unpackage %t.compressed-package.bc amdgcn-amd-amdhsa gfx900 \
// RUN:    %t.bad-size.bc
// RUN: cmp %t.miss.bc %t.bad-size.bc
//
// Truncate the entry so that it does not deserialize at all.
// RUN: truncate -s 4 $(ls %t.cache/llvmcache-*)
// RUN: unpackage %t.compressed-package.bc amdgcn-amd-amdhsa gfx900 \
// RUN:    %t.truncated.bc
// RUN: cmp %t.miss.bc %t.truncated.bc
//
// RUN: llvm-dis %t.miss.bc -o - | FileCheck %s
// CHECK: "target-cpu"="gfx900"

__attribute__((device))
void add_value(float* a, float* b, float* res) {
    *res = *a + *b;
}
