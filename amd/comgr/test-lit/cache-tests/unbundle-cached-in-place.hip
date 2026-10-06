// A cache hit uses the cached code object in place. The result must match a
// cache miss, also with save-temps, which writes the unbundled files to disk.
// unbundle-cached-bad-entry.hip covers damaged entries.
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
// Fill the cache, then read the entry back in place.
// RUN: unbundle %t.compressed-bundle.bc hip-amdgcn-amd-amdhsa-unknown-gfx900 \
// RUN:    %t.miss.bc
// RUN: unbundle %t.compressed-bundle.bc hip-amdgcn-amd-amdhsa-unknown-gfx900 \
// RUN:    %t.hit.bc
// RUN: cmp %t.miss.bc %t.hit.bc
//
// With save-temps, the cache hit is written to the temporary directory.
// RUN: rm -fr %t.temps && mkdir -p %t.temps
// RUN: TMPDIR=%t.temps AMD_COMGR_SAVE_TEMPS=1 unbundle \
// RUN:    %t.compressed-bundle.bc hip-amdgcn-amd-amdhsa-unknown-gfx900 \
// RUN:    %t.savetemps.bc
// RUN: cmp %t.miss.bc %t.savetemps.bc
// RUN: ls %t.temps/comgr-*/output | %FileCheck --check-prefix=TEMPS %s
// TEMPS: .bc
//
// RUN: %llvm-dis %t.hit.bc -o - | %FileCheck %s
// CHECK: "target-cpu"="gfx900"

__attribute__((device))
void add_value(float* a, float* b, float* res) {
    *res = *a + *b;
}
