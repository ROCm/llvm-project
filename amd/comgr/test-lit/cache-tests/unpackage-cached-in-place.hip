// A cache hit uses the cached image in place. The result must match a cache
// miss, also with save-temps, which writes the unpackaged file to disk.
// unpackage-cached-bad-entry.hip covers damaged entries.
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
// Fill the cache, then read the entry back in place.
// RUN: unpackage %t.compressed-package.bc amdgcn-amd-amdhsa gfx900 %t.miss.bc
// RUN: unpackage %t.compressed-package.bc amdgcn-amd-amdhsa gfx900 %t.hit.bc
// RUN: cmp %t.miss.bc %t.hit.bc
//
// With save-temps, the cache hit is written to the temporary directory.
// Windows reads TMP and TEMP instead of TMPDIR.
// RUN: rm -fr %t.temps && mkdir -p %t.temps
// RUN: TMPDIR=%t.temps TMP=%t.temps TEMP=%t.temps AMD_COMGR_SAVE_TEMPS=1 \
// RUN:    unpackage %t.compressed-package.bc amdgcn-amd-amdhsa gfx900 \
// RUN:    %t.savetemps.bc
// RUN: cmp %t.miss.bc %t.savetemps.bc
// RUN: ls %t.temps/comgr-*/output | FileCheck --check-prefix=TEMPS %s
// TEMPS: .bc
//
// RUN: llvm-dis %t.hit.bc -o - | FileCheck %s
// CHECK: "target-cpu"="gfx900"

__attribute__((device))
void add_value(float* a, float* b, float* res) {
    *res = *a + *b;
}
