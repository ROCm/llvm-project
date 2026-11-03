// Create a compressed bitcode package.
//
// RUN: %clang -c -x hip --offload-arch=gfx900 \
// RUN: -nogpulib -nogpuinc -emit-llvm \
// RUN: --offload-device-only %s -o %t.gfx900.bc
//
// RUN: %clang -c -x hip --offload-arch=gfx1030 \
// RUN: -nogpulib -nogpuinc -emit-llvm \
// RUN: --offload-device-only %s -o %t.gfx1030.bc
//
// Let llvm-offload-binary pick zstd or zlib (whichever this build supports);
// either is a compressed package as far as UnpackageCommand::canCache() is
// concerned.
// RUN: llvm-offload-binary --compress \
// RUN: --image=file=%t.gfx900.bc,triple=amdgcn-amd-amdhsa,arch=gfx900,kind=hip \
// RUN: --image=file=%t.gfx1030.bc,triple=amdgcn-amd-amdhsa,arch=gfx1030,kind=hip \
// RUN: -o %t.compressed-package.bc
//
// Clean the cache.
// RUN: rm -fr %t.cache
//
// With the cache enabled, test that we write one file to the cache.
// RUN: export AMD_COMGR_CACHE=1
// RUN: export AMD_COMGR_CACHE_DIR=%t.cache
// RUN: unpackage %t.compressed-package.bc amdgcn-amd-amdhsa gfx900 %t.cache_1.bc
// RUN: llvm-dis %t.cache_1.bc -o - | FileCheck --check-prefixes=BOTH,GFX9 %s
// RUN: COUNT=$(ls "%t.cache" | wc -l)
// RUN: [ 2 -eq $COUNT ]
//
// If there is a re-run, the cache contents remain the same.
// RUN: unpackage %t.compressed-package.bc amdgcn-amd-amdhsa gfx900 %t.cache_2.bc
// RUN: llvm-dis %t.cache_2.bc -o - | FileCheck --check-prefixes=BOTH,GFX9 %s
// RUN: COUNT=$(ls "%t.cache" | wc -l)
// RUN: [ 2 -eq $COUNT ]
// RUN: cmp %t.cache_1.bc %t.cache_2.bc
//
// A run with different package entry IDs results in new contents in the
// cache.
// RUN: unpackage %t.compressed-package.bc amdgcn-amd-amdhsa gfx1030 %t.cache_3.bc
// RUN: llvm-dis %t.cache_3.bc -o - | FileCheck --check-prefixes=BOTH,GFX10 %s
// RUN: COUNT=$(ls "%t.cache" | wc -l)
// RUN: [ 3 -eq $COUNT ]
//
// An uncompressed package is not cached at all: no new entries appear.
// RUN: llvm-offload-binary \
// RUN: --image=file=%t.gfx900.bc,triple=amdgcn-amd-amdhsa,arch=gfx900,kind=hip \
// RUN: --image=file=%t.gfx1030.bc,triple=amdgcn-amd-amdhsa,arch=gfx1030,kind=hip \
// RUN: -o %t.package.bc
// RUN: unpackage %t.package.bc amdgcn-amd-amdhsa gfx900 %t.uncompressed.bc
// RUN: llvm-dis %t.uncompressed.bc -o - | FileCheck --check-prefixes=BOTH,GFX9 %s
// RUN: COUNT=$(ls "%t.cache" | wc -l)
// RUN: [ 3 -eq $COUNT ]

// BOTH: target datalayout =
// GFX9: {{target triple = "amdgpu9\.00-amd-amdhsa"|"target-cpu"="gfx900"}}
// GFX10: {{target triple = "amdgpu10\.30-amd-amdhsa"|"target-cpu"="gfx1030"}}

__attribute__((device))
void add_value(float* a, float* b, float* res) {
    *res = *a + *b;
}
