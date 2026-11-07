// A version 2 llvm-offload-binary package (predating the InflatedSize header
// field added in version 3) must not be treated as compressed/cacheable.
//
// Regression test: UnpackageCommand::canCache() used to read
// OffloadBinary::Header::InflatedSize itself, without checking the header
// version first (that check now lives in OffloadBinary::isCompressed(), which
// canCache() defers to). A v2 header is 32 bytes and has no InflatedSize
// field, so the read aliased into whatever bytes happened to follow (the
// entries block), which could read as spuriously nonzero and make an
// uncompressed v2 package look compressed. That both wasted a cache entry on
// an uncompressed package and, incidentally, defeated the in-place
// (no-disk-I/O) extraction path, which is only taken when canCache() is false.
//
// RUN: %clang -c -x hip --offload-arch=gfx900 \
// RUN: -nogpulib -nogpuinc -emit-llvm \
// RUN: --offload-device-only %s -o %t.gfx900.bc
//
// RUN: %python %S/../make-v2-offload-package.py \
// RUN: amdgcn-amd-amdhsa gfx900 %t.gfx900.bc %t.v2.package.bc
//
// RUN: rm -fr %t.cache
// RUN: export AMD_COMGR_CACHE=1
// RUN: export AMD_COMGR_CACHE_DIR=%t.cache
// RUN: unpackage %t.v2.package.bc amdgcn-amd-amdhsa gfx900 %t.output.bc
// RUN: llvm-dis %t.output.bc -o - | FileCheck --check-prefixes=BOTH,GFX9 %s
//
// COM: an uncompressed package (v2 or v3) must not populate the cache: the
// COM: cache directory should not even be created.
// RUN: not test -d %t.cache

// BOTH: target datalayout =
// GFX9: {{target triple = "amdgpu9\.00-amd-amdhsa"|"target-cpu"="gfx900"}}

__attribute__((device))
void add_value(float* a, float* b, float* res) {
    *res = *a + *b;
}
