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
// RUN: export AMD_COMGR_EMIT_VERBOSE_LOGS=1
// RUN: export AMD_COMGR_REDIRECT_LOGS=stderr
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
// RUN:    %t.bad-size.bc 2> %t.bad-size.log
// RUN: %FileCheck --check-prefix=BAD-SIZE %s < %t.bad-size.log
// RUN: cmp %t.miss.bc %t.bad-size.bc
//
// Truncate the entry so that it does not deserialize.
// RUN: truncate -s 4 $(ls %t.cache/llvmcache-*)
// RUN: unbundle %t.compressed-bundle.bc hip-amdgcn-amd-amdhsa-unknown-gfx900 \
// RUN:    %t.truncated.bc 2> %t.truncated.log
// RUN: %FileCheck --check-prefix=TRUNCATED %s < %t.truncated.log
// RUN: cmp %t.miss.bc %t.truncated.bc
//
// RUN: %llvm-dis %t.miss.bc -o - | %FileCheck %s
// CHECK: "target-cpu"="gfx900"
// BAD-SIZE-NOT: Comgr cache: found entry
// BAD-SIZE: Comgr cache, when writing the command output:
// BAD-SIZE-NOT: Comgr cache: found entry
// TRUNCATED-NOT: Comgr cache: found entry
// TRUNCATED: Comgr cache, when reading the cache entry:
// TRUNCATED-NOT: Comgr cache: found entry
//
// Cache diagnostics stay silent when verbose logging is disabled. The verbose
// setting is part of the key, so populate a separate cache in quiet mode.
// RUN: export AMD_COMGR_EMIT_VERBOSE_LOGS=0
// RUN: export AMD_COMGR_CACHE_DIR=%t.quiet-cache
// RUN: rm -fr %t.quiet-cache
// RUN: unbundle %t.compressed-bundle.bc hip-amdgcn-amd-amdhsa-unknown-gfx900 \
// RUN:    %t.quiet-miss.bc
// RUN: printf '\377\377\377\377' | dd of=$(ls %t.quiet-cache/llvmcache-*) bs=1 \
// RUN:    seek=8 conv=notrunc 2>/dev/null
// RUN: unbundle %t.compressed-bundle.bc hip-amdgcn-amd-amdhsa-unknown-gfx900 \
// RUN:    %t.quiet-hit.bc 2> %t.quiet.log
// RUN: cmp %t.miss.bc %t.quiet-hit.bc
// RUN: %FileCheck --allow-empty --check-prefix=QUIET %s < %t.quiet.log
// QUIET-NOT: Comgr cache

__attribute__((device))
void add_value(float* a, float* b, float* res) {
    *res = *a + *b;
}
