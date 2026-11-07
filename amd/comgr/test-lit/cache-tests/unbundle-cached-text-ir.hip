// A cache hit must preserve textual IR's terminator for downstream linking.
// More than one output puts a size field immediately after the first slice.
// RUN: %clang -S -x hip --offload-arch=gfx900 -nogpulib -nogpuinc \
// RUN:   --offload-device-only -emit-llvm %s -o %t.gfx900.ll
// RUN: %clang -S -x hip --offload-arch=gfx906 -nogpulib -nogpuinc \
// RUN:   --offload-device-only -emit-llvm %s -o %t.gfx906.ll
// Make the second size odd so its first byte cannot terminate the first IR.
// RUN: %python %S/Inputs/odd-text-size.py %t.gfx906.ll
// RUN: clang-offload-bundler --type=bc --compress \
// RUN:   --targets=hip-amdgcn-amd-amdhsa-unknown-gfx900,hip-amdgcn-amd-amdhsa-unknown-gfx906 \
// RUN:   --input=%t.gfx900.ll --input=%t.gfx906.ll --output=%t.bundle.ll
// RUN: rm -fr %t.cache
// RUN: export AMD_COMGR_CACHE=1
// RUN: export AMD_COMGR_CACHE_DIR=%t.cache
// RUN: export AMD_COMGR_EMIT_VERBOSE_LOGS=1
// RUN: export AMD_COMGR_REDIRECT_LOGS=stderr
// RUN: unbundle --link %t.bundle.ll \
// RUN:   hip-amdgcn-amd-amdhsa-unknown-gfx900 %t.miss.0.bc \
// RUN:   hip-amdgcn-amd-amdhsa-unknown-gfx906 %t.miss.1.bc 2> %t.miss.log
// RUN: %FileCheck --check-prefix=MISS %s < %t.miss.log
// RUN: unbundle --link %t.bundle.ll \
// RUN:   hip-amdgcn-amd-amdhsa-unknown-gfx900 %t.hit.0.bc \
// RUN:   hip-amdgcn-amd-amdhsa-unknown-gfx906 %t.hit.1.bc 2> %t.hit.log
// RUN: %FileCheck --check-prefix=HIT %s < %t.hit.log
// RUN: cmp %t.miss.0.bc %t.hit.0.bc
// RUN: cmp %t.miss.1.bc %t.hit.1.bc
// RUN: %llvm-dis %t.hit.0.bc -o - | %FileCheck --check-prefix=GFX900 %s
// RUN: %llvm-dis %t.hit.1.bc -o - | %FileCheck --check-prefix=GFX906 %s
// MISS: Comgr cache: stored entry
// HIT: Comgr cache: found entry
// GFX900: "target-cpu"="gfx900"
// GFX906: "target-cpu"="gfx906"

__attribute__((device)) void add_value(float *a, float *b, float *res) {
  *res = *a + *b;
}
