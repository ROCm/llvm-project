// Cached ELF slices must support metadata and ISA parsing after the action.
// RUN: %clang -c -x hip --offload-arch=gfx900 -nogpulib -nogpuinc \
// RUN:   --gpu-bundle-output --offload-device-only --offload-compress %s -o %t.co
// RUN: rm -fr %t.cache
// RUN: export AMD_COMGR_CACHE=1
// RUN: export AMD_COMGR_CACHE_DIR=%t.cache
// RUN: export AMD_COMGR_EMIT_VERBOSE_LOGS=1
// RUN: export AMD_COMGR_REDIRECT_LOGS=stderr
// RUN: unbundle --object %t.co hip-amdgcn-amd-amdhsa-unknown-gfx900 %t.miss.o
// RUN: unbundle --object %t.co hip-amdgcn-amd-amdhsa-unknown-gfx900 %t.hit.o \
// RUN:   2> %t.hit.log
// RUN: %FileCheck %s < %t.hit.log
// RUN: cmp %t.miss.o %t.hit.o
// CHECK: Comgr cache: found entry

__attribute__((global)) void add_value(float *a, float *b, float *res) {
  *res = *a + *b;
}
