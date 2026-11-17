// RUN: %clangxx_asan -O0 -DSHARED_LIB -isystem %rocm_include %s -fPIC -shared \
// RUN:   -o %t-so.so -L%rocm_lib -lhsa-runtime64 -Wl,-rpath,%rocm_lib
// RUN: %clangxx_asan -O0 %s -o %t -Wl,-rpath,%compiler_rt_libdir
// RUN: not %run %t %t-so.so 2>&1 | FileCheck %s
//
// Regression test for HSA interception when ROCr is only reachable from an
// RTLD_LOCAL dlopen group (e.g. Python importing a HIP extension). RTLD_NEXT
// from the ASan runtime cannot see such a ROCr, yet the HSA interceptors still
// win symbol resolution; calling hsa_init must not jump to a null REAL().
//
// REQUIRES: linux, stable-runtime, rocm
// UNSUPPORTED: android

#ifdef SHARED_LIB

#  include "hsa_amd_test_helpers.h"

#  include <hsa/hsa.h>
#  include <hsa/hsa_ext_amd.h>

#  include <stdio.h>

extern "C" int run() {
  HSA_CHECK(hsa_init());
  fprintf(stderr, "hsa_init ok\n");

  HsaAmdPoolSearch ps;
  if (hsa_amd_test_find_first_runtime_alloc_pool(&ps))
    return 1;

  void *mem = nullptr;
  HSA_CHECK(hsa_amd_memory_pool_allocate(ps.pool, 64, 0, &mem));

  (void)hsa_amd_memory_pool_free(mem);
  (void)hsa_amd_memory_pool_free(mem);
  return 0;
}

#else

#  include <dlfcn.h>
#  include <stdio.h>

int main(int argc, char **argv) {
  void *h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (!h) {
    fprintf(stderr, "dlopen failed: %s\n", dlerror());
    return 1;
  }
  int (*run)() = (int (*)())dlsym(h, "run");
  return run();
}

#endif

// CHECK-NOT: SEGV on unknown address
// CHECK: hsa_init ok
// CHECK: ERROR: AddressSanitizer: attempting double-free
// CHECK: SUMMARY: AddressSanitizer: double-free {{.*}}hsa_rtld_local_dlopen.cpp:35:{{[0-9]+}} in run
