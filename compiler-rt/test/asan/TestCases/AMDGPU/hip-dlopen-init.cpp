// RUN: %clangxx_asan %s -ldl -o %t && %ROCM_ENV && %run %t 2>&1 | FileCheck %s
// CHECK: hipInit succeeded

// Verify that ASan can initialize before HIP and resolve the HSA allocator
// callbacks after libamdhip64 loads ROCr.

#include <dlfcn.h>
#include <stdio.h>

int main() {
  void *hip = dlopen("libamdhip64.so.7", RTLD_NOW | RTLD_GLOBAL);
  if (!hip) {
    fprintf(stderr, "dlopen failed: %s\n", dlerror());
    return 1;
  }

  using HipInit = int (*)(unsigned int);
  auto hip_init = reinterpret_cast<HipInit>(dlsym(hip, "hipInit"));
  if (!hip_init) {
    fprintf(stderr, "dlsym failed: %s\n", dlerror());
    return 1;
  }

  const int status = hip_init(0);
  if (status != 0) {
    fprintf(stderr, "hipInit failed: %d\n", status);
    return 1;
  }

  puts("hipInit succeeded");
  return 0;
}
