// With a runtime's table registered, an intercepted HSA call doesn't load a
// ROCr: the one named by SANITIZER_HSA_RUNTIME_PATH stays unloaded.
// REQUIRES: asan-64-bits
// UNSUPPORTED: android
// RUN: %clangxx -DFAKE_ROCR -fPIC -shared -I%p/../../../../lib %p/hsa_table_rocr.cpp -o %t-rocr.so
// RUN: %clangxx_asan -O0 -I%p/../../../../lib %s -o %t
// RUN: env SANITIZER_HSA_RUNTIME_PATH=%t-rocr.so %run %t %t-rocr.so 2>&1 | FileCheck %s

#include "hsa_table_fake.h"
#include <dlfcn.h>

int main(int argc, char **argv) {
  if (argc < 2 || !fake::Register())
    return 1;
  // CHECK: hsa_init without ROCr: 1
  printf("hsa_init without ROCr: %d\n",
         hsa_init() == HSA_STATUS_ERROR_NOT_INITIALIZED);
  // CHECK: ROCr loaded: 0
  printf("ROCr loaded: %d\n",
         dlopen(argv[1], RTLD_LAZY | RTLD_NOLOAD) != nullptr);
  return 0;
}
