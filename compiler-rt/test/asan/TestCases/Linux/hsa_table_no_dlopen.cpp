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
  char dst[8], src[8];
  hsa_agent_t agent = {};
  hsa_signal_t signal = {};
  // CHECK: hsa_memory_copy without ROCr: 1
  printf("hsa_memory_copy without ROCr: %d\n",
         hsa_memory_copy(dst, src, sizeof(src)) ==
             HSA_STATUS_ERROR_NOT_INITIALIZED);
  // CHECK: async copy without ROCr: 1
  printf("async copy without ROCr: %d\n",
         hsa_amd_memory_async_copy(dst, agent, src, agent, sizeof(src), 0,
                                   nullptr,
                                   signal) == HSA_STATUS_ERROR_NOT_INITIALIZED);
  // CHECK: async copy on engine without ROCr: 1
  printf("async copy on engine without ROCr: %d\n",
         hsa_amd_memory_async_copy_on_engine(
             dst, agent, src, agent, sizeof(src), 0, nullptr, signal,
             HSA_AMD_SDMA_ENGINE_0, false) == HSA_STATUS_ERROR_NOT_INITIALIZED);
  // CHECK: ROCr loaded: 0
  printf("ROCr loaded: %d\n",
         dlopen(argv[1], RTLD_LAZY | RTLD_NOLOAD) != nullptr);
  return 0;
}
