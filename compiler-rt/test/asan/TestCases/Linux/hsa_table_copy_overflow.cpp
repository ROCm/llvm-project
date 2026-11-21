// The copy entry points check host-visible ranges like the interceptors do.
// REQUIRES: asan-64-bits
// UNSUPPORTED: android
// RUN: %clangxx_asan -O0 -I%p/../../../../lib %s -o %t
// RUN: not %run %t copy 2>&1 | FileCheck %s
// RUN: not %run %t async 2>&1 | FileCheck %s
// RUN: not %run %t engine 2>&1 | FileCheck %s

#include "hsa_table_fake.h"
#include <stdlib.h>

int main(int argc, char **argv) {
  if (argc < 2 || !fake::Register())
    return 1;
  char *src = (char *)malloc(10);
  char *dst = (char *)malloc(20);
  const hsa_agent_t agent = {0};
  const hsa_signal_t signal = {0};
  if (!strcmp(argv[1], "copy"))
    __asan_hsa_memory_copy(dst, src, 20);
  else if (!strcmp(argv[1], "async"))
    __asan_hsa_amd_memory_async_copy(dst, agent, src, agent, 20, 0, nullptr,
                                     signal);
  else
    __asan_hsa_amd_memory_async_copy_on_engine(dst, agent, src, agent, 20, 0,
                                               nullptr, signal,
                                               HSA_AMD_SDMA_ENGINE_0, false);
  // CHECK: ERROR: AddressSanitizer: heap-buffer-overflow
  // CHECK: READ of size 20
  // CHECK: 0 bytes after 10-byte region
  return 0;
}
