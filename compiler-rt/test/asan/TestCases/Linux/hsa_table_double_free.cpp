// A second free of device memory from a registered table is reported.
// REQUIRES: asan-64-bits
// UNSUPPORTED: android
// RUN: %clangxx_asan -O0 -I%p/../../../../lib %s -o %t && not %run %t 2>&1 | FileCheck %s

#include "hsa_table_fake.h"

int main() {
  if (!fake::Register())
    return 1;
  void *p = nullptr;
  if (__asan_hsa_amd_memory_pool_allocate(fake::kPool, 100, 0, &p) !=
      HSA_STATUS_SUCCESS)
    return 2;
  __asan_hsa_amd_memory_pool_free(p);
  __asan_hsa_amd_memory_pool_free(p);
  // CHECK: ERROR: AddressSanitizer: attempting double-free
  // CHECK: #0 {{.*}} in __asan_hsa_amd_memory_pool_free
  return 0;
}
