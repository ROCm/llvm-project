// Freed device memory from a registered table stays poisoned in quarantine.
// REQUIRES: asan-64-bits
// UNSUPPORTED: android
// RUN: %clangxx_asan -O0 -I%p/../../../../lib %s -o %t && not %run %t 2>&1 | FileCheck %s

#include "hsa_table_fake.h"

int main() {
  if (!fake::Register())
    return 1;
  char *p = nullptr;
  if (__asan_hsa_amd_memory_pool_allocate(fake::kPool, 100, 0, (void **)&p) !=
      HSA_STATUS_SUCCESS)
    return 2;
  if (__asan_hsa_amd_memory_pool_free(p) != HSA_STATUS_SUCCESS)
    return 3;
  return p[10];
  // CHECK: ERROR: AddressSanitizer: heap-use-after-free
  // CHECK: freed by thread T0 here:
  // CHECK: #0 {{.*}} in __asan_hsa_amd_memory_pool_free
  // CHECK: previously allocated by thread T0 here:
  // CHECK: #0 {{.*}} in __asan_hsa_amd_memory_pool_allocate
}
