// Device memory from a registered table gets redzones and allocation stacks.
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
  p[100] = 1;
  // CHECK: ERROR: AddressSanitizer: heap-buffer-overflow
  // CHECK: WRITE of size 1
  // CHECK: 0 bytes after 100-byte region
  // CHECK: allocated by thread T0 here:
  // CHECK: #0 {{.*}} in __asan_hsa_amd_memory_pool_allocate
  return 0;
}
