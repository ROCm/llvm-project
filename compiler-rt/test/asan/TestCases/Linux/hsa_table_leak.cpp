// Leak checking reaches device memory through the table: a block still
// referenced isn't reported, a lost one is, with its allocation stack.
// UNSUPPORTED: android
// RUN: %clangxx_asan -O0 -I%p/../../../../lib %s -o %t
// RUN: %env_asan_opts=detect_leaks=1 %run %t kept 2>&1 | FileCheck %s --check-prefix=KEPT
// RUN: %env_asan_opts=detect_leaks=1 not %run %t lost 2>&1 | FileCheck %s --check-prefix=LOST
// REQUIRES: asan-64-bits, leak-detection

#include "hsa_table_fake.h"

static void *kept;

int main(int argc, char **argv) {
  if (argc < 2 || !fake::Register())
    return 1;
  void *p = nullptr;
  if (__asan_hsa_amd_memory_pool_allocate(fake::kPool, 100, 0, &p) !=
      HSA_STATUS_SUCCESS)
    return 2;
  if (!strcmp(argv[1], "kept"))
    kept = p;
  p = nullptr;
  fprintf(stderr, "done\n");
  return 0;
}

// KEPT: done
// KEPT-NOT: LeakSanitizer
// LOST: done
// LOST: ERROR: LeakSanitizer: detected memory leaks
// LOST: Direct leak of 100 byte(s)
// LOST: #0 {{.*}} in __asan_hsa_amd_memory_pool_allocate
