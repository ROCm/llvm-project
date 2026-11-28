// A ROCr in the same process: once a table is registered its calls pass through
// the interceptors untouched, and once its calls went through the interceptors
// registration fails and the interceptors keep tracking its memory.
// REQUIRES: asan-64-bits
// UNSUPPORTED: android
// RUN: %clangxx -DFAKE_ROCR -fPIC -shared -I%p/../../../../lib %s -o %t-rocr.so
// RUN: %clangxx_asan -O0 -I%p/../../../../lib %s %t-rocr.so -o %t
// RUN: env SANITIZER_HSA_RUNTIME_PATH=%t-rocr.so %run %t register-first 2>&1 | FileCheck %s --check-prefix=FIRST
// RUN: env SANITIZER_HSA_RUNTIME_PATH=%t-rocr.so %run %t rocr-first 2>&1 | FileCheck %s --check-prefix=ROCR

#include "hsa_table_fake.h"

#ifdef FAKE_ROCR
extern "C" {
int rocr_inits, rocr_allocs, rocr_handlers;

hsa_status_t hsa_init() {
  ++rocr_inits;
  return HSA_STATUS_SUCCESS;
}
hsa_status_t hsa_amd_memory_pool_allocate(hsa_amd_memory_pool_t pool,
                                          size_t size, uint32_t flags,
                                          void **ptr) {
  ++rocr_allocs;
  return fake::PoolAllocate(pool, size, flags, ptr);
}
hsa_status_t hsa_amd_memory_pool_free(void *ptr) { return fake::PoolFree(ptr); }
hsa_status_t hsa_amd_pointer_info(const void *ptr, hsa_amd_pointer_info_t *info,
                                  void *(*alloc)(size_t),
                                  uint32_t *num_agents_accessible,
                                  hsa_agent_t **accessible) {
  return fake::PointerInfo(ptr, info, alloc, num_agents_accessible, accessible);
}
hsa_status_t hsa_amd_vmem_address_reserve_align(void **ptr, size_t size,
                                                uint64_t address,
                                                uint64_t alignment,
                                                uint64_t flags) {
  return fake::VmemReserve(ptr, size, address, alignment, flags);
}
hsa_status_t hsa_amd_vmem_address_free(void *ptr, size_t size) {
  return fake::VmemFree(ptr, size);
}
hsa_status_t
hsa_amd_register_system_event_handler(hsa_amd_system_event_callback_t callback,
                                      void *data) {
  ++rocr_handlers;
  return fake::RegisterEventHandler(callback, data);
}
}
#else
#  include <sanitizer/asan_interface.h>

extern "C" int rocr_inits, rocr_allocs, rocr_handlers;

int main(int argc, char **argv) {
  const bool register_first = argc > 1 && !strcmp(argv[1], "register-first");
  if (register_first) {
    // FIRST: registered: 1
    printf("registered: %d\n", fake::Register());
  }
  if (hsa_init() != HSA_STATUS_SUCCESS)
    return 1;
  if (!register_first) {
    __sanitizer_amdgpu_hsa_table t = fake::Table();
    // ROCR: registered: 0
    printf("registered: %d\n", __sanitizer_amdgpu_set_hsa_table(&t));
    // ROCR: unregistered entry point: 1
    printf("unregistered entry point: %d\n",
           __asan_hsa_init() == HSA_STATUS_ERROR_NOT_INITIALIZED);
  }
  char *p = nullptr;
  if (hsa_amd_memory_pool_allocate(fake::kPool, 100, 0, (void **)&p) !=
      HSA_STATUS_SUCCESS)
    return 2;
  // FIRST: rocr: inits 1, allocs 1, shutdown handlers 0, redzone 0
  // ROCR: rocr: inits 1, allocs 1, shutdown handlers 1, redzone 1
  printf("rocr: inits %d, allocs %d, shutdown handlers %d, redzone %d\n",
         rocr_inits, rocr_allocs, rocr_handlers,
         __asan_address_is_poisoned(p + 100));
  if (!register_first) {
    hsa_amd_pointer_info_t info;
    info.size = sizeof(info);
    // ROCR: rocr pointer info: user range 1
    printf("rocr pointer info: user range %d\n",
           hsa_amd_pointer_info(p, &info, nullptr, nullptr, nullptr) ==
                   HSA_STATUS_SUCCESS &&
               info.agentBaseAddress == p && info.sizeInBytes == 100);
  }
  if (register_first) {
    char *q = nullptr;
    if (__asan_hsa_amd_memory_pool_allocate(fake::kPool, 100, 0, (void **)&q) !=
        HSA_STATUS_SUCCESS)
      return 3;
    // FIRST: table: allocs 1, redzone 1
    printf("table: allocs %d, redzone %d\n", fake::allocs,
           __asan_address_is_poisoned(q + 100));
    if (__asan_hsa_amd_memory_pool_free(q) != HSA_STATUS_SUCCESS)
      return 4;
  }
  return hsa_amd_memory_pool_free(p) == HSA_STATUS_SUCCESS ? 0 : 5;
}
#endif
