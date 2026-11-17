// Registration, pointer info, pass-through, IPC and shutdown rules of the
// table.
// REQUIRES: asan-64-bits
// UNSUPPORTED: android
// RUN: %clangxx_asan -O0 -I%p/../../../../lib %s -o %t
// RUN: env SANITIZER_HSA_RUNTIME_PATH=/nonexistent/libhsa-runtime64.so.1 %run %t 2>&1 | FileCheck %s

#include "hsa_table_fake.h"
#include <sanitizer/allocator_interface.h>
#include <sanitizer/asan_interface.h>
#include <stdint.h>

static hsa_amd_pointer_info_t Info(const void *p) {
  hsa_amd_pointer_info_t info;
  info.size = sizeof(info);
  if (__asan_hsa_amd_pointer_info(p, &info, nullptr, nullptr, nullptr) !=
      HSA_STATUS_SUCCESS)
    info.type = (hsa_amd_pointer_type_t)-1;
  return info;
}

int main() {
  // With no ROCr behind it, an intercepted call fails and refuses no
  // registration.
  // CHECK: hsa_init without ROCr: 1
  printf("hsa_init without ROCr: %d\n",
         hsa_init() == HSA_STATUS_ERROR_NOT_INITIALIZED);

  void *p = nullptr;
  // CHECK: unregistered: 1
  printf("unregistered: %d\n",
         __asan_hsa_amd_memory_pool_allocate(fake::kPool, 100, 0, &p) ==
             HSA_STATUS_ERROR_NOT_INITIALIZED);

  __sanitizer_amdgpu_hsa_table t = fake::Table();
  t.hsa_amd_pointer_info = nullptr;
  // CHECK: incomplete: 0
  printf("incomplete: %d\n", __sanitizer_amdgpu_set_hsa_table(&t));
  t = fake::Table();
  t.size = sizeof(t) - sizeof(void *);
  // CHECK: short: 0
  printf("short: %d\n", __sanitizer_amdgpu_set_hsa_table(&t));

  // CHECK: registered: 1
  printf("registered: %d\n", fake::Register());
  // CHECK: shutdown handler: 1
  printf("shutdown handler: %d\n", fake::event_handler != nullptr);
  t = fake::Table();
  // CHECK: same again: 1
  printf("same again: %d\n", __sanitizer_amdgpu_set_hsa_table(&t));
  t.hsa_init = [] { return HSA_STATUS_SUCCESS; };
  // CHECK: different: 0
  printf("different: %d\n", __sanitizer_amdgpu_set_hsa_table(&t));

  if (__asan_hsa_amd_memory_pool_allocate(fake::kPool, 100, 0, &p) !=
      HSA_STATUS_SUCCESS)
    return 1;
  hsa_amd_pointer_info_t info = Info(p);
  // CHECK: allocs 1, page aligned 1, user range 1
  printf("allocs %d, page aligned %d, user range %d\n", fake::allocs,
         (uintptr_t)p % 4096 == 0,
         info.type == HSA_EXT_POINTER_TYPE_HSA && info.agentBaseAddress == p &&
             info.sizeInBytes == 100);

  // CHECK: allow access to the block: 1
  __asan_hsa_amd_agents_allow_access(0, nullptr, nullptr, p);
  printf("allow access to the block: %d\n",
         fake::allowed_ptr == (char *)p - 4096);

  // The weak hsa_* exports pass through to a real ROCr, of which there is none.
  void *q = nullptr;
  hsa_status_t direct = hsa_amd_memory_pool_allocate(fake::kPool, 100, 0, &q);
  // CHECK: direct call: 1, allocs 1
  printf("direct call: %d, allocs %d\n",
         direct == HSA_STATUS_ERROR_NOT_INITIALIZED, fake::allocs);

  // IPC exports the whole block and maps an extra page in front of an import.
  hsa_amd_ipc_memory_t handle;
  if (__asan_hsa_amd_ipc_memory_create(p, 100, &handle) != HSA_STATUS_SUCCESS)
    return 2;
  // CHECK: exported block: 1, whole length: 1
  printf("exported block: %d, whole length: %d\n",
         fake::ipc_created_ptr == (char *)p - 4096,
         fake::ipc_created_len == fake::Find(p)->size);
  char *u = nullptr;
  if (__asan_hsa_amd_ipc_memory_attach(&handle, 100, 0, nullptr, (void **)&u) !=
      HSA_STATUS_SUCCESS)
    return 3;
  // CHECK: attached length 4196, header 1, user 0, tail 1
  printf("attached length %zu, header %d, user %d, tail %d\n",
         fake::ipc_attached_len, __asan_address_is_poisoned(u - 1),
         __asan_address_is_poisoned(u + 99),
         __asan_address_is_poisoned(u + 100));
  if (__asan_hsa_amd_ipc_memory_detach(u) != HSA_STATUS_SUCCESS)
    return 4;
  // CHECK: detached mapping: 1, header 0
  printf("detached mapping: %d, header %d\n",
         fake::ipc_detached_ptr == u - 4096, __asan_address_is_poisoned(u - 1));

  void *r = nullptr;
  if (__asan_hsa_amd_vmem_address_reserve_align(
          &r, 8192, 0, 0, HSA_AMD_VMEM_ADDRESS_NO_REGISTER) !=
      HSA_STATUS_SUCCESS)
    return 5;
  // CHECK: reserved: 1, freed: 1
  printf("reserved: %d, freed: %d\n", (uintptr_t)r % 4096 == 0,
         __asan_hsa_amd_vmem_address_free(r, 8192) == HSA_STATUS_SUCCESS);

  if (__asan_hsa_amd_memory_pool_free(p) != HSA_STATUS_SUCCESS)
    return 6;
  // CHECK: freed is unknown: 1
  printf("freed is unknown: %d\n",
         Info(p).type == HSA_EXT_POINTER_TYPE_UNKNOWN);

  // The purge hands quarantined blocks back to the table, each to the function
  // matching its pointer type.
  __sanitizer_purge_allocator();
  // CHECK: purged: pool 1, vmem 1
  printf("purged: pool %d, vmem %d\n", fake::pool_frees, fake::vmem_frees);

  // A reservation at a fixed address goes straight to the table, unguarded.
  void *f = nullptr;
  if (__asan_hsa_amd_vmem_address_reserve_align(&f, 8192, 0x10000000, 0, 0) !=
      HSA_STATUS_SUCCESS)
    return 8;
  const int unguarded = f == fake::reserved_ptr;
  // CHECK: fixed address: unguarded 1, freed at once 1
  printf("fixed address: unguarded %d, freed at once %d\n", unguarded,
         __asan_hsa_amd_vmem_address_free(f, 8192) == HSA_STATUS_SUCCESS &&
             fake::vmem_frees == 2);

  // After the shutdown event ASan stops calling the runtime; the runtime frees
  // what is left in quarantine.
  if (__asan_hsa_amd_memory_pool_allocate(fake::kPool, 100, 0, &p) !=
          HSA_STATUS_SUCCESS ||
      __asan_hsa_amd_memory_pool_free(p) != HSA_STATUS_SUCCESS)
    return 7;
  fake::RaiseShutdown();
  __sanitizer_purge_allocator();
  // CHECK: frees after shutdown: 0
  printf("frees after shutdown: %d\n", fake::pool_frees - 1);
  return 0;
}
