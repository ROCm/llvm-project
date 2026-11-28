// A host-memory backend registered through __sanitizer_amdgpu_set_hsa_table,
// standing in for a GPU runtime that exports no HSA API.
#include "sanitizer_common/sanitizer_offload_hsa.h"
#include <sanitizer/amdgpu_interface.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>

namespace fake {
enum Kind { kNoBlock, kPoolBlock, kReservedBlock, kIpcBlock };
struct Block {
  char *base;
  size_t size;
  Kind kind;
};
static Block blocks[64];
static int allocs, pool_frees, vmem_frees;
static void *ipc_created_ptr, *ipc_detached_ptr, *reserved_ptr;
static const void *allowed_ptr;
static size_t ipc_created_len, ipc_attached_len;
static hsa_amd_system_event_callback_t event_handler;
static void *event_data;

static Block *Find(const void *p) {
  for (Block &b : blocks)
    if (b.base && (const char *)p >= b.base &&
        (const char *)p < b.base + b.size)
      return &b;
  return nullptr;
}

static hsa_status_t Map(void **ptr, size_t size, Kind kind) {
  void *p = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED)
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  for (Block &b : blocks)
    if (!b.base) {
      b = {(char *)p, size, kind};
      break;
    }
  *ptr = p;
  if (kind != kIpcBlock)
    ++allocs;
  return HSA_STATUS_SUCCESS;
}

static hsa_status_t Unmap(void *ptr, Kind kind) {
  Block *b = Find(ptr);
  if (!b || b->base != ptr || b->kind != kind)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  munmap(b->base, b->size);
  *b = {};
  if (kind == kPoolBlock)
    ++pool_frees;
  else if (kind == kReservedBlock)
    ++vmem_frees;
  return HSA_STATUS_SUCCESS;
}

static hsa_status_t Init() { return HSA_STATUS_SUCCESS; }
static hsa_status_t MemoryCopy(void *dst, const void *src, size_t size) {
  memcpy(dst, src, size);
  return HSA_STATUS_SUCCESS;
}
static hsa_status_t PoolAllocate(hsa_amd_memory_pool_t, size_t size, uint32_t,
                                 void **ptr) {
  return Map(ptr, size, kPoolBlock);
}
static hsa_status_t PoolFree(void *ptr) { return Unmap(ptr, kPoolBlock); }
static hsa_status_t AllowAccess(uint32_t, const hsa_agent_t *, const uint32_t *,
                                const void *ptr) {
  allowed_ptr = ptr;
  return HSA_STATUS_SUCCESS;
}
static hsa_status_t AsyncCopy(void *dst, hsa_agent_t, const void *src,
                              hsa_agent_t, size_t size, uint32_t,
                              const hsa_signal_t *, hsa_signal_t) {
  return MemoryCopy(dst, src, size);
}
static hsa_status_t AsyncCopyOnEngine(void *dst, hsa_agent_t, const void *src,
                                      hsa_agent_t, size_t size, uint32_t,
                                      const hsa_signal_t *, hsa_signal_t,
                                      hsa_amd_sdma_engine_id_t, bool) {
  return MemoryCopy(dst, src, size);
}
static hsa_status_t IpcCreate(void *ptr, size_t len, hsa_amd_ipc_memory_t *) {
  ipc_created_ptr = ptr;
  ipc_created_len = len;
  return HSA_STATUS_SUCCESS;
}
static hsa_status_t IpcAttach(const hsa_amd_ipc_memory_t *, size_t len,
                              uint32_t, const hsa_agent_t *, void **ptr) {
  ipc_attached_len = len;
  return Map(ptr, (len + 4095) & ~size_t{4095}, kIpcBlock);
}
static hsa_status_t IpcDetach(void *ptr) {
  ipc_detached_ptr = ptr;
  return Unmap(ptr, kIpcBlock);
}
static hsa_status_t VmemReserve(void **ptr, size_t size, uint64_t, uint64_t,
                                uint64_t) {
  const hsa_status_t status = Map(ptr, size, kReservedBlock);
  reserved_ptr = *ptr;
  return status;
}
static hsa_status_t VmemFree(void *ptr, size_t) {
  return Unmap(ptr, kReservedBlock);
}
static hsa_status_t PointerInfo(const void *ptr, hsa_amd_pointer_info_t *info,
                                void *(*)(size_t), uint32_t *, hsa_agent_t **) {
  const uint32_t info_size = info->size;
  memset(info, 0, info_size);
  info->size = info_size;
  Block *b = Find(ptr);
  if (!b) {
    info->type = HSA_EXT_POINTER_TYPE_UNKNOWN;
    return HSA_STATUS_SUCCESS;
  }
  info->type = b->kind == kReservedBlock ? HSA_EXT_POINTER_TYPE_RESERVED_ADDR
               : b->kind == kIpcBlock    ? HSA_EXT_POINTER_TYPE_IPC
                                         : HSA_EXT_POINTER_TYPE_HSA;
  info->agentBaseAddress = b->base;
  info->hostBaseAddress = b->base;
  info->sizeInBytes = b->size;
  return HSA_STATUS_SUCCESS;
}
static hsa_status_t RegisterEventHandler(hsa_amd_system_event_callback_t cb,
                                         void *data) {
  event_handler = cb;
  event_data = data;
  return HSA_STATUS_SUCCESS;
}

static __sanitizer_amdgpu_hsa_table Table() {
  __sanitizer_amdgpu_hsa_table t;
  memset(&t, 0, sizeof(t));
  t.size = sizeof(t);
  t.hsa_init = Init;
  t.hsa_memory_copy = MemoryCopy;
  t.hsa_amd_memory_pool_allocate = PoolAllocate;
  t.hsa_amd_memory_pool_free = PoolFree;
  t.hsa_amd_agents_allow_access = AllowAccess;
  t.hsa_amd_memory_async_copy = AsyncCopy;
  t.hsa_amd_memory_async_copy_on_engine = AsyncCopyOnEngine;
  t.hsa_amd_ipc_memory_create = IpcCreate;
  t.hsa_amd_ipc_memory_attach = IpcAttach;
  t.hsa_amd_ipc_memory_detach = IpcDetach;
  t.hsa_amd_vmem_address_reserve_align = VmemReserve;
  t.hsa_amd_vmem_address_free = VmemFree;
  t.hsa_amd_pointer_info = PointerInfo;
  t.hsa_amd_register_system_event_handler = RegisterEventHandler;
  return t;
}

// Registers the table and initializes as a runtime would; returns false on
// failure.
static bool Register() {
  __sanitizer_amdgpu_hsa_table t = Table();
  return __sanitizer_amdgpu_set_hsa_table(&t) &&
         __asan_hsa_init() == HSA_STATUS_SUCCESS;
}

static void RaiseShutdown() {
  hsa_amd_event_t event;
  memset(&event, 0, sizeof(event));
  event.event_type = HSA_AMD_SYSTEM_SHUTDOWN_EVENT;
  event_handler(&event, event_data);
}

static const hsa_amd_memory_pool_t kPool = {1};
} // namespace fake
