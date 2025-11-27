//===-- sanitizer_allocator_amdgpu.cpp --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Part of the Sanitizer Allocator.
//
//===----------------------------------------------------------------------===//
#if SANITIZER_AMDGPU
#  include <dlfcn.h>  // For dlsym
#  include "sanitizer_allocator.h"
#  include "sanitizer_atomic.h"

namespace __sanitizer {
struct HsaFunctions {
  // ---------------- Memory Functions ----------------
  hsa_status_t (*memory_pool_allocate)(hsa_amd_memory_pool_t memory_pool,
                                       size_t size, uint32_t flags, void **ptr);
  hsa_status_t (*memory_pool_free)(void *ptr);
  hsa_status_t (*pointer_info)(void *ptr, hsa_amd_pointer_info_t *info,
                               void *(*alloc)(size_t),
                               uint32_t *num_agents_accessible,
                               hsa_agent_t **accessible);
  hsa_status_t (*vmem_address_reserve_align)(void** ptr, size_t size,
                                             uint64_t address,
                                             uint64_t alignment,
                                             uint64_t flags);
  hsa_status_t (*vmem_address_free)(void *ptr, size_t size);

  // ----------------Event Functions ----------------
  hsa_status_t (*register_system_event_handler)(
      hsa_amd_system_event_callback_t callback, void *data);
};

static HsaFunctions hsa_amd;

// Always align to page boundary to match current ROCr behavior
static const size_t kPageSize_ = 4096;

static atomic_uint8_t amdgpu_runtime_shutdown{0};
static atomic_uint8_t amdgpu_event_registered{0};

// Check if AMDGPU runtime shutdown state
bool AmdgpuMemFuncs::IsAmdgpuRuntimeShutdown() {
  return static_cast<bool>(
      atomic_load(&amdgpu_runtime_shutdown, memory_order_acquire));
}

// Notify AMDGPU runtime shutdown to allocator
void AmdgpuMemFuncs::NotifyAmdgpuRuntimeShutdown() {
  uint8_t shutdown = 0;
  if (atomic_compare_exchange_strong(&amdgpu_runtime_shutdown, &shutdown, 1,
                                     memory_order_acq_rel)) {
    VReport(1, " Amdgpu Allocator: AMDGPU runtime shutdown detected\n");
  }
}

bool AmdgpuMemFuncs::Init() {
  hsa_amd.memory_pool_allocate =
      (decltype(hsa_amd.memory_pool_allocate))dlsym(
          RTLD_NEXT, "hsa_amd_memory_pool_allocate");
  hsa_amd.memory_pool_free = (decltype(hsa_amd.memory_pool_free))dlsym(
      RTLD_NEXT, "hsa_amd_memory_pool_free");
  hsa_amd.pointer_info = (decltype(hsa_amd.pointer_info))dlsym(
      RTLD_NEXT, "hsa_amd_pointer_info");
  hsa_amd.vmem_address_reserve_align =
      (decltype(hsa_amd.vmem_address_reserve_align))dlsym(
          RTLD_NEXT, "hsa_amd_vmem_address_reserve_align");
  hsa_amd.vmem_address_free = (decltype(hsa_amd.vmem_address_free))dlsym(
      RTLD_NEXT, "hsa_amd_vmem_address_free");
  hsa_amd.register_system_event_handler =
      (decltype(hsa_amd.register_system_event_handler))dlsym(
          RTLD_NEXT, "hsa_amd_register_system_event_handler");
  if (!hsa_amd.memory_pool_allocate || !hsa_amd.memory_pool_free ||
      !hsa_amd.pointer_info || !hsa_amd.vmem_address_reserve_align ||
      !hsa_amd.vmem_address_free || !hsa_amd.register_system_event_handler)
    return false;
  return true;
}

void *AmdgpuMemFuncs::Allocate(uptr size, uptr alignment,
                               DeviceAllocationInfo *da_info) {
  // Do not allocate if AMDGPU runtime is shutdown
  if (IsAmdgpuRuntimeShutdown())
    return nullptr;
  AmdgpuAllocationInfo *aa_info =
      reinterpret_cast<AmdgpuAllocationInfo *>(da_info);
  if (!aa_info->memory_pool.handle) {
    aa_info->status = hsa_amd.vmem_address_reserve_align(
        &aa_info->ptr, size, aa_info->address, aa_info->alignment,
        aa_info->flags64);
  } else {
    aa_info->status = hsa_amd.memory_pool_allocate(
        aa_info->memory_pool, size, aa_info->flags, &aa_info->ptr);
  }
  if (aa_info->status != HSA_STATUS_SUCCESS)
    return nullptr;

  return aa_info->ptr;
}

void AmdgpuMemFuncs::Deallocate(void *p) {
  // Deallocate does nothing after AMDGPU runtime shutdown
  if (IsAmdgpuRuntimeShutdown())
    return;
  DevicePointerInfo DevPtrInfo;
  if (AmdgpuMemFuncs::GetPointerInfo(reinterpret_cast<uptr>(p), &DevPtrInfo)) {
    if (DevPtrInfo.type == HSA_EXT_POINTER_TYPE_HSA) {
      UNUSED hsa_status_t status = hsa_amd.memory_pool_free(p);
    } else if (DevPtrInfo.type == HSA_EXT_POINTER_TYPE_RESERVED_ADDR) {
      UNUSED hsa_status_t status =
          hsa_amd.vmem_address_free(p, DevPtrInfo.map_size);
    }
  }
}

bool AmdgpuMemFuncs::GetPointerInfo(uptr ptr, DevicePointerInfo* ptr_info) {
  hsa_amd_pointer_info_t info;
  info.size = sizeof(hsa_amd_pointer_info_t);
  hsa_status_t status =
    hsa_amd.pointer_info(reinterpret_cast<void *>(ptr), &info, 0, 0, 0);

  if (status != HSA_STATUS_SUCCESS)
    return false;

  if (info.type == HSA_EXT_POINTER_TYPE_RESERVED_ADDR)
    ptr_info->map_beg = reinterpret_cast<uptr>(info.hostBaseAddress);
  else if (info.type == HSA_EXT_POINTER_TYPE_HSA)
    ptr_info->map_beg = reinterpret_cast<uptr>(info.agentBaseAddress);
  ptr_info->map_size = info.sizeInBytes;
  ptr_info->type = reinterpret_cast<hsa_amd_pointer_type_t>(info.type);

  return true;
}
 // Register shutdown system event handler only once
 // TODO: Register multiple event handlers if needed in future
void AmdgpuMemFuncs::RegisterSystemEventHandlers() {
  // Check if already registered
  if (atomic_load(&amdgpu_event_registered, memory_order_acquire) == 0) {
    // Callback to just detect runtime shutdown
    hsa_amd_system_event_callback_t callback = [](const hsa_amd_event_t* event,
                                                  void* data) {
      if (!event)
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
      if (event->event_type == HSA_AMD_SYSTEM_SHUTDOWN_EVENT)
        AmdgpuMemFuncs::NotifyAmdgpuRuntimeShutdown();
      return HSA_STATUS_SUCCESS;
    };
    // Register the callback
    hsa_status_t status =
        hsa_amd.register_system_event_handler(callback, nullptr);
    // Mark as registered if successful
    if (status == HSA_STATUS_SUCCESS)
      atomic_store(&amdgpu_event_registered, 1, memory_order_release);
  }
}

uptr AmdgpuMemFuncs::GetPageSize() { return kPageSize_; }
}  // namespace __sanitizer
#endif  // SANITIZER_AMDGPU
