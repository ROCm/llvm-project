//===-- sanitizer_allocator_amdhsa.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Part of the Sanitizer Allocator. HSA device memory backend for
// DeviceAllocatorT.
//
//===----------------------------------------------------------------------===//
#ifndef SANITIZER_ALLOCATOR_H
#  error This file must be included after sanitizer_allocator.h
#endif

#if SANITIZER_AMDHSA

static const DeviceAllocationType DAT_AMDHSA =
    static_cast<DeviceAllocationType>(1);

// dlopen handle for the loaded ROCr (libhsa-runtime64), or null. Unless
// allow_dlopen, only a ROCr that is already loaded is found.
void* GetHsaRuntimeHandle(bool allow_dlopen = true);

class AmdHsaDeviceAllocator {
 public:
  static constexpr bool kEnableDeviceBackend = true;

  // The ROCr functions the device allocator uses.
  struct HsaFunctions {
    hsa_status_t (*memory_pool_allocate)(hsa_amd_memory_pool_t memory_pool,
                                         size_t size, uint32_t flags,
                                         void** ptr);
    hsa_status_t (*memory_pool_free)(void* ptr);
    hsa_status_t (*pointer_info)(const void* ptr, hsa_amd_pointer_info_t* info,
                                 void* (*alloc)(size_t),
                                 uint32_t* num_agents_accessible,
                                 hsa_agent_t** accessible);
    hsa_status_t (*vmem_address_reserve_align)(void** ptr, size_t size,
                                               uint64_t address,
                                               uint64_t alignment,
                                               uint64_t flags);
    hsa_status_t (*vmem_address_free)(void* ptr, size_t size);
    hsa_status_t (*register_system_event_handler)(
        hsa_amd_system_event_callback_t callback, void* data);
  };

  static bool Init(bool allow_dlopen = false);
  // Uses these functions instead of the ROCr's found by name, which Init then
  // no longer looks for. Only while the device allocator holds no chunk and no
  // thread can be in Allocate, Deallocate or GetPointerInfo.
  static void SetFunctions(const HsaFunctions& fns);
  static void* Allocate(uptr size, uptr alignment,
                        DeviceAllocationInfo* da_info);
  static void Deallocate(void* p);
  static bool GetPointerInfo(uptr ptr, DevicePointerInfo* ptr_info);
  static uptr GetPageSize();
  static void RegisterSystemEventHandlers();
  static bool IsRuntimeShutdown();
  static void ClearRuntimeShutdownState();
  static void NoteDeviceAllocatorFailure(DeviceAllocationInfo* da_info,
                                         DeviceAllocFailure failure);

 private:
  static void NotifyRuntimeShutdown();
};

// GPU-only hsa_amd_vmem_address_reserve_align (without
// HSA_AMD_VMEM_ADDRESS_NO_REGISTER) returns KFD "address only" VA that is not
// host-writable. Sanitizer runtimes that cannot place metadata in the reserved
// range track such reservations here for double-free detection on free.
class VmemGpuReserveTracker {
 public:
  enum FreeResult {
    kNotTracked,
    kFirstFree,
    kSizeMismatch,
    kDoubleFree,
  };

  // Provenance returned for a double-freed reservation so the reporter can
  // print the reserve and first-free stacks (captured as StackDepot ids).
  struct ReservationInfo {
    uptr base;
    uptr size;
    u32 reserve_stack_id;
    u32 reserve_tid;
    u32 first_free_stack_id;
    u32 first_free_tid;
  };

  static VmemGpuReserveTracker& Get();

  void OnReserve(uptr ptr, uptr size, u32 reserve_stack_id, u32 reserve_tid);
  // CheckFree validates without updating state; call MarkFreed only after the
  // real hsa_amd_vmem_address_free succeeds.
  FreeResult CheckFree(uptr ptr, uptr size);
  void MarkFreed(uptr ptr, uptr size, u32 free_stack_id, u32 free_tid);
  // Fills *out for the reservation at ptr. Returns false when untracked.
  bool GetReservationInfo(uptr ptr, ReservationInfo* out);

 private:
  struct VmemGpuReservation {
    uptr ptr;
    uptr size;
    bool freed;
    u32 reserve_stack_id;
    u32 reserve_tid;
    u32 first_free_stack_id;
    u32 first_free_tid;
  };

  void EnsureInited();

  StaticSpinMutex mu_;
  InternalMmapVectorNoCtor<VmemGpuReservation> reservations_;
  atomic_uint8_t inited_;
};

// True when ROCr reserves host-accessible VA (e.g. HIP managed / SVM).
inline bool AmdHsaVmemReserveUsesHostMapping(u64 flags) {
  return (flags & HSA_AMD_VMEM_ADDRESS_NO_REGISTER) != 0;
}

struct AmdHsaAllocationInfo : public DeviceAllocationInfo {
  AmdHsaAllocationInfo() : DeviceAllocationInfo(DAT_AMDHSA) {
    status = HSA_STATUS_SUCCESS;
    alloc_func = nullptr;
  }
  // If allocation fails without an HSA API status, record one so callers never
  // see *ptr == nullptr with status still SUCCESS.
  void EnsureFailureStatus(hsa_status_t err) {
    if (status == HSA_STATUS_SUCCESS)
      status = err;
  }

  hsa_status_t status;
  void* alloc_func;
  hsa_amd_memory_pool_t memory_pool;
  u64 alignment;
  u64 address;
  u64 flags64;
  usize size;
  u32 flags;
  void* ptr;
};

// AMDGPU device heap for CombinedAllocator's third template parameter:
// DeviceAllocatorT<Primary, AmdHsaDeviceAllocator>.
template <class PrimaryAllocator>
using AmdHsaDeviceAllocatorT =
    DeviceAllocatorT<PrimaryAllocator, AmdHsaDeviceAllocator>;

#endif  // SANITIZER_AMDHSA
