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
  // -------------- Memory Allocate/Deallocate Functions ----------------
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

  // ----------------- System Event Register Function -------------------
  hsa_status_t (*register_system_event_handler)(
      hsa_amd_system_event_callback_t callback, void *data);
};

static HsaFunctions hsa_amd;

// Always align to page boundary to match current ROCr behavior
static const size_t kPageSize_ = 4096;

static atomic_uint8_t amdgpu_runtime_shutdown{0};
static atomic_uint8_t amdgpu_event_registered{0};

// ---------------------------------------------------------------------------
// Re-entrancy guard against ROCr agent_memory_lock_ self-deadlock.
//
// ROCr's MemoryRegion::Allocate()/Free() both take the same non-recursive
// agent_memory_lock_. With ASan's quarantine, a libc free() performed by ROCr
// *while it holds that lock* (e.g. bind_mem_to_numa() inside an allocation)
// can be intercepted, trigger a quarantine eviction, and recycle an older
// device chunk -- issuing a REAL hsa_amd_memory_pool_free() that re-enters
// MemoryRegion::Free() on the same thread and dead-locks on the same mutex.
//
// To break the cycle we never issue a REAL device free while this thread is
// already executing inside a REAL device allocate/free. Instead the pointer is
// parked on a thread-local list and flushed once the outermost real-HSA call
// has fully unwound (and ROCr has released agent_memory_lock_).
// ---------------------------------------------------------------------------

// Nesting depth of REAL hsa allocate/free calls on the current thread.
__attribute__((
    tls_model("initial-exec"))) static THREADLOCAL uptr real_hsa_depth = 0;

// Thread-local stack of device pointers whose REAL free was deferred because
// it was requested from within a REAL hsa call. Bounded, intrusive storage to
// avoid any allocation on this path.
static const uptr kMaxDeferredFrees = 64;
__attribute__((tls_model("initial-exec"))) static THREADLOCAL void*
    deferred_free_ptrs[kMaxDeferredFrees];
__attribute__((
    tls_model("initial-exec"))) static THREADLOCAL uptr deferred_free_count = 0;

namespace {
// RAII marker for "this thread is inside a REAL hsa allocate/free".
struct RealHsaScope {
  RealHsaScope() { ++real_hsa_depth; }
  ~RealHsaScope() { --real_hsa_depth; }
};
}  // namespace

// Issue the REAL device free for a single pointer. Must only be called when it
// is safe to take ROCr's agent_memory_lock_ (i.e. not nested inside a REAL hsa
// call on this thread).
static void RealDeviceFree(void* p) {
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

// Drain any device frees that were deferred while nested inside a REAL hsa
// call. Called once the outermost real-HSA scope has unwound.
static void FlushDeferredDeviceFrees() {
  // Snapshot-and-clear style: each RealDeviceFree() runs outside the guard.
  while (deferred_free_count > 0) {
    void* p = deferred_free_ptrs[--deferred_free_count];
    RealDeviceFree(p);
  }
}

#  define LOAD_HSA_FUNC_WITH_ERROR_CHECK(func, name, success)         \
    func = (decltype(func))dlsym(RTLD_NEXT, name);                    \
    if (!func) {                                                      \
      VReport(2, "Amdgpu Init: Failed to load " #name " function\n"); \
      success = false;                                                \
    }

// Check AMDGPU runtime shutdown state
bool AmdgpuMemFuncs::IsAmdgpuRuntimeShutdown() {
  return static_cast<bool>(
      atomic_load(&amdgpu_runtime_shutdown, memory_order_acquire));
}

// Notify AMDGPU runtime shutdown to allocator
void AmdgpuMemFuncs::NotifyAmdgpuRuntimeShutdown() {
  uint8_t shutdown = 0;
  if (atomic_compare_exchange_strong(&amdgpu_runtime_shutdown, &shutdown, 1,
                                     memory_order_acq_rel)) {
    VReport(2, "Amdgpu Allocator: AMDGPU runtime shutdown detected\n");
  }
}

// Clear shutdown state when hsa_init() succeeds again (re-init after shutdown).
// Resets amdgpu_runtime_shutdown so allocator operations are enabled, and
// amdgpu_event_registered so RegisterSystemEventHandlers() will register the
// shutdown callback for the new runtime instance.
void AmdgpuMemFuncs::ClearAmdgpuRuntimeShutdownState() {
  atomic_store(&amdgpu_runtime_shutdown, 0, memory_order_release);
  atomic_store(&amdgpu_event_registered, 0, memory_order_release);
}

bool AmdgpuMemFuncs::Init() {
  bool success = true;
  LOAD_HSA_FUNC_WITH_ERROR_CHECK(hsa_amd.memory_pool_allocate,
                                 "hsa_amd_memory_pool_allocate", success);
  LOAD_HSA_FUNC_WITH_ERROR_CHECK(hsa_amd.memory_pool_free,
                                 "hsa_amd_memory_pool_free", success);
  LOAD_HSA_FUNC_WITH_ERROR_CHECK(hsa_amd.pointer_info, "hsa_amd_pointer_info",
                                 success);
  LOAD_HSA_FUNC_WITH_ERROR_CHECK(hsa_amd.vmem_address_reserve_align,
                                 "hsa_amd_vmem_address_reserve_align", success);
  LOAD_HSA_FUNC_WITH_ERROR_CHECK(hsa_amd.vmem_address_free,
                                 "hsa_amd_vmem_address_free", success);
  LOAD_HSA_FUNC_WITH_ERROR_CHECK(hsa_amd.register_system_event_handler,
                                 "hsa_amd_register_system_event_handler",
                                 success);
  if (!success) {
    VReport(1, "Amdgpu Init: Failed to load AMDGPU runtime functions\n");
    return false;
  }
  return true;
}

void *AmdgpuMemFuncs::Allocate(uptr size, uptr alignment,
                               DeviceAllocationInfo *da_info) {
  // Do not allocate if AMDGPU runtime is shutdown
  if (UNLIKELY(IsAmdgpuRuntimeShutdown())) {
    VReport(1,
            "Amdgpu Allocate: Runtime shutdown, skipping allocation for size "
            "%zu alignment %zu\n",
            size, alignment);
    return nullptr;
  }

  AmdgpuAllocationInfo *aa_info =
      reinterpret_cast<AmdgpuAllocationInfo *>(da_info);
  {
    // Mark this thread as being inside a REAL hsa call. ROCr takes
    // agent_memory_lock_ for the duration; any quarantine-driven device free
    // that fires underneath must be deferred, not issued, to avoid re-locking
    // the same non-recursive mutex on this thread.
    RealHsaScope real_hsa_scope;
    if (!aa_info->memory_pool.handle) {
      aa_info->status = hsa_amd.vmem_address_reserve_align(
          &aa_info->ptr, size, aa_info->address, aa_info->alignment,
          aa_info->flags64);
    } else {
      aa_info->status = hsa_amd.memory_pool_allocate(
          aa_info->memory_pool, size, aa_info->flags, &aa_info->ptr);
    }
  }
  // Outermost real-HSA scope has unwound here; ROCr has released its lock so
  // it is now safe to issue any device frees that were deferred underneath.
  if (real_hsa_depth == 0 && deferred_free_count > 0)
    FlushDeferredDeviceFrees();
  if (aa_info->status != HSA_STATUS_SUCCESS)
    return nullptr;

  return aa_info->ptr;
}

void AmdgpuMemFuncs::Deallocate(void *p) {
  // Deallocate does nothing after AMDGPU runtime shutdown
  if (UNLIKELY(IsAmdgpuRuntimeShutdown())) {
    VReport(
        1,
        "Amdgpu Deallocate: Runtime shutdown, skipping deallocation for %p\n",
        reinterpret_cast<void*>(p));
    return;
  }

  // If we are nested inside a REAL hsa allocate/free on this thread, ROCr is
  // currently holding agent_memory_lock_. Issuing the REAL free now would
  // re-enter MemoryRegion::Free() and dead-lock on that same non-recursive
  // mutex. Defer the free until the outermost real-HSA call unwinds.
  if (real_hsa_depth > 0) {
    if (LIKELY(deferred_free_count < kMaxDeferredFrees)) {
      deferred_free_ptrs[deferred_free_count++] = p;
      return;
    }
    // Deferral list is full (pathologically deep nesting). Fall through and
    // free directly; this is exceedingly unlikely and preserves correctness
    // over the (already remote) risk of re-entrancy at this depth.
  }

  RealDeviceFree(p);
}

bool AmdgpuMemFuncs::GetPointerInfo(uptr ptr, DevicePointerInfo* ptr_info) {
  // GetPointerInfo returns false after AMDGPU runtime shutdown
  if (UNLIKELY(IsAmdgpuRuntimeShutdown())) {
    VReport(1,
            "Amdgpu GetPointerInfo: Runtime shutdown, skipping query for %p\n",
            reinterpret_cast<void*>(ptr));
    return false;
  }

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
  uint8_t registered = 0;
  // Check if shutdown event handler is already registered
  if (atomic_compare_exchange_strong(&amdgpu_event_registered, &registered, 1,
                                     memory_order_acq_rel)) {
    // Callback to detect and notify AMDGPU runtime shutdown
    hsa_amd_system_event_callback_t callback = [](const hsa_amd_event_t* event,
                                                  void* data) {
      if (!event)
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
      if (event->event_type == HSA_AMD_SYSTEM_SHUTDOWN_EVENT)
        AmdgpuMemFuncs::NotifyAmdgpuRuntimeShutdown();
      return HSA_STATUS_SUCCESS;
    };
    // Register the event callback
    hsa_status_t status =
        hsa_amd.register_system_event_handler(callback, nullptr);
    // Check as registered if successful
    if (status == HSA_STATUS_SUCCESS)
      VReport(
          1,
          "Amdgpu RegisterSystemEventHandlers: Registered shutdown event \n");
    else {
      VReport(1,
              "Amdgpu RegisterSystemEventHandlers: Failed to register shutdown "
              "event \n");
      atomic_store(&amdgpu_event_registered, 0, memory_order_release);
    }
  }
}

uptr AmdgpuMemFuncs::GetPageSize() { return kPageSize_; }
}  // namespace __sanitizer
#endif  // SANITIZER_AMDGPU
