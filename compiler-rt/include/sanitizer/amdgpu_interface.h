//===-- sanitizer/amdgpu_interface.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer.
//
// Public interface for GPU runtimes that do not export the HSA API. Such a
// runtime registers the HSA functions AddressSanitizer uses, then calls the
// __asan_hsa_* entry points wherever a call to the HSA function of the same
// name would be intercepted. Include <hsa/hsa.h> and <hsa/hsa_ext_amd.h>
// before this header. Linux only. The functions exist only in ASan runtimes
// built with SANITIZER_AMDHSA (64-bit Linux), so a runtime looks them up
// (dlsym) or references them weakly.
//===----------------------------------------------------------------------===//
#ifndef SANITIZER_AMDGPU_INTERFACE_H
#define SANITIZER_AMDGPU_INTERFACE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Fields are only ever appended. `size` must cover at least the fields below;
// fields appended later are optional.
typedef struct __sanitizer_amdgpu_hsa_table {
  // sizeof(__sanitizer_amdgpu_hsa_table) as seen by the caller.
  size_t size;
  hsa_status_t (*hsa_init)(void);
  hsa_status_t (*hsa_memory_copy)(void *dst, const void *src, size_t size);
  hsa_status_t (*hsa_amd_memory_pool_allocate)(hsa_amd_memory_pool_t pool,
                                               size_t size, uint32_t flags,
                                               void **ptr);
  hsa_status_t (*hsa_amd_memory_pool_free)(void *ptr);
  hsa_status_t (*hsa_amd_agents_allow_access)(uint32_t num_agents,
                                              const hsa_agent_t *agents,
                                              const uint32_t *flags,
                                              const void *ptr);
  hsa_status_t (*hsa_amd_memory_async_copy)(void *dst, hsa_agent_t dst_agent,
                                            const void *src,
                                            hsa_agent_t src_agent, size_t size,
                                            uint32_t num_dep_signals,
                                            const hsa_signal_t *dep_signals,
                                            hsa_signal_t completion_signal);
  hsa_status_t (*hsa_amd_memory_async_copy_on_engine)(
      void *dst, hsa_agent_t dst_agent, const void *src, hsa_agent_t src_agent,
      size_t size, uint32_t num_dep_signals, const hsa_signal_t *dep_signals,
      hsa_signal_t completion_signal, hsa_amd_sdma_engine_id_t engine_id,
      bool force_copy_on_sdma);
  hsa_status_t (*hsa_amd_ipc_memory_create)(void *ptr, size_t len,
                                            hsa_amd_ipc_memory_t *handle);
  hsa_status_t (*hsa_amd_ipc_memory_attach)(const hsa_amd_ipc_memory_t *handle,
                                            size_t len, uint32_t num_agents,
                                            const hsa_agent_t *mapping_agents,
                                            void **mapped_ptr);
  hsa_status_t (*hsa_amd_ipc_memory_detach)(void *mapped_ptr);
  hsa_status_t (*hsa_amd_vmem_address_reserve_align)(void **ptr, size_t size,
                                                     uint64_t address,
                                                     uint64_t alignment,
                                                     uint64_t flags);
  hsa_status_t (*hsa_amd_vmem_address_free)(void *ptr, size_t size);
  hsa_status_t (*hsa_amd_pointer_info)(const void *ptr,
                                       hsa_amd_pointer_info_t *info,
                                       void *(*alloc)(size_t),
                                       uint32_t *num_agents_accessible,
                                       hsa_agent_t **accessible);
  hsa_status_t (*hsa_amd_register_system_event_handler)(
      hsa_amd_system_event_callback_t callback, void *data);
} __sanitizer_amdgpu_hsa_table;

// Makes AddressSanitizer use a copy of `table` instead of the HSA functions it
// finds by name; the runtime then calls __asan_hsa_init, which registers for
// its shutdown event. Returns non-zero on success; on failure GPU memory is not
// tracked. Only the first successful registration counts: repeating it
// succeeds only with an identical table. Registration fails once an HSA call
// other than a copy went through the hsa_* interceptors, for example from a
// ROCr in the same process, so a runtime registers during its initialization,
// before such calls. Once a table is registered, the interceptors other than
// the copy ones pass a ROCr's calls through.
//
// ASan calls the table's hsa_amd_pointer_info, hsa_amd_memory_pool_free and
// hsa_amd_vmem_address_free from inside any thread's free(); pointer_info also
// with its device allocator locked, and during leak checking with its
// allocator locked and all other threads stopped. These functions must not
// call __asan_hsa_* or wait on a lock the runtime may hold while it frees host
// memory or calls __asan_hsa_*. pointer_info must not allocate or free host
// memory, and during leak checking must not wait on any lock. On a thread ASan
// no longer tracks (during thread teardown) the free functions may run with
// ASan's fallback allocator locked, where allocating or freeing host memory
// deadlocks.
//
// The table's functions must stay callable until the runtime raises
// HSA_AMD_SYSTEM_SHUTDOWN_EVENT to the handler __asan_hsa_init registered.
// After the event ASan calls them only for the runtime's own __asan_hsa_* calls
// until __asan_hsa_init runs again, and the blocks still in its quarantine are
// the runtime's to reclaim.
int __sanitizer_amdgpu_set_hsa_table(const __sanitizer_amdgpu_hsa_table *table);

// Same behavior as the interceptors of the HSA functions of the same name,
// forwarding to the registered table. Without a table they return
// HSA_STATUS_ERROR_NOT_INITIALIZED.
hsa_status_t __asan_hsa_init(void);
hsa_status_t __asan_hsa_memory_copy(void *dst, const void *src, size_t size);
hsa_status_t __asan_hsa_amd_memory_pool_allocate(hsa_amd_memory_pool_t pool,
                                                 size_t size, uint32_t flags,
                                                 void **ptr);
hsa_status_t __asan_hsa_amd_memory_pool_free(void *ptr);
hsa_status_t __asan_hsa_amd_agents_allow_access(uint32_t num_agents,
                                                const hsa_agent_t *agents,
                                                const uint32_t *flags,
                                                const void *ptr);
hsa_status_t __asan_hsa_amd_memory_async_copy(
    void *dst, hsa_agent_t dst_agent, const void *src, hsa_agent_t src_agent,
    size_t size, uint32_t num_dep_signals, const hsa_signal_t *dep_signals,
    hsa_signal_t completion_signal);
hsa_status_t __asan_hsa_amd_memory_async_copy_on_engine(
    void *dst, hsa_agent_t dst_agent, const void *src, hsa_agent_t src_agent,
    size_t size, uint32_t num_dep_signals, const hsa_signal_t *dep_signals,
    hsa_signal_t completion_signal, hsa_amd_sdma_engine_id_t engine_id,
    bool force_copy_on_sdma);
hsa_status_t __asan_hsa_amd_ipc_memory_create(void *ptr, size_t len,
                                              hsa_amd_ipc_memory_t *handle);
hsa_status_t __asan_hsa_amd_ipc_memory_attach(
    const hsa_amd_ipc_memory_t *handle, size_t len, uint32_t num_agents,
    const hsa_agent_t *mapping_agents, void **mapped_ptr);
hsa_status_t __asan_hsa_amd_ipc_memory_detach(void *mapped_ptr);
hsa_status_t __asan_hsa_amd_vmem_address_reserve_align(void **ptr, size_t size,
                                                       uint64_t address,
                                                       uint64_t alignment,
                                                       uint64_t flags);
hsa_status_t __asan_hsa_amd_vmem_address_free(void *ptr, size_t size);
hsa_status_t __asan_hsa_amd_pointer_info(const void *ptr,
                                         hsa_amd_pointer_info_t *info,
                                         void *(*alloc)(size_t),
                                         uint32_t *num_agents_accessible,
                                         hsa_agent_t **accessible);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // SANITIZER_AMDGPU_INTERFACE_H
