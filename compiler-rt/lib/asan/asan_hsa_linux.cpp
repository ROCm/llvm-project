//===-- asan_hsa_linux.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Linux host HSA API interception for AddressSanitizer (SANITIZER_AMDHSA).
//
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_platform.h"

#if SANITIZER_AMDHSA

#  include <dlfcn.h>

#  include "asan_hsa_linux.h"
#  include "asan_interceptors.h"
#  include "asan_interceptors_memintrinsics.h"
#  include "asan_internal.h"
#  include "asan_report.h"
#  include "asan_stack.h"
#  include "asan_suppressions.h"

using namespace __asan;

// This TU provides the HSA interceptors, so it must also define the underlying
// REAL() slots (e.g. __interception::real_hsa_init). Otherwise the shared
// runtime fails to link when any interceptor references REAL(hsa_*).
DEFINE_REAL(hsa_status_t, hsa_init, );
DEFINE_REAL(hsa_status_t, hsa_memory_copy, void*, const void*, size_t)
DEFINE_REAL(hsa_status_t, hsa_amd_agents_allow_access, uint32_t num_agents,
            const hsa_agent_t* agents, const uint32_t* flags, const void* ptr)
DEFINE_REAL(hsa_status_t, hsa_amd_memory_pool_allocate,
            hsa_amd_memory_pool_t memory_pool, size_t size, uint32_t flags,
            void** ptr)
DEFINE_REAL(hsa_status_t, hsa_amd_memory_pool_free, void* ptr)
DEFINE_REAL(hsa_status_t, hsa_memory_free, void* ptr)
DEFINE_REAL(hsa_status_t, hsa_amd_memory_async_copy, void*, hsa_agent_t,
            const void*, hsa_agent_t, size_t, uint32_t, const hsa_signal_t*,
            hsa_signal_t)
DEFINE_REAL(hsa_status_t, hsa_amd_memory_async_copy_on_engine, void*,
            hsa_agent_t, const void*, hsa_agent_t, size_t, uint32_t,
            const hsa_signal_t*, hsa_signal_t, hsa_amd_sdma_engine_id_t, bool)
DEFINE_REAL(hsa_status_t, hsa_amd_ipc_memory_create, void* ptr, size_t len,
            hsa_amd_ipc_memory_t* handle)
DEFINE_REAL(hsa_status_t, hsa_amd_ipc_memory_attach,
            const hsa_amd_ipc_memory_t* handle, size_t len, uint32_t num_agents,
            const hsa_agent_t* mapping_agents, void** mapped_ptr)
DEFINE_REAL(hsa_status_t, hsa_amd_ipc_memory_detach, void* mapped_ptr)
DEFINE_REAL(hsa_status_t, hsa_amd_vmem_address_reserve_align, void** ptr,
            size_t size, uint64_t address, uint64_t alignment, uint64_t flags)
DEFINE_REAL(hsa_status_t, hsa_amd_vmem_address_free, void* ptr, size_t size)
DEFINE_REAL(hsa_status_t, hsa_amd_pointer_info, const void* ptr,
            hsa_amd_pointer_info_t* info, void* (*alloc)(size_t),
            uint32_t* num_agents_accessible, hsa_agent_t** accessible)

namespace __asan {

// Defined after the INTERCEPTORs: each INTERCEPTOR in this namespace creates
// its own __asan::__interception::real_* slot, which REAL() only resolves to
// once that INTERCEPTOR has been seen.
static void ENSURE_HSA_INITED();

// With a runtime's table registered, the interceptors only reach a real ROCr.
// Without one, a call is no reason to refuse a registration.
#  define HSA_PASS_THROUGH_IF_TABLE(func, ...)   \
    do {                                         \
      if (!REAL(func))                           \
        return HSA_STATUS_ERROR_NOT_INITIALIZED; \
      if (AsanHsaPassThrough())                  \
        return REAL(func)(__VA_ARGS__);          \
    } while (0)

#  define HSA_REQUIRE_TABLE()                    \
    do {                                         \
      if (!AsanHsaTableRegistered())             \
        return HSA_STATUS_ERROR_NOT_INITIALIZED; \
    } while (0)

#  define HSA_CHECK_COPY(name, dst, src, size)              \
    do {                                                    \
      if (flags()->replace_intrin) {                        \
        if (dst != src) {                                   \
          CHECK_RANGES_OVERLAP(name, dst, size, src, size); \
        }                                                   \
        ASAN_READ_RANGE(nullptr, src, size);                \
        ASAN_WRITE_RANGE(nullptr, dst, size);               \
      }                                                     \
    } while (0)

INTERCEPTOR(hsa_status_t, hsa_amd_memory_pool_allocate,
            hsa_amd_memory_pool_t memory_pool, size_t size, uint32_t flags,
            void** ptr) {
  AsanInitFromRtl();
  ENSURE_HSA_INITED();
  HSA_PASS_THROUGH_IF_TABLE(hsa_amd_memory_pool_allocate, memory_pool, size,
                            flags, ptr);
  GET_STACK_TRACE_MALLOC;
  return asan_hsa_amd_memory_pool_allocate(memory_pool, size, flags, ptr,
                                           &stack);
}

INTERCEPTOR(hsa_status_t, hsa_amd_memory_pool_free, void* ptr) {
  AsanInitFromRtl();
  ENSURE_HSA_INITED();
  HSA_PASS_THROUGH_IF_TABLE(hsa_amd_memory_pool_free, ptr);
  GET_STACK_TRACE_FREE;
  return asan_hsa_amd_memory_pool_free(ptr, &stack);
}

INTERCEPTOR(hsa_status_t, hsa_memory_free, void* ptr) {
  AsanInitFromRtl();
  ENSURE_HSA_INITED();
  HSA_PASS_THROUGH_IF_TABLE(hsa_memory_free, ptr);
  GET_STACK_TRACE_FREE;
  return asan_hsa_amd_memory_pool_free(ptr, &stack);
}

INTERCEPTOR(hsa_status_t, hsa_amd_agents_allow_access, uint32_t num_agents,
            const hsa_agent_t* agents, const uint32_t* flags, const void* ptr) {
  AsanInitFromRtl();
  ENSURE_HSA_INITED();
  HSA_PASS_THROUGH_IF_TABLE(hsa_amd_agents_allow_access, num_agents, agents,
                            flags, ptr);
  GET_STACK_TRACE_FREE;
  return asan_hsa_amd_agents_allow_access(num_agents, agents, flags, ptr,
                                          &stack);
}

INTERCEPTOR(hsa_status_t, hsa_memory_copy, void* dst, const void* src,
            size_t size) {
  AsanInitFromRtl();
  ENSURE_HSA_INITED();
  if (flags()->replace_intrin) {
    if (dst != src) {
      CHECK_RANGES_OVERLAP("hsa_memory_copy", dst, size, src, size);
    }
    ASAN_READ_RANGE(nullptr, src, size);
    ASAN_WRITE_RANGE(nullptr, dst, size);
  }
  if (!REAL(hsa_memory_copy))
    return HSA_STATUS_ERROR_NOT_INITIALIZED;
  return REAL(hsa_memory_copy)(dst, src, size);
}

INTERCEPTOR(hsa_status_t, hsa_amd_memory_async_copy, void* dst,
            hsa_agent_t dst_agent, const void* src, hsa_agent_t src_agent,
            size_t size, uint32_t num_dep_signals,
            const hsa_signal_t* dep_signals, hsa_signal_t completion_signal) {
  AsanInitFromRtl();
  ENSURE_HSA_INITED();
  if (flags()->replace_intrin) {
    if (dst != src) {
      CHECK_RANGES_OVERLAP("hsa_amd_memory_async_copy", dst, size, src, size);
    }
    ASAN_READ_RANGE(nullptr, src, size);
    ASAN_WRITE_RANGE(nullptr, dst, size);
  }
  if (!REAL(hsa_amd_memory_async_copy))
    return HSA_STATUS_ERROR_NOT_INITIALIZED;
  return REAL(hsa_amd_memory_async_copy)(dst, dst_agent, src, src_agent, size,
                                         num_dep_signals, dep_signals,
                                         completion_signal);
}

INTERCEPTOR(hsa_status_t, hsa_amd_memory_async_copy_on_engine, void* dst,
            hsa_agent_t dst_agent, const void* src, hsa_agent_t src_agent,
            size_t size, uint32_t num_dep_signals,
            const hsa_signal_t* dep_signals, hsa_signal_t completion_signal,
            hsa_amd_sdma_engine_id_t engine_id, bool force_copy_on_sdma) {
  AsanInitFromRtl();
  ENSURE_HSA_INITED();
  if (flags()->replace_intrin) {
    if (dst != src) {
      CHECK_RANGES_OVERLAP("hsa_amd_memory_async_copy_on_engine", dst, size,
                           src, size);
    }
    ASAN_READ_RANGE(nullptr, src, size);
    ASAN_WRITE_RANGE(nullptr, dst, size);
  }
  if (!REAL(hsa_amd_memory_async_copy_on_engine))
    return HSA_STATUS_ERROR_NOT_INITIALIZED;
  return REAL(hsa_amd_memory_async_copy_on_engine)(
      dst, dst_agent, src, src_agent, size, num_dep_signals, dep_signals,
      completion_signal, engine_id, force_copy_on_sdma);
}

INTERCEPTOR(hsa_status_t, hsa_amd_ipc_memory_create, void* ptr, size_t len,
            hsa_amd_ipc_memory_t* handle) {
  AsanInitFromRtl();
  ENSURE_HSA_INITED();
  HSA_PASS_THROUGH_IF_TABLE(hsa_amd_ipc_memory_create, ptr, len, handle);
  return asan_hsa_amd_ipc_memory_create(ptr, len, handle);
}

INTERCEPTOR(hsa_status_t, hsa_amd_ipc_memory_attach,
            const hsa_amd_ipc_memory_t* handle, size_t len, uint32_t num_agents,
            const hsa_agent_t* mapping_agents, void** mapped_ptr) {
  AsanInitFromRtl();
  ENSURE_HSA_INITED();
  HSA_PASS_THROUGH_IF_TABLE(hsa_amd_ipc_memory_attach, handle, len, num_agents,
                            mapping_agents, mapped_ptr);
  return asan_hsa_amd_ipc_memory_attach(handle, len, num_agents, mapping_agents,
                                        mapped_ptr);
}

INTERCEPTOR(hsa_status_t, hsa_amd_ipc_memory_detach, void* mapped_ptr) {
  AsanInitFromRtl();
  ENSURE_HSA_INITED();
  HSA_PASS_THROUGH_IF_TABLE(hsa_amd_ipc_memory_detach, mapped_ptr);
  return asan_hsa_amd_ipc_memory_detach(mapped_ptr);
}

INTERCEPTOR(hsa_status_t, hsa_amd_vmem_address_reserve_align, void** ptr,
            size_t size, uint64_t address, uint64_t alignment, uint64_t flags) {
  AsanInitFromRtl();
  ENSURE_HSA_INITED();
  HSA_PASS_THROUGH_IF_TABLE(hsa_amd_vmem_address_reserve_align, ptr, size,
                            address, alignment, flags);
  GET_STACK_TRACE_MALLOC;
  return asan_hsa_amd_vmem_address_reserve_align(ptr, size, address, alignment,
                                                 flags, &stack);
}

INTERCEPTOR(hsa_status_t, hsa_amd_vmem_address_free, void* ptr, size_t size) {
  AsanInitFromRtl();
  ENSURE_HSA_INITED();
  HSA_PASS_THROUGH_IF_TABLE(hsa_amd_vmem_address_free, ptr, size);
  GET_STACK_TRACE_FREE;
  return asan_hsa_amd_vmem_address_free(ptr, size, &stack);
}

INTERCEPTOR(hsa_status_t, hsa_amd_pointer_info, const void* ptr,
            hsa_amd_pointer_info_t* info, void* (*alloc)(size_t),
            uint32_t* num_agents_accessible, hsa_agent_t** accessible) {
  AsanInitFromRtl();
  ENSURE_HSA_INITED();
  HSA_PASS_THROUGH_IF_TABLE(hsa_amd_pointer_info, ptr, info, alloc,
                            num_agents_accessible, accessible);
  return asan_hsa_amd_pointer_info(ptr, info, alloc, num_agents_accessible,
                                   accessible);
}

INTERCEPTOR(hsa_status_t, hsa_init) {
  AsanInitFromRtl();
  ENSURE_HSA_INITED();
  HSA_PASS_THROUGH_IF_TABLE(hsa_init, );
  return asan_hsa_init();
}

static StaticSpinMutex hsa_reals_mu;

// RTLD_NEXT cannot see a ROCr that was loaded into an RTLD_LOCAL group after
// ASan initialized, yet the interceptors still win symbol resolution because
// the ASan runtime is in the global scope. Bind such REAL() slots from the
// loaded ROCr handle instead.
static void BindHSARealsFromRuntime() {
  // With a runtime's table registered no ROCr is needed: only a loaded one is
  // bound, and none is loaded for the calls it makes.
  void* handle = __sanitizer::GetHsaRuntimeHandle(!AsanHsaTableRegistered());
  if (!handle)
    return;
#  define BIND_HSA_REAL(name) \
    if (!REAL(name))          \
    REAL(name) = reinterpret_cast<decltype(REAL(name))>(dlsym(handle, #name))
  BIND_HSA_REAL(hsa_memory_copy);
  BIND_HSA_REAL(hsa_amd_memory_pool_allocate);
  BIND_HSA_REAL(hsa_amd_memory_pool_free);
  BIND_HSA_REAL(hsa_memory_free);
  BIND_HSA_REAL(hsa_amd_agents_allow_access);
  BIND_HSA_REAL(hsa_amd_memory_async_copy);
  BIND_HSA_REAL(hsa_amd_memory_async_copy_on_engine);
  BIND_HSA_REAL(hsa_amd_ipc_memory_create);
  BIND_HSA_REAL(hsa_amd_ipc_memory_attach);
  BIND_HSA_REAL(hsa_amd_ipc_memory_detach);
  BIND_HSA_REAL(hsa_amd_vmem_address_reserve_align);
  BIND_HSA_REAL(hsa_amd_vmem_address_free);
  BIND_HSA_REAL(hsa_amd_pointer_info);
#  undef BIND_HSA_REAL
  // Bound last: a non-null REAL(hsa_init) publishes the whole set.
  auto init =
      reinterpret_cast<decltype(REAL(hsa_init))>(dlsym(handle, "hsa_init"));
  if (!init) {
    const char* err = dlerror();
    VReport(1, "AddressSanitizer: dlsym(hsa_init) failed: %s\n",
            err ? err : "unknown error");
    return;
  }
  __atomic_store_n(&REAL(hsa_init), init, __ATOMIC_RELEASE);
  VReport(1, "AddressSanitizer: bound HSA interceptors from loaded ROCr\n");
}

static void ENSURE_HSA_INITED() {
  // The HSA interceptors are initialized lazily: if the program calls into HSA
  // before ASan's global interceptor init runs, we still need the REAL() slots
  // to be populated so we can call through to ROCr.
  if (LIKELY(__atomic_load_n(&REAL(hsa_init), __ATOMIC_ACQUIRE)))
    return;
  SpinMutexLock l(&hsa_reals_mu);
  if (REAL(hsa_init))
    return;
  InitializeHSAInterceptors();
  if (!REAL(hsa_init))
    BindHSARealsFromRuntime();
}

void InitializeHSAInterceptors() {
  ASAN_INTERCEPT_FUNC(hsa_init);
  ASAN_INTERCEPT_FUNC(hsa_memory_copy);
  ASAN_INTERCEPT_FUNC(hsa_amd_memory_pool_allocate);
  ASAN_INTERCEPT_FUNC(hsa_amd_memory_pool_free);
  ASAN_INTERCEPT_FUNC(hsa_amd_agents_allow_access);
  ASAN_INTERCEPT_FUNC(hsa_amd_memory_async_copy);
  ASAN_INTERCEPT_FUNC(hsa_amd_memory_async_copy_on_engine);
  ASAN_INTERCEPT_FUNC(hsa_amd_ipc_memory_create);
  ASAN_INTERCEPT_FUNC(hsa_amd_ipc_memory_attach);
  ASAN_INTERCEPT_FUNC(hsa_amd_ipc_memory_detach);
  ASAN_INTERCEPT_FUNC(hsa_amd_vmem_address_reserve_align);
  ASAN_INTERCEPT_FUNC(hsa_amd_vmem_address_free);
  ASAN_INTERCEPT_FUNC(hsa_amd_pointer_info);
}

}  // namespace __asan

// Entry points for runtimes that registered a table with
// __sanitizer_amdgpu_set_hsa_table: the interceptor bodies above, forwarding to
// the table.
extern "C" {
SANITIZER_INTERFACE_ATTRIBUTE hsa_status_t __asan_hsa_init() {
  AsanInitFromRtl();
  HSA_REQUIRE_TABLE();
  return asan_hsa_init();
}

SANITIZER_INTERFACE_ATTRIBUTE hsa_status_t
__asan_hsa_memory_copy(void* dst, const void* src, size_t size) {
  AsanInitFromRtl();
  HSA_REQUIRE_TABLE();
  HSA_CHECK_COPY("hsa_memory_copy", dst, src, size);
  return GetAsanHsaTable().hsa_memory_copy(dst, src, size);
}

SANITIZER_INTERFACE_ATTRIBUTE hsa_status_t
__asan_hsa_amd_memory_pool_allocate(hsa_amd_memory_pool_t memory_pool,
                                    size_t size, uint32_t flags, void** ptr) {
  AsanInitFromRtl();
  HSA_REQUIRE_TABLE();
  GET_STACK_TRACE_MALLOC;
  return asan_hsa_amd_memory_pool_allocate(memory_pool, size, flags, ptr,
                                           &stack);
}

SANITIZER_INTERFACE_ATTRIBUTE hsa_status_t
__asan_hsa_amd_memory_pool_free(void* ptr) {
  AsanInitFromRtl();
  HSA_REQUIRE_TABLE();
  GET_STACK_TRACE_FREE;
  return asan_hsa_amd_memory_pool_free(ptr, &stack);
}

SANITIZER_INTERFACE_ATTRIBUTE hsa_status_t __asan_hsa_amd_agents_allow_access(
    uint32_t num_agents, const hsa_agent_t* agents, const uint32_t* flags,
    const void* ptr) {
  AsanInitFromRtl();
  HSA_REQUIRE_TABLE();
  return asan_hsa_amd_agents_allow_access(num_agents, agents, flags, ptr,
                                          nullptr);
}

SANITIZER_INTERFACE_ATTRIBUTE hsa_status_t __asan_hsa_amd_memory_async_copy(
    void* dst, hsa_agent_t dst_agent, const void* src, hsa_agent_t src_agent,
    size_t size, uint32_t num_dep_signals, const hsa_signal_t* dep_signals,
    hsa_signal_t completion_signal) {
  AsanInitFromRtl();
  HSA_REQUIRE_TABLE();
  HSA_CHECK_COPY("hsa_amd_memory_async_copy", dst, src, size);
  return GetAsanHsaTable().hsa_amd_memory_async_copy(
      dst, dst_agent, src, src_agent, size, num_dep_signals, dep_signals,
      completion_signal);
}

SANITIZER_INTERFACE_ATTRIBUTE hsa_status_t
__asan_hsa_amd_memory_async_copy_on_engine(
    void* dst, hsa_agent_t dst_agent, const void* src, hsa_agent_t src_agent,
    size_t size, uint32_t num_dep_signals, const hsa_signal_t* dep_signals,
    hsa_signal_t completion_signal, hsa_amd_sdma_engine_id_t engine_id,
    bool force_copy_on_sdma) {
  AsanInitFromRtl();
  HSA_REQUIRE_TABLE();
  HSA_CHECK_COPY("hsa_amd_memory_async_copy_on_engine", dst, src, size);
  return GetAsanHsaTable().hsa_amd_memory_async_copy_on_engine(
      dst, dst_agent, src, src_agent, size, num_dep_signals, dep_signals,
      completion_signal, engine_id, force_copy_on_sdma);
}

SANITIZER_INTERFACE_ATTRIBUTE hsa_status_t __asan_hsa_amd_ipc_memory_create(
    void* ptr, size_t len, hsa_amd_ipc_memory_t* handle) {
  AsanInitFromRtl();
  HSA_REQUIRE_TABLE();
  return asan_hsa_amd_ipc_memory_create(ptr, len, handle);
}

SANITIZER_INTERFACE_ATTRIBUTE hsa_status_t __asan_hsa_amd_ipc_memory_attach(
    const hsa_amd_ipc_memory_t* handle, size_t len, uint32_t num_agents,
    const hsa_agent_t* mapping_agents, void** mapped_ptr) {
  AsanInitFromRtl();
  HSA_REQUIRE_TABLE();
  return asan_hsa_amd_ipc_memory_attach(handle, len, num_agents, mapping_agents,
                                        mapped_ptr);
}

SANITIZER_INTERFACE_ATTRIBUTE hsa_status_t
__asan_hsa_amd_ipc_memory_detach(void* mapped_ptr) {
  AsanInitFromRtl();
  HSA_REQUIRE_TABLE();
  return asan_hsa_amd_ipc_memory_detach(mapped_ptr);
}

SANITIZER_INTERFACE_ATTRIBUTE hsa_status_t
__asan_hsa_amd_vmem_address_reserve_align(void** ptr, size_t size,
                                          uint64_t address, uint64_t alignment,
                                          uint64_t flags) {
  AsanInitFromRtl();
  HSA_REQUIRE_TABLE();
  GET_STACK_TRACE_MALLOC;
  return asan_hsa_amd_vmem_address_reserve_align(ptr, size, address, alignment,
                                                 flags, &stack);
}

SANITIZER_INTERFACE_ATTRIBUTE hsa_status_t
__asan_hsa_amd_vmem_address_free(void* ptr, size_t size) {
  AsanInitFromRtl();
  HSA_REQUIRE_TABLE();
  GET_STACK_TRACE_FREE;
  return asan_hsa_amd_vmem_address_free(ptr, size, &stack);
}

SANITIZER_INTERFACE_ATTRIBUTE hsa_status_t __asan_hsa_amd_pointer_info(
    const void* ptr, hsa_amd_pointer_info_t* info, void* (*alloc)(size_t),
    uint32_t* num_agents_accessible, hsa_agent_t** accessible) {
  AsanInitFromRtl();
  HSA_REQUIRE_TABLE();
  return asan_hsa_amd_pointer_info(ptr, info, alloc, num_agents_accessible,
                                   accessible);
}
}  // extern "C"

#endif  // SANITIZER_AMDHSA
