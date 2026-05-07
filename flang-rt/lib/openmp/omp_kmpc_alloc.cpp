//===-- lib/openmp/omp_kmpc_alloc.cpp ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the host OpenMP allocator path that backs the Flang lowering of
// the OpenMP 5.0+ `!$omp allocators` construct.  The Flang lowering emits
// _FortranAOmpAllocatorStamp(descriptor, handle, align) immediately before
// the matching _FortranAAllocatableAllocate runtime call so that the
// descriptor's allocator slot, the OpenMP allocator handle, and the
// (optional) alignment travel through to the OpenMP runtime.
//
// `__kmpc_alloc` / `__kmpc_aligned_alloc` / `__kmpc_free` /
// `__kmpc_global_thread_num` are resolved lazily via dlsym so flang_rt.openmp
// does not take a build-time dependency on libomp.
//
// The (handle, align) pair stamped on a particular allocate cannot be
// expressed through the AllocFct signature (`void *(*)(size_t,
// std::int64_t *)`); we therefore stash it in a thread-local single-slot
// pending stamp that the OpenMP allocate adapter consumes on the next call.
// The Flang lowering guarantees the stamp call is emitted immediately before
// the matching _FortranAAllocatableAllocate, so a single slot is sufficient.
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/OpenMP/omp_kmpc_alloc.h"

#include "flang-rt/runtime/allocator-registry.h"
#include "flang-rt/runtime/descriptor.h"
#include "flang/Common/api-attrs.h"

#include <atomic>
#include <cstddef>
#include <cstdint>

// flang_rt.openmp is also cross-compiled for AMDGPU / NVPTX as part of the
// flang-rt device runtime build (LLVM_RUNTIME_TARGETS).  The host OpenMP
// runtime entry points (`__kmpc_alloc` / `__kmpc_aligned_alloc` /
// `__kmpc_free` / `__kmpc_global_thread_num`) and `<dlfcn.h>` do not exist
// in the freestanding GPU sysroot, `thread_local` is not modelled by the
// AMDGPU back end, and `!$omp allocators` is a host-side directive: the
// compiler emits the stamp call from host code, not from the
// device-compiled translation unit.  When RT_GPU_TARGET is set (i.e.  we
// are natively compiling for AMDGPU or NVPTX), we therefore compile no-op
// stub bodies for the public entry points and skip the full host
// implementation entirely.
#if !defined(RT_GPU_TARGET)

#if !defined(_WIN32)
#define FLANG_RT_OPENMP_HAVE_DLSYM 1
#include <dlfcn.h>
#endif

namespace Fortran::runtime::omp {

namespace {

using KmpcAllocFn = void *(*)(int /*gtid*/, std::size_t /*size*/,
    std::uintptr_t /*allocator*/);
using KmpcAlignedAllocFn = void *(*)(int /*gtid*/, std::size_t /*align*/,
    std::size_t /*size*/, std::uintptr_t /*allocator*/);
using KmpcFreeFn = void (*)(int /*gtid*/, void *,
    std::uintptr_t /*allocator*/);
using KmpcGtidFn = int (*)(void * /*ident*/);

struct OmpRuntime {
  KmpcAllocFn alloc{nullptr};
  KmpcAlignedAllocFn alignedAlloc{nullptr};
  KmpcFreeFn free{nullptr};
  KmpcGtidFn gtid{nullptr};
  bool resolved{false};
  bool available{false};
};

// Initialised on first use; protected by the std::atomic flag.  The function
// pointers are written before the `resolved` flag flips to true so other
// threads see a consistent state.
OmpRuntime &Runtime() {
  static OmpRuntime rt;
  return rt;
}

std::atomic<bool> resolveInProgress{false};

void Resolve() {
  OmpRuntime &rt{Runtime()};
  if (rt.resolved) {
    return;
  }
  bool expected{false};
  if (!resolveInProgress.compare_exchange_strong(expected, true)) {
    while (!rt.resolved) {
      // Another thread is resolving; busy-wait briefly.
    }
    return;
  }

#if defined(FLANG_RT_OPENMP_HAVE_DLSYM)
  // RTLD_DEFAULT searches symbols already loaded into the process; if libomp
  // is linked into the executable (or loaded as a shared lib) the symbols
  // resolve.  Otherwise the OMP allocator path is unavailable and we fall
  // back gracefully.  __kmpc_aligned_alloc is optional -- if absent,
  // OmpAllocate falls back on __kmpc_alloc for the align=0 path and returns
  // nullptr for an aligned request.
  void *allocSym{dlsym(RTLD_DEFAULT, "__kmpc_alloc")};
  void *alignedAllocSym{dlsym(RTLD_DEFAULT, "__kmpc_aligned_alloc")};
  void *freeSym{dlsym(RTLD_DEFAULT, "__kmpc_free")};
  void *gtidSym{dlsym(RTLD_DEFAULT, "__kmpc_global_thread_num")};
  rt.alloc = reinterpret_cast<KmpcAllocFn>(allocSym);
  rt.alignedAlloc = reinterpret_cast<KmpcAlignedAllocFn>(alignedAllocSym);
  rt.free = reinterpret_cast<KmpcFreeFn>(freeSym);
  rt.gtid = reinterpret_cast<KmpcGtidFn>(gtidSym);
  rt.available = (rt.alloc != nullptr) && (rt.free != nullptr);
#else
  // Host build without dlsym (e.g. Windows MSVC).  Leave `available`
  // false; OmpAllocate returns nullptr and OmpFree is a no-op so the link
  // surface is satisfied without dragging in libomp.  GPU device builds
  // take the outer RT_GPU_TARGET stub path further down.
  rt.available = false;
#endif

  rt.resolved = true;
}

int CurrentGtid() {
  OmpRuntime &rt{Runtime()};
  if (rt.gtid) {
    // Passing nullptr as the ident is permitted by libomp; it elides source
    // location tracking but otherwise behaves like a normal kmpc call.
    return rt.gtid(nullptr);
  }
  return 0;
}

//===----------------------------------------------------------------------===//
// Thread-local pending-stamp slot
//===----------------------------------------------------------------------===//
//
// Single-slot TL store.  The Flang lowering guarantees that at most one
// OmpAllocatorStamp call is live at a time on any given thread -- the stamp
// is emitted immediately before the matching _FortranAAllocatableAllocate.
// A simple scalar slot is therefore sufficient and avoids pulling in
// <thread> / <mutex>.
struct PendingStamp {
  std::uintptr_t handle{0};
  std::size_t align{0};
  bool set{false};
};

thread_local PendingStamp tlStamp;

} // namespace

RT_API_ATTRS void *OmpAllocate(
    std::size_t byteSize, std::size_t align, std::uintptr_t handle) {
  Resolve();
  OmpRuntime &rt{Runtime()};
  if (!rt.available) {
    return nullptr;
  }
  if (align == 0) {
    return rt.alloc(CurrentGtid(), byteSize, handle);
  }
  if (rt.alignedAlloc == nullptr) {
    // Aligned-alloc requested but libomp does not expose the aligned entry
    // point.  We cannot honour the alignment request silently; return null
    // so the caller reports a proper allocation failure.
    return nullptr;
  }
  return rt.alignedAlloc(CurrentGtid(), align, byteSize, handle);
}

RT_API_ATTRS void OmpFree(void *ptr, std::uintptr_t handle) {
  Resolve();
  OmpRuntime &rt{Runtime()};
  if (!rt.available || !ptr) {
    return;
  }
  rt.free(CurrentGtid(), ptr, handle);
}

RT_API_ATTRS void SetPendingOmpAllocStamp(
    std::uintptr_t handle, std::size_t align) {
  tlStamp.handle = handle;
  tlStamp.align = align;
  tlStamp.set = true;
}

RT_API_ATTRS bool ConsumePendingOmpAllocStamp(
    std::uintptr_t &handle, std::size_t &align) {
  if (!tlStamp.set) {
    return false;
  }
  handle = tlStamp.handle;
  align = tlStamp.align;
  tlStamp.set = false;
  tlStamp.handle = 0;
  tlStamp.align = 0;
  return true;
}

RT_API_ATTRS void *OmpAllocateAdapter(
    std::size_t byteSize, std::int64_t * /*asyncObject*/) {
  // The matching OmpAllocatorStamp call (emitted by the Flang lowering)
  // populated the TL slot just above us.  If the slot is empty (e.g. a
  // reallocation of a descriptor that still carries kOmpAllocatorPos but
  // wasn't re-stamped), fall back on omp_null_allocator (handle 0) with
  // default alignment, which libomp resolves to the default memory space.
  std::uintptr_t handle{0};
  std::size_t align{0};
  ConsumePendingOmpAllocStamp(handle, align);
  return OmpAllocate(byteSize, align, handle);
}

RT_API_ATTRS void OmpFreeAdapter(void *ptr) {
  // We pass handle 0 (omp_null_allocator) to __kmpc_free; libomp recovers
  // the original allocator from the pointer's chunk metadata.  This keeps
  // the descriptor addendum layout ABI-compatible with non-OpenMP
  // descriptors.
  OmpFree(ptr, /*handle=*/0);
}

void RegisterOmpAllocator() {
  static std::atomic<bool> registered{false};
  bool expected{false};
  if (!registered.compare_exchange_strong(expected, true)) {
    return;
  }
  Fortran::runtime::allocatorRegistry.Register(
      kOmpAllocatorPos, {&OmpAllocateAdapter, &OmpFreeAdapter});
}

} // namespace Fortran::runtime::omp

namespace Fortran::runtime {

extern "C" {
RT_EXT_API_GROUP_BEGIN

void RTDEF(OmpRegisterAllocator)() { omp::RegisterOmpAllocator(); }

void RTDEF(OmpAllocatorStamp)(
    Descriptor &descriptor, std::uintptr_t handle, std::size_t align) {
  // First call wins; subsequent calls are cheap.  We register here so that
  // programs that use !$omp allocators do not need an explicit ctor pass --
  // any stamp call ensures the registry slot is populated before the
  // matching Allocate / Deallocate dispatches through it.
  omp::RegisterOmpAllocator();
  // Route a subsequent AllocatableAllocate through the OpenMP runtime.  We
  // deliberately do not assert the descriptor's state here (e.g.
  // IsAllocated) because the call is emitted by the lowering immediately
  // before the user's allocate statement, and the descriptor may
  // legitimately be in the "initialized, not yet allocated" state.
  descriptor.SetAllocIdx(kOmpAllocatorPos);
  // Stash (handle, align) in the thread-local pending-stamp slot.  The
  // OmpAllocateAdapter (registered under kOmpAllocatorPos) will consume it
  // on the immediately-following allocation call.  OmpFreeAdapter does not
  // need the handle: it passes 0 to __kmpc_free, and libomp recovers the
  // original allocator from the pointer's chunk header.
  omp::SetPendingOmpAllocStamp(handle, align);
}

RT_EXT_API_GROUP_END
}

} // namespace Fortran::runtime

#else // RT_GPU_TARGET

// GPU device build: provide no-op stubs that satisfy the link surface but
// never invoke any host-only kmpc/dlsym/TLS machinery.  `!$omp allocators`
// is emitted by the host lowering, so device code never calls these
// directly; we still emit definitions so any inadvertent reference from
// device code links cleanly.
namespace Fortran::runtime::omp {

RT_API_ATTRS void *OmpAllocate(std::size_t /*byteSize*/, std::size_t /*align*/,
    std::uintptr_t /*handle*/) {
  return nullptr;
}

RT_API_ATTRS void OmpFree(void * /*ptr*/, std::uintptr_t /*handle*/) {}

RT_API_ATTRS void SetPendingOmpAllocStamp(
    std::uintptr_t /*handle*/, std::size_t /*align*/) {}

RT_API_ATTRS bool ConsumePendingOmpAllocStamp(
    std::uintptr_t & /*handle*/, std::size_t & /*align*/) {
  return false;
}

RT_API_ATTRS void *OmpAllocateAdapter(
    std::size_t /*byteSize*/, std::int64_t * /*asyncObject*/) {
  return nullptr;
}

RT_API_ATTRS void OmpFreeAdapter(void * /*ptr*/) {}

void RegisterOmpAllocator() {}

} // namespace Fortran::runtime::omp

namespace Fortran::runtime {

extern "C" {
RT_EXT_API_GROUP_BEGIN

void RTDEF(OmpRegisterAllocator)() {}

void RTDEF(OmpAllocatorStamp)(Descriptor & /*descriptor*/,
    std::uintptr_t /*handle*/, std::size_t /*align*/) {}

RT_EXT_API_GROUP_END
}

} // namespace Fortran::runtime

#endif // RT_GPU_TARGET
