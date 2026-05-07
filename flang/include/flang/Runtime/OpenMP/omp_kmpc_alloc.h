//===-- include/flang/Runtime/OpenMP/omp_kmpc_alloc.h -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Public API + internal helpers for the host OpenMP allocator path that lives
// in the `flang_rt.openmp` library and dispatches Fortran ALLOCATABLE storage
// through `__kmpc_alloc` / `__kmpc_aligned_alloc` / `__kmpc_free`.
//
// This is the OpenMP analogue of `flang/Runtime/CUDA/allocator.h` and sits
// alongside `flang/Runtime/OpenMP/omp_alloc.h` (the offload-device allocator
// path that registers `omp_target_alloc` / `omp_target_free` into the
// allocator registry).
//
// The Flang lowering of the OpenMP 5.0+ `!$omp allocators` construct emits a
// call to RTNAME(OmpAllocatorStamp) immediately before the matching
// _FortranAAllocatableAllocate runtime call, so that Descriptor::Allocate can
// route the allocation through `__kmpc_alloc` / `__kmpc_aligned_alloc` and
// the matching deallocation through `__kmpc_free`.
//
// The OpenMP runtime symbols are resolved lazily via `dlsym(RTLD_DEFAULT, ...)`
// on first use, so flang_rt.openmp does not take a build-time dependency on
// libomp.
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_OPENMP_OMP_KMPC_ALLOC_H_
#define FORTRAN_RUNTIME_OPENMP_OMP_KMPC_ALLOC_H_

#include "flang/Common/api-attrs.h"
#include "flang/Runtime/descriptor-consts.h"
#include "flang/Runtime/entry-names.h"

#include <cstddef>
#include <cstdint>

namespace Fortran::runtime {
class Descriptor;

extern "C" {

/// Mark `descriptor` as OpenMP-allocated (kOmpAllocatorPos in the registry)
/// and stash the requested allocator handle and (optional) alignment in a
/// thread-local pending-stamp slot consumed by the next Descriptor::Allocate
/// call on the same thread.  This makes the OpenMP allocation path work even
/// for descriptors with no DescriptorAddendum (i.e. plain intrinsic-type
/// allocatables).  The stamp also lazily registers the OpenMP allocate / free
/// pair into the global allocator registry; subsequent Allocate / Deallocate
/// dispatch through the registry like every other allocator family.
void RTDECL(OmpAllocatorStamp)(
    Descriptor &, std::uintptr_t handle, std::size_t align = 0);

/// Register the OpenMP allocate / free entry into the global
/// allocator registry under kOmpAllocatorPos. Calling this is optional;
/// RTNAME(OmpAllocatorStamp) calls it lazily on first use.
void RTDECL(OmpRegisterAllocator)();

} // extern "C"
} // namespace Fortran::runtime

namespace Fortran::runtime::omp {

/// Allocate `byteSize` bytes via the OpenMP runtime using the supplied
/// allocator handle.  When `align` is non-zero the allocation goes through
/// `__kmpc_aligned_alloc`; otherwise `__kmpc_alloc` is used.  Returns nullptr
/// if the OpenMP runtime is not available or the allocation failed.
RT_API_ATTRS void *OmpAllocate(
    std::size_t byteSize, std::size_t align, std::uintptr_t handle);

/// Free a pointer previously returned by OmpAllocate, using the supplied
/// OpenMP allocator handle.  Silently no-ops if the OpenMP runtime is not
/// available.
RT_API_ATTRS void OmpFree(void *ptr, std::uintptr_t handle);

/// Record a pending (handle, align) pair on the calling thread.  The next
/// invocation of OmpAllocateAdapter on the same thread will consume the slot.
/// Overwrites any previously-stamped pair.
RT_API_ATTRS void SetPendingOmpAllocStamp(
    std::uintptr_t handle, std::size_t align);

/// Consume the pending stamp.  On a hit returns true and writes the stamped
/// (handle, align) pair into the out parameters; on a miss returns false and
/// leaves the out parameters untouched.
RT_API_ATTRS bool ConsumePendingOmpAllocStamp(
    std::uintptr_t &handle, std::size_t &align);

/// Adapter functions registered into the global Fortran::runtime
/// allocator-registry under kOmpAllocatorPos.  They consult the thread-local
/// pending stamp (set by RTNAME(OmpAllocatorStamp)) and dispatch through
/// OmpAllocate / OmpFree.
RT_API_ATTRS void *OmpAllocateAdapter(
    std::size_t byteSize, std::int64_t *asyncObject);
RT_API_ATTRS void OmpFreeAdapter(void *ptr);

/// Register the OmpAllocateAdapter / OmpFreeAdapter pair into the global
/// Fortran::runtime::allocatorRegistry under kOmpAllocatorPos.  Idempotent
/// and thread-safe; cheap to call repeatedly.
void RegisterOmpAllocator();

} // namespace Fortran::runtime::omp

#endif // FORTRAN_RUNTIME_OPENMP_OMP_KMPC_ALLOC_H_
