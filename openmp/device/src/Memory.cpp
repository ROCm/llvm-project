//===--------- Memory.cpp - OpenMP device allocator interfaces --- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Allocator.h"
#include "DeviceTypes.h"

using namespace ompx;

extern "C" {
void *__kmpc_impl_alloc_aligned(size_t Size, size_t Align);
void __kmpc_impl_free_aligned(void *Ptr);
void *omp_alloc(size_t Size, omp_allocator_handle_t Allocator);
void omp_free(void *Ptr, omp_allocator_handle_t Allocator);
}

namespace {

static omp_allocator_handle_t DefaultAllocator;

static omp_allocator_handle_t getDefaultAllocator() {
  return DefaultAllocator ? DefaultAllocator : omp_default_mem_alloc;
}

static omp_allocator_handle_t
mapMemspaceToAllocator(omp_memspace_handle_t Memspace) {
  switch (Memspace) {
  case omp_default_mem_space:
    return omp_default_mem_alloc;
  case omp_large_cap_mem_space:
    return omp_large_cap_mem_alloc;
  case omp_const_mem_space:
    return omp_const_mem_alloc;
  case omp_high_bw_mem_space:
    return omp_high_bw_mem_alloc;
  case omp_low_lat_mem_space:
    return omp_low_lat_mem_alloc;
  default:
    return omp_null_allocator;
  }
}

static bool isGlobalAllocator(omp_allocator_handle_t Allocator) {
  switch (Allocator) {
  case omp_default_mem_alloc:
  case omp_large_cap_mem_alloc:
  case omp_const_mem_alloc:
  case omp_high_bw_mem_alloc:
  case omp_low_lat_mem_alloc:
    return true;
  default:
    return false;
  }
}

static bool isPowerOfTwo(size_t Value) {
  return Value && !(Value & (Value - 1));
}

} // namespace

extern "C" {

void *__kmpc_impl_alloc_aligned(size_t Size, size_t Align) {
  if (!isPowerOfTwo(Align))
    return nullptr;

  size_t Extra = sizeof(allocator_api::AlignedAllocHeader) + Align - 1;
  void *Base = allocator::alloc(Size + Extra);
  if (!Base)
    return nullptr;

  uintptr_t Ptr = reinterpret_cast<uintptr_t>(Base);
  uintptr_t Aligned =
      __builtin_align_up(Ptr + sizeof(allocator_api::AlignedAllocHeader), Align);
  auto *Header =
      reinterpret_cast<allocator_api::AlignedAllocHeader *>(Aligned) - 1;
  Header->BasePtr = Base;
  Header->Magic = allocator_api::AlignedAllocMagic;
  return reinterpret_cast<void *>(Aligned);
}

void __kmpc_impl_free_aligned(void *Ptr) {
  if (!Ptr)
    return;

  auto *Header =
      reinterpret_cast<allocator_api::AlignedAllocHeader *>(Ptr) - 1;
  allocator::free(Header->BasePtr);
}

omp_allocator_handle_t omp_init_allocator(omp_memspace_handle_t Memspace,
                                          int NumTraits,
                                          omp_alloctrait_t *Traits) {
  (void)NumTraits;
  (void)Traits;
  return mapMemspaceToAllocator(Memspace);
}

void omp_destroy_allocator(omp_allocator_handle_t Allocator) {
  (void)Allocator;
}

void omp_set_default_allocator(omp_allocator_handle_t Allocator) {
  DefaultAllocator =
      Allocator == omp_null_allocator ? omp_default_mem_alloc : Allocator;
}

omp_allocator_handle_t omp_get_default_allocator(void) {
  return getDefaultAllocator();
}

void *omp_aligned_alloc(size_t Align, size_t Size,
                        omp_allocator_handle_t Allocator) {
  Allocator = Allocator == omp_null_allocator ? getDefaultAllocator()
                                              : Allocator;
  if (!isGlobalAllocator(Allocator))
    return nullptr;
  if (!isPowerOfTwo(Align))
    return nullptr;
  return __kmpc_impl_alloc_aligned(Size, Align);
}

void *omp_calloc(size_t NumMembers, size_t Size,
                 omp_allocator_handle_t Allocator) {
  size_t TotalSize = 0;
  if (__builtin_mul_overflow(NumMembers, Size, &TotalSize))
    return nullptr;

  void *Ptr = omp_alloc(TotalSize, Allocator);
  if (Ptr)
    __builtin_memset(Ptr, 0, TotalSize);
  return Ptr;
}

void *omp_aligned_calloc(size_t Align, size_t NumMembers, size_t Size,
                         omp_allocator_handle_t Allocator) {
  size_t TotalSize = 0;
  if (__builtin_mul_overflow(NumMembers, Size, &TotalSize))
    return nullptr;

  void *Ptr = omp_aligned_alloc(Align, TotalSize, Allocator);
  if (Ptr)
    __builtin_memset(Ptr, 0, TotalSize);
  return Ptr;
}

void *omp_realloc(void *Ptr, size_t Size, omp_allocator_handle_t Allocator,
                  omp_allocator_handle_t FreeAllocator) {
  if (!Ptr)
    return omp_alloc(Size, Allocator);
  if (!Size) {
    omp_free(Ptr, FreeAllocator);
    return nullptr;
  }
  return nullptr;
}

} // extern "C"
