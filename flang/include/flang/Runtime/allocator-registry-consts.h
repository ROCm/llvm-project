//===-- include/flang/Runtime/allocator-registry-consts.h -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_ALLOCATOR_REGISTRY_CONSTS_H_
#define FORTRAN_RUNTIME_ALLOCATOR_REGISTRY_CONSTS_H_

RT_OFFLOAD_VAR_GROUP_BEGIN

static constexpr unsigned kDefaultAllocator = 0;

// Allocator used for CUF
static constexpr unsigned kPinnedAllocatorPos = 1;
static constexpr unsigned kDeviceAllocatorPos = 2;
static constexpr unsigned kManagedAllocatorPos = 3;
static constexpr unsigned kUnifiedAllocatorPos = 4;
// Allocator used for the OpenMP `allocators` construct over Fortran
// allocatable arrays.  The OpenMP allocator handle (e.g. omp_high_bw_mem_alloc
// or a user-defined handle from omp_init_allocator) is *not* stored on the
// descriptor itself; instead the matching __kmpc_free is given handle 0
// (omp_null_allocator) and libomp recovers the original allocator from the
// pointer's chunk metadata.  This keeps the descriptor layout
// ABI-compatible with non-OpenMP descriptors and avoids needing to allocate
// an addendum for plain intrinsic-type allocatables.  See
// flang/include/flang/Runtime/OpenMP/omp_kmpc_alloc.h for the detailed
// stamp / consume / dispatch protocol.
static constexpr unsigned kOmpAllocatorPos = 5;
static constexpr unsigned kDefaultArrayAlignment = 64;
RT_OFFLOAD_VAR_GROUP_END

#endif /* FORTRAN_RUNTIME_ALLOCATOR_REGISTRY_CONSTS_H_ */
