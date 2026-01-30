//===-------- Allocator.h - OpenMP memory allocator interface ---- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_ALLOCATOR_H
#define OMPTARGET_ALLOCATOR_H

#include "DeviceTypes.h"

namespace ompx {

namespace allocator {

static uint64_t constexpr ALIGNMENT = 16;

/// Allocate \p Size bytes.
[[gnu::alloc_size(1), gnu::assume_aligned(ALIGNMENT), gnu::malloc]] void *
alloc(uint64_t Size);

/// Free the allocation pointed to by \p Ptr.
void free(void *Ptr);

#if defined(__AMDGPU__) && defined(SANITIZER_AMDGPU)
/// Allocate \p Size bytes with caller PC for ASAN.
[[gnu::alloc_size(1), gnu::assume_aligned(ALIGNMENT), gnu::malloc]] void *
alloc(uint64_t Size, uint64_t PC);

/// Free the allocation pointed to by \p Ptr with caller PC for ASAN.
void free(void *Ptr, uint64_t PC);
#endif

} // namespace allocator

} // namespace ompx

extern "C" {
void *malloc(size_t Size);
void free(void *Ptr);
}

#endif
