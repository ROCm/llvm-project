//===-- unittests/Runtime/OmpAllocator.cpp ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Verifies that Descriptor::Allocate / Descriptor::Deallocate dispatch through
// the OpenMP runtime when the descriptor's allocator-index is kOmpAllocatorPos
// and that the OpenMP allocator handle and alignment stamped via
// _FortranAOmpAllocatorStamp are forwarded through to
// __kmpc_alloc / __kmpc_aligned_alloc / __kmpc_free.
//
// We provide local definitions of __kmpc_alloc, __kmpc_aligned_alloc,
// __kmpc_free, and __kmpc_global_thread_num so that the lazy
// dlsym(RTLD_DEFAULT, ...) lookup in flang_rt.openmp's omp_kmpc_alloc.cpp
// finds them in the test binary.  The CMakeLists.txt for this unittest
// uses a linker dynamic-list (OmpAllocatorKmpcSymbols.list) to ensure
// only these four __kmpc_* symbols are exported as dynamic symbols, so
// the dlsym lookup cannot accidentally bind to anything else.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "flang-rt/runtime/descriptor.h"
#include "flang/Common/ISO_Fortran_binding_wrapper.h"
#include "flang/Runtime/OpenMP/omp_kmpc_alloc.h"
#include "flang/Runtime/allocator-registry-consts.h"
#include <cstdint>
#include <cstdlib>

namespace {

// Counters and last-seen handle, captured by our fake __kmpc_* shims.  The
// flang-rt OmpAllocate / OmpFree resolve these via dlsym(RTLD_DEFAULT, ...)
// the first time they are called and cache the function pointers.
int gAllocCount = 0;
int gAlignedAllocCount = 0;
int gFreeCount = 0;
std::uintptr_t gLastAllocHandle = 0;
std::uintptr_t gLastFreeHandle = 0;
std::size_t gLastAllocAlign = 0;
std::size_t gLastAllocSize = 0;

void ResetCounters() {
  gAllocCount = 0;
  gAlignedAllocCount = 0;
  gFreeCount = 0;
  gLastAllocHandle = 0;
  gLastFreeHandle = 0;
  gLastAllocAlign = 0;
  gLastAllocSize = 0;
}

} // namespace

extern "C" void *__kmpc_alloc(
    int /*gtid*/, std::size_t size, std::uintptr_t allocator) {
  ++gAllocCount;
  gLastAllocHandle = allocator;
  gLastAllocAlign = 0;
  gLastAllocSize = size;
  return std::malloc(size);
}

extern "C" void *__kmpc_aligned_alloc(int /*gtid*/, std::size_t align,
    std::size_t size, std::uintptr_t allocator) {
  ++gAlignedAllocCount;
  gLastAllocHandle = allocator;
  gLastAllocAlign = align;
  gLastAllocSize = size;
  // Real libomp's __kmpc_aligned_alloc tolerates `size` not being a multiple
  // of `align` (it rounds up internally).  POSIX's std::aligned_alloc, which
  // we wrap here for the test, does not -- it returns nullptr if size is not
  // a multiple of align.  Pad the size up to the next alignment boundary so
  // the fake exhibits the libomp-compatible behaviour.  Production code in
  // flang-rt's OmpAllocate intentionally relies on __kmpc_aligned_alloc to
  // do this padding for it.
  std::size_t padded = ((size + align - 1) / align) * align;
  return std::aligned_alloc(align, padded);
}

extern "C" void __kmpc_free(int /*gtid*/, void *ptr, std::uintptr_t allocator) {
  ++gFreeCount;
  gLastFreeHandle = allocator;
  std::free(ptr);
}

extern "C" int __kmpc_global_thread_num(void * /*ident*/) { return 0; }

using namespace Fortran::runtime;

TEST(OmpAllocatorTest, AllocateAndDeallocateUseOmpRuntime) {
  using Fortran::common::TypeCategory;

  ResetCounters();

  // INTEGER(4), ALLOCATABLE :: a(:).  The descriptor carries no addendum (as
  // is typical for intrinsic-type allocatables); the OMP allocator handle is
  // delivered via the thread-local pending-stamp slot that the runtime stamp
  // entry populates.  The deallocate path passes handle 0 to __kmpc_free and
  // relies on libomp recovering the original allocator from the chunk header.
  auto desc{Descriptor::Create(TypeCode{TypeCategory::Integer, 4}, 4,
      /*p=*/nullptr, /*rank=*/1, /*extent=*/nullptr,
      CFI_attribute_allocatable)};
  ASSERT_TRUE(desc);
  desc->GetDimension(0).SetBounds(1, 16);

  constexpr std::uintptr_t kTestHandle = 0xC0FFEEUL;
  // Make sure the OpenMP adapter is registered into the global allocator
  // registry (RTNAME(OmpAllocatorStamp) does this lazily; we go a level
  // lower in this test so we ask for it explicitly here).
  omp::RegisterOmpAllocator();
  desc->SetAllocIdx(kOmpAllocatorPos);
  omp::SetPendingOmpAllocStamp(kTestHandle, /*align=*/0);

  ASSERT_EQ(desc->Allocate(/*asyncObject=*/nullptr), 0);
  EXPECT_TRUE(desc->IsAllocated());
  EXPECT_EQ(gAllocCount, 1);
  EXPECT_EQ(gAlignedAllocCount, 0);
  EXPECT_EQ(gLastAllocHandle, kTestHandle);
  EXPECT_EQ(gLastAllocAlign, 0u);

  ASSERT_EQ(desc->Deallocate(), CFI_SUCCESS);
  EXPECT_FALSE(desc->IsAllocated());
  EXPECT_EQ(gFreeCount, 1);
  // Deallocate always passes handle 0 to __kmpc_free; libomp recovers the
  // original allocator from the chunk header.
  EXPECT_EQ(gLastFreeHandle, 0u);
}

TEST(OmpAllocatorTest, DefaultAllocatorBypassesOmpRuntime) {
  using Fortran::common::TypeCategory;

  ResetCounters();

  auto desc{Descriptor::Create(TypeCode{TypeCategory::Integer, 4}, 4,
      /*p=*/nullptr, /*rank=*/1, /*extent=*/nullptr,
      CFI_attribute_allocatable)};
  ASSERT_TRUE(desc);
  desc->GetDimension(0).SetBounds(1, 8);

  ASSERT_EQ(desc->Allocate(/*asyncObject=*/nullptr), 0);
  EXPECT_TRUE(desc->IsAllocated());
  ASSERT_EQ(desc->Deallocate(), CFI_SUCCESS);
  EXPECT_FALSE(desc->IsAllocated());

  EXPECT_EQ(gAllocCount, 0);
  EXPECT_EQ(gAlignedAllocCount, 0);
  EXPECT_EQ(gFreeCount, 0);
}

TEST(OmpAllocatorTest, RuntimeStampRoutesToOmp) {
  using Fortran::common::TypeCategory;

  ResetCounters();

  // Same as AllocateAndDeallocateUseOmpRuntime, but instead of stamping the
  // descriptor inline we go through the runtime entry point
  // _FortranAOmpAllocatorStamp that the Flang `!$omp allocators` lowering
  // uses.  align=0 => unaligned __kmpc_alloc path.
  auto desc{Descriptor::Create(TypeCode{TypeCategory::Integer, 4}, 4,
      /*p=*/nullptr, /*rank=*/1, /*extent=*/nullptr, CFI_attribute_allocatable,
      /*addendum=*/true)};
  ASSERT_TRUE(desc);
  desc->GetDimension(0).SetBounds(1, 4);

  constexpr std::uintptr_t kTestHandle = 0xBEEFU;
  RTNAME(OmpAllocatorStamp)(*desc, kTestHandle, /*align=*/0);

  ASSERT_EQ(desc->Allocate(/*asyncObject=*/nullptr), 0);
  EXPECT_EQ(gAllocCount, 1);
  EXPECT_EQ(gAlignedAllocCount, 0);
  EXPECT_EQ(gLastAllocHandle, kTestHandle);

  ASSERT_EQ(desc->Deallocate(), CFI_SUCCESS);
  EXPECT_EQ(gFreeCount, 1);
  // Deallocate passes handle 0 to __kmpc_free; libomp recovers the allocator
  // from the pointer's chunk header.
  EXPECT_EQ(gLastFreeHandle, 0u);
}

TEST(OmpAllocatorTest, RuntimeStampAlignRoutesToAlignedAlloc) {
  using Fortran::common::TypeCategory;

  ResetCounters();

  // Stamp with a non-zero alignment; allocate should go through
  // __kmpc_aligned_alloc carrying the alignment value.
  auto desc{Descriptor::Create(TypeCode{TypeCategory::Integer, 4}, 4,
      /*p=*/nullptr, /*rank=*/1, /*extent=*/nullptr, CFI_attribute_allocatable,
      /*addendum=*/true)};
  ASSERT_TRUE(desc);
  desc->GetDimension(0).SetBounds(1, 8);

  constexpr std::uintptr_t kTestHandle = 0xABCDU;
  constexpr std::size_t kAlign = 64;
  RTNAME(OmpAllocatorStamp)(*desc, kTestHandle, kAlign);

  ASSERT_EQ(desc->Allocate(/*asyncObject=*/nullptr), 0);
  EXPECT_EQ(gAllocCount, 0);
  EXPECT_EQ(gAlignedAllocCount, 1);
  EXPECT_EQ(gLastAllocHandle, kTestHandle);
  EXPECT_EQ(gLastAllocAlign, kAlign);

  ASSERT_EQ(desc->Deallocate(), CFI_SUCCESS);
  EXPECT_EQ(gFreeCount, 1);
  // Deallocate passes handle 0 to __kmpc_free; libomp recovers the allocator
  // from the pointer's chunk header.
  EXPECT_EQ(gLastFreeHandle, 0u);
}

TEST(OmpAllocatorTest, RuntimeStampWorksWithoutAddendum) {
  using Fortran::common::TypeCategory;

  ResetCounters();

  // Intrinsic-type allocatable with no addendum.  The stamp's (handle, align)
  // travels through the thread-local pending-stamp slot, so allocate still
  // dispatches through the OpenMP runtime even though addendum()==nullptr.
  auto desc{Descriptor::Create(TypeCode{TypeCategory::Integer, 4}, 4,
      /*p=*/nullptr, /*rank=*/1, /*extent=*/nullptr, CFI_attribute_allocatable,
      /*addendum=*/false)};
  ASSERT_TRUE(desc);
  desc->GetDimension(0).SetBounds(1, 4);
  ASSERT_EQ(desc->Addendum(), nullptr);

  constexpr std::uintptr_t kTestHandle = 0xFADEU;
  RTNAME(OmpAllocatorStamp)(*desc, kTestHandle, /*align=*/0);

  ASSERT_EQ(desc->Allocate(/*asyncObject=*/nullptr), 0);
  EXPECT_EQ(gAllocCount, 1);
  EXPECT_EQ(gLastAllocHandle, kTestHandle);

  ASSERT_EQ(desc->Deallocate(), CFI_SUCCESS);
  EXPECT_EQ(gFreeCount, 1);
  // Without an addendum the deallocate path has no stashed handle; it falls
  // back on handle 0 (omp_null_allocator), which libomp interprets as
  // "read the allocator from the pointer's chunk metadata".
  EXPECT_EQ(gLastFreeHandle, 0u);
}

TEST(OmpAllocatorTest, OrphanedStampDoesNotLeakIntoNextAllocate) {
  using Fortran::common::TypeCategory;

  ResetCounters();

  // Regression test for the orphaned-stamp invariant described in
  // flang-rt/lib/openmp/omp_kmpc_alloc.cpp:
  //
  //   * Stamping a descriptor without a matching Descriptor::Allocate
  //     leaves the thread-local pending-stamp slot live.
  //   * A *new* stamp on a *different* descriptor must overwrite that slot
  //     (last writer wins), not silently propagate the old handle.
  //
  // We exercise the second descriptor's allocate to confirm it sees only
  // the second stamp's handle.  In debug builds the runtime will also
  // emit a stderr warning about the orphaned first stamp; that is
  // diagnostic-only and intentionally not asserted on here so the test
  // works in both debug and release builds.
  auto descA{Descriptor::Create(TypeCode{TypeCategory::Integer, 4}, 4,
      /*p=*/nullptr, /*rank=*/1, /*extent=*/nullptr, CFI_attribute_allocatable,
      /*addendum=*/false)};
  auto descB{Descriptor::Create(TypeCode{TypeCategory::Integer, 4}, 4,
      /*p=*/nullptr, /*rank=*/1, /*extent=*/nullptr, CFI_attribute_allocatable,
      /*addendum=*/false)};
  ASSERT_TRUE(descA);
  ASSERT_TRUE(descB);
  descA->GetDimension(0).SetBounds(1, 4);
  descB->GetDimension(0).SetBounds(1, 4);

  constexpr std::uintptr_t kOrphanHandle = 0xDEAD0001UL;
  constexpr std::uintptr_t kWinningHandle = 0xDEAD0002UL;

  // Orphan: stamp descA, then never call Allocate on it.
  RTNAME(OmpAllocatorStamp)(*descA, kOrphanHandle, /*align=*/0);
  // Stomp on the orphan with a new stamp targeted at descB.
  RTNAME(OmpAllocatorStamp)(*descB, kWinningHandle, /*align=*/0);

  ASSERT_EQ(descB->Allocate(/*asyncObject=*/nullptr), 0);
  EXPECT_EQ(gAllocCount, 1);
  EXPECT_EQ(gLastAllocHandle, kWinningHandle)
      << "Orphan stamp's handle leaked into descB's allocate";

  ASSERT_EQ(descB->Deallocate(), CFI_SUCCESS);
  EXPECT_EQ(gFreeCount, 1);
  EXPECT_EQ(gLastFreeHandle, 0u);
}
