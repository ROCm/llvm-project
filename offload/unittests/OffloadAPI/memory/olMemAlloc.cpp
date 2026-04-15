//===------- Offload API tests - olMemAlloc -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

#include <atomic>
#include <vector>

using olMemAllocTest = OffloadDeviceTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olMemAllocTest);

TEST_P(olMemAllocTest, SuccessAllocManaged) {
  void *Alloc = nullptr;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED, 1024, &Alloc));
  ASSERT_NE(Alloc, nullptr);
  olMemFree(Alloc);
}

TEST_P(olMemAllocTest, SuccessAllocHost) {
  void *Alloc = nullptr;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_HOST, 1024, &Alloc));
  ASSERT_NE(Alloc, nullptr);
  olMemFree(Alloc);
}

TEST_P(olMemAllocTest, SuccessAllocDevice) {
  void *Alloc = nullptr;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_DEVICE, 1024, &Alloc));
  ASSERT_NE(Alloc, nullptr);
  olMemFree(Alloc);
}

TEST_P(olMemAllocTest, SuccessAllocMany) {
  std::vector<void *> Allocs;
  Allocs.reserve(1000);

  constexpr ol_alloc_type_t TYPES[3] = {
      OL_ALLOC_TYPE_DEVICE, OL_ALLOC_TYPE_MANAGED, OL_ALLOC_TYPE_HOST};

  for (size_t I = 1; I < 1000; I++) {
    void *Alloc = nullptr;
    ASSERT_SUCCESS(olMemAlloc(Device, TYPES[I % 3], 1024 * I, &Alloc));
    ASSERT_NE(Alloc, nullptr);

    Allocs.push_back(Alloc);
  }

  for (auto *A : Allocs) {
    olMemFree(A);
  }
}

TEST_P(olMemAllocTest, SuccessAllocTinyChurn) {
  constexpr size_t Iterations = 20000;
  constexpr size_t TinyAllocBytes = 8;

  for (size_t I = 0; I < Iterations; ++I) {
    void *Alloc = nullptr;
    if (auto Err =
            olMemAlloc(Device, OL_ALLOC_TYPE_DEVICE, TinyAllocBytes, &Alloc)) {
      if (Err->Code == OL_ERRC_UNSUPPORTED)
        GTEST_SKIP() << "Device allocation unsupported for tiny churn test";
      GTEST_FAIL() << "olMemAlloc failed with " << Err->Code << ": "
                   << Err->Details;
    }

    ASSERT_NE(Alloc, nullptr);
    ASSERT_SUCCESS(olMemFree(Alloc));
  }
}

TEST_P(olMemAllocTest, SuccessAllocTinyParallelChurn) {
  constexpr size_t IterationsPerThread = 2000;
  constexpr size_t TinyAllocBytes = 8;

  void *ProbeAlloc = nullptr;
  if (auto Err =
          olMemAlloc(Device, OL_ALLOC_TYPE_DEVICE, TinyAllocBytes, &ProbeAlloc)) {
    if (Err->Code == OL_ERRC_UNSUPPORTED)
      GTEST_SKIP() << "Device allocation unsupported for tiny churn test";
    GTEST_FAIL() << "olMemAlloc failed with " << Err->Code << ": "
                 << Err->Details;
  }
  ASSERT_NE(ProbeAlloc, nullptr);
  ASSERT_SUCCESS(olMemFree(ProbeAlloc));

  std::atomic<int> FirstErrorCode{OL_ERRC_SUCCESS};
  threadify([&](size_t) {
    for (size_t I = 0; I < IterationsPerThread; ++I) {
      if (FirstErrorCode.load(std::memory_order_relaxed) != OL_ERRC_SUCCESS)
        return;

      void *Alloc = nullptr;
      if (auto Err =
              olMemAlloc(Device, OL_ALLOC_TYPE_DEVICE, TinyAllocBytes, &Alloc)) {
        FirstErrorCode.store(Err->Code, std::memory_order_relaxed);
        return;
      }
      if (!Alloc) {
        FirstErrorCode.store(OL_ERRC_UNKNOWN, std::memory_order_relaxed);
        return;
      }

      if (auto Err = olMemFree(Alloc)) {
        FirstErrorCode.store(Err->Code, std::memory_order_relaxed);
        return;
      }
    }
  });

  ASSERT_EQ(FirstErrorCode.load(std::memory_order_relaxed), OL_ERRC_SUCCESS);
}

TEST_P(olMemAllocTest, InvalidNullDevice) {
  void *Alloc = nullptr;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olMemAlloc(nullptr, OL_ALLOC_TYPE_DEVICE, 1024, &Alloc));
}

TEST_P(olMemAllocTest, InvalidNullOutPtr) {
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olMemAlloc(Device, OL_ALLOC_TYPE_DEVICE, 1024, nullptr));
}
