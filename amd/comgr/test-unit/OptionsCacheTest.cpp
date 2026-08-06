//===- OptionsCacheTest.cpp - Tests for comgr-options-cache.cpp ----------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Exercises COMGR::acquireOptionsScope()'s locking/fingerprint semantics in
// isolation. The real parseLLVMOptions()/clearLLVMOptions() (comgr.cpp,
// comgr-compiler.cpp) and CachedCommandAdaptor::addUInt()/addString()
// (comgr-cache-command.cpp) are not linked in -- see the link-seam
// definitions below -- so only comgr-options-cache.cpp's own logic is under
// test here.
//
//===----------------------------------------------------------------------===//

#include "comgr-cache-command.h"
#include "comgr-options-cache.h"
#include "comgr.h"

#include "gtest/gtest.h"

#include <atomic>
#include <condition_variable>
#include <cstring>
#include <future>
#include <mutex>
#include <thread>
#include <vector>

// A cyclic barrier for exactly N threads. std::barrier is C++20; the
// test-unit targets are pinned to C++17 (see comgr_configure_test_target()
// in CMakeLists.txt), so this is a minimal, correctness-only substitute.
namespace {
class SimpleBarrier {
public:
  explicit SimpleBarrier(unsigned Count) : Count(Count), Waiting(0), Gen(0) {}

  void wait() {
    std::unique_lock<std::mutex> Lock(Mutex);
    unsigned MyGen = Gen;
    if (++Waiting == Count) {
      Waiting = 0;
      ++Gen;
      CV.notify_all();
      return;
    }
    CV.wait(Lock, [&] { return Gen != MyGen; });
  }

private:
  std::mutex Mutex;
  std::condition_variable CV;
  unsigned Count;
  unsigned Waiting;
  unsigned Gen;
};
} // namespace

namespace COMGR {

// -- Link seams -----------------------------------------------------------
//
// The real implementations live in comgr-compiler.cpp/comgr.cpp, which pull
// in the full Clang driver, and comgr-cache-command.cpp, which pulls in
// device-lib/version identifiers unrelated to this test. Providing our own
// definitions here lets us compile-but-not-link the rest of comgr while
// still testing the real comgr-options-cache.cpp logic against them.

std::atomic<int> ParseCount{0};
std::atomic<int> ClearCount{0};
std::atomic<bool> ForceParseFailure{false};

amd_comgr_status_t parseLLVMOptions(const std::vector<std::string> &Options) {
  (void)Options;
  ParseCount.fetch_add(1, std::memory_order_relaxed);
  if (ForceParseFailure.load(std::memory_order_relaxed)) {
    return AMD_COMGR_STATUS_ERROR;
  }
  return AMD_COMGR_STATUS_SUCCESS;
}

void clearLLVMOptions() { ClearCount.fetch_add(1, std::memory_order_relaxed); }

// Copied verbatim from comgr-cache-command.cpp.
void CachedCommandAdaptor::addUInt(CachedCommandAdaptor::HashAlgorithm &H,
                                    uint64_t I) {
  uint8_t Bytes[sizeof(I)];
  memcpy(&Bytes, &I, sizeof(I));
  H.update(Bytes);
}

void CachedCommandAdaptor::addString(CachedCommandAdaptor::HashAlgorithm &H,
                                      llvm::StringRef S) {
  addUInt(H, S.size());
  H.update(S);
}

} // namespace COMGR

using namespace COMGR;

namespace {

void resetFakes() {
  resetOptionsCacheStateForTest();
  ParseCount.store(0, std::memory_order_relaxed);
  ClearCount.store(0, std::memory_order_relaxed);
  ForceParseFailure.store(false, std::memory_order_relaxed);
}

} // namespace

TEST(OptionsCache, SameFingerprintRunsConcurrently) {
  resetFakes();

  // Warm the cache to a known, non-empty option set.
  {
    OptionsScopeGuard Warm = acquireOptionsScope({"-a"});
    EXPECT_TRUE(Warm.isExclusive());
  }
  EXPECT_EQ(ParseCount.load(), 1);

  constexpr unsigned NumThreads = 8;
  SimpleBarrier Barrier(NumThreads);
  std::atomic<int> ConcurrentCount{0};
  std::atomic<int> MaxConcurrentCount{0};
  // Not std::vector<bool>: its bit-packed representation makes concurrent
  // writes to different indices a data race even though the indices are
  // logically distinct.
  std::vector<char> WasExclusive(NumThreads, true);
  std::vector<std::thread> Threads;

  for (unsigned I = 0; I < NumThreads; ++I) {
    Threads.emplace_back([&, I] {
      OptionsScopeGuard Guard = acquireOptionsScope({"-a"});
      WasExclusive[I] = Guard.isExclusive();

      int Now = ConcurrentCount.fetch_add(1, std::memory_order_relaxed) + 1;
      int Prev = MaxConcurrentCount.load(std::memory_order_relaxed);
      while (Now > Prev &&
             !MaxConcurrentCount.compare_exchange_weak(Prev, Now)) {
      }

      // All threads rendezvous while still holding their guard, proving
      // the shared lock allows them to overlap.
      Barrier.wait();
      ConcurrentCount.fetch_sub(1, std::memory_order_relaxed);
    });
  }
  for (std::thread &T : Threads) {
    T.join();
  }

  EXPECT_EQ(MaxConcurrentCount.load(), static_cast<int>(NumThreads));
  for (unsigned I = 0; I < NumThreads; ++I) {
    EXPECT_FALSE(WasExclusive[I]) << "thread " << I;
  }
  // No reapply should have happened during the concurrent section: the
  // fingerprint never changed. Both counters should still read their
  // warm-up values (one clear+parse to get from the reset state to "-a").
  EXPECT_EQ(ParseCount.load(), 1);
  EXPECT_EQ(ClearCount.load(), 1);
}

TEST(OptionsCache, DifferentFingerprintBlocksUntilFirstReleases) {
  resetFakes();

  std::promise<void> HolderReady;
  std::future<void> HolderReadyFuture = HolderReady.get_future();
  std::promise<void> ReleaseHolder;
  std::future<void> ReleaseHolderFuture = ReleaseHolder.get_future();
  std::atomic<bool> SecondAcquired{false};

  std::thread Holder([&] {
    OptionsScopeGuard Guard = acquireOptionsScope({"-x"});
    EXPECT_TRUE(Guard.isExclusive());
    HolderReady.set_value();
    ReleaseHolderFuture.wait();
  });

  HolderReadyFuture.wait();

  std::thread Second([&] {
    OptionsScopeGuard Guard = acquireOptionsScope({"-y"});
    EXPECT_TRUE(Guard.isExclusive());
    SecondAcquired.store(true, std::memory_order_relaxed);
  });

  // Give the second thread ample opportunity to (incorrectly) proceed while
  // the holder still owns the exclusive lock.
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_FALSE(SecondAcquired.load(std::memory_order_relaxed));

  ReleaseHolder.set_value();
  Holder.join();
  Second.join();

  EXPECT_TRUE(SecondAcquired.load(std::memory_order_relaxed));
  EXPECT_EQ(ParseCount.load(), 2);
}

TEST(OptionsCache, RacingColdCallsConvergeToOneWinner) {
  resetFakes();

  constexpr unsigned NumThreads = 8;
  SimpleBarrier Barrier(NumThreads);
  std::vector<std::thread> Threads;
  for (unsigned I = 0; I < NumThreads; ++I) {
    Threads.emplace_back([&] {
      Barrier.wait();
      // Not all racing threads are guaranteed to observe the exclusive
      // path: a thread still blocked on its initial shared-lock attempt
      // when the winner finishes may wake to find the fingerprint already
      // applied, and correctly take the fast/shared path instead. Only the
      // aggregate invariants below (exactly one reset+reapply happened)
      // are guaranteed.
      OptionsScopeGuard Guard = acquireOptionsScope({"-z"});
      (void)Guard;
    });
  }
  for (std::thread &T : Threads) {
    T.join();
  }

  // Regardless of how many threads took the exclusive-lock path, only one
  // should have actually reset and reapplied the registry.
  EXPECT_EQ(ParseCount.load(), 1);
  EXPECT_EQ(ClearCount.load(), 1);

  // A subsequent same-fingerprint call should now be warm.
  OptionsScopeGuard Guard = acquireOptionsScope({"-z"});
  EXPECT_FALSE(Guard.isExclusive());
  EXPECT_EQ(ParseCount.load(), 1);
}

TEST(OptionsCache, EmptyFingerprintIsWarmAfterReset) {
  resetFakes();

  OptionsScopeGuard Guard = acquireOptionsScope({});
  EXPECT_FALSE(Guard.isExclusive());
  EXPECT_EQ(ParseCount.load(), 0);
  EXPECT_EQ(ClearCount.load(), 0);
}

TEST(OptionsCache, FailedReapplyPoisonsCacheUntilNextSuccess) {
  resetFakes();

  ForceParseFailure.store(true, std::memory_order_relaxed);
  {
    OptionsScopeGuard Guard = acquireOptionsScope({"-fail"});
    EXPECT_TRUE(Guard.isExclusive());
  }
  EXPECT_EQ(ParseCount.load(), 1);

  ForceParseFailure.store(false, std::memory_order_relaxed);

  // Even a request for the empty set -- the cache's initial state -- must
  // not be trusted after a failed reapply, since the registry's actual
  // contents are now unknown.
  {
    OptionsScopeGuard Guard = acquireOptionsScope({});
    EXPECT_TRUE(Guard.isExclusive());
  }
  EXPECT_EQ(ParseCount.load(), 2);

  // Now that the reapply succeeded, the same request should be warm.
  OptionsScopeGuard Guard = acquireOptionsScope({});
  EXPECT_FALSE(Guard.isExclusive());
  EXPECT_EQ(ParseCount.load(), 2);
}
