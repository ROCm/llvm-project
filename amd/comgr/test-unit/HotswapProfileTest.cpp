//===- HotswapProfileTest.cpp - Unit tests for the HotSwap profiler -------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Tests for the typed HotSwap rewrite profiler in comgr-hotswap-internal.h
/// (HotswapProfile session, HotswapProfileSink, HotswapProfileSession). The
/// profiler is compiled in only under ENABLE_HOTSWAP_PROFILE, which the
/// amd_comgr library sets PRIVATE -- so the enabled branch never reaches the
/// other unit-test binaries and would otherwise be dead in CI. This target
/// defines the macro itself (see test-unit/CMakeLists.txt) and covers both
/// runtime modes:
///   * HOTSWAP_PROFILE unset -- the session is dormant and reads no clock;
///   * HOTSWAP_PROFILE set   -- typed timings / counts accumulate locally and
///                              merge into the process-wide sink once.
///
//===----------------------------------------------------------------------===//

#ifndef ENABLE_HOTSWAP_PROFILE
#error "HotswapProfileTest must be built with ENABLE_HOTSWAP_PROFILE"
#endif

#include "comgr-hotswap-internal.h"
#include "gtest/gtest.h"

#include <cstdlib>

using namespace COMGR::hotswap;

namespace {

// Portable set/clear of the HOTSWAP_PROFILE runtime gate. An empty value reads
// back as disabled (the session checks for a non-empty, non-"0" value).
void setProfileEnv(bool On) {
#ifdef _WIN32
  _putenv_s("HOTSWAP_PROFILE", On ? "1" : "");
#else
  if (On)
    setenv("HOTSWAP_PROFILE", "1", /*overwrite=*/1);
  else
    unsetenv("HOTSWAP_PROFILE");
#endif
}

} // namespace

// A session constructed with the runtime gate off is inert: enabled() is false
// and neither time() nor count() records anything. (The Scope constructor only
// samples the clock when the session is enabled, so a disabled session performs
// no clock reads at all.)
TEST(HotswapProfile, RuntimeDisabledIsInert) {
  setProfileEnv(false);
  HotswapProfile Profile;
  EXPECT_FALSE(Profile.enabled());

  {
    HotswapProfile::Scope S = Profile.time(HotswapMetric::RewriteTotal);
    S.addPatches(7);
  }
  Profile.count(HotswapMetric::JumpNopSled, 5);

  const HotswapSample &Total = Profile.sample(HotswapMetric::RewriteTotal);
  EXPECT_EQ(Total.Calls, 0u);
  EXPECT_EQ(Total.Nanos, 0u);
  EXPECT_EQ(Total.Patches, 0u);
  EXPECT_EQ(Profile.sample(HotswapMetric::JumpNopSled).Calls, 0u);

  // merge() on a disabled session is a no-op and must not touch the sink.
  const uint64_t Before =
      HotswapProfileSink::get().total(HotswapMetric::RewriteTotal).Calls;
  Profile.merge();
  EXPECT_EQ(HotswapProfileSink::get().total(HotswapMetric::RewriteTotal).Calls,
            Before);
}

// A session constructed with the runtime gate on records typed timings and
// counts into its local samples.
TEST(HotswapProfile, RuntimeEnabledRecordsLocally) {
  setProfileEnv(true);
  HotswapProfile Profile;
  ASSERT_TRUE(Profile.enabled());

  for (int I = 0; I < 2; ++I) {
    HotswapProfile::Scope S = Profile.time(HotswapMetric::TrampolineDs2Addr);
    S.addPatches(1);
  }
  Profile.count(HotswapMetric::JumpShort, 3);

  const HotswapSample &Ds2Addr =
      Profile.sample(HotswapMetric::TrampolineDs2Addr);
  EXPECT_EQ(Ds2Addr.Calls, 2u);
  EXPECT_EQ(Ds2Addr.Patches, 2u);
  // Both bounds were assigned from real clock reads, so the invariant holds
  // (equal is fine when the scopes are faster than the clock resolution).
  EXPECT_LE(Ds2Addr.MinNanos, Ds2Addr.MaxNanos);

  EXPECT_EQ(Profile.sample(HotswapMetric::JumpShort).Calls, 3u);
}

// An enabled session merges its local samples into the process-wide sink
// exactly once, adding to whatever the sink already holds.
TEST(HotswapProfile, MergeAccumulatesIntoSink) {
  setProfileEnv(true);
  HotswapProfileSink &Sink = HotswapProfileSink::get();
  const uint64_t CallsBefore =
      Sink.total(HotswapMetric::TrampolineAddtid).Calls;
  const uint64_t PatchesBefore =
      Sink.total(HotswapMetric::TrampolineAddtid).Patches;
  const uint64_t JumpBefore = Sink.total(HotswapMetric::JumpLong).Calls;

  {
    HotswapProfile Profile;
    ASSERT_TRUE(Profile.enabled());
    {
      // Close the timed scope before merging so its sample is recorded.
      HotswapProfile::Scope S = Profile.time(HotswapMetric::TrampolineAddtid);
      S.addPatches(4);
    }
    Profile.count(HotswapMetric::JumpLong, 2);
    Profile.merge();
  }

  EXPECT_EQ(Sink.total(HotswapMetric::TrampolineAddtid).Calls,
            CallsBefore + 1u);
  EXPECT_EQ(Sink.total(HotswapMetric::TrampolineAddtid).Patches,
            PatchesBefore + 4u);
  EXPECT_EQ(Sink.total(HotswapMetric::JumpLong).Calls, JumpBefore + 2u);
}

// The RAII session times phase:rewrite_total and merges exactly once when it
// leaves scope.
TEST(HotswapProfile, SessionTimesRewriteTotalOnce) {
  setProfileEnv(true);
  HotswapProfileSink &Sink = HotswapProfileSink::get();
  const uint64_t Before = Sink.total(HotswapMetric::RewriteTotal).Calls;
  {
    HotswapProfileSession Session;
    ASSERT_TRUE(Session.profile().enabled());
    HotswapProfile::Scope S = Session.profile().time(HotswapMetric::Decode);
  }
  EXPECT_EQ(Sink.total(HotswapMetric::RewriteTotal).Calls, Before + 1u);
}
