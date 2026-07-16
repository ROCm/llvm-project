//===- HotswapProfileTest.cpp - Unit tests for HotSwap profiling ----------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for the opt-in HotSwap rewrite profiler
/// (comgr-hotswap-internal.h). The profiler is always compiled and gated only
/// at run time (AMD_COMGR_TIME_STATISTICS), so these tests exercise both the
/// disabled and enabled per-rewrite sessions directly. See review on
/// ROCm/llvm-project#3364 and #3388.
///
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-internal.h"

#include "gtest/gtest.h"

using namespace COMGR::hotswap;

// Runtime gate: with AMD_COMGR_TIME_STATISTICS unset, the session must report
// disabled so no hot-path clock read happens.
TEST(HotswapProfile, ProfilingDisabledByDefault) {
  EXPECT_FALSE(hotswapProfilingEnabled());
}

// A disabled session is fully inert: every hook is a no-op and nothing lands in
// the local samples (runtime-off mode).
TEST(HotswapProfile, DisabledSessionRecordsNothing) {
  HotswapProfile Profile(/*Enabled=*/false);
  EXPECT_FALSE(Profile.enabled());

  {
    HotswapProfile::Scope S = Profile.time(HotswapMetric::Decode);
    S.addPatches(7);
  }
  Profile.count(HotswapMetric::JumpShort, 4);
  Profile.add(HotswapMetric::Trampoline, 1000, 2);

  EXPECT_EQ(Profile.sample(HotswapMetric::Decode).Calls, 0u);
  EXPECT_EQ(Profile.sample(HotswapMetric::JumpShort).Calls, 0u);
  EXPECT_EQ(Profile.sample(HotswapMetric::Trampoline).Calls, 0u);
  EXPECT_EQ(Profile.sample(HotswapMetric::Trampoline).Nanos, 0u);
}

// add() folds a pre-measured interval as one call, tracking totals and min/max.
TEST(HotswapProfile, EnabledSessionAddAccumulates) {
  HotswapProfile Profile(/*Enabled=*/true);
  EXPECT_TRUE(Profile.enabled());

  Profile.add(HotswapMetric::Decode, 500, 2);
  Profile.add(HotswapMetric::Decode, 1500, 3);

  const HotswapSample &S = Profile.sample(HotswapMetric::Decode);
  EXPECT_EQ(S.Calls, 2u);
  EXPECT_EQ(S.Nanos, 2000u);
  EXPECT_EQ(S.Patches, 5u);
  EXPECT_EQ(S.MinNanos, 500u);
  EXPECT_EQ(S.MaxNanos, 1500u);
}

// count() bumps only the call count (jump-outcome rows carry no wall time).
TEST(HotswapProfile, CountOnlyRecordsCalls) {
  HotswapProfile Profile(/*Enabled=*/true);
  Profile.count(HotswapMetric::JumpShort);
  Profile.count(HotswapMetric::JumpShort, 2);

  const HotswapSample &S = Profile.sample(HotswapMetric::JumpShort);
  EXPECT_EQ(S.Calls, 3u);
  EXPECT_EQ(S.Nanos, 0u);
  EXPECT_EQ(S.Patches, 0u);
}

// The RAII Scope records exactly one call (with its patches) on destruction.
TEST(HotswapProfile, ScopeRecordsOnceOnDestruction) {
  HotswapProfile Profile(/*Enabled=*/true);
  {
    HotswapProfile::Scope S = Profile.time(HotswapMetric::TrampolineDs2Addr);
    S.addPatches(1);
  }
  const HotswapSample &S = Profile.sample(HotswapMetric::TrampolineDs2Addr);
  EXPECT_EQ(S.Calls, 1u);
  EXPECT_EQ(S.Patches, 1u);
}

// finish() is idempotent: an explicit finish() followed by destruction records
// a single call, not two.
TEST(HotswapProfile, ScopeFinishIsIdempotent) {
  HotswapProfile Profile(/*Enabled=*/true);
  {
    HotswapProfile::Scope S = Profile.time(HotswapMetric::WmmaSplit);
    S.addPatches(2);
    S.finish();
    S.finish();
  }
  const HotswapSample &S = Profile.sample(HotswapMetric::WmmaSplit);
  EXPECT_EQ(S.Calls, 1u);
  EXPECT_EQ(S.Patches, 2u);
}

// A Scope from a disabled session never records, even when patches are added.
TEST(HotswapProfile, DisabledScopeRecordsNothing) {
  HotswapProfile Profile(/*Enabled=*/false);
  {
    HotswapProfile::Scope S = Profile.time(HotswapMetric::Trampoline);
    S.addPatches(9);
  }
  EXPECT_EQ(Profile.sample(HotswapMetric::Trampoline).Calls, 0u);
}

// The static label/parent/partition table must stay in lockstep with the enum:
// every row has a label, every child points at a valid top-level parent, and
// the phases that partition rewrite_total exist while total/unaccounted do not.
TEST(HotswapProfile, MetricInfoTableWellFormed) {
  size_t PartitionCount = 0;
  for (size_t I = 0; I < HotswapMetricCount; ++I) {
    const HotswapMetricInfo &Info = hotswapMetricInfo[I];
    ASSERT_NE(Info.Label, nullptr);
    EXPECT_NE(Info.Label[0], '\0');
    if (Info.Parent != HotswapMetric::Count) {
      // A child's parent must itself be a top-level row.
      const size_t ParentIdx = static_cast<size_t>(Info.Parent);
      ASSERT_LT(ParentIdx, HotswapMetricCount);
      EXPECT_EQ(hotswapMetricInfo[ParentIdx].Parent, HotswapMetric::Count);
    }
    if (Info.PartitionsTotal)
      ++PartitionCount;
  }
  EXPECT_GT(PartitionCount, 0u);
  EXPECT_FALSE(
      hotswapMetricInfo[static_cast<size_t>(HotswapMetric::RewriteTotal)]
          .PartitionsTotal);
  EXPECT_FALSE(
      hotswapMetricInfo[static_cast<size_t>(HotswapMetric::Unaccounted)]
          .PartitionsTotal);
  EXPECT_STREQ(
      hotswapMetricInfo[static_cast<size_t>(HotswapMetric::RewriteTotal)].Label,
      "phase:rewrite_total");
}
