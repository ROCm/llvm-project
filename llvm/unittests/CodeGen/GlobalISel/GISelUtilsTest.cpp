//===- GISelUtilsTest.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GISelMITest.h"
#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {
static const LLT S1 = LLT::scalar(1);
static const LLT S8 = LLT::scalar(8);
static const LLT S16 = LLT::scalar(16);
static const LLT S32 = LLT::scalar(32);
static const LLT S64 = LLT::scalar(64);
static const LLT P0 = LLT::pointer(0, 64);
static const LLT P1 = LLT::pointer(1, 32);

static const LLT V2S8 = LLT::fixed_vector(2, 8);
static const LLT V4S8 = LLT::fixed_vector(4, 8);
static const LLT V8S8 = LLT::fixed_vector(8, 8);

static const LLT V2S16 = LLT::fixed_vector(2, 16);
static const LLT V3S16 = LLT::fixed_vector(3, 16);
static const LLT V4S16 = LLT::fixed_vector(4, 16);

static const LLT V2S32 = LLT::fixed_vector(2, 32);
static const LLT V3S32 = LLT::fixed_vector(3, 32);
static const LLT V4S32 = LLT::fixed_vector(4, 32);
static const LLT V6S32 = LLT::fixed_vector(6, 32);

static const LLT V2S64 = LLT::fixed_vector(2, 64);
static const LLT V3S64 = LLT::fixed_vector(3, 64);
static const LLT V4S64 = LLT::fixed_vector(4, 64);

static const LLT V2P0 = LLT::fixed_vector(2, P0);
static const LLT V3P0 = LLT::fixed_vector(3, P0);
static const LLT V4P0 = LLT::fixed_vector(4, P0);
static const LLT V6P0 = LLT::fixed_vector(6, P0);

static const LLT V2P1 = LLT::fixed_vector(2, P1);
static const LLT V4P1 = LLT::fixed_vector(4, P1);

static const LLT NXV1S1 = LLT::scalable_vector(1, S1);
static const LLT NXV2S1 = LLT::scalable_vector(2, S1);
static const LLT NXV3S1 = LLT::scalable_vector(3, S1);
static const LLT NXV4S1 = LLT::scalable_vector(4, S1);
static const LLT NXV12S1 = LLT::scalable_vector(12, S1);
static const LLT NXV32S1 = LLT::scalable_vector(32, S1);
static const LLT NXV64S1 = LLT::scalable_vector(64, S1);
static const LLT NXV128S1 = LLT::scalable_vector(128, S1);
static const LLT NXV384S1 = LLT::scalable_vector(384, S1);

static const LLT NXV1S32 = LLT::scalable_vector(1, S32);
static const LLT NXV2S32 = LLT::scalable_vector(2, S32);
static const LLT NXV3S32 = LLT::scalable_vector(3, S32);
static const LLT NXV4S32 = LLT::scalable_vector(4, S32);
static const LLT NXV8S32 = LLT::scalable_vector(8, S32);
static const LLT NXV12S32 = LLT::scalable_vector(12, S32);
static const LLT NXV24S32 = LLT::scalable_vector(24, S32);

static const LLT NXV1S64 = LLT::scalable_vector(1, S64);
static const LLT NXV2S64 = LLT::scalable_vector(2, S64);
static const LLT NXV3S64 = LLT::scalable_vector(3, S64);
static const LLT NXV4S64 = LLT::scalable_vector(4, S64);
static const LLT NXV6S64 = LLT::scalable_vector(6, S64);
static const LLT NXV12S64 = LLT::scalable_vector(12, S64);

static const LLT NXV1P0 = LLT::scalable_vector(1, P0);
static const LLT NXV2P0 = LLT::scalable_vector(2, P0);
static const LLT NXV3P0 = LLT::scalable_vector(3, P0);
static const LLT NXV4P0 = LLT::scalable_vector(4, P0);
static const LLT NXV12P0 = LLT::scalable_vector(12, P0);

static void collectNonCopyMI(SmallVectorImpl<MachineInstr *> &MIList,
                             MachineFunction *MF) {
  for (auto &MBB : *MF)
    for (MachineInstr &MI : MBB) {
      if (MI.getOpcode() != TargetOpcode::COPY)
        MIList.push_back(&MI);
    }
}

TEST(GISelUtilsTest, getGCDType) {
  EXPECT_EQ(S1, getGCDType(S1, S1));
  EXPECT_EQ(S32, getGCDType(S32, S32));
  EXPECT_EQ(S1, getGCDType(S1, S32));
  EXPECT_EQ(S1, getGCDType(S32, S1));
  EXPECT_EQ(S16, getGCDType(S16, S32));
  EXPECT_EQ(S16, getGCDType(S32, S16));

  EXPECT_EQ(V2S32, getGCDType(V2S32, V2S32));
  EXPECT_EQ(S32, getGCDType(V3S32, V2S32));
  EXPECT_EQ(S32, getGCDType(V2S32, V3S32));

  EXPECT_EQ(V2S16, getGCDType(V4S16, V2S16));
  EXPECT_EQ(V2S16, getGCDType(V2S16, V4S16));

  EXPECT_EQ(V2S32, getGCDType(V4S32, V2S32));
  EXPECT_EQ(V2S32, getGCDType(V2S32, V4S32));

  EXPECT_EQ(S16, getGCDType(P0, S16));
  EXPECT_EQ(S16, getGCDType(S16, P0));

  EXPECT_EQ(S32, getGCDType(P0, S32));
  EXPECT_EQ(S32, getGCDType(S32, P0));

  EXPECT_EQ(P0, getGCDType(P0, S64));
  EXPECT_EQ(S64, getGCDType(S64, P0));

  EXPECT_EQ(S32, getGCDType(P0, P1));
  EXPECT_EQ(S32, getGCDType(P1, P0));

  EXPECT_EQ(P0, getGCDType(V3P0, V2P0));
  EXPECT_EQ(P0, getGCDType(V2P0, V3P0));

  EXPECT_EQ(P0, getGCDType(P0, V2P0));
  EXPECT_EQ(P0, getGCDType(V2P0, P0));


  EXPECT_EQ(V2P0, getGCDType(V2P0, V2P0));
  EXPECT_EQ(P0, getGCDType(V3P0, V2P0));
  EXPECT_EQ(P0, getGCDType(V2P0, V3P0));
  EXPECT_EQ(V2P0, getGCDType(V4P0, V2P0));

  EXPECT_EQ(V2P0, getGCDType(V2P0, V4P1));
  EXPECT_EQ(V4P1, getGCDType(V4P1, V2P0));

  EXPECT_EQ(V2P0, getGCDType(V4P0, V4P1));
  EXPECT_EQ(V4P1, getGCDType(V4P1, V4P0));

  // Elements have same size, but have different pointeriness, so prefer the
  // original element type.
  EXPECT_EQ(V2P0, getGCDType(V2P0, V4S64));
  EXPECT_EQ(V2S64, getGCDType(V4S64, V2P0));

  EXPECT_EQ(V2S16, getGCDType(V2S16, V4P1));
  EXPECT_EQ(P1, getGCDType(V4P1, V2S16));
  EXPECT_EQ(V2P1, getGCDType(V4P1, V4S16));
  EXPECT_EQ(V4S16, getGCDType(V4S16, V2P1));

  EXPECT_EQ(P0, getGCDType(P0, V2S64));
  EXPECT_EQ(S64, getGCDType(V2S64, P0));

  EXPECT_EQ(S16, getGCDType(V2S16, V3S16));
  EXPECT_EQ(S16, getGCDType(V3S16, V2S16));
  EXPECT_EQ(S16, getGCDType(V3S16, S16));
  EXPECT_EQ(S16, getGCDType(S16, V3S16));

  EXPECT_EQ(V2S16, getGCDType(V2S16, V2S32));
  EXPECT_EQ(S32, getGCDType(V2S32, V2S16));

  EXPECT_EQ(V4S8, getGCDType(V4S8, V2S32));
  EXPECT_EQ(S32, getGCDType(V2S32, V4S8));

  // Test cases where neither element type nicely divides.
  EXPECT_EQ(LLT::scalar(3),
            getGCDType(LLT::fixed_vector(3, 5), LLT::fixed_vector(2, 6)));
  EXPECT_EQ(LLT::scalar(3),
            getGCDType(LLT::fixed_vector(2, 6), LLT::fixed_vector(3, 5)));

  // Have to go smaller than a pointer element.
  EXPECT_EQ(LLT::scalar(3), getGCDType(LLT::fixed_vector(2, LLT::pointer(3, 6)),
                                       LLT::fixed_vector(3, 5)));
  EXPECT_EQ(LLT::scalar(3),
            getGCDType(LLT::fixed_vector(3, 5),
                       LLT::fixed_vector(2, LLT::pointer(3, 6))));

  EXPECT_EQ(V4S8, getGCDType(V4S8, S32));
  EXPECT_EQ(S32, getGCDType(S32, V4S8));
  EXPECT_EQ(V4S8, getGCDType(V4S8, P1));
  EXPECT_EQ(P1, getGCDType(P1, V4S8));

  EXPECT_EQ(V2S8, getGCDType(V2S8, V4S16));
  EXPECT_EQ(S16, getGCDType(V4S16, V2S8));

  EXPECT_EQ(S8, getGCDType(V2S8, LLT::fixed_vector(4, 2)));
  EXPECT_EQ(LLT::fixed_vector(4, 2), getGCDType(LLT::fixed_vector(4, 2), S8));

  EXPECT_EQ(LLT::pointer(4, 8),
            getGCDType(LLT::fixed_vector(2, LLT::pointer(4, 8)),
                       LLT::fixed_vector(4, 2)));

  EXPECT_EQ(LLT::fixed_vector(4, 2),
            getGCDType(LLT::fixed_vector(4, 2),
                       LLT::fixed_vector(2, LLT::pointer(4, 8))));

  EXPECT_EQ(LLT::scalar(4), getGCDType(LLT::fixed_vector(3, 4), S8));
  EXPECT_EQ(LLT::scalar(4), getGCDType(S8, LLT::fixed_vector(3, 4)));

  // Scalable -> Scalable
  EXPECT_EQ(NXV1S1, getGCDType(NXV1S1, NXV1S32));
  EXPECT_EQ(NXV1S32, getGCDType(NXV1S64, NXV1S32));
  EXPECT_EQ(NXV1S32, getGCDType(NXV1S32, NXV1S64));
  EXPECT_EQ(NXV1P0, getGCDType(NXV1P0, NXV1S64));
  EXPECT_EQ(NXV1S64, getGCDType(NXV1S64, NXV1P0));

  EXPECT_EQ(NXV4S1, getGCDType(NXV4S1, NXV4S32));
  EXPECT_EQ(NXV2S64, getGCDType(NXV4S64, NXV4S32));
  EXPECT_EQ(NXV4S32, getGCDType(NXV4S32, NXV4S64));
  EXPECT_EQ(NXV4P0, getGCDType(NXV4P0, NXV4S64));
  EXPECT_EQ(NXV4S64, getGCDType(NXV4S64, NXV4P0));

  EXPECT_EQ(NXV4S1, getGCDType(NXV4S1, NXV2S32));
  EXPECT_EQ(NXV1S64, getGCDType(NXV4S64, NXV2S32));
  EXPECT_EQ(NXV4S32, getGCDType(NXV4S32, NXV2S64));
  EXPECT_EQ(NXV2P0, getGCDType(NXV4P0, NXV2S64));
  EXPECT_EQ(NXV2S64, getGCDType(NXV4S64, NXV2P0));

  EXPECT_EQ(NXV2S1, getGCDType(NXV2S1, NXV4S32));
  EXPECT_EQ(NXV2S64, getGCDType(NXV2S64, NXV4S32));
  EXPECT_EQ(NXV2S32, getGCDType(NXV2S32, NXV4S64));
  EXPECT_EQ(NXV2P0, getGCDType(NXV2P0, NXV4S64));
  EXPECT_EQ(NXV2S64, getGCDType(NXV2S64, NXV4P0));

  EXPECT_EQ(NXV1S1, getGCDType(NXV3S1, NXV4S32));
  EXPECT_EQ(NXV1S64, getGCDType(NXV3S64, NXV4S32));
  EXPECT_EQ(NXV1S32, getGCDType(NXV3S32, NXV4S64));
  EXPECT_EQ(NXV1P0, getGCDType(NXV3P0, NXV4S64));
  EXPECT_EQ(NXV1S64, getGCDType(NXV3S64, NXV4P0));

  EXPECT_EQ(NXV1S1, getGCDType(NXV3S1, NXV4S1));
  EXPECT_EQ(NXV1S32, getGCDType(NXV3S32, NXV4S32));
  EXPECT_EQ(NXV1S64, getGCDType(NXV3S64, NXV4S64));
  EXPECT_EQ(NXV1P0, getGCDType(NXV3P0, NXV4P0));

  // Scalable, Scalar

  EXPECT_EQ(S1, getGCDType(NXV1S1, S1));
  EXPECT_EQ(S1, getGCDType(NXV1S1, S32));
  EXPECT_EQ(S1, getGCDType(NXV1S32, S1));
  EXPECT_EQ(S32, getGCDType(NXV1S32, S32));
  EXPECT_EQ(S32, getGCDType(NXV1S32, S64));
  EXPECT_EQ(S1, getGCDType(NXV2S32, S1));
  EXPECT_EQ(S32, getGCDType(NXV2S32, S32));
  EXPECT_EQ(S32, getGCDType(NXV2S32, S64));

  EXPECT_EQ(S1, getGCDType(S1, NXV1S1));
  EXPECT_EQ(S1, getGCDType(S32, NXV1S1));
  EXPECT_EQ(S1, getGCDType(S1, NXV1S32));
  EXPECT_EQ(S32, getGCDType(S32, NXV1S32));
  EXPECT_EQ(S32, getGCDType(S64, NXV1S32));
  EXPECT_EQ(S1, getGCDType(S1, NXV2S32));
  EXPECT_EQ(S32, getGCDType(S32, NXV2S32));
  EXPECT_EQ(S32, getGCDType(S64, NXV2S32));
}

TEST(GISelUtilsTest, getLCMType) {
  EXPECT_EQ(S1, getLCMType(S1, S1));
  EXPECT_EQ(S32, getLCMType(S32, S1));
  EXPECT_EQ(S32, getLCMType(S1, S32));
  EXPECT_EQ(S32, getLCMType(S32, S32));

  EXPECT_EQ(S32, getLCMType(S32, S16));
  EXPECT_EQ(S32, getLCMType(S16, S32));

  EXPECT_EQ(S64, getLCMType(S64, P0));
  EXPECT_EQ(P0, getLCMType(P0, S64));

  EXPECT_EQ(P0, getLCMType(S32, P0));
  EXPECT_EQ(P0, getLCMType(P0, S32));

  EXPECT_EQ(S32, getLCMType(S32, P1));
  EXPECT_EQ(P1, getLCMType(P1, S32));
  EXPECT_EQ(P0, getLCMType(P0, P0));
  EXPECT_EQ(P1, getLCMType(P1, P1));

  EXPECT_EQ(P0, getLCMType(P0, P1));
  EXPECT_EQ(P0, getLCMType(P1, P0));

  EXPECT_EQ(V2S32, getLCMType(V2S32, V2S32));
  EXPECT_EQ(V2S32, getLCMType(V2S32, S32));
  EXPECT_EQ(V2S32, getLCMType(S32, V2S32));
  EXPECT_EQ(V2S32, getLCMType(V2S32, V2S32));
  EXPECT_EQ(V6S32, getLCMType(V2S32, V3S32));
  EXPECT_EQ(V6S32, getLCMType(V3S32, V2S32));
  EXPECT_EQ(LLT::fixed_vector(12, S32), getLCMType(V4S32, V3S32));
  EXPECT_EQ(LLT::fixed_vector(12, S32), getLCMType(V3S32, V4S32));

  EXPECT_EQ(V2P0, getLCMType(V2P0, V2P0));
  EXPECT_EQ(V2P0, getLCMType(V2P0, P0));
  EXPECT_EQ(V2P0, getLCMType(P0, V2P0));
  EXPECT_EQ(V2P0, getLCMType(V2P0, V2P0));
  EXPECT_EQ(V6P0, getLCMType(V2P0, V3P0));
  EXPECT_EQ(V6P0, getLCMType(V3P0, V2P0));
  EXPECT_EQ(LLT::fixed_vector(12, P0), getLCMType(V4P0, V3P0));
  EXPECT_EQ(LLT::fixed_vector(12, P0), getLCMType(V3P0, V4P0));

  EXPECT_EQ(LLT::fixed_vector(12, S64), getLCMType(V4S64, V3P0));
  EXPECT_EQ(LLT::fixed_vector(12, P0), getLCMType(V3P0, V4S64));

  EXPECT_EQ(LLT::fixed_vector(12, P0), getLCMType(V4P0, V3S64));
  EXPECT_EQ(LLT::fixed_vector(12, S64), getLCMType(V3S64, V4P0));

  EXPECT_EQ(V2P0, getLCMType(V2P0, S32));
  EXPECT_EQ(V4S32, getLCMType(S32, V2P0));
  EXPECT_EQ(V2P0, getLCMType(V2P0, S64));
  EXPECT_EQ(V2S64, getLCMType(S64, V2P0));


  EXPECT_EQ(V2P0, getLCMType(V2P0, V2P1));
  EXPECT_EQ(V4P1, getLCMType(V2P1, V2P0));

  EXPECT_EQ(V2P0, getLCMType(V2P0, V4P1));
  EXPECT_EQ(V4P1, getLCMType(V4P1, V2P0));


  EXPECT_EQ(V2S32, getLCMType(V2S32, S64));
  EXPECT_EQ(S64, getLCMType(S64, V2S32));

  EXPECT_EQ(V4S16, getLCMType(V4S16, V2S32));
  EXPECT_EQ(V2S32, getLCMType(V2S32, V4S16));

  EXPECT_EQ(V2S32, getLCMType(V2S32, V4S8));
  EXPECT_EQ(V8S8, getLCMType(V4S8, V2S32));

  EXPECT_EQ(V2S16, getLCMType(V2S16, V4S8));
  EXPECT_EQ(V4S8, getLCMType(V4S8, V2S16));

  EXPECT_EQ(LLT::fixed_vector(6, S16), getLCMType(V3S16, V4S8));
  EXPECT_EQ(LLT::fixed_vector(12, S8), getLCMType(V4S8, V3S16));
  EXPECT_EQ(V4S16, getLCMType(V4S16, V4S8));
  EXPECT_EQ(V8S8, getLCMType(V4S8, V4S16));

  EXPECT_EQ(LLT::fixed_vector(6, 4), getLCMType(LLT::fixed_vector(3, 4), S8));
  EXPECT_EQ(LLT::fixed_vector(3, 8), getLCMType(S8, LLT::fixed_vector(3, 4)));

  EXPECT_EQ(LLT::fixed_vector(6, 4),
            getLCMType(LLT::fixed_vector(3, 4), LLT::pointer(4, 8)));
  EXPECT_EQ(LLT::fixed_vector(3, LLT::pointer(4, 8)),
            getLCMType(LLT::pointer(4, 8), LLT::fixed_vector(3, 4)));

  EXPECT_EQ(V2S64, getLCMType(V2S64, P0));
  EXPECT_EQ(V2P0, getLCMType(P0, V2S64));

  EXPECT_EQ(V2S64, getLCMType(V2S64, P1));
  EXPECT_EQ(V4P1, getLCMType(P1, V2S64));

  // Scalable, Scalable
  EXPECT_EQ(NXV32S1, getLCMType(NXV1S1, NXV1S32));
  EXPECT_EQ(NXV1S64, getLCMType(NXV1S64, NXV1S32));
  EXPECT_EQ(NXV2S32, getLCMType(NXV1S32, NXV1S64));
  EXPECT_EQ(NXV1P0, getLCMType(NXV1P0, NXV1S64));
  EXPECT_EQ(NXV1S64, getLCMType(NXV1S64, NXV1P0));

  EXPECT_EQ(NXV128S1, getLCMType(NXV4S1, NXV4S32));
  EXPECT_EQ(NXV4S64, getLCMType(NXV4S64, NXV4S32));
  EXPECT_EQ(NXV8S32, getLCMType(NXV4S32, NXV4S64));
  EXPECT_EQ(NXV4P0, getLCMType(NXV4P0, NXV4S64));
  EXPECT_EQ(NXV4S64, getLCMType(NXV4S64, NXV4P0));

  EXPECT_EQ(NXV64S1, getLCMType(NXV4S1, NXV2S32));
  EXPECT_EQ(NXV4S64, getLCMType(NXV4S64, NXV2S32));
  EXPECT_EQ(NXV4S32, getLCMType(NXV4S32, NXV2S64));
  EXPECT_EQ(NXV4P0, getLCMType(NXV4P0, NXV2S64));
  EXPECT_EQ(NXV4S64, getLCMType(NXV4S64, NXV2P0));

  EXPECT_EQ(NXV128S1, getLCMType(NXV2S1, NXV4S32));
  EXPECT_EQ(NXV2S64, getLCMType(NXV2S64, NXV4S32));
  EXPECT_EQ(NXV8S32, getLCMType(NXV2S32, NXV4S64));
  EXPECT_EQ(NXV4P0, getLCMType(NXV2P0, NXV4S64));
  EXPECT_EQ(NXV4S64, getLCMType(NXV2S64, NXV4P0));

  EXPECT_EQ(NXV384S1, getLCMType(NXV3S1, NXV4S32));
  EXPECT_EQ(NXV6S64, getLCMType(NXV3S64, NXV4S32));
  EXPECT_EQ(NXV24S32, getLCMType(NXV3S32, NXV4S64));
  EXPECT_EQ(NXV12P0, getLCMType(NXV3P0, NXV4S64));
  EXPECT_EQ(NXV12S64, getLCMType(NXV3S64, NXV4P0));

  EXPECT_EQ(NXV12S1, getLCMType(NXV3S1, NXV4S1));
  EXPECT_EQ(NXV12S32, getLCMType(NXV3S32, NXV4S32));
  EXPECT_EQ(NXV12S64, getLCMType(NXV3S64, NXV4S64));
  EXPECT_EQ(NXV12P0, getLCMType(NXV3P0, NXV4P0));

  // Scalable, Scalar

  EXPECT_EQ(NXV1S1, getLCMType(NXV1S1, S1));
  EXPECT_EQ(NXV32S1, getLCMType(NXV1S1, S32));
  EXPECT_EQ(NXV1S32, getLCMType(NXV1S32, S1));
  EXPECT_EQ(NXV1S32, getLCMType(NXV1S32, S32));
  EXPECT_EQ(NXV2S32, getLCMType(NXV1S32, S64));
  EXPECT_EQ(NXV2S32, getLCMType(NXV2S32, S1));
  EXPECT_EQ(NXV2S32, getLCMType(NXV2S32, S32));
  EXPECT_EQ(NXV2S32, getLCMType(NXV2S32, S64));

  EXPECT_EQ(NXV1S1, getLCMType(S1, NXV1S1));
  EXPECT_EQ(NXV1S32, getLCMType(S32, NXV1S1));
  EXPECT_EQ(NXV32S1, getLCMType(S1, NXV1S32));
  EXPECT_EQ(NXV1S32, getLCMType(S32, NXV1S32));
  EXPECT_EQ(NXV1S64, getLCMType(S64, NXV1S32));
  EXPECT_EQ(NXV64S1, getLCMType(S1, NXV2S32));
  EXPECT_EQ(NXV2S32, getLCMType(S32, NXV2S32));
  EXPECT_EQ(NXV1S64, getLCMType(S64, NXV2S32));
}

TEST_F(AArch64GISelMITest, ConstFalseTest) {
  setUp();
  if (!TM)
    GTEST_SKIP();
  const auto &TLI = *B.getMF().getSubtarget().getTargetLowering();
  bool BooleanChoices[2] = {true, false};

  // AArch64 uses ZeroOrOneBooleanContent for scalars, and
  // ZeroOrNegativeOneBooleanContent for vectors.
  for (auto IsVec : BooleanChoices) {
    for (auto IsFP : BooleanChoices) {
      EXPECT_TRUE(isConstFalseVal(TLI, 0, IsVec, IsFP));
      EXPECT_FALSE(isConstFalseVal(TLI, 1, IsVec, IsFP));

      // This would be true with UndefinedBooleanContent.
      EXPECT_FALSE(isConstFalseVal(TLI, 2, IsVec, IsFP));
    }
  }
}

TEST_F(AMDGPUGISelMITest, isConstantOrConstantSplatVectorFP) {
  StringRef MIRString =
      "  %cst0:_(s32) = G_FCONSTANT float 2.000000e+00\n"
      "  %cst1:_(s32) = G_FCONSTANT float 0.0\n"
      "  %cst2:_(s64) = G_FCONSTANT double 3.000000e-02\n"
      "  %cst3:_(s32) = G_CONSTANT i32 2\n"
      "  %cst4:_(<2 x s32>) = G_BUILD_VECTOR %cst0(s32), %cst0(s32)\n"
      "  %cst5:_(<2 x s32>) = G_BUILD_VECTOR %cst1(s32), %cst0(s32)\n"
      "  %cst6:_(<2 x s64>) = G_BUILD_VECTOR %cst2(s64), %cst2(s64)\n"
      "  %cst7:_(<2 x s32>) = G_BUILD_VECTOR %cst3(s32), %cst3:_(s32)\n"
      "  %cst8:_(<4 x s32>) = G_CONCAT_VECTORS %cst4:_(<2 x s32>), %cst4:_(<2 "
      "x s32>)\n"
      "  %cst9:_(<4 x s64>) = G_CONCAT_VECTORS %cst6:_(<2 x s64>), %cst6:_(<2 "
      "x s64>)\n"
      "  %cst10:_(<4 x s32>) = G_CONCAT_VECTORS %cst4:_(<2 x s32>), %cst5:_(<2 "
      "x s32>)\n"
      "  %cst11:_(<4 x s32>) = G_CONCAT_VECTORS %cst7:_(<2 x s32>), %cst7:_(<2 "
      "x s32>)\n"
      "  %cst12:_(s32) = G_IMPLICIT_DEF \n"
      "  %cst13:_(<2 x s32>) = G_BUILD_VECTOR %cst12(s32), %cst12(s32)\n"
      "  %cst14:_(<2 x s32>) = G_BUILD_VECTOR %cst0(s32), %cst12(s32)\n"
      "  %cst15:_(<4 x s32>) = G_CONCAT_VECTORS %cst4:_(<2 x s32>), "
      "%cst14:_(<2 "
      "x s32>)\n";

  SmallVector<MachineInstr *, 16> MIList;

  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  collectNonCopyMI(MIList, MF);

  EXPECT_TRUE(isConstantOrConstantSplatVectorFP(*MIList[0], *MRI).has_value());
  auto val = isConstantOrConstantSplatVectorFP(*MIList[0], *MRI).value();
  EXPECT_EQ(2.0, val.convertToFloat());

  EXPECT_TRUE(isConstantOrConstantSplatVectorFP(*MIList[1], *MRI).has_value());
  val = isConstantOrConstantSplatVectorFP(*MIList[1], *MRI).value();
  EXPECT_EQ(0.0, val.convertToFloat());

  EXPECT_TRUE(isConstantOrConstantSplatVectorFP(*MIList[2], *MRI).has_value());
  val = isConstantOrConstantSplatVectorFP(*MIList[2], *MRI).value();
  EXPECT_EQ(0.03, val.convertToDouble());

  EXPECT_FALSE(isConstantOrConstantSplatVectorFP(*MIList[3], *MRI).has_value());

  EXPECT_TRUE(isConstantOrConstantSplatVectorFP(*MIList[4], *MRI).has_value());
  val = isConstantOrConstantSplatVectorFP(*MIList[4], *MRI).value();
  EXPECT_EQ(2.0, val.convertToFloat());

  EXPECT_FALSE(isConstantOrConstantSplatVectorFP(*MIList[5], *MRI).has_value());

  EXPECT_TRUE(isConstantOrConstantSplatVectorFP(*MIList[6], *MRI).has_value());
  val = isConstantOrConstantSplatVectorFP(*MIList[6], *MRI).value();
  EXPECT_EQ(0.03, val.convertToDouble());

  EXPECT_FALSE(isConstantOrConstantSplatVectorFP(*MIList[7], *MRI).has_value());

  EXPECT_TRUE(isConstantOrConstantSplatVectorFP(*MIList[8], *MRI).has_value());
  val = isConstantOrConstantSplatVectorFP(*MIList[8], *MRI).value();
  EXPECT_EQ(2.0, val.convertToFloat());

  EXPECT_TRUE(isConstantOrConstantSplatVectorFP(*MIList[9], *MRI).has_value());
  val = isConstantOrConstantSplatVectorFP(*MIList[9], *MRI).value();
  EXPECT_EQ(0.03, val.convertToDouble());

  EXPECT_FALSE(
      isConstantOrConstantSplatVectorFP(*MIList[10], *MRI).has_value());

  EXPECT_FALSE(
      isConstantOrConstantSplatVectorFP(*MIList[11], *MRI).has_value());

  EXPECT_FALSE(
      isConstantOrConstantSplatVectorFP(*MIList[12], *MRI).has_value());

  EXPECT_FALSE(
      isConstantOrConstantSplatVectorFP(*MIList[13], *MRI).has_value());

  EXPECT_FALSE(
      isConstantOrConstantSplatVectorFP(*MIList[14], *MRI).has_value());

  EXPECT_FALSE(
      isConstantOrConstantSplatVectorFP(*MIList[15], *MRI).has_value());
}
}
