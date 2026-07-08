//===- CanonicalOpAttrsTest.cpp - EXEC-router attribute tests -------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Regression tests for the SPE A-level EXEC-writer allow-list
// (RoutesExecThroughStoreExec). The raiser's Phase-1.5 gate aborts with
// SPE-unmodeled-EXEC-writer for any opcode that can write EXEC but is not
// marked here. The s_lshl{1,2,3,4}_add_u32 family was missing, which made
// certain autotuned Triton kernels (falcon-mamba selective scan) fail to
// transpile gfx1250->gfx942/gfx950. See handle-sop2.cpp.
//
//===----------------------------------------------------------------------===//

#include "hotswap/canonical-op-attrs.h"
#include "hotswap/canonical-op.h"

#include "gtest/gtest.h"

using namespace COMGR::hotswap;

// The fix: the shift-add family must be EXEC-routers, exactly like the plain
// shift/bitwise SOP2 ops already are.
TEST(CanonicalOpAttrs, LshlAddFamilyRoutesExecThroughStoreExec) {
  EXPECT_TRUE(getCanonicalOpAttrs(CanonicalOp::S_LSHL1_ADD_U32)
                  .RoutesExecThroughStoreExec);
  EXPECT_TRUE(getCanonicalOpAttrs(CanonicalOp::S_LSHL2_ADD_U32)
                  .RoutesExecThroughStoreExec);
  EXPECT_TRUE(getCanonicalOpAttrs(CanonicalOp::S_LSHL3_ADD_U32)
                  .RoutesExecThroughStoreExec);
  EXPECT_TRUE(getCanonicalOpAttrs(CanonicalOp::S_LSHL4_ADD_U32)
                  .RoutesExecThroughStoreExec);
}

// Control: the plain shift/bitwise ops the fix mirrors stay marked.
TEST(CanonicalOpAttrs, ShiftAndBitwiseRoutersStayMarked) {
  EXPECT_TRUE(
      getCanonicalOpAttrs(CanonicalOp::S_LSHL_B32).RoutesExecThroughStoreExec);
  EXPECT_TRUE(
      getCanonicalOpAttrs(CanonicalOp::S_AND_B32).RoutesExecThroughStoreExec);
  EXPECT_TRUE(
      getCanonicalOpAttrs(CanonicalOp::S_OR_B64).RoutesExecThroughStoreExec);
}

// Control: an op that does NOT route EXEC stays unmarked, so the test would
// catch an accidental blanket-true regression.
TEST(CanonicalOpAttrs, NonExecRouterStaysUnmarked) {
  EXPECT_FALSE(
      getCanonicalOpAttrs(CanonicalOp::Unknown).RoutesExecThroughStoreExec);
}
