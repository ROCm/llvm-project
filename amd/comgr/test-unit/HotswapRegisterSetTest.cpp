//===- HotswapRegisterSetTest.cpp - Unit tests for reglive::RegisterSet --===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Tests for the ISA-independent register set/reference types in
/// comgr-hotswap-liveness.h. These types are the leaf data layer of the
/// HotSwap register-liveness port and have no MC dependency, so the suite is
/// self-contained.
///
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-liveness.h"
#include "gtest/gtest.h"

#include <vector>

using namespace COMGR::hotswap::reglive;

TEST(HotswapRegisterSet, KeepsRegisterClassesSeparate) {
  RegisterSet Set;
  Set.expand({RegClass::SGPR, 4, 1});

  EXPECT_TRUE(Set.contains({RegClass::SGPR, 4, 1}));
  EXPECT_FALSE(Set.contains({RegClass::VGPR, 4, 1}));
  EXPECT_FALSE(Set.contains({RegClass::ACC_VGPR, 4, 1}));
}

TEST(HotswapRegisterSet, IgnoresSpecialRegisterClasses) {
  RegisterSet Set;
  Set.expand({RegClass::EXEC, 0, 2});
  Set.expand({RegClass::SCC, 0, 1});
  Set.expand({RegClass::FLAT_SCRATCH, 0, 2});

  EXPECT_TRUE(Set.none());
  EXPECT_FALSE(Set.contains({RegClass::EXEC, 0, 1}));
  EXPECT_FALSE(Set.contains({RegClass::SCC, 0, 1}));
  EXPECT_FALSE(Set.contains({RegClass::FLAT_SCRATCH, 0, 2}));
}

TEST(HotswapRegisterSet, ExpandMultiLaneSetsAllLanes) {
  RegisterSet Set;
  Set.expand({RegClass::VGPR, 4, 4});

  EXPECT_TRUE(Set.contains({RegClass::VGPR, 4, 4}));
  for (uint16_t I = 4; I < 8; ++I)
    EXPECT_TRUE(Set.contains({RegClass::VGPR, I, 1}));
  EXPECT_FALSE(Set.contains({RegClass::VGPR, 3, 1}));
  EXPECT_FALSE(Set.contains({RegClass::VGPR, 8, 1}));
  // A wider ref that runs past the set lanes is not fully contained.
  EXPECT_FALSE(Set.contains({RegClass::VGPR, 4, 5}));
}

TEST(HotswapRegisterSet, EraseIsLanePrecise) {
  RegisterSet Set;
  Set.expand({RegClass::SGPR, 6, 2});
  Set.erase({RegClass::SGPR, 6, 1});

  EXPECT_FALSE(Set.contains({RegClass::SGPR, 6, 1}));
  EXPECT_TRUE(Set.contains({RegClass::SGPR, 7, 1}));
  EXPECT_FALSE(Set.contains({RegClass::SGPR, 6, 2}));
}

TEST(HotswapRegisterSet, ClearClassOnlyClearsThatClass) {
  RegisterSet Set;
  Set.expand({RegClass::SGPR, 4, 1});
  Set.expand({RegClass::VGPR, 4, 1});
  Set.expand({RegClass::ACC_VGPR, 4, 1});

  Set.clearClass(RegClass::VGPR);

  EXPECT_TRUE(Set.contains({RegClass::SGPR, 4, 1}));
  EXPECT_FALSE(Set.contains({RegClass::VGPR, 4, 1}));
  EXPECT_TRUE(Set.contains({RegClass::ACC_VGPR, 4, 1}));
}

TEST(HotswapRegisterSet, UnionSubtractIntersectAreMemberWise) {
  RegisterSet A;
  A.expand({RegClass::VGPR, 0, 1});
  A.expand({RegClass::SGPR, 2, 1});

  RegisterSet B;
  B.expand({RegClass::VGPR, 1, 1});
  B.expand({RegClass::SGPR, 2, 1});

  RegisterSet Union = A | B;
  EXPECT_TRUE(Union.contains({RegClass::VGPR, 0, 1}));
  EXPECT_TRUE(Union.contains({RegClass::VGPR, 1, 1}));
  EXPECT_TRUE(Union.contains({RegClass::SGPR, 2, 1}));

  RegisterSet Inter = A & B;
  EXPECT_FALSE(Inter.contains({RegClass::VGPR, 0, 1}));
  EXPECT_TRUE(Inter.contains({RegClass::SGPR, 2, 1}));
  EXPECT_EQ(Inter.size(), 1u);

  RegisterSet Diff = A - B;
  EXPECT_TRUE(Diff.contains({RegClass::VGPR, 0, 1}));
  EXPECT_FALSE(Diff.contains({RegClass::SGPR, 2, 1}));
}

TEST(HotswapRegisterSet, IntersectsDetectsSharedLane) {
  RegisterSet A;
  A.expand({RegClass::VGPR, 5, 1});
  RegisterSet B;
  B.expand({RegClass::VGPR, 5, 1});
  RegisterSet C;
  C.expand({RegClass::VGPR, 6, 1});

  EXPECT_TRUE(A.intersects(B));
  EXPECT_FALSE(A.intersects(C));
  // Same index in a different class does not intersect.
  RegisterSet D;
  D.expand({RegClass::SGPR, 5, 1});
  EXPECT_FALSE(A.intersects(D));
}

TEST(HotswapRegisterSet, SizeCountsLanesAcrossClasses) {
  RegisterSet Set;
  EXPECT_EQ(Set.size(), 0u);
  EXPECT_TRUE(Set.none());

  Set.expand({RegClass::SGPR, 0, 2});
  Set.expand({RegClass::VGPR, 10, 1});
  Set.expand({RegClass::ACC_VGPR, 3, 4});

  EXPECT_EQ(Set.size(), 2u + 1u + 4u);
  EXPECT_FALSE(Set.none());
}

TEST(HotswapRegisterSet, ForEachVisitsAscendingSgprThenVgprThenAcc) {
  RegisterSet Set;
  Set.expand({RegClass::ACC_VGPR, 2, 1});
  Set.expand({RegClass::VGPR, 7, 1});
  Set.expand({RegClass::VGPR, 1, 1});
  Set.expand({RegClass::SGPR, 5, 1});

  std::vector<RegisterRef> Visited;
  Set.forEach([&](RegisterRef Ref) { Visited.push_back(Ref); });

  ASSERT_EQ(Visited.size(), 4u);
  EXPECT_EQ(Visited[0], (RegisterRef{RegClass::SGPR, 5, 1}));
  EXPECT_EQ(Visited[1], (RegisterRef{RegClass::VGPR, 1, 1}));
  EXPECT_EQ(Visited[2], (RegisterRef{RegClass::VGPR, 7, 1}));
  EXPECT_EQ(Visited[3], (RegisterRef{RegClass::ACC_VGPR, 2, 1}));
}

TEST(HotswapRegisterSet, EqualityIsMemberWise) {
  RegisterSet A;
  A.expand({RegClass::VGPR, 3, 2});
  RegisterSet B;
  B.expand({RegClass::VGPR, 3, 1});
  EXPECT_FALSE(A == B);

  B.expand({RegClass::VGPR, 4, 1});
  EXPECT_TRUE(A == B);
}
