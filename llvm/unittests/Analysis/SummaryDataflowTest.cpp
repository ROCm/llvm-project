//===- SummaryDataflowTest.cpp - Summary dataflow unit tests --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/SummaryDataflow.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"
#include <set>

using namespace llvm;

namespace {

static std::unique_ptr<ModuleSummaryIndex> parseIndex(StringRef Summary) {
  SMDiagnostic Err;
  std::unique_ptr<ModuleSummaryIndex> Index =
      parseSummaryIndexAssemblyString(Summary, Err);
  if (!Index)
    Err.print("SummaryDataflowTest", errs());
  return Index;
}

// Builds the call graph of an index. Every copy prevails.
class Graph {
public:
  Graph(ModuleSummaryIndex &Index,
        DenseSet<GlobalValue::GUID> PreservedSymbols = {})
      : Index(Index), PreservedSymbols(std::move(PreservedSymbols)) {
    AM.registerPass([] { return PassInstrumentationAnalysis(); });
    AM.registerPass([&] {
      return ThinLinkInfoAnalysis(
          this->PreservedSymbols,
          [](GlobalValue::GUID, const GlobalValueSummary *) { return true; });
    });
    AM.registerPass([] { return SummaryCallGraphAnalysis(); });
    AM.registerPass([] { return SummaryAddressTakenAnalysis(); });
  }

  const SummaryCallGraph &get() {
    return AM.getResult<SummaryCallGraphAnalysis>(Index);
  }

private:
  ModuleSummaryIndex &Index;
  DenseSet<GlobalValue::GUID> PreservedSymbols;
  SummaryAnalysisManager AM;
};

// A top-down problem: the roots that reach each function. The roots are like
// kernels. The code outside the summaries is the root 0.
struct ReachingRoots {
  using ValueT = std::set<GlobalValue::GUID>;

  DenseSet<GlobalValue::GUID> Roots;

  ValueT getInitialValue(const SummaryCallGraph::Node &) { return {}; }
  ValueT getExternalValue() { return {0}; }
  void join(ValueT &Into, const ValueT &V) { Into.insert(V.begin(), V.end()); }
  ValueT transfer(const SummaryCallGraph::Node &N, const ValueT &In) {
    ValueT Out = In;
    if (Roots.contains(N.getValueInfo().getGUID()))
      Out.insert(N.getValueInfo().getGUID());
    return Out;
  }
};

// A bottom-up problem: whether a function can reach a use of a resource. The
// code outside the summaries can use it.
struct UsesResource {
  using ValueT = bool;

  DenseSet<GlobalValue::GUID> Users;

  ValueT getInitialValue(const SummaryCallGraph::Node &) { return false; }
  ValueT getExternalValue() { return true; }
  void join(ValueT &Into, const ValueT &V) { Into |= V; }
  ValueT transfer(const SummaryCallGraph::Node &N, const ValueT &In) {
    return In || Users.contains(N.getValueInfo().getGUID());
  }
};

// The functions 1 and 2 call 3, which is in a cycle with 4. Code outside the
// summaries calls 5, which calls 4. Nothing calls 6.
TEST(SummaryDataflowTest, TopDown) {
  std::unique_ptr<ModuleSummaryIndex> Index = parseIndex(R"(
^0 = module: (path: "a.o", hash: (0, 0, 0, 0, 0))
^1 = gv: (guid: 1, summaries: (function: (module: ^0, flags: (linkage: external), insts: 1, calls: ((callee: ^3)))))
^2 = gv: (guid: 2, summaries: (function: (module: ^0, flags: (linkage: external), insts: 1, calls: ((callee: ^3)))))
^3 = gv: (guid: 3, summaries: (function: (module: ^0, flags: (linkage: external), insts: 1, calls: ((callee: ^4)))))
^4 = gv: (guid: 4, summaries: (function: (module: ^0, flags: (linkage: external), insts: 1, calls: ((callee: ^3)))))
^5 = gv: (guid: 5, summaries: (function: (module: ^0, flags: (linkage: external), insts: 1, calls: ((callee: ^4)))))
^6 = gv: (guid: 6, summaries: (function: (module: ^0, flags: (linkage: external), insts: 1)))
)");
  ASSERT_NE(Index, nullptr);
  Graph G(*Index, /*PreservedSymbols=*/{5});
  const SummaryCallGraph &CG = G.get();

  ReachingRoots Problem;
  Problem.Roots = {1, 2};
  auto Out =
      solveSummaryDataflow(CG, Problem, SummaryDataflowDirection::TopDown);
  using Set = ReachingRoots::ValueT;
  EXPECT_EQ(Out.lookup(CG.lookup(1)), Set({1}));
  EXPECT_EQ(Out.lookup(CG.lookup(2)), Set({2}));
  EXPECT_EQ(Out.lookup(CG.lookup(5)), Set({0}));
  EXPECT_EQ(Out.lookup(CG.lookup(3)), Set({0, 1, 2}));
  EXPECT_EQ(Out.lookup(CG.lookup(4)), Set({0, 1, 2}));
  EXPECT_EQ(Out.lookup(CG.lookup(6)), Set());
  EXPECT_EQ(Out.lookup(CG.getExternalCallingNode()), Set({0}));
}

// The chain 1, 2, 3 reaches the use in 3. The function 4 calls itself. The
// function 5 calls the function 8, which has no summary. The functions 6 and
// 7 form a cycle, and 7 also calls 3.
TEST(SummaryDataflowTest, BottomUp) {
  std::unique_ptr<ModuleSummaryIndex> Index = parseIndex(R"(
^0 = module: (path: "a.o", hash: (0, 0, 0, 0, 0))
^1 = gv: (guid: 1, summaries: (function: (module: ^0, flags: (linkage: external), insts: 1, calls: ((callee: ^2)))))
^2 = gv: (guid: 2, summaries: (function: (module: ^0, flags: (linkage: external), insts: 1, calls: ((callee: ^3)))))
^3 = gv: (guid: 3, summaries: (function: (module: ^0, flags: (linkage: external), insts: 1)))
^4 = gv: (guid: 4, summaries: (function: (module: ^0, flags: (linkage: external), insts: 1, calls: ((callee: ^4)))))
^5 = gv: (guid: 5, summaries: (function: (module: ^0, flags: (linkage: external), insts: 1, calls: ((callee: ^8)))))
^6 = gv: (guid: 6, summaries: (function: (module: ^0, flags: (linkage: external), insts: 1, calls: ((callee: ^7)))))
^7 = gv: (guid: 7, summaries: (function: (module: ^0, flags: (linkage: external), insts: 1, calls: ((callee: ^6), (callee: ^3)))))
^8 = gv: (guid: 8)
)");
  ASSERT_NE(Index, nullptr);
  Graph G(*Index);
  const SummaryCallGraph &CG = G.get();

  UsesResource Problem;
  Problem.Users = {3};
  auto Out =
      solveSummaryDataflow(CG, Problem, SummaryDataflowDirection::BottomUp);
  for (GlobalValue::GUID GUID : {1, 2, 3, 5, 6, 7, 8})
    EXPECT_TRUE(Out.lookup(CG.lookup(GUID))) << "GUID " << GUID;
  EXPECT_FALSE(Out.lookup(CG.lookup(4)));
  EXPECT_TRUE(Out.lookup(CG.getCallsExternalNode()));
}

// A solver can leave out the indirect edges. The function 1 has an indirect
// call that can reach the function 2, whose address the variable 3 takes.
TEST(SummaryDataflowTest, IndirectEdges) {
  std::unique_ptr<ModuleSummaryIndex> Index = parseIndex(R"(
^0 = module: (path: "a.o", hash: (0, 0, 0, 0, 0))
^1 = gv: (guid: 1, summaries: (function: (module: ^0, flags: (linkage: external), insts: 1, indirectCallTypeIds: (100))))
^2 = gv: (guid: 2, summaries: (function: (module: ^0, flags: (linkage: external), insts: 1, callTypeId: 100)))
^3 = gv: (guid: 3, summaries: (variable: (module: ^0, flags: (linkage: external), varFlags: (readonly: 0), refs: (^2))))
)");
  ASSERT_NE(Index, nullptr);
  Graph G(*Index);
  const SummaryCallGraph &CG = G.get();

  UsesResource Problem;
  Problem.Users = {2};
  auto All =
      solveSummaryDataflow(CG, Problem, SummaryDataflowDirection::BottomUp);
  EXPECT_TRUE(All.lookup(CG.lookup(1)));
  auto Calls =
      solveSummaryDataflow(CG, Problem, SummaryDataflowDirection::BottomUp,
                           /*IncludeIndirect=*/false);
  EXPECT_FALSE(Calls.lookup(CG.lookup(1)));
}

} // end anonymous namespace
