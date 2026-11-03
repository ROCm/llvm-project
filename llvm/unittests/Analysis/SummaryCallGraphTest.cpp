//===- SummaryCallGraphTest.cpp - Summary call graph unit tests -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/SummaryCallGraph.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

using GUIDList = std::vector<GlobalValue::GUID>;

static std::unique_ptr<ModuleSummaryIndex> parseIndex(StringRef Summary) {
  SMDiagnostic Err;
  std::unique_ptr<ModuleSummaryIndex> Index =
      parseSummaryIndexAssemblyString(Summary, Err);
  if (!Index)
    Err.print("SummaryCallGraphTest", errs());
  return Index;
}

static bool allPrevail(GlobalValue::GUID, const GlobalValueSummary *) {
  return true;
}

static GUIDList getGUIDs(ArrayRef<SummaryCallGraph::Node *> Nodes) {
  GUIDList GUIDs;
  for (const SummaryCallGraph::Node *N : Nodes)
    GUIDs.push_back(N->getValueInfo().getGUID());
  return GUIDs;
}

// Returns the GUIDs of each SCC in bottom-up order. The GUIDs in an SCC are
// sorted.
static std::vector<GUIDList> getSCCGUIDs(const SummaryCallGraph &CG) {
  std::vector<GUIDList> SCCs;
  for (SummaryCallGraph::SCC C : CG.sccs()) {
    SCCs.push_back(getGUIDs(C));
    llvm::sort(SCCs.back());
  }
  return SCCs;
}

TEST(SummaryCallGraphTest, Edges) {
  std::unique_ptr<ModuleSummaryIndex> Index = parseIndex(R"(
^0 = module: (path: "a.o", hash: (0, 0, 0, 0, 0))
^1 = gv: (guid: 1, summaries: (function: (module: ^0, flags: (linkage: external), insts: 1, calls: ((callee: ^2), (callee: ^3)))))
^2 = gv: (guid: 2, summaries: (function: (module: ^0, flags: (linkage: external), insts: 1, calls: ((callee: ^3)))))
^3 = gv: (guid: 3, summaries: (function: (module: ^0, flags: (linkage: external), insts: 1)))
)");
  ASSERT_NE(Index, nullptr);
  SummaryCallGraph CG(*Index, /*GUIDPreservedSymbols=*/{}, allPrevail);

  EXPECT_EQ(getGUIDs(CG.nodes()), GUIDList({1, 2, 3}));
  SummaryCallGraph::Node *N1 = CG.lookup(1), *N3 = CG.lookup(3);
  ASSERT_NE(N1, nullptr);
  ASSERT_NE(N3, nullptr);
  EXPECT_EQ(N1->getSummary(), Index->getValueInfo(1).getSummaryList()[0].get());
  EXPECT_EQ(getGUIDs(N1->callees()), GUIDList({2, 3}));
  EXPECT_TRUE(N1->callers().empty());
  EXPECT_TRUE(N3->callees().empty());
  EXPECT_EQ(getGUIDs(N3->callers()), GUIDList({1, 2}));
  EXPECT_FALSE(N1->hasUnknownCallers());
  EXPECT_EQ(CG.lookup(4), nullptr);
  EXPECT_EQ(getSCCGUIDs(CG), std::vector<GUIDList>({{3}, {2}, {1}}));
}

// The SCC walk visits every function, also a cycle that no function outside
// the cycle calls.
TEST(SummaryCallGraphTest, Cycles) {
  std::unique_ptr<ModuleSummaryIndex> Index = parseIndex(R"(
^0 = module: (path: "a.o", hash: (0, 0, 0, 0, 0))
^1 = gv: (guid: 1, summaries: (function: (module: ^0, flags: (linkage: external), insts: 1, calls: ((callee: ^2)))))
^2 = gv: (guid: 2, summaries: (function: (module: ^0, flags: (linkage: external), insts: 1, calls: ((callee: ^1), (callee: ^3)))))
^3 = gv: (guid: 3, summaries: (function: (module: ^0, flags: (linkage: external), insts: 1, calls: ((callee: ^3)))))
)");
  ASSERT_NE(Index, nullptr);
  SummaryCallGraph CG(*Index, /*GUIDPreservedSymbols=*/{}, allPrevail);

  EXPECT_EQ(getSCCGUIDs(CG), std::vector<GUIDList>({{3}, {1, 2}}));
  SummaryCallGraph::Node *N3 = CG.lookup(3);
  ASSERT_NE(N3, nullptr);
  EXPECT_EQ(getGUIDs(N3->callees()), GUIDList({3}));
  EXPECT_EQ(getGUIDs(N3->callers()), GUIDList({2, 3}));
}

// The edges come from the copy that prevails.
TEST(SummaryCallGraphTest, PrevailingCopy) {
  std::unique_ptr<ModuleSummaryIndex> Index = parseIndex(R"(
^0 = module: (path: "a.o", hash: (0, 0, 0, 0, 0))
^1 = module: (path: "b.o", hash: (0, 0, 0, 0, 1))
^2 = gv: (guid: 1, summaries: (function: (module: ^0, flags: (linkage: linkonce_odr), insts: 1, calls: ((callee: ^3))), function: (module: ^1, flags: (linkage: linkonce_odr), insts: 1, calls: ((callee: ^4)))))
^3 = gv: (guid: 2, summaries: (function: (module: ^0, flags: (linkage: external), insts: 1)))
^4 = gv: (guid: 3, summaries: (function: (module: ^1, flags: (linkage: external), insts: 1)))
)");
  ASSERT_NE(Index, nullptr);
  SummaryCallGraph CG(*Index, /*GUIDPreservedSymbols=*/{},
                      [](GlobalValue::GUID GUID, const GlobalValueSummary *S) {
                        return GUID != 1 || S->modulePath() == "b.o";
                      });

  SummaryCallGraph::Node *N1 = CG.lookup(1), *N2 = CG.lookup(2);
  ASSERT_NE(N1, nullptr);
  ASSERT_NE(N2, nullptr);
  ASSERT_NE(N1->getSummary(), nullptr);
  EXPECT_EQ(N1->getSummary()->modulePath(), "b.o");
  EXPECT_EQ(getGUIDs(N1->callees()), GUIDList({3}));
  EXPECT_TRUE(N2->callers().empty());
}

// A local copy and a promoted copy are in use without a prevailing check.
TEST(SummaryCallGraphTest, LocalAndPromotedCopies) {
  std::unique_ptr<ModuleSummaryIndex> Index = parseIndex(R"(
^0 = module: (path: "a.o", hash: (0, 0, 0, 0, 0))
^1 = gv: (guid: 1, summaries: (function: (module: ^0, flags: (linkage: internal), insts: 1)))
^2 = gv: (guid: 2, summaries: (function: (module: ^0, flags: (linkage: internal), insts: 1)))
^3 = gv: (guid: 3, summaries: (function: (module: ^0, flags: (linkage: external), insts: 1)))
)");
  ASSERT_NE(Index, nullptr);
  Index->getValueInfo(2).getSummaryList()[0]->promote();
  SummaryCallGraph CG(
      *Index, /*GUIDPreservedSymbols=*/{},
      [](GlobalValue::GUID, const GlobalValueSummary *) { return false; });

  EXPECT_EQ(getGUIDs(CG.nodes()), GUIDList({1, 2}));
  ASSERT_NE(CG.lookup(2), nullptr);
  EXPECT_NE(CG.lookup(2)->getSummary(), nullptr);
}

// A callee gets a node without a summary if no copy of it is in use.
TEST(SummaryCallGraphTest, UnknownCallees) {
  std::unique_ptr<ModuleSummaryIndex> Index = parseIndex(R"(
^0 = module: (path: "a.o", hash: (0, 0, 0, 0, 0))
^1 = module: (path: "b.o", hash: (0, 0, 0, 0, 1))
^2 = gv: (guid: 1, summaries: (function: (module: ^0, flags: (linkage: external), insts: 1, calls: ((callee: ^3), (callee: ^4), (callee: ^5)))))
^3 = gv: (guid: 2)
^4 = gv: (guid: 3, summaries: (function: (module: ^0, flags: (linkage: available_externally), insts: 1)))
^5 = gv: (guid: 4, summaries: (function: (module: ^0, flags: (linkage: internal), insts: 1), function: (module: ^1, flags: (linkage: internal), insts: 1)))
)");
  ASSERT_NE(Index, nullptr);
  SummaryCallGraph CG(*Index, /*GUIDPreservedSymbols=*/{}, allPrevail);

  EXPECT_EQ(getGUIDs(CG.nodes()), GUIDList({1, 2, 3, 4}));
  for (GlobalValue::GUID GUID : {2, 3, 4}) {
    SummaryCallGraph::Node *N = CG.lookup(GUID);
    ASSERT_NE(N, nullptr);
    EXPECT_EQ(N->getSummary(), nullptr);
    EXPECT_TRUE(N->callees().empty());
    EXPECT_EQ(getGUIDs(N->callers()), GUIDList({1}));
  }
}

// A dead function has no node, and its calls are not edges.
TEST(SummaryCallGraphTest, DeadFunctions) {
  std::unique_ptr<ModuleSummaryIndex> Index = parseIndex(R"(
^0 = module: (path: "a.o", hash: (0, 0, 0, 0, 0))
^1 = gv: (guid: 1, summaries: (function: (module: ^0, flags: (linkage: external, live: 1), insts: 1, calls: ((callee: ^2)))))
^2 = gv: (guid: 2, summaries: (function: (module: ^0, flags: (linkage: external, live: 1), insts: 1)))
^3 = gv: (guid: 3, summaries: (function: (module: ^0, flags: (linkage: external, live: 0), insts: 1, calls: ((callee: ^2)))))
^4 = flags: 1
)");
  ASSERT_NE(Index, nullptr);
  ASSERT_TRUE(Index->withGlobalValueDeadStripping());
  SummaryCallGraph CG(*Index, /*GUIDPreservedSymbols=*/{}, allPrevail);

  EXPECT_EQ(getGUIDs(CG.nodes()), GUIDList({1, 2}));
  EXPECT_EQ(getGUIDs(CG.lookup(2)->callers()), GUIDList({1}));
}

// A call to an alias is an edge to the function that the alias resolves to,
// also through an alias in another module. A preserved alias gives its aliasee
// unknown callers.
TEST(SummaryCallGraphTest, Aliases) {
  // GUID 20 is an alias of GUID 21 in a.o. The copy of GUID 21 that prevails
  // is an alias of GUID 2 in b.o. GUIDs 30 and 31 are aliases of each other.
  // An aliasee comes before its alias, so that the parser finds its copy in
  // the module of the alias.
  std::unique_ptr<ModuleSummaryIndex> Index = parseIndex(R"(
^0 = module: (path: "a.o", hash: (0, 0, 0, 0, 0))
^1 = module: (path: "b.o", hash: (0, 0, 0, 0, 1))
^2 = gv: (guid: 1, summaries: (function: (module: ^0, flags: (linkage: external), insts: 1, calls: ((callee: ^4), (callee: ^6), (callee: ^7)))))
^3 = gv: (guid: 2, summaries: (function: (module: ^1, flags: (linkage: external), insts: 1)))
^4 = gv: (guid: 10, summaries: (alias: (module: ^1, flags: (linkage: external), aliasee: ^3)))
^5 = gv: (guid: 21, summaries: (function: (module: ^0, flags: (linkage: linkonce_odr), insts: 1), alias: (module: ^1, flags: (linkage: linkonce_odr), aliasee: ^3)))
^6 = gv: (guid: 20, summaries: (alias: (module: ^0, flags: (linkage: linkonce_odr), aliasee: ^5)))
^7 = gv: (guid: 30, summaries: (alias: (module: ^0, flags: (linkage: linkonce_odr), aliasee: ^8), function: (module: ^1, flags: (linkage: linkonce_odr), insts: 1)))
^8 = gv: (guid: 31, summaries: (function: (module: ^0, flags: (linkage: linkonce_odr), insts: 1), alias: (module: ^1, flags: (linkage: linkonce_odr), aliasee: ^7)))
)");
  ASSERT_NE(Index, nullptr);
  DenseSet<GlobalValue::GUID> Preserved = {10};
  SummaryCallGraph CG(*Index, Preserved,
                      [](GlobalValue::GUID GUID, const GlobalValueSummary *S) {
                        if (GUID == 21 || GUID == 31)
                          return S->modulePath() == "b.o";
                        return S->modulePath() == "a.o" || GUID < 20;
                      });

  // The aliases have no nodes.
  for (GlobalValue::GUID GUID : {10, 20, 21})
    EXPECT_EQ(CG.lookup(GUID), nullptr);
  SummaryCallGraph::Node *N1 = CG.lookup(1), *N2 = CG.lookup(2);
  ASSERT_NE(N1, nullptr);
  ASSERT_NE(N2, nullptr);
  // The calls to GUIDs 10 and 20 reach GUID 2. The call to the cycle of
  // aliases reaches a function without a summary.
  ASSERT_EQ(N1->callees().size(), 2u);
  EXPECT_EQ(N1->callees()[0], N2);
  EXPECT_EQ(N1->callees()[1]->getSummary(), nullptr);
  EXPECT_EQ(getGUIDs(N2->callers()), GUIDList({1}));
  EXPECT_TRUE(N2->hasUnknownCallers());
  EXPECT_FALSE(N1->hasUnknownCallers());
}

static const char *AddressTakenSummary = R"(
^0 = module: (path: "a.o", hash: (0, 0, 0, 0, 0))
^1 = gv: (guid: 1, summaries: (function: (module: ^0, flags: (linkage: external, live: 1), insts: 1, refs: (^2))))
^2 = gv: (guid: 2, summaries: (function: (module: ^0, flags: (linkage: external, live: 1), insts: 1)))
^3 = gv: (guid: 3, summaries: (function: (module: ^0, flags: (linkage: external, live: 0), insts: 1, refs: (^4))))
^4 = gv: (guid: 4, summaries: (function: (module: ^0, flags: (linkage: external, live: 1), insts: 1)))
^5 = gv: (guid: 5, summaries: (function: (module: ^0, flags: (linkage: external, live: 1), insts: 1)))
^6 = gv: (guid: 8, summaries: (function: (module: ^0, flags: (linkage: external, live: 1), insts: 1)))
^7 = gv: (guid: 7, summaries: (alias: (module: ^0, flags: (linkage: external, live: 1), aliasee: ^6)))
^8 = gv: (guid: 6, summaries: (variable: (module: ^0, flags: (linkage: external, live: 1), varFlags: (readonly: 0), refs: (^7))))
^9 = flags: 1
)";

static void registerAnalyses(SummaryAnalysisManager &AM,
                             const DenseSet<GlobalValue::GUID> &Preserved) {
  AM.registerPass([] { return PassInstrumentationAnalysis(); });
  AM.registerPass([&] { return ThinLinkInfoAnalysis(Preserved, allPrevail); });
  AM.registerPass([] { return SummaryCallGraphAnalysis(); });
  AM.registerPass([] { return SummaryAddressTakenAnalysis(); });
}

// The address of a value is taken if a live summary refers to it or to an
// alias of it, or if it is a preserved symbol.
TEST(SummaryCallGraphTest, AddressTaken) {
  std::unique_ptr<ModuleSummaryIndex> Index = parseIndex(AddressTakenSummary);
  ASSERT_NE(Index, nullptr);
  DenseSet<GlobalValue::GUID> Preserved = {5};
  SummaryAnalysisManager AM;
  registerAnalyses(AM, Preserved);

  const DenseSet<GlobalValue::GUID> &AddressTaken =
      AM.getResult<SummaryAddressTakenAnalysis>(*Index);
  GUIDList GUIDs(AddressTaken.begin(), AddressTaken.end());
  llvm::sort(GUIDs);
  EXPECT_EQ(GUIDs, GUIDList({2, 5, 7, 8}));
}

TEST(SummaryCallGraphTest, AnalysisAndPrinter) {
  std::unique_ptr<ModuleSummaryIndex> Index = parseIndex(R"(
^0 = module: (path: "a.o", hash: (0, 0, 0, 0, 0))
^1 = gv: (guid: 1, summaries: (function: (module: ^0, flags: (linkage: external), insts: 1, calls: ((callee: ^2), (callee: ^3)))))
^2 = gv: (guid: 2, summaries: (function: (module: ^0, flags: (linkage: internal), insts: 1, calls: ((callee: ^2)))))
^3 = gv: (guid: 3)
)");
  ASSERT_NE(Index, nullptr);
  DenseSet<GlobalValue::GUID> Preserved = {1};
  SummaryAnalysisManager AM;
  registerAnalyses(AM, Preserved);

  // The result stays in the cache until a pass invalidates it.
  SummaryCallGraph *CG = &AM.getResult<SummaryCallGraphAnalysis>(*Index);
  EXPECT_EQ(AM.getCachedResult<SummaryCallGraphAnalysis>(*Index), CG);
  AM.invalidate(*Index, PreservedAnalyses::none());
  EXPECT_EQ(AM.getCachedResult<SummaryCallGraphAnalysis>(*Index), nullptr);

  std::string Output;
  raw_string_ostream OS(Output);
  SummaryPassManager PM;
  PM.addPass(SummaryCallGraphPrinterPass(OS));
  PM.run(*Index, AM);
  EXPECT_EQ(Output, "Summary call graph:\n"
                    "  1 [a.o] [unknown callers]\n"
                    "    calls 2\n"
                    "    calls 3\n"
                    "  2 [a.o]\n"
                    "    calls 2\n"
                    "  3 [no summary]\n"
                    "SCCs in bottom-up order:\n"
                    "  [2]\n"
                    "  [3]\n"
                    "  [1]\n");
}

} // end anonymous namespace
