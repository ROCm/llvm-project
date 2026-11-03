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

// The IDs of the special nodes in the GUID lists of the tests.
constexpr GlobalValue::GUID ExternalCallers = ~0ULL;
constexpr GlobalValue::GUID ExternalCallees = ~0ULL - 1;

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

// Runs the analyses on an index.
class Analyses {
public:
  Analyses(ModuleSummaryIndex &Index,
           DenseSet<GlobalValue::GUID> PreservedSymbols = {},
           ThinLinkInfoAnalysis::IsPrevailingFn IsPrevailing = allPrevail)
      : Index(Index), PreservedSymbols(std::move(PreservedSymbols)) {
    AM.registerPass([] { return PassInstrumentationAnalysis(); });
    AM.registerPass([&] {
      return ThinLinkInfoAnalysis(this->PreservedSymbols, IsPrevailing);
    });
    AM.registerPass([] { return SummaryCallGraphAnalysis(); });
    AM.registerPass([] { return SummaryAddressTakenAnalysis(); });
  }

  SummaryCallGraph &graph() {
    return AM.getResult<SummaryCallGraphAnalysis>(Index);
  }

  ModuleSummaryIndex &Index;
  DenseSet<GlobalValue::GUID> PreservedSymbols;
  SummaryAnalysisManager AM;
};

static GlobalValue::GUID getID(const SummaryCallGraph &CG,
                               const SummaryCallGraph::Node *N) {
  if (N == CG.getExternalCallingNode())
    return ExternalCallers;
  if (N == CG.getCallsExternalNode())
    return ExternalCallees;
  return N->getValueInfo().getGUID();
}

static GUIDList getIDs(const SummaryCallGraph &CG,
                       ArrayRef<SummaryCallGraph::Node *> Nodes) {
  GUIDList IDs;
  for (const SummaryCallGraph::Node *N : Nodes)
    IDs.push_back(getID(CG, N));
  return IDs;
}

static GUIDList getIDs(const SummaryCallGraph &CG,
                       ArrayRef<SummaryCallGraph::Edge> Edges) {
  GUIDList IDs;
  for (const SummaryCallGraph::Edge &E : Edges)
    IDs.push_back(getID(CG, E.getNode()));
  return IDs;
}

// Returns the IDs of each SCC in bottom-up order. The IDs in an SCC are
// sorted.
static std::vector<GUIDList> getSCCIDs(const SummaryCallGraph &CG,
                                       bool IncludeIndirect = true) {
  std::vector<GUIDList> SCCs;
  for (SummaryCallGraph::SCC C : CG.sccs(IncludeIndirect)) {
    SCCs.push_back(getIDs(CG, C));
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
  Analyses A(*Index, /*PreservedSymbols=*/{1});
  SummaryCallGraph &CG = A.graph();

  EXPECT_EQ(getIDs(CG, CG.nodes()), GUIDList({1, 2, 3}));
  SummaryCallGraph::Node *N1 = CG.lookup(1), *N3 = CG.lookup(3);
  ASSERT_NE(N1, nullptr);
  ASSERT_NE(N3, nullptr);
  EXPECT_EQ(N1->getSummary(), Index->getValueInfo(1).getSummaryList()[0].get());
  EXPECT_EQ(N1->copies().size(), 1u);
  EXPECT_EQ(getIDs(CG, N1->callees()), GUIDList({2, 3}));
  for (const SummaryCallGraph::Edge &E : N1->callees()) {
    EXPECT_TRUE(E.isCall());
    EXPECT_FALSE(E.isIndirect());
  }
  EXPECT_EQ(getIDs(CG, N3->callers()), GUIDList({1, 2}));
  EXPECT_TRUE(N3->callees().empty());
  EXPECT_EQ(CG.lookup(4), nullptr);

  // Code outside the summaries can call the preserved symbol 1.
  EXPECT_EQ(getIDs(CG, CG.getExternalCallingNode()->callees()), GUIDList({1}));
  EXPECT_EQ(getIDs(CG, N1->callers()), GUIDList({ExternalCallers}));
  EXPECT_TRUE(CG.getCallsExternalNode()->callers().empty());

  EXPECT_EQ(getSCCIDs(CG),
            std::vector<GUIDList>(
                {{3}, {2}, {1}, {ExternalCallers}, {ExternalCallees}}));
}

// The SCC walk visits every node, also a cycle that no node outside the cycle
// calls.
TEST(SummaryCallGraphTest, Cycles) {
  std::unique_ptr<ModuleSummaryIndex> Index = parseIndex(R"(
^0 = module: (path: "a.o", hash: (0, 0, 0, 0, 0))
^1 = gv: (guid: 1, summaries: (function: (module: ^0, flags: (linkage: external), insts: 1, calls: ((callee: ^2)))))
^2 = gv: (guid: 2, summaries: (function: (module: ^0, flags: (linkage: external), insts: 1, calls: ((callee: ^1), (callee: ^3)))))
^3 = gv: (guid: 3, summaries: (function: (module: ^0, flags: (linkage: external), insts: 1, calls: ((callee: ^3)))))
)");
  ASSERT_NE(Index, nullptr);
  Analyses A(*Index);
  SummaryCallGraph &CG = A.graph();

  EXPECT_EQ(getSCCIDs(CG),
            std::vector<GUIDList>(
                {{ExternalCallers}, {3}, {1, 2}, {ExternalCallees}}));
  SummaryCallGraph::Node *N3 = CG.lookup(3);
  ASSERT_NE(N3, nullptr);
  EXPECT_EQ(getIDs(CG, N3->callees()), GUIDList({3}));
  EXPECT_EQ(getIDs(CG, N3->callers()), GUIDList({2, 3}));
}

// The edges come from all live copies. A module can inline its own copy of a
// template function that does not prevail, so the calls of that copy also run.
TEST(SummaryCallGraphTest, Copies) {
  std::unique_ptr<ModuleSummaryIndex> Index = parseIndex(R"(
^0 = module: (path: "a.o", hash: (0, 0, 0, 0, 0))
^1 = module: (path: "b.o", hash: (0, 0, 0, 0, 1))
^2 = gv: (guid: 1, summaries: (function: (module: ^0, flags: (linkage: linkonce_odr), insts: 1, calls: ((callee: ^3))), function: (module: ^1, flags: (linkage: linkonce_odr), insts: 1, calls: ((callee: ^4)))))
^3 = gv: (guid: 2, summaries: (function: (module: ^0, flags: (linkage: external), insts: 1)))
^4 = gv: (guid: 3, summaries: (function: (module: ^1, flags: (linkage: external), insts: 1)))
)");
  ASSERT_NE(Index, nullptr);
  Analyses A(*Index, /*PreservedSymbols=*/{},
             [](GlobalValue::GUID GUID, const GlobalValueSummary *S) {
               return GUID != 1 || S->modulePath() == "b.o";
             });
  SummaryCallGraph &CG = A.graph();

  SummaryCallGraph::Node *N1 = CG.lookup(1);
  ASSERT_NE(N1, nullptr);
  ASSERT_NE(N1->getSummary(), nullptr);
  EXPECT_EQ(N1->getSummary()->modulePath(), "b.o");
  ASSERT_EQ(N1->copies().size(), 2u);
  EXPECT_EQ(getIDs(CG, N1->callees()), GUIDList({2, 3}));
}

// A local copy and a promoted copy are in use without a prevailing check. If
// no copy prevails, the node calls the calls external node.
TEST(SummaryCallGraphTest, LocalAndPromotedCopies) {
  std::unique_ptr<ModuleSummaryIndex> Index = parseIndex(R"(
^0 = module: (path: "a.o", hash: (0, 0, 0, 0, 0))
^1 = gv: (guid: 1, summaries: (function: (module: ^0, flags: (linkage: internal), insts: 1)))
^2 = gv: (guid: 2, summaries: (function: (module: ^0, flags: (linkage: internal), insts: 1)))
^3 = gv: (guid: 3, summaries: (function: (module: ^0, flags: (linkage: external), insts: 1)))
)");
  ASSERT_NE(Index, nullptr);
  Index->getValueInfo(2).getSummaryList()[0]->promote();
  Analyses A(
      *Index, /*PreservedSymbols=*/{},
      [](GlobalValue::GUID, const GlobalValueSummary *) { return false; });
  SummaryCallGraph &CG = A.graph();

  EXPECT_EQ(getIDs(CG, CG.nodes()), GUIDList({1, 2, 3}));
  EXPECT_NE(CG.lookup(1)->getSummary(), nullptr);
  EXPECT_NE(CG.lookup(2)->getSummary(), nullptr);
  EXPECT_EQ(CG.lookup(3)->getSummary(), nullptr);
  EXPECT_EQ(getIDs(CG, CG.lookup(3)->callees()), GUIDList({ExternalCallees}));
}

// A node without a known definition calls the calls external node.
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
  Analyses A(*Index);
  SummaryCallGraph &CG = A.graph();

  EXPECT_EQ(getIDs(CG, CG.nodes()), GUIDList({1, 2, 3, 4}));
  for (GlobalValue::GUID GUID : {2, 3, 4}) {
    SummaryCallGraph::Node *N = CG.lookup(GUID);
    ASSERT_NE(N, nullptr);
    EXPECT_EQ(N->getSummary(), nullptr);
    EXPECT_EQ(getIDs(CG, N->callees()), GUIDList({ExternalCallees}));
    EXPECT_TRUE(N->callees()[0].isCall());
    EXPECT_EQ(getIDs(CG, N->callers()), GUIDList({1}));
  }
  EXPECT_TRUE(CG.lookup(2)->copies().empty());
  EXPECT_EQ(CG.lookup(3)->copies().size(), 1u);
  EXPECT_EQ(CG.lookup(4)->copies().size(), 2u);
  EXPECT_EQ(getIDs(CG, CG.getCallsExternalNode()->callers()),
            GUIDList({2, 3, 4}));
}

// A dead function has no node, and a dead copy adds no edges.
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
  Analyses A(*Index);
  SummaryCallGraph &CG = A.graph();

  EXPECT_EQ(getIDs(CG, CG.nodes()), GUIDList({1, 2}));
  EXPECT_EQ(getIDs(CG, CG.lookup(2)->callers()), GUIDList({1}));
}

// An alias calls its aliasee.
TEST(SummaryCallGraphTest, Aliases) {
  // GUID 10 is an alias of GUID 2. GUID 20 is an alias of GUID 21 in a.o. GUID
  // 21 is a function in a.o, and its copy that prevails is an alias of GUID 2
  // in b.o (like a C++ constructor alias). GUIDs 30 and 31 are aliases of each
  // other. An aliasee comes before its alias, so that the parser finds its copy
  // in the module of the alias.
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
  Analyses A(*Index, /*PreservedSymbols=*/{10},
             [](GlobalValue::GUID GUID, const GlobalValueSummary *S) {
               if (GUID == 21 || GUID == 31)
                 return S->modulePath() == "b.o";
               return S->modulePath() == "a.o" || GUID < 20;
             });
  SummaryCallGraph &CG = A.graph();

  EXPECT_EQ(getIDs(CG, CG.nodes()), GUIDList({1, 2, 10, 20, 21, 30, 31}));
  SummaryCallGraph::Node *N10 = CG.lookup(10), *N21 = CG.lookup(21);
  ASSERT_NE(N10, nullptr);
  ASSERT_NE(N21, nullptr);
  EXPECT_TRUE(isa_and_nonnull<AliasSummary>(N10->getPrevailingCopy()));
  EXPECT_EQ(N10->getSummary(), nullptr);
  EXPECT_TRUE(N10->copies().empty());
  // The function copy of GUID 21 in a.o can still run if a.o inlines it.
  EXPECT_TRUE(isa_and_nonnull<AliasSummary>(N21->getPrevailingCopy()));
  EXPECT_EQ(N21->copies().size(), 1u);

  EXPECT_EQ(getIDs(CG, CG.lookup(1)->callees()), GUIDList({10, 20, 30}));
  EXPECT_EQ(getIDs(CG, N10->callees()), GUIDList({2}));
  EXPECT_EQ(getIDs(CG, N10->callers()), GUIDList({ExternalCallers, 1}));
  EXPECT_EQ(getIDs(CG, CG.lookup(20)->callees()), GUIDList({21}));
  EXPECT_EQ(getIDs(CG, N21->callees()), GUIDList({2}));
  EXPECT_EQ(getIDs(CG, CG.lookup(2)->callers()), GUIDList({10, 21}));
  // A cycle of aliases is a cycle in the graph.
  EXPECT_EQ(getIDs(CG, CG.lookup(30)->callees()), GUIDList({31}));
  EXPECT_EQ(getIDs(CG, CG.lookup(31)->callees()), GUIDList({30}));
  EXPECT_TRUE(is_contained(getSCCIDs(CG), GUIDList({30, 31})));
}

// The variable with GUID 5 refers to the functions 1, 2, 3, 9, and the alias 6
// of the function 7. The functions 1, 2, 4, 7, and 9 have the call type ID
// 100, and 3 has 200. No summary refers to 4. The function 9 is dead. The
// function 1 has a copy in a second module, and the copy in test.o calls 8.
// The function 8 has indirect calls with the IDs 100 and 300, and an unknown
// call.
static const char *IndirectCallSummary = R"Summary(
^0 = module: (path: "test.o", hash: (0, 0, 0, 0, 0))
^1 = module: (path: "test2.o", hash: (0, 0, 0, 0, 0))
^2 = gv: (guid: 1, summaries: (function: (module: ^0, flags: (linkage: linkonce_odr, live: 1), insts: 1, calls: ((callee: ^9)), callTypeId: 100), function: (module: ^1, flags: (linkage: linkonce_odr, live: 1), insts: 1, callTypeId: 100)))
^3 = gv: (guid: 2, summaries: (function: (module: ^0, flags: (linkage: external, live: 1), insts: 1, callTypeId: 100)))
^4 = gv: (guid: 3, summaries: (function: (module: ^0, flags: (linkage: external, live: 1), insts: 1, callTypeId: 200)))
^5 = gv: (guid: 4, summaries: (function: (module: ^0, flags: (linkage: external, live: 1), insts: 1, callTypeId: 100)))
^6 = gv: (guid: 5, summaries: (variable: (module: ^0, flags: (linkage: external, live: 1), varFlags: (readonly: 0), refs: (^2, ^3, ^4, ^7, ^10))))
^7 = gv: (guid: 6, summaries: (alias: (module: ^0, flags: (linkage: external, live: 1), aliasee: ^8)))
^8 = gv: (guid: 7, summaries: (function: (module: ^0, flags: (linkage: external, live: 1), insts: 1, callTypeId: 100)))
^9 = gv: (guid: 8, summaries: (function: (module: ^0, flags: (linkage: external, live: 1), insts: 2, funcFlags: (hasUnknownCall: 1), indirectCallTypeIds: (100, 300))))
^10 = gv: (guid: 9, summaries: (function: (module: ^0, flags: (linkage: external, live: 0), insts: 1, callTypeId: 100)))
^11 = flags: 1
)Summary";

// An indirect call can reach the live functions with its call type ID whose
// address can be taken. For the alias, the aliasee is a target.
TEST(SummaryCallGraphTest, IndirectCalls) {
  std::unique_ptr<ModuleSummaryIndex> Index = parseIndex(IndirectCallSummary);
  ASSERT_NE(Index, nullptr);
  Analyses A(*Index);
  SummaryCallGraph &CG = A.graph();

  // The unknown call can be a call to inline assembly or to an ifunc, which
  // has no call type ID.
  SummaryCallGraph::Node *N8 = CG.lookup(8);
  ASSERT_NE(N8, nullptr);
  EXPECT_EQ(getIDs(CG, N8->callees()), GUIDList({1, 2, 7, ExternalCallees}));
  for (const SummaryCallGraph::Edge &E : N8->callees()) {
    EXPECT_FALSE(E.isCall());
    EXPECT_TRUE(E.isIndirect());
  }
  EXPECT_TRUE(N8->callees(/*IncludeIndirect=*/false).empty());
  EXPECT_EQ(getIDs(CG, CG.lookup(2)->callers()), GUIDList({8}));
  EXPECT_TRUE(CG.lookup(2)->callers(/*IncludeIndirect=*/false).empty());

  // The call from 1 to 8 and the indirect call from 8 to 1 form a cycle only
  // with the indirect edges.
  std::vector<GUIDList> SCCs = getSCCIDs(CG);
  EXPECT_TRUE(is_contained(SCCs, GUIDList({1, 8})));
  std::vector<GUIDList> CallSCCs = getSCCIDs(CG, /*IncludeIndirect=*/false);
  EXPECT_TRUE(is_contained(CallSCCs, GUIDList({1})));
  EXPECT_TRUE(is_contained(CallSCCs, GUIDList({8})));
}

// A function that code outside the summaries can call is also a target of the
// indirect calls in the summaries, also if no summary refers to it.
TEST(SummaryCallGraphTest, IndirectCallsToPreservedSymbols) {
  std::unique_ptr<ModuleSummaryIndex> Index = parseIndex(IndirectCallSummary);
  ASSERT_NE(Index, nullptr);
  Analyses A(*Index, /*PreservedSymbols=*/{4});
  SummaryCallGraph &CG = A.graph();

  EXPECT_EQ(getIDs(CG, CG.lookup(8)->callees()),
            GUIDList({1, 2, 4, 7, ExternalCallees}));
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

// The address of a value is taken if a live summary refers to it or to an
// alias of it, or if it is a preserved symbol.
TEST(SummaryCallGraphTest, AddressTaken) {
  std::unique_ptr<ModuleSummaryIndex> Index = parseIndex(AddressTakenSummary);
  ASSERT_NE(Index, nullptr);
  Analyses A(*Index, /*PreservedSymbols=*/{5});

  const DenseSet<GlobalValue::GUID> &AddressTaken =
      A.AM.getResult<SummaryAddressTakenAnalysis>(*Index);
  GUIDList GUIDs(AddressTaken.begin(), AddressTaken.end());
  llvm::sort(GUIDs);
  EXPECT_EQ(GUIDs, GUIDList({2, 5, 7, 8}));
}

TEST(SummaryCallGraphTest, AnalysisAndPrinter) {
  std::unique_ptr<ModuleSummaryIndex> Index = parseIndex(R"(
^0 = module: (path: "a.o", hash: (0, 0, 0, 0, 0))
^1 = module: (path: "b.o", hash: (0, 0, 0, 0, 1))
^2 = gv: (guid: 1, summaries: (function: (module: ^0, flags: (linkage: external), insts: 1, calls: ((callee: ^3), (callee: ^4)))))
^3 = gv: (guid: 2, summaries: (function: (module: ^0, flags: (linkage: linkonce_odr), insts: 1, calls: ((callee: ^3))), function: (module: ^1, flags: (linkage: linkonce_odr), insts: 1)))
^4 = gv: (guid: 3)
)");
  ASSERT_NE(Index, nullptr);
  Analyses A(*Index, /*PreservedSymbols=*/{1},
             [](GlobalValue::GUID, const GlobalValueSummary *S) {
               return S->modulePath() == "a.o";
             });

  // The result stays in the cache until a pass invalidates it.
  SummaryCallGraph *CG = &A.graph();
  EXPECT_EQ(A.AM.getCachedResult<SummaryCallGraphAnalysis>(*Index), CG);
  A.AM.invalidate(*Index, PreservedAnalyses::none());
  EXPECT_EQ(A.AM.getCachedResult<SummaryCallGraphAnalysis>(*Index), nullptr);

  std::string Output;
  raw_string_ostream OS(Output);
  SummaryPassManager PM;
  PM.addPass(SummaryCallGraphPrinterPass(OS));
  PM.run(*Index, A.AM);
  EXPECT_EQ(Output, "Summary call graph:\n"
                    "  <external callers>\n"
                    "    calls 1\n"
                    "  1 [a.o]\n"
                    "    calls 2\n"
                    "    calls 3\n"
                    "  2 [a.o] [other copies: b.o]\n"
                    "    calls 2\n"
                    "  3 [unknown]\n"
                    "    calls <external callees>\n"
                    "  <external callees>\n"
                    "SCCs in bottom-up order:\n"
                    "  [2]\n"
                    "  [<external callees>]\n"
                    "  [3]\n"
                    "  [1]\n"
                    "  [<external callers>]\n");
}

} // end anonymous namespace
