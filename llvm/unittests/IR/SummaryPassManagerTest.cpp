//===- SummaryPassManagerTest.cpp - Summary pass manager unit tests -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/SummaryPassManager.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

static std::unique_ptr<ModuleSummaryIndex> parseIndex(const char *Summary) {
  SMDiagnostic Err;
  std::unique_ptr<ModuleSummaryIndex> Index =
      parseSummaryIndexAssemblyString(Summary, Err);
  if (!Index)
    Err.print("SummaryPassManagerTest", errs());
  return Index;
}

static const char *TwoFunctions = R"(
^0 = module: (path: "a.o", hash: (0, 0, 0, 0, 0))
^1 = gv: (guid: 1, summaries: (function: (module: ^0, flags: (linkage: external), insts: 1, calls: ((callee: ^2)))))
^2 = gv: (guid: 2, summaries: (function: (module: ^0, flags: (linkage: external), insts: 1)))
)";

// Counts the function summaries. Records how often it runs.
class CountFunctionsAnalysis
    : public AnalysisInfoMixin<CountFunctionsAnalysis> {
public:
  struct Result {
    unsigned NumFunctions;
  };

  CountFunctionsAnalysis(unsigned &Runs) : Runs(Runs) {}

  Result run(ModuleSummaryIndex &Index, SummaryAnalysisManager &) {
    ++Runs;
    unsigned NumFunctions = 0;
    for (const auto &[GUID, Info] : Index)
      for (const auto &Summary : Info.getSummaryList())
        NumFunctions += isa<FunctionSummary>(Summary.get());
    return {NumFunctions};
  }

private:
  friend AnalysisInfoMixin<CountFunctionsAnalysis>;
  static AnalysisKey Key;
  unsigned &Runs;
};

AnalysisKey CountFunctionsAnalysis::Key;

// Marks every summary live, reads the analysis, and preserves either all or
// no analyses.
struct MarkLivePass : RequiredPassInfoMixin<MarkLivePass> {
  unsigned &NumFunctions;
  bool PreserveAll;

  MarkLivePass(unsigned &NumFunctions, bool PreserveAll)
      : NumFunctions(NumFunctions), PreserveAll(PreserveAll) {}

  PreservedAnalyses run(ModuleSummaryIndex &Index, SummaryAnalysisManager &AM) {
    NumFunctions = AM.getResult<CountFunctionsAnalysis>(Index).NumFunctions;
    for (const auto &[GUID, Info] : Index)
      for (const auto &Summary : Info.getSummaryList())
        Summary->setLive(true);
    return PreserveAll ? PreservedAnalyses::all() : PreservedAnalyses::none();
  }
};

TEST(SummaryPassManagerTest, RunsPassesAndCachesAnalyses) {
  std::unique_ptr<ModuleSummaryIndex> Index = parseIndex(TwoFunctions);
  ASSERT_NE(Index, nullptr);

  unsigned Runs = 0;
  SummaryAnalysisManager AM;
  AM.registerPass([] { return PassInstrumentationAnalysis(); });
  AM.registerPass([&] { return CountFunctionsAnalysis(Runs); });

  unsigned Seen1 = 0, Seen2 = 0, Seen3 = 0;
  SummaryPassManager PM;
  PM.addPass(MarkLivePass(Seen1, /*PreserveAll=*/true));
  PM.addPass(MarkLivePass(Seen2, /*PreserveAll=*/false));
  PM.addPass(MarkLivePass(Seen3, /*PreserveAll=*/true));
  PM.run(*Index, AM);

  EXPECT_EQ(Seen1, 2u);
  EXPECT_EQ(Seen2, 2u);
  EXPECT_EQ(Seen3, 2u);
  // The first pass preserves the analysis, so the second pass gets the cached
  // result. The second pass preserves nothing, so the third pass computes it
  // again.
  EXPECT_EQ(Runs, 2u);
  for (const auto &[GUID, Info] : *Index)
    for (const auto &Summary : Info.getSummaryList())
      EXPECT_TRUE(Summary->isLive());
}

TEST(SummaryPassManagerTest, ThinLinkInfo) {
  std::unique_ptr<ModuleSummaryIndex> Index = parseIndex(TwoFunctions);
  ASSERT_NE(Index, nullptr);

  DenseSet<GlobalValue::GUID> Preserved = {2};
  SummaryAnalysisManager AM;
  AM.registerPass([] { return PassInstrumentationAnalysis(); });
  AM.registerPass([&] {
    return ThinLinkInfoAnalysis(
        Preserved, [](GlobalValue::GUID GUID, const GlobalValueSummary *) {
          return GUID == 1;
        });
  });

  auto &Info = AM.getResult<ThinLinkInfoAnalysis>(*Index);
  EXPECT_TRUE(Info.GUIDPreservedSymbols.contains(2));
  EXPECT_FALSE(Info.GUIDPreservedSymbols.contains(1));
  EXPECT_TRUE(Info.IsPrevailing(1, nullptr));
  EXPECT_FALSE(Info.IsPrevailing(2, nullptr));

  // The link facts survive an invalidation of all analyses.
  AM.invalidate(*Index, PreservedAnalyses::none());
  EXPECT_NE(AM.getCachedResult<ThinLinkInfoAnalysis>(*Index), nullptr);
}

TEST(SummaryPassManagerTest, Instrumentation) {
  std::unique_ptr<ModuleSummaryIndex> Index = parseIndex(TwoFunctions);
  ASSERT_NE(Index, nullptr);

  // Instrumentation callbacks get the index as the IR unit.
  PassInstrumentationCallbacks PIC;
  std::vector<std::string> Passes;
  PIC.registerBeforeNonSkippedPassCallback([&](StringRef Name, IRUnitRef IR) {
    EXPECT_EQ(dyn_cast<ModuleSummaryIndex>(IR), Index.get());
    Passes.push_back(Name.str());
  });

  unsigned Runs = 0, Seen = 0;
  SummaryAnalysisManager AM;
  AM.registerPass([&] { return PassInstrumentationAnalysis(&PIC); });
  AM.registerPass([&] { return CountFunctionsAnalysis(Runs); });
  SummaryPassManager PM;
  PM.addPass(MarkLivePass(Seen, /*PreserveAll=*/true));
  PM.run(*Index, AM);

  ASSERT_EQ(Passes.size(), 1u);
  EXPECT_NE(Passes[0].find("MarkLivePass"), std::string::npos);
}

TEST(SummaryPassManagerTest, StackTraceName) {
  std::unique_ptr<ModuleSummaryIndex> Index = parseIndex(TwoFunctions);
  ASSERT_NE(Index, nullptr);
  std::string Name;
  raw_string_ostream OS(Name);
  printIRUnitNameForStackTrace(OS, *Index);
  EXPECT_EQ(Name, "summary index of 1 module");
}

} // end anonymous namespace
