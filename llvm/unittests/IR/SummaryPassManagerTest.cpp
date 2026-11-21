//===- SummaryPassManagerTest.cpp - Summary pass manager unit tests -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/SummaryPassManager.h"
#include "llvm/Analysis/SummaryCallGraph.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Testing/Support/Error.h"
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
  EXPECT_FALSE(Info.ClosedWorld);

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

// The pass builder registers the summary analyses and parses a pipeline of
// summary passes.
TEST(SummaryPassManagerTest, ParsePipeline) {
  std::unique_ptr<ModuleSummaryIndex> Index = parseIndex(TwoFunctions);
  ASSERT_NE(Index, nullptr);

  LLVMContext Ctx;
  PassBuilder PB(Ctx);
  DenseSet<GlobalValue::GUID> Preserved;
  SummaryAnalysisManager AM;
  PB.registerSummaryAnalyses(AM);
  AM.registerPass([&] {
    return ThinLinkInfoAnalysis(
        Preserved,
        [](GlobalValue::GUID, const GlobalValueSummary *) { return true; });
  });

  SummaryPassManager PM;
  ASSERT_THAT_ERROR(PB.parsePassPipeline(PM, "require<summary-callgraph>,"
                                             "require<summary-address-taken>,"
                                             "invalidate<summary-callgraph>"),
                    Succeeded());
  PM.run(*Index, AM);
  EXPECT_EQ(AM.getCachedResult<SummaryCallGraphAnalysis>(*Index), nullptr);
  EXPECT_NE(AM.getCachedResult<SummaryAddressTakenAnalysis>(*Index), nullptr);

  // The printer of the call graph takes a parameter.
  SummaryPassManager PrintPM;
  EXPECT_THAT_ERROR(
      PB.parsePassPipeline(PrintPM, "print<summary-callgraph><indirect-calls>"),
      Succeeded());
}

TEST(SummaryPassManagerTest, ParsePipelineErrors) {
  LLVMContext Ctx;
  PassBuilder PB(Ctx);
  SummaryPassManager PM;
  EXPECT_THAT_ERROR(
      PB.parsePassPipeline(PM, "require<summary-callgraph>)"),
      FailedWithMessage(
          "invalid summary pass pipeline 'require<summary-callgraph>)'"));
  EXPECT_THAT_ERROR(PB.parsePassPipeline(PM, "no-such-pass"),
                    FailedWithMessage("unknown summary pass 'no-such-pass'"));
  // A pass over modules is not a summary pass.
  EXPECT_THAT_ERROR(PB.parsePassPipeline(PM, "no-op-module"),
                    FailedWithMessage("unknown summary pass 'no-op-module'"));
  EXPECT_THAT_ERROR(
      PB.parsePassPipeline(PM, "require<summary-callgraph>(require<"
                               "summary-callgraph>)"),
      FailedWithMessage("unknown summary pass 'require<summary-callgraph>'"));
  EXPECT_THAT_ERROR(
      PB.parsePassPipeline(PM, "print<summary-callgraph><no-such-option>"),
      FailedWithMessage("invalid SummaryCallGraphPrinterPass pass parameter "
                        "'no-such-option'"));
}

// Callbacks add passes and analyses that the pass builder does not know.
TEST(SummaryPassManagerTest, ParsingCallbacks) {
  std::unique_ptr<ModuleSummaryIndex> Index = parseIndex(TwoFunctions);
  ASSERT_NE(Index, nullptr);

  LLVMContext Ctx;
  PassBuilder PB(Ctx);
  unsigned Runs = 0, Seen = 0;
  PB.registerAnalysisRegistrationCallback([&](SummaryAnalysisManager &AM) {
    AM.registerPass([&] { return CountFunctionsAnalysis(Runs); });
  });
  PB.registerPipelineParsingCallback(
      [&](StringRef Name, SummaryPassManager &PM,
          ArrayRef<PassBuilder::PipelineElement> InnerPipeline) {
        if (Name != "mark-live" || !InnerPipeline.empty())
          return false;
        PM.addPass(MarkLivePass(Seen, /*PreserveAll=*/false));
        return true;
      });

  SummaryAnalysisManager AM;
  PB.registerSummaryAnalyses(AM);
  SummaryPassManager PM;
  ASSERT_THAT_ERROR(PB.parsePassPipeline(PM, "mark-live,mark-live"),
                    Succeeded());
  PM.run(*Index, AM);
  EXPECT_EQ(Seen, 2u);
  EXPECT_EQ(Runs, 2u);
  for (const auto &[GUID, Info] : *Index)
    for (const auto &Summary : Info.getSummaryList())
      EXPECT_TRUE(Summary->isLive());
}

// A callback of the extension point adds passes to the default pipeline of the
// thin link.
TEST(SummaryPassManagerTest, PipelineExtensionPoint) {
  std::unique_ptr<ModuleSummaryIndex> Index = parseIndex(TwoFunctions);
  ASSERT_NE(Index, nullptr);

  LLVMContext Ctx;
  PassBuilder PB(Ctx);
  unsigned Runs = 0, Seen = 0;
  PB.registerAnalysisRegistrationCallback([&](SummaryAnalysisManager &AM) {
    AM.registerPass([&] { return CountFunctionsAnalysis(Runs); });
  });
  PB.registerSummaryPipelineEPCallback([&](SummaryPassManager &PM) {
    PM.addPass(MarkLivePass(Seen, /*PreserveAll=*/true));
  });

  SummaryAnalysisManager AM;
  PB.registerSummaryAnalyses(AM);
  SummaryPassManager PM;
  PB.invokeSummaryPipelineEPCallbacks(PM);
  PM.run(*Index, AM);
  EXPECT_EQ(Seen, 2u);
}

// The default thin-link pipeline runs the generic passes, then the passes of
// the extension point. The pipeline text "default" adds the same passes.
TEST(SummaryPassManagerTest, DefaultPipeline) {
  std::unique_ptr<ModuleSummaryIndex> Index = parseIndex(TwoFunctions);
  ASSERT_NE(Index, nullptr);

  LLVMContext Ctx;
  PassInstrumentationCallbacks PIC;
  std::vector<std::string> Passes;
  PIC.registerBeforeNonSkippedPassCallback(
      [&](StringRef Name, IRUnitRef) { Passes.push_back(Name.str()); });
  PassBuilder PB(Ctx, /*TM=*/nullptr, /*PTO=*/std::nullopt,
                 /*PGOOpt=*/std::nullopt, &PIC);
  unsigned Runs = 0, Seen = 0;
  PB.registerSummaryPipelineEPCallback([&](SummaryPassManager &PM) {
    PM.addPass(MarkLivePass(Seen, /*PreserveAll=*/true));
  });
  DenseSet<GlobalValue::GUID> Preserved;
  SummaryAnalysisManager AM;
  PB.registerSummaryAnalyses(AM);
  AM.registerPass([&] { return CountFunctionsAnalysis(Runs); });
  AM.registerPass([&] {
    return ThinLinkInfoAnalysis(
        Preserved,
        [](GlobalValue::GUID, const GlobalValueSummary *) { return true; });
  });

  auto CheckPasses = [&] {
    ASSERT_EQ(Passes.size(), 2u);
    EXPECT_NE(Passes[0].find("ThinLinkFunctionAttrsPass"), std::string::npos);
    EXPECT_NE(Passes[1].find("MarkLivePass"), std::string::npos);
  };
  SummaryPassManager PM = PB.buildThinLinkDefaultPipeline();
  PM.run(*Index, AM);
  CheckPasses();

  Passes.clear();
  SummaryPassManager TextPM;
  ASSERT_THAT_ERROR(PB.parsePassPipeline(TextPM, "default"), Succeeded());
  TextPM.run(*Index, AM);
  CheckPasses();
}

static unsigned TargetPassRuns = 0;

// A pass of a test target.
struct TargetPass : RequiredPassInfoMixin<TargetPass> {
  PreservedAnalyses run(ModuleSummaryIndex &, SummaryAnalysisManager &) {
    ++TargetPassRuns;
    return PreservedAnalyses::all();
  }
};

static void addTargetPasses(PassBuilder &PB) {
  PB.registerSummaryPipelineEPCallback(
      [](SummaryPassManager &PM) { PM.addPass(TargetPass()); });
}

// A target adds its passes to the thin link without a TargetMachine.
TEST(SummaryPassManagerTest, TargetCallback) {
  std::unique_ptr<ModuleSummaryIndex> Index = parseIndex(TwoFunctions);
  ASSERT_NE(Index, nullptr);

  Target T;
  TargetRegistry::RegisterThinLinkPassBuilderCallback(T, addTargetPasses);
  LLVMContext Ctx;
  PassBuilder PB(Ctx);
  T.registerThinLinkPassBuilderCallbacks(PB);

  SummaryAnalysisManager AM;
  PB.registerSummaryAnalyses(AM);
  SummaryPassManager PM;
  PB.invokeSummaryPipelineEPCallbacks(PM);
  TargetPassRuns = 0;
  PM.run(*Index, AM);
  EXPECT_EQ(TargetPassRuns, 1u);

  // A target without a callback adds no passes.
  Target Other;
  PassBuilder OtherPB(Ctx);
  Other.registerThinLinkPassBuilderCallbacks(OtherPB);
  SummaryPassManager OtherPM;
  OtherPB.invokeSummaryPipelineEPCallbacks(OtherPM);
  OtherPM.run(*Index, AM);
  EXPECT_EQ(TargetPassRuns, 1u);
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
