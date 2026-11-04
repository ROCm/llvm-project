//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Passes/RunCodeGen.h"
#include "PassesOptions.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/ModuleSummaryAnalysis.h"
#include "llvm/Analysis/RuntimeLibcallInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/MachineFunctionAnalysisManager.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/ModuleSummaryIndex.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/CGPassBuilderOption.h"

using namespace llvm;

static Error runCodeGenPipelineLegacy(TargetMachine &TM, Module &M,
                                      raw_pwrite_stream &OS,
                                      raw_pwrite_stream *DwoOS,
                                      CodeGenFileType CGFT,
                                      const CodeGenPipelineConfig &Config) {
  legacy::PassManager CodeGenPasses;
  CodeGenPasses.add(
      createTargetTransformInfoWrapperPass(TM.getTargetIRAnalysis()));
  // Add LibraryInfo.
  TargetLibraryInfoImpl TLII(TM.getTargetTriple(), TM.Options.VecLib);
  if (Config.DisableSimplifyLibCalls)
    TLII.disableAllFunctions();
  CodeGenPasses.add(new TargetLibraryInfoWrapperPass(TLII));

  const TargetOptions &Options = TM.Options;
  CodeGenPasses.add(
      new RuntimeLibraryInfoWrapper(Options.MCOptions.ABIName, Options.VecLib));

  if (Config.SummaryIndex)
    CodeGenPasses.add(
        createImmutableModuleSummaryIndexWrapperPass(Config.SummaryIndex));

  if (Config.PreCodeGenPassesHook)
    Config.PreCodeGenPassesHook(CodeGenPasses);

  if (TM.addPassesToEmitFile(CodeGenPasses, OS, DwoOS, CGFT,
                             Config.DisableVerify))
    return createStringError("Failed to construct CodeGen pipeline");
  CodeGenPasses.run(M);

  return Error::success();
}

static Error runCodeGenPipelineNewPM(TargetMachine &TM, Module &M,
                                     raw_pwrite_stream &OS,
                                     raw_pwrite_stream *DwoOS,
                                     CodeGenFileType CGFT,
                                     const CodeGenPipelineConfig &Config) {
  ModulePassManager MPM;
  MachineFunctionAnalysisManager MFAM;
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;
  CGPassBuilderOption Opt = getCGPassBuilderOption(TM);
  Opt.DisableVerify = Config.DisableVerify;
  MachineModuleInfo MMI(&TM);
  PassInstrumentationCallbacks PIC;
  PipelineTuningOptions PTOptions;
  StandardInstrumentations SI(M.getContext(), Config.DebugPassManager,
                              Config.VerifyEach);
  SI.registerCallbacks(PIC, &MAM);

  // Add LibraryInfo.
  TargetLibraryInfoImpl TLII(TM.getTargetTriple(), TM.Options.VecLib);
  if (Config.DisableSimplifyLibCalls)
    TLII.disableAllFunctions();
  FAM.registerPass([&] { return TargetLibraryAnalysis(TLII); });

  MAM.registerPass([&] { return MachineModuleAnalysis(MMI); });
  MAM.registerPass([&] {
    const TargetOptions &Options = TM.Options;
    return RuntimeLibraryAnalysis(Options.MCOptions.ABIName, Options.VecLib);
  });

  if (Config.SummaryIndex)
    MAM.registerPass([&] {
      return ImmutableModuleSummaryIndexAnalysis(Config.SummaryIndex);
    });

  PassBuilder PB(M.getContext(), &TM, PTOptions, std::nullopt, &PIC,
                 Config.VFS ? Config.VFS : vfs::getRealFileSystem());
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.registerMachineFunctionAnalyses(MFAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM, &MFAM);

  Error BuildPipelineError = TM.buildCodeGenPipeline(
      MPM, MAM, OS, DwoOS, CGFT, Opt, MMI.getContext(), &PIC);
  if (BuildPipelineError)
    return BuildPipelineError;

  MPM.run(M, MAM);
  return Error::success();
}

Error llvm::runCodeGenPipeline(TargetMachine &TM, Module &M,
                               raw_pwrite_stream &OS, raw_pwrite_stream *DwoOS,
                               CodeGenFileType CGFT,
                               const CodeGenPipelineConfig &Config) {
  const PassesOptions &Opts = getPassesOptions(M.getContext());
  if (valueOr(Opts.force_new_pm_codegen, TM.shouldDefaultToNewPM())) {
    return runCodeGenPipelineNewPM(TM, M, OS, DwoOS, CGFT, Config);
  }

  return runCodeGenPipelineLegacy(TM, M, OS, DwoOS, CGFT, Config);
}
