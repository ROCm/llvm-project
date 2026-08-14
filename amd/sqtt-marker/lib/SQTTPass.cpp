//===- SQTTPass.cpp - SQTT pass plugin ------------------------------------===//
//
// Part of AMD SQTT Marker, under the MIT License. See
// amd/sqtt-marker/LICENSE.txt for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Orchestrates SQTT instrumentation phases and registers the pass plugin.
///
//===----------------------------------------------------------------------===//

#include "SQTTPass.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Passes/PassBuilder.h"
#if __has_include("llvm/Plugins/PassPlugin.h")
#include "llvm/Plugins/PassPlugin.h"
#else
#include "llvm/Passes/PassPlugin.h"
#endif
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Transforms/Utils/Cloning.h"

using namespace llvm;

namespace {

constexpr StringLiteral MarkerSentinelNames[] = {
    "sqtt_marker_enter",         "sqtt_marker_exit",
    "sqtt_marker_point",         "sqtt_marker_data",
    "__sqtt_named_marker_enter", "__sqtt_named_marker_exit",
    "__sqtt_named_marker_point", "__sqtt_named_marker_data"};

} // namespace

static void eraseUnusedMarkerSentinels(Module &M) {
  for (StringRef Name : MarkerSentinelNames)
    if (Function *F = M.getFunction(Name); F && F->use_empty())
      F->eraseFromParent();
}

static bool inlineMarkerWrappers(Module &M) {
  SmallVector<Function *, 4> Wrappers;
  for (StringRef Name : MarkerSentinelNames) {
    Function *Sentinel = M.getFunction(Name);
    if (!Sentinel)
      continue;
    for (User *U : Sentinel->users()) {
      auto *Call = dyn_cast<CallInst>(U);
      Function *Wrapper = Call ? Call->getFunction() : nullptr;
      if (Wrapper && Wrapper != Sentinel &&
          llvm::find(Wrappers, Wrapper) == Wrappers.end())
        Wrappers.push_back(Wrapper);
    }
  }

  bool Changed = false;
  for (Function *Wrapper : Wrappers) {
    SmallVector<CallInst *, 8> CallSites;
    for (User *U : Wrapper->users())
      if (auto *Call = dyn_cast<CallInst>(U))
        CallSites.push_back(Call);
    for (CallInst *Call : CallSites) {
      InlineFunctionInfo IFI;
      InlineFunction(*Call, IFI);
      Changed = true;
    }
  }
  return Changed;
}

template <typename Callable>
static bool visitTargetFunctions(Module &M, Callable &&Visit) {
  bool Changed = false;
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    GfxGen Gen = getGfxGen(F);
    if (Gen != GfxGen::Unknown)
      Changed |= Visit(F, Gen);
  }
  return Changed;
}

PreservedAnalyses SQTTInstrumentPass::run(Module &M,
                                          ModuleAnalysisManager &MAM) {
  if (!Triple(M.getTargetTriple()).isAMDGPU())
    return PreservedAnalyses::all();
  return PassMode == Mode::Early ? runEarly(M) : runLate(M);
}

PreservedAnalyses SQTTInstrumentPass::runEarly(Module &M) {
  bool Changed = false;
  LLVMContext &Ctx = M.getContext();

  // Force-inline all callers of the named marker sentinels.
  Changed |= inlineMarkerWrappers(M);

  // Now resolve sentinel calls that are directly visible.
  Changed |= visitTargetFunctions(M, [&](Function &F, GfxGen Gen) {
    return processMarkerCalls(F, Gen, /*useBareTrace=*/true);
  });

  eraseUnusedMarkerSentinels(M);

  Changed |= visitTargetFunctions(M, [&](Function &F, GfxGen Gen) {
    if (F.getCallingConv() == CallingConv::AMDGPU_KERNEL ||
        Config.FunctionThreshold == 0 || hasMustTailCall(F))
      return false;
    uint32_t Id = NextEventID++;
    Type *I32 = Type::getInt32Ty(Ctx);
    F.setMetadata(
        "sqtt.func.id",
        MDNode::get(Ctx, {ConstantAsMetadata::get(ConstantInt::get(I32, Id))}));
    Markers.push_back({Id, MarkerKind::Function, F.getName().str(),
                       getFunctionSourceLoc(F),
                       computeFunctionSize(F, Config.Mode)});
    insertFunctionMarkers(F, Id, Gen, /*useBareTrace=*/true);
    return true;
  });

  if (!Markers.empty())
    storeEarlyMarkerMetadata(M, Ctx);

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

PreservedAnalyses SQTTInstrumentPass::runLate(Module &M) {
  bool Changed = false;

  bool HadEarlyFuncInst = false;
  bool HadEarlyPass = recoverEarlyMarkerMetadata(M, HadEarlyFuncInst);

  if (HadEarlyFuncInst)
    Changed |= finalizeEarlyFunctionMarkers(M);

  auto AddSystemMarkers = [&](std::initializer_list<const char *> Names) {
    uint32_t FirstId = NextEventID;
    for (const char *Name : Names)
      Markers.push_back({NextEventID++, MarkerKind::SystemPoint, Name});
    return FirstId;
  };
  if (Config.InstrumentBarriers)
    FirstBarrierID =
        AddSystemMarkers({"barrier_signal", "barrier_wait", "barrier"});
  if (Config.MemoryChunkSize)
    FirstVmemID = AddSystemMarkers({"vmem_load", "vmem_store"});

  // A nonzero clock field must wait until every payload-producing protocol
  // has been discovered. The default no-clock path can lower each function
  // as soon as all of its markers have been inserted.
  const bool DeferFullTraceFinalization = Config.ShaderClockBits != 0;
  Changed |= visitTargetFunctions(M, [&](Function &F, GfxGen Gen) {
    CurScopeCheck = nullptr; // reset per function
    bool IsKernel = F.getCallingConv() == CallingConv::AMDGPU_KERNEL;
    if (IsKernel)
      Markers.push_back(
          {0, MarkerKind::Kernel, F.getName().str(), getFunctionSourceLoc(F)});

    bool Changed = finalizeExistingMarkers(F, Gen);
    Changed |= processMarkerCalls(F, Gen, /*useBareTrace=*/false);
    if (Config.InstrumentBarriers)
      Changed |= instrumentBarriers(F, Gen);
    if (Config.MemoryChunkSize)
      Changed |= instrumentMemoryOps(F, Gen);
    if (Config.hasAddressTracing())
      Changed |= instrumentAddressTraces(F, Gen);
    if (!HadEarlyPass && Config.FunctionThreshold > 0 && !IsKernel)
      Changed |= instrumentFunctionDirect(F, Gen);
    if (!DeferFullTraceFinalization)
      Changed |= finalizeFullTraces(F, Gen);
    return Changed;
  });

  if (Config.ShaderClockBits != 0 &&
      llvm::any_of(Markers, [](const MarkerRecord &Entry) {
        return Entry.ExtraPayloadCount != 0;
      }))
    report_fatal_error("sqtt payload markers require SQTT_SHADER_CLOCK_BITS=0");

  if (DeferFullTraceFinalization)
    Changed |= visitTargetFunctions(
        M, [&](Function &F, GfxGen Gen) { return finalizeFullTraces(F, Gen); });

  eraseUnusedMarkerSentinels(M);

  if (!Markers.empty() || ShaderClockBitsUsed > 0) {
    emitFuncMap(M);
    Changed = true;
  }

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

// ============================================================================
// Plugin entry point
// ============================================================================

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "SQTTMarkerPass", SQTT_MARKER_VERSION_STRING,
          [](PassBuilder &PB) {
            using Mode = SQTTInstrumentPass::Mode;
            SQTTConfig Cfg = SQTTConfig::fromEnvironment();

            PB.registerPipelineEarlySimplificationEPCallback(
                [Cfg](ModulePassManager &MPM, OptimizationLevel OL,
                      ThinOrFullLTOPhase) {
                  if (OL != OptimizationLevel::O0)
                    MPM.addPass(SQTTInstrumentPass(Cfg, Mode::Early));
                });

            PB.registerOptimizerLastEPCallback([Cfg](ModulePassManager &MPM,
                                                     OptimizationLevel OL,
                                                     ThinOrFullLTOPhase) {
              if (OL != OptimizationLevel::O0)
                MPM.addPass(SQTTInstrumentPass(Cfg, Mode::Late));
            });

            PB.registerPipelineStartEPCallback(
                [Cfg](ModulePassManager &MPM, OptimizationLevel OL) {
                  if (OL == OptimizationLevel::O0)
                    MPM.addPass(SQTTInstrumentPass(Cfg, Mode::Late));
                });
          }};
}
