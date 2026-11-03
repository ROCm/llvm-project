//===- AMDGPUThinLinkPasses.cpp - AMDGPU passes of the thin link ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUThinLinkPasses.h"
#include "llvm/Analysis/SummaryCallGraph.h"
#include "llvm/Analysis/SummaryDataflow.h"
#include "llvm/IR/ModuleSummaryIndex.h"
#include "llvm/IR/ThinLinkFnAttrs.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <limits>
#include <optional>
#include <string>

using namespace llvm;

#define DEBUG_TYPE "amdgpu-thinlink-occupancy"

namespace {

/// The occupancy attributes of a function. A missing maximum of the waves per
/// EU is the maximum of the target.
struct OccupancyAttrs {
  uint32_t FlatWGSizeMin = 1;
  uint32_t FlatWGSizeMax = 1024;
  uint32_t WavesPerEUMin = 1;
  std::optional<uint32_t> WavesPerEUMax;
  uint32_t MaxNumWGX = std::numeric_limits<uint32_t>::max();
  uint32_t MaxNumWGY = std::numeric_limits<uint32_t>::max();
  uint32_t MaxNumWGZ = std::numeric_limits<uint32_t>::max();

  /// Gets the attributes of \p Summary. An attribute that the function does
  /// not have gets its default value.
  explicit OccupancyAttrs(const AMDGPUFunctionSummary &Summary) {
    if (Summary.FlatWorkGroupSize) {
      FlatWGSizeMin = Summary.FlatWorkGroupSize->first;
      FlatWGSizeMax = Summary.FlatWorkGroupSize->second;
    }
    if (Summary.WavesPerEU) {
      WavesPerEUMin = Summary.WavesPerEU->Min;
      WavesPerEUMax = Summary.WavesPerEU->Max;
    }
    if (Summary.MaxNumWorkgroups) {
      MaxNumWGX = (*Summary.MaxNumWorkgroups)[0];
      MaxNumWGY = (*Summary.MaxNumWorkgroups)[1];
      MaxNumWGZ = (*Summary.MaxNumWorkgroups)[2];
    }
  }

  /// Merges \p Other into these attributes, so that the result allows the
  /// launches of both.
  void merge(const OccupancyAttrs &Other) {
    FlatWGSizeMin = std::min(FlatWGSizeMin, Other.FlatWGSizeMin);
    FlatWGSizeMax = std::max(FlatWGSizeMax, Other.FlatWGSizeMax);
    WavesPerEUMin = std::max(WavesPerEUMin, Other.WavesPerEUMin);
    if (WavesPerEUMax && Other.WavesPerEUMax)
      WavesPerEUMax = std::max(*WavesPerEUMax, *Other.WavesPerEUMax);
    else
      WavesPerEUMax = std::nullopt;
    MaxNumWGX = std::max(MaxNumWGX, Other.MaxNumWGX);
    MaxNumWGY = std::max(MaxNumWGY, Other.MaxNumWGY);
    MaxNumWGZ = std::max(MaxNumWGZ, Other.MaxNumWGZ);
  }

  bool hasValidWavesPerEU() const {
    return !WavesPerEUMax || WavesPerEUMin <= *WavesPerEUMax;
  }

  bool operator==(const OccupancyAttrs &Other) const {
    return FlatWGSizeMin == Other.FlatWGSizeMin &&
           FlatWGSizeMax == Other.FlatWGSizeMax &&
           WavesPerEUMin == Other.WavesPerEUMin &&
           WavesPerEUMax == Other.WavesPerEUMax &&
           MaxNumWGX == Other.MaxNumWGX && MaxNumWGY == Other.MaxNumWGY &&
           MaxNumWGZ == Other.MaxNumWGZ;
  }
};

/// A top-down problem: the value of a function is the merge of the occupancy
/// attributes of the kernels that can reach it, or none if no kernel can.
struct KernelOccupancyProblem {
  using ValueT = std::optional<OccupancyAttrs>;

  const AMDGPUSummaryMap &Summaries;

  /// Returns the summary of \p N if it is a kernel, or null.
  const AMDGPUFunctionSummary *getKernel(const SummaryCallGraph::Node &N) {
    auto It = Summaries.find(N.getValueInfo().getGUID());
    if (It == Summaries.end() || It->second.CC != CallingConv::AMDGPU_KERNEL)
      return nullptr;
    return &It->second;
  }

  ValueT getInitialValue(const SummaryCallGraph::Node &) {
    return std::nullopt;
  }

  /// The code outside the summaries launches kernels. The pass assumes that it
  /// calls no device function.
  ValueT getExternalValue() { return std::nullopt; }

  void join(ValueT &Into, const ValueT &V) {
    if (!V)
      return;
    if (Into)
      Into->merge(*V);
    else
      Into = V;
  }

  /// Code cannot call a kernel, so the value of a kernel is its own
  /// attributes.
  ValueT transfer(const SummaryCallGraph::Node &N, const ValueT &In) {
    if (const AMDGPUFunctionSummary *Kernel = getKernel(N))
      return OccupancyAttrs(*Kernel);
    return In;
  }
};

} // end anonymous namespace

PreservedAnalyses AMDGPUThinLinkOccupancyPass::run(ModuleSummaryIndex &Index,
                                                   SummaryAnalysisManager &AM) {
  const AMDGPUSummaryMap &Summaries = Index.getAMDGPUSummaries();
  // Build the call graph only for a link with AMDGPU summaries.
  if (Summaries.empty())
    return PreservedAnalyses::all();

  const SummaryCallGraph &CG = AM.getResult<SummaryCallGraphAnalysis>(Index);
  KernelOccupancyProblem Problem{Summaries};
  auto Values =
      solveSummaryDataflow(CG, Problem, SummaryDataflowDirection::TopDown);

  for (const SummaryCallGraph::Node *N : CG.nodes()) {
    // A kernel keeps its own attributes. An alias has no code.
    std::optional<OccupancyAttrs> Result = Values.lookup(N);
    if (!Result || Problem.getKernel(*N) || N->copies().empty())
      continue;

    // A function with its own attributes gets their merge with the
    // attributes of the kernels.
    auto It = Summaries.find(N->getValueInfo().getGUID());
    if (It != Summaries.end())
      Result->merge(OccupancyAttrs(It->second));

    ThinLinkFnAttrs Attrs;
    Attrs.add("amdgpu-flat-work-group-size",
              std::to_string(Result->FlatWGSizeMin) + "," +
                  std::to_string(Result->FlatWGSizeMax));
    // The merge of valid attributes is valid. If an input is not valid, the
    // function does not get the attribute.
    if (Result->hasValidWavesPerEU()) {
      std::string WavesPerEU = std::to_string(Result->WavesPerEUMin);
      if (Result->WavesPerEUMax)
        WavesPerEU += "," + std::to_string(*Result->WavesPerEUMax);
      Attrs.add("amdgpu-waves-per-eu", WavesPerEU);
    } else {
      LLVM_DEBUG(dbgs() << "AMDGPU thin link: the waves per EU of "
                        << N->getValueInfo() << " are not valid\n");
    }
    Attrs.add("amdgpu-max-num-workgroups",
              std::to_string(Result->MaxNumWGX) + "," +
                  std::to_string(Result->MaxNumWGY) + "," +
                  std::to_string(Result->MaxNumWGZ));
    LLVM_DEBUG(dbgs() << "AMDGPU thin link: set the occupancy attributes of "
                      << N->getValueInfo() << "\n");
    ThinLinkFnAttrs::addToSummaries(Index, N->getValueInfo(), Attrs);
  }

  // The pass adds only results, which no analysis uses.
  return PreservedAnalyses::all();
}

void llvm::registerAMDGPUThinLinkPasses(PassBuilder &PB) {
  if (PassInstrumentationCallbacks *PIC = PB.getPassInstrumentationCallbacks())
    PIC->addClassToPassName(AMDGPUThinLinkOccupancyPass::name(),
                            "amdgpu-thinlink-occupancy");
  PB.registerSummaryPipelineEPCallback([](SummaryPassManager &SPM) {
    SPM.addPass(AMDGPUThinLinkOccupancyPass());
  });
  PB.registerPipelineParsingCallback(
      [](StringRef Name, SummaryPassManager &SPM,
         ArrayRef<PassBuilder::PipelineElement> InnerPipeline) {
        if (Name != "amdgpu-thinlink-occupancy" || !InnerPipeline.empty())
          return false;
        SPM.addPass(AMDGPUThinLinkOccupancyPass());
        return true;
      });
}
