//===- AMDGPUThinLinkPasses.cpp - AMDGPU passes of the thin link ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUThinLinkPasses.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/SummaryCallGraph.h"
#include "llvm/Analysis/SummaryDataflow.h"
#include "llvm/IR/Function.h"
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

/// The summary extension with the occupancy facts: an entry for each kernel,
/// and for each function with an occupancy attribute. An entry holds the
/// calling convention of the function, then the name and the value of each
/// occupancy attribute of the function. Only the thin link uses the facts, so
/// the extension is not required.
static constexpr StringLiteral OccupancyFactsName = "amdgpu-occupancy";
static constexpr uint64_t OccupancyFactsVersion = 1;

static constexpr StringLiteral FlatWorkGroupSizeAttr =
    "amdgpu-flat-work-group-size";
static constexpr StringLiteral MaxNumWorkgroupsAttr =
    "amdgpu-max-num-workgroups";
static constexpr StringLiteral WavesPerEUAttr = "amdgpu-waves-per-eu";

void llvm::addAMDGPUThinLinkFacts(const Function &F,
                                  const GlobalValueSummary &Summary,
                                  ModuleSummaryIndex &Index) {
  SummaryExtension::Entry E;
  for (StringRef Name :
       {FlatWorkGroupSizeAttr, MaxNumWorkgroupsAttr, WavesPerEUAttr}) {
    Attribute A = F.getFnAttribute(Name);
    if (!A.isStringAttribute())
      continue;
    E.Strings.push_back(Name.str());
    E.Strings.push_back(A.getValueAsString().str());
  }
  // A kernel has facts also without occupancy attributes.
  if (E.Strings.empty() && F.getCallingConv() != CallingConv::AMDGPU_KERNEL)
    return;
  E.Ints.push_back(F.getCallingConv());
  Index
      .getOrInsertExtension(OccupancyFactsName, OccupancyFactsVersion,
                            /*Required=*/false)
      .setEntry(&Summary, std::move(E));
}

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

  /// Sets the attribute \p Name to \p Value. If \p Value is not valid, the
  /// attribute keeps its default value.
  void set(StringRef Name, StringRef Value) {
    if (Name == FlatWorkGroupSizeAttr) {
      auto [MinS, MaxS] = Value.split(',');
      uint32_t Min, Max;
      if (!MinS.trim().getAsInteger(0, Min) &&
          !MaxS.trim().getAsInteger(0, Max)) {
        FlatWGSizeMin = Min;
        FlatWGSizeMax = Max;
      }
    } else if (Name == WavesPerEUAttr) {
      auto [MinS, MaxS] = Value.split(',');
      MaxS = MaxS.trim();
      uint32_t Min, Max;
      if (MinS.trim().getAsInteger(0, Min))
        return;
      if (MaxS.empty()) {
        WavesPerEUMin = Min;
        WavesPerEUMax = std::nullopt;
      } else if (!MaxS.getAsInteger(0, Max)) {
        WavesPerEUMin = Min;
        WavesPerEUMax = Max;
      }
    } else if (Name == MaxNumWorkgroupsAttr) {
      SmallVector<StringRef, 3> Parts;
      Value.split(Parts, ',');
      uint32_t X, Y, Z;
      if (Parts.size() == 3 && !Parts[0].trim().getAsInteger(0, X) &&
          !Parts[1].trim().getAsInteger(0, Y) &&
          !Parts[2].trim().getAsInteger(0, Z)) {
        MaxNumWGX = X;
        MaxNumWGY = Y;
        MaxNumWGZ = Z;
      }
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

/// The occupancy facts of a function: its calling convention and its own
/// occupancy attributes. An attribute that the function does not have gets
/// its default value.
struct OccupancyFacts {
  CallingConv::ID CC = CallingConv::C;
  OccupancyAttrs Attrs;
};

/// Reads the occupancy facts in \p E. Returns nullopt if \p E is not valid.
std::optional<OccupancyFacts> readFacts(const SummaryExtension::Entry &E) {
  if (E.Ints.size() != 1 || E.Ints[0] > CallingConv::MaxID ||
      E.Strings.size() % 2 != 0)
    return std::nullopt;
  OccupancyFacts Facts;
  Facts.CC = static_cast<CallingConv::ID>(E.Ints[0]);
  for (size_t I = 0; I < E.Strings.size(); I += 2)
    Facts.Attrs.set(E.Strings[I], E.Strings[I + 1]);
  return Facts;
}

/// A top-down problem: the value of a function is the merge of the occupancy
/// attributes of the kernels that can reach it, or none if no kernel can.
struct KernelOccupancyProblem {
  using ValueT = std::optional<OccupancyAttrs>;

  /// The facts of the copy of each function that prevails.
  const DenseMap<const SummaryCallGraph::Node *, OccupancyFacts> &Facts;

  /// Returns the attributes of \p N if it is a kernel, or null.
  const OccupancyAttrs *getKernelAttrs(const SummaryCallGraph::Node &N) const {
    auto It = Facts.find(&N);
    if (It == Facts.end() || It->second.CC != CallingConv::AMDGPU_KERNEL)
      return nullptr;
    return &It->second.Attrs;
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
    if (const OccupancyAttrs *Kernel = getKernelAttrs(N))
      return *Kernel;
    return In;
  }
};

} // end anonymous namespace

PreservedAnalyses AMDGPUThinLinkOccupancyPass::run(ModuleSummaryIndex &Index,
                                                   SummaryAnalysisManager &AM) {
  const SummaryExtension *FactsExt =
      Index.getExtension(OccupancyFactsName, OccupancyFactsVersion);
  // Build the call graph only for a link with occupancy facts.
  if (!FactsExt)
    return PreservedAnalyses::all();

  const SummaryCallGraph &CG = AM.getResult<SummaryCallGraphAnalysis>(Index);
  DenseMap<const SummaryCallGraph::Node *, OccupancyFacts> Facts;
  for (const SummaryCallGraph::Node *N : CG.nodes()) {
    const FunctionSummary *S = N->getSummary();
    const SummaryExtension::Entry *E = S ? FactsExt->getEntry(S) : nullptr;
    if (!E)
      continue;
    if (std::optional<OccupancyFacts> F = readFacts(*E))
      Facts.try_emplace(N, *F);
    else
      LLVM_DEBUG(dbgs() << "AMDGPU thin link: the occupancy facts of "
                        << N->getValueInfo() << " are not valid\n");
  }

  KernelOccupancyProblem Problem{Facts};
  auto Values =
      solveSummaryDataflow(CG, Problem, SummaryDataflowDirection::TopDown);

  for (const SummaryCallGraph::Node *N : CG.nodes()) {
    // A kernel keeps its own attributes. An alias has no code.
    std::optional<OccupancyAttrs> Result = Values.lookup(N);
    if (!Result || Problem.getKernelAttrs(*N) || N->copies().empty())
      continue;

    // A function with its own attributes gets their merge with the
    // attributes of the kernels.
    if (auto It = Facts.find(N); It != Facts.end())
      Result->merge(It->second.Attrs);

    ThinLinkFnAttrs Attrs;
    Attrs.add(FlatWorkGroupSizeAttr, std::to_string(Result->FlatWGSizeMin) +
                                         "," +
                                         std::to_string(Result->FlatWGSizeMax));
    // The merge of valid attributes is valid. If an input is not valid, the
    // function does not get the attribute.
    if (Result->hasValidWavesPerEU()) {
      std::string WavesPerEU = std::to_string(Result->WavesPerEUMin);
      if (Result->WavesPerEUMax)
        WavesPerEU += "," + std::to_string(*Result->WavesPerEUMax);
      Attrs.add(WavesPerEUAttr, WavesPerEU);
    } else {
      LLVM_DEBUG(dbgs() << "AMDGPU thin link: the waves per EU of "
                        << N->getValueInfo() << " are not valid\n");
    }
    Attrs.add(MaxNumWorkgroupsAttr, std::to_string(Result->MaxNumWGX) + "," +
                                        std::to_string(Result->MaxNumWGY) +
                                        "," +
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
