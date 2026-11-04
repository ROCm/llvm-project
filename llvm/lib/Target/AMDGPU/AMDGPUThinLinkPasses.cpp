//===- AMDGPUThinLinkPasses.cpp - AMDGPU passes of the thin link ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUThinLinkPasses.h"
#include "AMDGPU.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/STLExtras.h"
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

#define DEBUG_TYPE "amdgpu-thinlink"

/// The summary extension with the launch facts: an entry for each entry
/// function (a kernel or a shader), and for each function with a launch
/// attribute. An entry holds the calling convention of the function, then the
/// name and the value of each launch attribute of the function. Only the thin
/// link uses the facts, so the extension is not required.
static constexpr StringLiteral LaunchFactsName = "amdgpu-launch-attrs";
static constexpr uint64_t LaunchFactsVersion = 1;

/// The launch attributes: the attributes about the launches of the kernels
/// that can reach a function.
static constexpr StringLiteral FlatWorkGroupSizeName =
    "amdgpu-flat-work-group-size";
static constexpr StringLiteral MaxNumWorkgroupsName =
    "amdgpu-max-num-workgroups";
static constexpr StringLiteral WavesPerEUName = "amdgpu-waves-per-eu";
static constexpr StringLiteral UniformWorkGroupSizeName =
    "uniform-work-group-size";
static constexpr StringLiteral ClusterDimsName = "amdgpu-cluster-dims";

/// The summary extension with the implicit arguments that the code of each
/// function uses, without the functions that it calls. An entry holds their
/// names: the names of their attributes without the "amdgpu-no-" prefix. A
/// function without an entry can use all implicit arguments. Bump the version
/// when getLocalImplicitArgUses() finds the uses of a new implicit argument.
static constexpr StringLiteral ImplicitArgFactsName = "amdgpu-implicit-args";
static constexpr uint64_t ImplicitArgFactsVersion = 1;

/// The attributes for the implicit arguments that a function does not use, in
/// the order of the bits of AMDGPU::getLocalImplicitArgUses().
#define AMDGPU_ATTRIBUTE(Name, Str) Str,
static constexpr StringLiteral ImplicitArgAttrs[] = {
#include "AMDGPUAttributes.def"
};
static constexpr uint32_t AllImplicitArgs =
    (1u << std::size(ImplicitArgAttrs)) - 1;

/// Returns the name of the implicit argument of the attribute \p Attr.
static StringRef getImplicitArgName(StringRef Attr) {
  [[maybe_unused]] bool HasPrefix = Attr.consume_front("amdgpu-no-");
  assert(HasPrefix && "unexpected implicit argument attribute");
  return Attr;
}

static void addLaunchFacts(const Function &F, const GlobalValueSummary &Summary,
                           ModuleSummaryIndex &Index) {
  SummaryExtension::Entry E;
  for (StringRef Name :
       {FlatWorkGroupSizeName, MaxNumWorkgroupsName, WavesPerEUName,
        UniformWorkGroupSizeName, ClusterDimsName}) {
    Attribute A = F.getFnAttribute(Name);
    if (!A.isStringAttribute())
      continue;
    E.Strings.push_back(Name.str());
    E.Strings.push_back(A.getValueAsString().str());
  }
  // An entry function has facts also without launch attributes.
  if (E.Strings.empty() && !AMDGPU::isEntryFunctionCC(F.getCallingConv()))
    return;
  E.Ints.push_back(F.getCallingConv());
  Index
      .getOrInsertExtension(LaunchFactsName, LaunchFactsVersion,
                            /*Required=*/false)
      .setEntry(&Summary, std::move(E));
}

static void addImplicitArgFacts(const Function &F,
                                const GlobalValueSummary &Summary,
                                ModuleSummaryIndex &Index) {
  // Like the attributor, do not change an optnone function. Without facts, it
  // and its callers can use all implicit arguments.
  if (F.hasOptNone())
    return;
  std::optional<uint32_t> Uses = AMDGPU::getLocalImplicitArgUses(F);
  if (!Uses)
    return;
  SummaryExtension::Entry E;
  for (auto [I, Attr] : enumerate(ImplicitArgAttrs))
    if (*Uses & (1u << I))
      E.Strings.push_back(getImplicitArgName(Attr).str());
  Index
      .getOrInsertExtension(ImplicitArgFactsName, ImplicitArgFactsVersion,
                            /*Required=*/false)
      .setEntry(&Summary, std::move(E));
}

void llvm::addAMDGPUThinLinkFacts(const Function &F,
                                  const GlobalValueSummary &Summary,
                                  ModuleSummaryIndex &Index) {
  addLaunchFacts(F, Summary, Index);
  addImplicitArgFacts(F, Summary, Index);
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
    if (Name == FlatWorkGroupSizeName) {
      // Like the backend, ignore sizes that are not valid.
      auto [MinS, MaxS] = Value.split(',');
      uint32_t Min, Max;
      if (!MinS.trim().getAsInteger(0, Min) &&
          !MaxS.trim().getAsInteger(0, Max) && Min >= 1 && Min <= Max &&
          Max <= 1024) {
        FlatWGSizeMin = Min;
        FlatWGSizeMax = Max;
      }
    } else if (Name == WavesPerEUName) {
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
    } else if (Name == MaxNumWorkgroupsName) {
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

/// The launch facts of a function: its calling convention and its own launch
/// attributes. An attribute that the function does not have gets its default
/// value.
struct LaunchFacts {
  CallingConv::ID CC = CallingConv::C;
  OccupancyAttrs Occupancy;
  /// True if the function has an occupancy attribute.
  bool HasOccupancy = false;
  bool UniformWorkGroupSize = false;
  AMDGPU::ClusterDimsAttr ClusterDims;

  bool isKernel() const { return CC == CallingConv::AMDGPU_KERNEL; }
  bool isEntryFunction() const { return AMDGPU::isEntryFunctionCC(CC); }
};

/// Reads the launch facts in \p E. Returns nullopt if \p E is not valid.
std::optional<LaunchFacts> readLaunchFacts(const SummaryExtension::Entry &E) {
  if (E.Ints.size() != 1 || E.Ints[0] > CallingConv::MaxID ||
      E.Strings.size() % 2 != 0)
    return std::nullopt;
  LaunchFacts Facts;
  Facts.CC = static_cast<CallingConv::ID>(E.Ints[0]);
  for (size_t I = 0; I < E.Strings.size(); I += 2) {
    StringRef Name = E.Strings[I], Value = E.Strings[I + 1];
    if (Name == UniformWorkGroupSizeName) {
      Facts.UniformWorkGroupSize = true;
    } else if (Name == ClusterDimsName) {
      Facts.ClusterDims = AMDGPU::ClusterDimsAttr::get(Value);
    } else {
      Facts.Occupancy.set(Name, Value);
      Facts.HasOccupancy = true;
    }
  }
  return Facts;
}

/// The launch facts of the functions, from the copy of each function that
/// prevails.
using LaunchFactsMap = DenseMap<const SummaryCallGraph::Node *, LaunchFacts>;

/// Returns the launch facts of the functions in \p CG, or nullopt if \p Index
/// has no launch facts.
std::optional<LaunchFactsMap> getLaunchFacts(const ModuleSummaryIndex &Index,
                                             const SummaryCallGraph &CG) {
  const SummaryExtension *Ext =
      Index.getExtension(LaunchFactsName, LaunchFactsVersion);
  if (!Ext)
    return std::nullopt;
  LaunchFactsMap Facts;
  for (const SummaryCallGraph::Node *N : CG.nodes()) {
    const FunctionSummary *S = N->getSummary();
    const SummaryExtension::Entry *E = S ? Ext->getEntry(S) : nullptr;
    if (!E)
      continue;
    if (std::optional<LaunchFacts> F = readLaunchFacts(*E))
      Facts.try_emplace(N, *F);
    else
      LLVM_DEBUG(dbgs() << "AMDGPU thin link: the launch facts of "
                        << N->getValueInfo() << " are not valid\n");
  }
  return Facts;
}

/// Returns the facts of \p N if it is a kernel, or null.
const LaunchFacts *getKernelFacts(const LaunchFactsMap &Facts,
                                  const SummaryCallGraph::Node &N) {
  auto It = Facts.find(&N);
  if (It == Facts.end() || !It->second.isKernel())
    return nullptr;
  return &It->second;
}

/// Returns the facts of \p N if it is an entry function, or null.
const LaunchFacts *getEntryFacts(const LaunchFactsMap &Facts,
                                 const SummaryCallGraph::Node &N) {
  auto It = Facts.find(&N);
  if (It == Facts.end() || !It->second.isEntryFunction())
    return nullptr;
  return &It->second;
}

/// A top-down problem: the value of a function is the merge of the occupancy
/// attributes of the entry functions that can reach it, or none if no entry
/// function can.
struct KernelOccupancyProblem {
  using ValueT = std::optional<OccupancyAttrs>;

  const LaunchFactsMap &Facts;

  ValueT getInitialValue(const SummaryCallGraph::Node &) {
    return std::nullopt;
  }

  /// Like in the attributor, the code outside the summaries can call a function
  /// that it can see from a kernel with any launch: the default attributes.
  /// They absorb each value except the minimum number of waves per EU, for
  /// which the merge keeps the largest value. That attribute only limits the
  /// registers of the function, so a smaller minimum still gives correct code.
  ValueT getExternalValue() { return OccupancyAttrs(); }

  void join(ValueT &Into, const ValueT &V) {
    if (!V)
      return;
    if (Into)
      Into->merge(*V);
    else
      Into = V;
  }

  /// Code cannot call an entry function, so the value of an entry function is
  /// its own attributes.
  ValueT transfer(const SummaryCallGraph::Node &N, const ValueT &In) {
    if (const LaunchFacts *Entry = getEntryFacts(Facts, N))
      return Entry->Occupancy;
    return In;
  }
};

/// A top-down problem: the value of a function is true if all kernels that can
/// reach it have the attribute uniform-work-group-size, false if some kernel
/// does not have it, and none if no kernel can reach the function.
struct UniformWorkGroupSizeProblem {
  using ValueT = std::optional<bool>;

  const LaunchFactsMap &Facts;

  ValueT getInitialValue(const SummaryCallGraph::Node &) {
    return std::nullopt;
  }

  /// Like in the attributor, the code outside the summaries can call a function
  /// that it can see from a kernel without the attribute.
  ValueT getExternalValue() { return false; }

  void join(ValueT &Into, const ValueT &V) {
    if (V)
      Into = Into ? *Into && *V : *V;
  }

  ValueT transfer(const SummaryCallGraph::Node &N, const ValueT &In) {
    if (const LaunchFacts *Kernel = getKernelFacts(Facts, N))
      return Kernel->UniformWorkGroupSize;
    return In;
  }
};

/// Joins the cluster dimensions \p A and \p B like the attributor: the result
/// is unknown if only one of them uses clusters, and variable if both use
/// clusters with different dimensions.
AMDGPU::ClusterDimsAttr joinClusterDims(AMDGPU::ClusterDimsAttr A,
                                        const AMDGPU::ClusterDimsAttr &B) {
  if (A == B)
    return A;
  if (A.isUnknown() || B.isUnknown() || A.isNoCluster() || B.isNoCluster())
    return AMDGPU::ClusterDimsAttr();
  A.setVariableDims();
  return A;
}

/// A top-down problem: the value of a function is the join of the cluster
/// dimensions of the entry functions that can reach it and of the functions
/// on the way, or none if they have no cluster dimensions. An entry function
/// without the attribute amdgpu-cluster-dims makes the value unknown.
struct ClusterDimsProblem {
  using ValueT = std::optional<AMDGPU::ClusterDimsAttr>;

  const LaunchFactsMap &Facts;

  ValueT getInitialValue(const SummaryCallGraph::Node &) {
    return std::nullopt;
  }

  /// Like in the attributor, the cluster dimensions of the callers outside the
  /// summaries are unknown.
  ValueT getExternalValue() { return AMDGPU::ClusterDimsAttr(); }

  void join(ValueT &Into, const ValueT &V) {
    if (!V)
      return;
    Into = Into ? joinClusterDims(*Into, *V) : *V;
  }

  /// Like in the attributor, an entry function gets its own attribute, and
  /// another function joins its own attribute with the value of its callers.
  ValueT transfer(const SummaryCallGraph::Node &N, const ValueT &In) {
    auto It = Facts.find(&N);
    if (It == Facts.end())
      return In;
    const LaunchFacts &F = It->second;
    if (AMDGPU::isEntryFunctionCC(F.CC))
      return F.ClusterDims;
    ValueT Out = In;
    if (!F.ClusterDims.isUnknown())
      join(Out, F.ClusterDims);
    return Out;
  }
};

/// Returns the implicit arguments that a copy of a function with the facts
/// \p E uses. A copy without valid facts can use all of them.
uint32_t readImplicitArgUses(const SummaryExtension::Entry *E) {
  if (!E || !E->Ints.empty())
    return AllImplicitArgs;
  uint32_t Uses = 0;
  for (StringRef Name : E->Strings) {
    const auto *It = find_if(ImplicitArgAttrs, [&](StringRef Attr) {
      return getImplicitArgName(Attr) == Name;
    });
    if (It == std::end(ImplicitArgAttrs))
      return AllImplicitArgs;
    Uses |= 1u << (It - std::begin(ImplicitArgAttrs));
  }
  return Uses;
}

/// A bottom-up problem: the value of a function is the set of the implicit
/// arguments that the function and the functions that it can call use.
struct ImplicitArgUsesProblem {
  using ValueT = uint32_t;

  /// The implicit arguments that the code of each function uses.
  const DenseMap<const SummaryCallGraph::Node *, uint32_t> &LocalUses;

  ValueT getInitialValue(const SummaryCallGraph::Node &) { return 0; }

  /// The code outside the summaries can use all implicit arguments.
  ValueT getExternalValue() { return AllImplicitArgs; }

  void join(ValueT &Into, const ValueT &V) { Into |= V; }

  ValueT transfer(const SummaryCallGraph::Node &N, const ValueT &In) {
    return In | LocalUses.lookup(&N);
  }
};

} // end anonymous namespace

PreservedAnalyses AMDGPUThinLinkOccupancyPass::run(ModuleSummaryIndex &Index,
                                                   SummaryAnalysisManager &AM) {
  // Build the call graph only for a link with launch facts.
  if (!Index.getExtension(LaunchFactsName, LaunchFactsVersion))
    return PreservedAnalyses::all();
  const SummaryCallGraph &CG =
      AM.getResult<SummaryCallGraphAnalysis>(Index).getGraph(
          /*IncludeIndirectCalls=*/true);
  LaunchFactsMap Facts = *getLaunchFacts(Index, CG);

  KernelOccupancyProblem Problem{Facts};
  auto Values =
      solveSummaryDataflow(CG, Problem, SummaryDataflowDirection::TopDown);

  for (const SummaryCallGraph::Node *N : CG.nodes()) {
    // An entry function keeps its own attributes. An alias has no code.
    std::optional<OccupancyAttrs> Result = Values.lookup(N);
    if (!Result || getEntryFacts(Facts, *N) || N->copies().empty())
      continue;

    // A function with its own occupancy attributes gets their merge with the
    // attributes of the entry functions.
    if (auto It = Facts.find(N); It != Facts.end() && It->second.HasOccupancy)
      Result->merge(It->second.Occupancy);

    ThinLinkFnAttrs Attrs;
    Attrs.add(FlatWorkGroupSizeName, std::to_string(Result->FlatWGSizeMin) +
                                         "," +
                                         std::to_string(Result->FlatWGSizeMax));
    // The merge of valid attributes is valid. If an input is not valid, the
    // function does not get the attribute.
    if (Result->hasValidWavesPerEU()) {
      std::string WavesPerEU = std::to_string(Result->WavesPerEUMin);
      if (Result->WavesPerEUMax)
        WavesPerEU += "," + std::to_string(*Result->WavesPerEUMax);
      Attrs.add(WavesPerEUName, WavesPerEU);
    } else {
      LLVM_DEBUG(dbgs() << "AMDGPU thin link: the waves per EU of "
                        << N->getValueInfo() << " are not valid\n");
    }
    Attrs.add(MaxNumWorkgroupsName, std::to_string(Result->MaxNumWGX) + "," +
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

PreservedAnalyses
AMDGPUThinLinkUniformWorkGroupSizePass::run(ModuleSummaryIndex &Index,
                                            SummaryAnalysisManager &AM) {
  if (!Index.getExtension(LaunchFactsName, LaunchFactsVersion))
    return PreservedAnalyses::all();
  const SummaryCallGraph &CG =
      AM.getResult<SummaryCallGraphAnalysis>(Index).getGraph(
          /*IncludeIndirectCalls=*/true);
  LaunchFactsMap Facts = *getLaunchFacts(Index, CG);

  UniformWorkGroupSizeProblem Problem{Facts};
  auto Values =
      solveSummaryDataflow(CG, Problem, SummaryDataflowDirection::TopDown);

  for (const SummaryCallGraph::Node *N : CG.nodes()) {
    // A kernel keeps its own attribute. An alias has no code.
    if (Values.lookup(N) != true || getKernelFacts(Facts, *N) ||
        N->copies().empty())
      continue;
    ThinLinkFnAttrs Attrs;
    Attrs.add(UniformWorkGroupSizeName);
    ThinLinkFnAttrs::addToSummaries(Index, N->getValueInfo(), Attrs);
  }
  return PreservedAnalyses::all();
}

PreservedAnalyses
AMDGPUThinLinkClusterDimsPass::run(ModuleSummaryIndex &Index,
                                   SummaryAnalysisManager &AM) {
  if (!Index.getExtension(LaunchFactsName, LaunchFactsVersion))
    return PreservedAnalyses::all();
  const SummaryCallGraph &CG =
      AM.getResult<SummaryCallGraphAnalysis>(Index).getGraph(
          /*IncludeIndirectCalls=*/true);
  LaunchFactsMap Facts = *getLaunchFacts(Index, CG);

  ClusterDimsProblem Problem{Facts};
  auto Values =
      solveSummaryDataflow(CG, Problem, SummaryDataflowDirection::TopDown);

  for (const SummaryCallGraph::Node *N : CG.nodes()) {
    // An alias has no code.
    if (N->copies().empty())
      continue;
    std::optional<AMDGPU::ClusterDimsAttr> Result = Values.lookup(N);
    if (!Result || Result->isUnknown())
      continue;
    // An entry function keeps its own attribute. Do not set the attribute that
    // a function has.
    auto It = Facts.find(N);
    if (It != Facts.end() && (AMDGPU::isEntryFunctionCC(It->second.CC) ||
                              It->second.ClusterDims == *Result))
      continue;
    ThinLinkFnAttrs Attrs;
    Attrs.add(ClusterDimsName, Result->to_string());
    ThinLinkFnAttrs::addToSummaries(Index, N->getValueInfo(), Attrs);
  }
  return PreservedAnalyses::all();
}

PreservedAnalyses
AMDGPUThinLinkImplicitArgsPass::run(ModuleSummaryIndex &Index,
                                    SummaryAnalysisManager &AM) {
  const SummaryExtension *FactsExt =
      Index.getExtension(ImplicitArgFactsName, ImplicitArgFactsVersion);
  // Build the call graph only for a link with facts about implicit arguments.
  if (!FactsExt)
    return PreservedAnalyses::all();

  const SummaryCallGraph &CG =
      AM.getResult<SummaryCallGraphAnalysis>(Index).getGraph(
          /*IncludeIndirectCalls=*/true);
  // Each copy of a function can run, so the code of a function uses the
  // implicit arguments that any of its copies uses.
  DenseMap<const SummaryCallGraph::Node *, uint32_t> LocalUses;
  for (const SummaryCallGraph::Node *N : CG.nodes()) {
    uint32_t Uses = 0;
    for (const FunctionSummary *Copy : N->copies())
      Uses |= readImplicitArgUses(FactsExt->getEntry(Copy));
    if (Uses)
      LocalUses[N] = Uses;
  }

  ImplicitArgUsesProblem Problem{LocalUses};
  auto Values =
      solveSummaryDataflow(CG, Problem, SummaryDataflowDirection::BottomUp);

  for (const SummaryCallGraph::Node *N : CG.nodes()) {
    // An alias has no code.
    auto It = Values.find(N);
    if (N->copies().empty() || It == Values.end())
      continue;
    uint32_t Unused = AllImplicitArgs & ~It->second;
    if (!Unused)
      continue;
    ThinLinkFnAttrs Attrs;
    for (auto [I, Attr] : enumerate(ImplicitArgAttrs))
      if (Unused & (1u << I))
        Attrs.add(Attr);
    LLVM_DEBUG(dbgs() << "AMDGPU thin link: set the implicit argument "
                         "attributes of "
                      << N->getValueInfo() << "\n");
    ThinLinkFnAttrs::addToSummaries(Index, N->getValueInfo(), Attrs);
  }

  // The pass adds only results, which no analysis uses.
  return PreservedAnalyses::all();
}

void llvm::registerAMDGPUThinLinkPasses(PassBuilder &PB) {
  if (PassInstrumentationCallbacks *PIC =
          PB.getPassInstrumentationCallbacks()) {
    PIC->addClassToPassName(AMDGPUThinLinkOccupancyPass::name(),
                            "amdgpu-thinlink-occupancy");
    PIC->addClassToPassName(AMDGPUThinLinkUniformWorkGroupSizePass::name(),
                            "amdgpu-thinlink-uniform-work-group-size");
    PIC->addClassToPassName(AMDGPUThinLinkClusterDimsPass::name(),
                            "amdgpu-thinlink-cluster-dims");
    PIC->addClassToPassName(AMDGPUThinLinkImplicitArgsPass::name(),
                            "amdgpu-thinlink-implicit-args");
  }
  PB.registerSummaryPipelineEPCallback([](SummaryPassManager &SPM) {
    SPM.addPass(AMDGPUThinLinkOccupancyPass());
    SPM.addPass(AMDGPUThinLinkUniformWorkGroupSizePass());
    SPM.addPass(AMDGPUThinLinkClusterDimsPass());
    SPM.addPass(AMDGPUThinLinkImplicitArgsPass());
  });
  PB.registerPipelineParsingCallback(
      [](StringRef Name, SummaryPassManager &SPM,
         ArrayRef<PassBuilder::PipelineElement> InnerPipeline) {
        if (!InnerPipeline.empty())
          return false;
        if (Name == "amdgpu-thinlink-occupancy")
          SPM.addPass(AMDGPUThinLinkOccupancyPass());
        else if (Name == "amdgpu-thinlink-uniform-work-group-size")
          SPM.addPass(AMDGPUThinLinkUniformWorkGroupSizePass());
        else if (Name == "amdgpu-thinlink-cluster-dims")
          SPM.addPass(AMDGPUThinLinkClusterDimsPass());
        else if (Name == "amdgpu-thinlink-implicit-args")
          SPM.addPass(AMDGPUThinLinkImplicitArgsPass());
        else
          return false;
        return true;
      });
}
