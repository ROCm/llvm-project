//===- AMDGPUThinLinkPasses.h - AMDGPU passes of the thin link --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The AMDGPU passes of the ThinLTO thin link. They run on the combined summary
// index, and they store their results as function attributes
// (ThinLinkFnAttrs), which each ThinLTO backend sets on its functions. The
// passes use the facts that the compile step stores in the summaries.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUTHINLINKPASSES_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUTHINLINKPASSES_H

#include "llvm/IR/SummaryPassManager.h"

namespace llvm {

class Function;
class PassBuilder;

/// Adds the facts about \p F that the AMDGPU thin-link passes use to the
/// summary extensions of \p Index. \p Summary is the summary of \p F in
/// \p Index.
void addAMDGPUThinLinkFacts(const Function &F,
                            const GlobalValueSummary &Summary,
                            ModuleSummaryIndex &Index);

/// Propagates the occupancy attributes of the kernels
/// (amdgpu-flat-work-group-size, amdgpu-waves-per-eu, and
/// amdgpu-max-num-workgroups) to the device functions that they can reach. A
/// function gets the merge of the attributes of these kernels and of its own
/// attributes.
class AMDGPUThinLinkOccupancyPass
    : public RequiredPassInfoMixin<AMDGPUThinLinkOccupancyPass> {
public:
  PreservedAnalyses run(ModuleSummaryIndex &Index, SummaryAnalysisManager &AM);
};

/// Gives the attribute uniform-work-group-size to a device function if all
/// kernels that can reach it have the attribute.
class AMDGPUThinLinkUniformWorkGroupSizePass
    : public RequiredPassInfoMixin<AMDGPUThinLinkUniformWorkGroupSizePass> {
public:
  PreservedAnalyses run(ModuleSummaryIndex &Index, SummaryAnalysisManager &AM);
};

/// Propagates the attribute amdgpu-cluster-dims of the entry functions to the
/// device functions that they can reach.
class AMDGPUThinLinkClusterDimsPass
    : public RequiredPassInfoMixin<AMDGPUThinLinkClusterDimsPass> {
public:
  PreservedAnalyses run(ModuleSummaryIndex &Index, SummaryAnalysisManager &AM);
};

/// Finds the implicit arguments that a function and the functions that it can
/// call do not use, and gives the function the amdgpu-no-* attributes of these
/// arguments. A backend cannot find them for a function that calls a function
/// in another module. For now, no module gets the facts that this pass reads
/// (see addImplicitArgFacts() in AMDGPUThinLinkPasses.cpp), so it does nothing.
class AMDGPUThinLinkImplicitArgsPass
    : public RequiredPassInfoMixin<AMDGPUThinLinkImplicitArgsPass> {
public:
  PreservedAnalyses run(ModuleSummaryIndex &Index, SummaryAnalysisManager &AM);
};

/// Adds the AMDGPU passes to the ThinLTO thin link that \p PB builds.
void registerAMDGPUThinLinkPasses(PassBuilder &PB);

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUTHINLINKPASSES_H
