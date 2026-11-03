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
// (ThinLinkFnAttrs), which each ThinLTO backend sets on its functions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUTHINLINKPASSES_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUTHINLINKPASSES_H

#include "llvm/IR/SummaryPassManager.h"

namespace llvm {

class PassBuilder;

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

/// Adds the AMDGPU passes to the ThinLTO thin link that \p PB builds.
void registerAMDGPUThinLinkPasses(PassBuilder &PB);

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUTHINLINKPASSES_H
