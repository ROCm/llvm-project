//===- AMDGPUUnifyDivergentExitNodes.h ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a variant of the UnifyFunctionExitNodes pass. Rather than ensuring
// there is at most one ret and one unreachable instruction, it ensures there is
// at most one divergent exiting block.
//
// StructurizeCFG can't deal with multi-exit regions formed by branches to
// multiple return nodes. It is not desirable to structurize regions with
// uniform branches, so unifying those to the same return block as divergent
// branches inhibits use of scalar branching. It still can't deal with the case
// where one branch goes to return, and one unreachable. Replace unreachable in
// this case with a return.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUUNIFYDIVERGENTEXITNODES_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUUNIFYDIVERGENTEXITNODES_H

#include "AMDGPU.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/UniformityAnalysis.h"

namespace llvm {
class BasicBlock;
class DomTreeUpdater;
class TargetTransformInfo;

class AMDGPUUnifyDivergentExitNodesPass
    : public RequiredPassInfoMixin<AMDGPUUnifyDivergentExitNodesPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

/// \returns true if \p BB is reachable through only uniform branches.
bool isUniformlyReached(const UniformityInfo &UA, BasicBlock &BB);

/// Merge the blocks in \p ReturningBlocks, which end with a return, into a new
/// block named \p Name that holds the only return. \p DTU is updated if it is
/// not null. \returns the new return block.
BasicBlock *unifyReturnBlockSet(Function &F, DomTreeUpdater *DTU,
                                ArrayRef<BasicBlock *> ReturningBlocks,
                                StringRef Name, const TargetTransformInfo &TTI);

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUUNIFYDIVERGENTEXITNODES_H
