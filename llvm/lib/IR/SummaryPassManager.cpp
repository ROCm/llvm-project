//===- SummaryPassManager.cpp - Pass manager for the summary index --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/SummaryPassManager.h"
#include "llvm/IR/PassManagerImpl.h"
#include "llvm/Support/Compiler.h"

using namespace llvm;

namespace llvm {
template class LLVM_EXPORT_TEMPLATE AllAnalysesOn<ModuleSummaryIndex>;
template class LLVM_EXPORT_TEMPLATE AnalysisManager<ModuleSummaryIndex>;
template class LLVM_EXPORT_TEMPLATE PassManager<ModuleSummaryIndex>;
} // namespace llvm

template <>
void llvm::printIRUnitNameForStackTrace<ModuleSummaryIndex>(
    raw_ostream &OS, const ModuleSummaryIndex &IR) {
  size_t NumModules = IR.modulePaths().size();
  OS << "summary index of " << NumModules
     << (NumModules == 1 ? " module" : " modules");
}

AnalysisKey ThinLinkInfoAnalysis::Key;
