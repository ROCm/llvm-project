//===- SummaryPassManager.h - Summary index pass manager --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A pass manager and an analysis manager for passes over a module summary
// index. The ThinLTO thin link runs such passes on the combined index.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_SUMMARYPASSMANAGER_H
#define LLVM_IR_SUMMARYPASSMANAGER_H

#include "llvm/ADT/DenseSet.h"
#include "llvm/IR/ModuleSummaryIndex.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/Compiler.h"
#include <functional>

namespace llvm {

extern template class LLVM_TEMPLATE_ABI AllAnalysesOn<ModuleSummaryIndex>;
extern template class LLVM_TEMPLATE_ABI AnalysisManager<ModuleSummaryIndex>;

template <>
LLVM_ABI void
printIRUnitNameForStackTrace<ModuleSummaryIndex>(raw_ostream &OS,
                                                 const ModuleSummaryIndex &IR);

extern template class LLVM_TEMPLATE_ABI PassManager<ModuleSummaryIndex>;

/// Convenience typedef for an analysis manager over a summary index.
using SummaryAnalysisManager = AnalysisManager<ModuleSummaryIndex>;

/// Convenience typedef for a pass manager over a summary index.
///
/// A summary index has no LLVMContext, so a pass over it must not call
/// RequiredPassInfoMixin::shouldSkipOptimizationForOptBisect().
using SummaryPassManager = PassManager<ModuleSummaryIndex>;

/// Facts about the link that thin-link passes need, but that the index does not
/// hold: the symbols that are visible outside the summaries (for example to
/// native objects), which copy of a symbol prevails, and whether the link
/// assumes a closed world. The link sets them when it registers the analysis.
/// They do not change while passes run.
class ThinLinkInfoAnalysis : public AnalysisInfoMixin<ThinLinkInfoAnalysis> {
public:
  using IsPrevailingFn =
      std::function<bool(GlobalValue::GUID, const GlobalValueSummary *)>;

  struct Result {
    const DenseSet<GlobalValue::GUID> &GUIDPreservedSymbols;
    IsPrevailingFn IsPrevailing;
    /// True if the link assumes a closed world:
    /// - Code outside the summaries calls a function in the summaries only by
    ///   the symbol of a preserved function, or as a constructor, destructor,
    ///   or used value of a module. It never calls through a function pointer.
    /// - An indirect call in the summaries reaches only functions in the
    ///   summaries.
    /// - Inline assembly calls no function.
    /// In an open world, code outside the summaries can also call each
    /// function whose address escapes to it, and an indirect call can reach
    /// code outside the summaries.
    bool ClosedWorld;

    bool invalidate(ModuleSummaryIndex &, const PreservedAnalyses &,
                    SummaryAnalysisManager::Invalidator &) {
      return false;
    }
  };

  ThinLinkInfoAnalysis(const DenseSet<GlobalValue::GUID> &GUIDPreservedSymbols
                           LLVM_LIFETIME_BOUND,
                       IsPrevailingFn IsPrevailing, bool ClosedWorld = false)
      : GUIDPreservedSymbols(GUIDPreservedSymbols),
        IsPrevailing(std::move(IsPrevailing)), ClosedWorld(ClosedWorld) {}

  Result run(ModuleSummaryIndex &, SummaryAnalysisManager &) {
    return {GUIDPreservedSymbols, IsPrevailing, ClosedWorld};
  }

private:
  friend AnalysisInfoMixin<ThinLinkInfoAnalysis>;
  LLVM_ABI static AnalysisKey Key;

  const DenseSet<GlobalValue::GUID> &GUIDPreservedSymbols;
  IsPrevailingFn IsPrevailing;
  bool ClosedWorld;
};

} // end namespace llvm

#endif // LLVM_IR_SUMMARYPASSMANAGER_H
