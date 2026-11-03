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
/// The standard pass instrumentation (StandardInstrumentations) and the
/// opt-bisect check of RequiredPassInfoMixin do not support this unit yet.
/// Register PassInstrumentationAnalysis without callbacks, or with callbacks
/// that handle a ModuleSummaryIndex.
using SummaryPassManager = PassManager<ModuleSummaryIndex>;

/// Facts about the link that thin-link passes need, but that the index does not
/// hold: the symbols that are visible outside the summaries (for example to
/// native objects), and which copy of a symbol prevails. The link sets them
/// when it registers the analysis. They do not change while passes run.
class ThinLinkInfoAnalysis : public AnalysisInfoMixin<ThinLinkInfoAnalysis> {
public:
  using IsPrevailingFn =
      std::function<bool(GlobalValue::GUID, const GlobalValueSummary *)>;

  struct Result {
    const DenseSet<GlobalValue::GUID> &GUIDPreservedSymbols;
    IsPrevailingFn IsPrevailing;

    bool invalidate(ModuleSummaryIndex &, const PreservedAnalyses &,
                    SummaryAnalysisManager::Invalidator &) {
      return false;
    }
  };

  ThinLinkInfoAnalysis(const DenseSet<GlobalValue::GUID> &GUIDPreservedSymbols
                           LLVM_LIFETIME_BOUND,
                       IsPrevailingFn IsPrevailing)
      : GUIDPreservedSymbols(GUIDPreservedSymbols),
        IsPrevailing(std::move(IsPrevailing)) {}

  Result run(ModuleSummaryIndex &, SummaryAnalysisManager &) {
    return {GUIDPreservedSymbols, IsPrevailing};
  }

private:
  friend AnalysisInfoMixin<ThinLinkInfoAnalysis>;
  LLVM_ABI static AnalysisKey Key;

  const DenseSet<GlobalValue::GUID> &GUIDPreservedSymbols;
  IsPrevailingFn IsPrevailing;
};

} // end namespace llvm

#endif // LLVM_IR_SUMMARYPASSMANAGER_H
