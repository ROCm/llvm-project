//===--- AMDGPUAsyncMarkScheduling.cpp - AMDGPU Async Mark Scheduling -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Adds the minimal artificial scheduling DAG edges that preserve the
/// semantics of ASYNCMARK / WAIT_ASYNCMARK without modeling ASYNCMARK as a
/// global memory barrier.
///
/// ASYNCMARK is a meta marker placed in the stream of asynchronous
/// requests; WAIT_ASYNCMARK(N) waits for the Nth previous mark to retire.
/// Treating ASYNCMARK as a global memory object would over-constrain the
/// scheduler, because everything around it would be pinned. Instead, this
/// mutation expresses the strictly necessary ordering as artificial edges:
///
///   1. Relative order of ASYNCMARKs and WAIT_ASYNCMARKs is kept.
///
///   2. Each async load/store gets an edge to the next ASYNCMARK and an
///      edge from the previous ASYNCMARK.
///
/// This scheme exempts ASYNCMARKs from dependencies with non-async
/// load/store.
///
/// WAIT_ASYNCMARK is intentionally left as a global-memory object so that
/// ordinary loads/stores remain anchored across waits.
///
/// At post-RA scheduling the SU may represent a BUNDLE rather than a single
/// instruction (created by SIPostRABundler / SIInsertHardClauses). When that
/// happens the relevant async loads/stores and ASYNCMARKs are bundle members
/// rather than top-level SUnits, so the helpers below classify the SU based
/// on what is *contained* in it.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUAsyncMarkScheduling.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIInstrInfo.h"
#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineInstrBundle.h"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/CodeGen/ScheduleDAGInstrs.h"

using namespace llvm;

namespace {

LLVM_ENABLE_BITMASK_ENUMS_IN_NAMESPACE();

// Classification of an SU's relevance to async marking. A bundle SU may
// hold any combination of async loads/stores, ASYNCMARKs, and
// WAIT_ASYNCMARKs, so this is a bitmask rather than a single tag.
enum class SUKind : unsigned {
  None = 0,
  // SU contains an async mem-op with no preceding ASYNCMARK in the same SU.
  PreMarkAsyncMemOp = 1u << 0,
  // SU contains an async mem-op with no following ASYNCMARK in the same SU.
  PostMarkAsyncMemOp = 1u << 1,
  AsyncMark = 1u << 2,
  WaitAsyncMark = 1u << 3,
  LLVM_MARK_AS_BITMASK_ENUM(/*LargestValue=*/WaitAsyncMark)
};

inline SUKind classify(const MachineInstr &MI, const SIInstrInfo &TII) {
  unsigned Opc = MI.getOpcode();
  if (Opc == AMDGPU::ASYNCMARK)
    return SUKind::AsyncMark;
  if (Opc == AMDGPU::WAIT_ASYNCMARK)
    return SUKind::WaitAsyncMark;
  if (TII.isAsyncLDSDMA(MI))
    return SUKind::PreMarkAsyncMemOp | SUKind::PostMarkAsyncMemOp;
  return SUKind::None;
}

inline SUKind classifySU(const SUnit &SU, const SIInstrInfo &TII) {
  const MachineInstr *MI = SU.getInstr();
  if (!MI)
    return SUKind::None;

  if (!MI->isBundle())
    return classify(*MI, TII);

  SUKind Bits = SUKind::None;
  for (auto It = std::next(MI->getIterator()),
            End = getBundleEnd(MI->getIterator());
       It != End; ++It) {
    SUKind New = classify(*It, TII);
    // Mem-ops following an in-bundle ASYNCMARK are not "pre-mark".
    if (any(Bits & SUKind::AsyncMark))
      Bits |= New & ~SUKind::PreMarkAsyncMemOp;
    else
      Bits |= New;
    // Once we see an ASYNCMARK in the bundle, any prior async mem-ops are
    // tagged by it, so they no longer count as "post-mark".
    if (any(New & SUKind::AsyncMark))
      Bits &= ~SUKind::PostMarkAsyncMemOp;
  }
  return Bits;
}

class AsyncMarkSched : public ScheduleDAGMutation {
public:
  void apply(ScheduleDAGInstrs *DAG) override;
};

void AsyncMarkSched::apply(ScheduleDAGInstrs *DAG) {
  const SIInstrInfo &TII = *static_cast<const SIInstrInfo *>(DAG->TII);

  SmallVector<SUnit *, 8> PendingAsync;
  SUnit *LastMark = nullptr;
  SUnit *LastWait = nullptr;

  for (SUnit &SU : DAG->SUnits) {
    SUKind Kind = classifySU(SU, TII);
    if (!any(Kind))
      continue;

    auto AddPred = [&](SUnit *Pred) {
      DAG->addEdge(&SU, SDep(Pred, SDep::Artificial));
    };

    bool IsMark = any(Kind & SUKind::AsyncMark);
    bool IsWait = any(Kind & SUKind::WaitAsyncMark);
    bool IsPre = any(Kind & SUKind::PreMarkAsyncMemOp);
    bool IsPost = any(Kind & SUKind::PostMarkAsyncMemOp);

    if (LastMark && (IsMark || IsWait || IsPre))
      AddPred(LastMark);

    if (IsMark) {
      for (SUnit *Pred : PendingAsync)
        AddPred(Pred);
      if (LastWait)
        AddPred(LastWait);
      PendingAsync.clear();
      LastMark = &SU;
    }

    if (IsPost)
      PendingAsync.push_back(&SU);

    if (IsWait)
      LastWait = &SU;
  }
}

} // end anonymous namespace

std::unique_ptr<ScheduleDAGMutation>
llvm::createAMDGPUAsyncMarkSchedDAGMutation() {
  return std::make_unique<AsyncMarkSched>();
}
