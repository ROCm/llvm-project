//===- setpc-analysis.h - Hotswap transpiler ------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_SETPC_ANALYSIS_H
#define HOTSWAP_TRANSPILER_SETPC_ANALYSIS_H

#include "decoded-inst.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <set>
#include <string>

namespace COMGR::hotswap {

struct MCState;

// Static analysis pass that classifies every `s_set_pc_i64` site in a
// decoded kernel. See canonical-op.h's S_SET_PC_I64 doc for the three
// principled lowering shapes (Pattern A direct br, Pattern B / DispatchSet
// cmp+br cascade via `emitEnumeratedDispatch`, Unresolvable refusal).
//
// The analysis is conservative: any SGPR pair whose value path it
// cannot follow drops out of the symbolic-PC table immediately, so
// downstream `s_set_pc_i64` sites that read a dropped pair are
// classified Unresolvable rather than guessed. The handler refuses
// loudly on Unresolvable -- never silently emit a stub branch.

struct SetPcSiteInfo {
  enum class Kind {
    // Statically resolvable single-target intra-kernel branch. Source
    // SGPR pair was produced by a complete `s_get_pc_i64 + s_add_co_u32
    // + s_add_co_ci_u32` chain in the same basic block as the
    // s_set_pc_i64 / s_swap_pc_i64. Lowering emits `br label %BB_target`.
    DirectA,
    // Subroutine-return shape: the source SGPR pair is the ret-pair
    // populated by a caller's chainTerminator hook. Lowering emits a
    // `cmp eq + br` cascade (via `emitEnumeratedDispatch` in
    // handle-sop1.cpp) over the resolved return targets, terminating
    // in an `unreachable` trap BB. Pattern B is asymmetric -- call-
    // side and return-side participate through chainTerminators +
    // pendingB enumeration in Pass 4.
    IndirectB,
    // Multi-target dispatch shape: source SGPR pair holds one of N
    // statically-known absolute targets reaching the use site through
    // distinct CFG paths (e.g. tensilelite's "activation function
    // dispatcher" -- each predecessor block writes a different chain
    // target into the same pair, then a join block consumes it). The
    // inter-block PC-chain dataflow in Pass 3 enumerates the targets;
    // Pass 5 retains every contributing chain terminator so the raiser
    // hook writes the per-predecessor i64 marker (the resolved
    // callee's source-MC byte offset) into the pair at each
    // predecessor (mirroring the IndirectB return-side mechanism).
    // The lowering emits the same cmp+br enumerated-dispatch cascade
    // as IndirectB (comparing the marker against each enumerated
    // target offset); for s_swap_pc_i64 it ALSO writes the
    // return-address marker into sdst before the cascade (same as
    // DirectA). Order is deterministic (ascending) so lit fixtures
    // can pin shape. See `emitEnumeratedDispatch`'s rationale block
    // in handle-sop1.cpp for why a cascade (FixIrreducible
    // compatibility under irreducible CFGs) and why integer markers
    // rather than `ptrtoint(blockaddress)` (AMDGPU ISel cannot
    // materialise a `BlockAddress` as an i64).
    DispatchSet,
    // Refused. The handler converts this into
    // `RaiseFailure::unsupportedInstructionForm` with `refusalReason`.
    Unresolvable,
  };
  Kind SiteKind = Kind::Unresolvable;
  // DirectA: the absolute kernel offset to branch to. The lowering
  // emits `br label %BB_<directTarget>`.
  uint64_t DirectTarget = 0;
  // IndirectB: the enumerated set of possible return targets (each is
  // the absolute kernel offset of a basic block following one of the
  // identified call sites). Order is deterministic (ascending) so
  // lit fixtures can pin shape.
  // DispatchSet: the enumerated set of possible callee/branch targets
  // (each is the absolute kernel offset of a basic block leader that
  // is a chain-resolved value of the source pair on some incoming CFG
  // path). Same ordering contract.
  llvm::SmallVector<uint64_t, 4> IndirectTargets;
  // IndirectB: the SGPR low index of the source pair (for diagnostics
  // and so the handler can read the right pair).
  // DispatchSet: same -- the SGPR low index of the source pair the
  // handler reads to drive the cmp+br cascade.
  unsigned IndirectRetPairLowReg = 0;
  // Unresolvable: human-readable reason for the refusal diagnostic.
  std::string RefusalReason;
};

struct SetPcCallSiteInfo {
  // Absolute kernel offset of the instruction immediately following
  // the call-site `s_branch` (i.e. the return target the call site
  // expected). The Pattern B lowering will list this offset's BB as
  // one of the cascade's enumerated targets.
  uint64_t ResolvedReturnAddr = 0;
  // SGPR low index of the ret-pair this call site populates. Pattern
  // B `s_set_pc_i64 sX:Y` enumerates call sites whose
  // retPairLowReg == X.
  unsigned RetPairLowReg = 0;
};

struct SetPcAnalysis {
  // Keyed by the s_set_pc_i64 instruction's absolute kernel offset.
  llvm::DenseMap<uint64_t, SetPcSiteInfo> SetpcSites;

  // Chain-terminator hooks: keyed by the absolute offset of the
  // s_add_co_ci_u32 (high-half add) that completes a call-site PC
  // chain. The raiser runs the SOP2 handler normally for that
  // instruction and then overwrites the ret-pair SGPR with the
  // plain i64 marker `resolvedReturnAddr` (the source-MC byte
  // offset of the BB the chain was intended to return to), so the
  // i64 value carried in the ret-pair across the call is a normal
  // integer constant on each contributing predecessor path rather
  // than a binary runtime PC. This makes each downstream
  // enumerated-dispatch cascade `icmp eq i64 %marker, <offset>`
  // foldable to `i1 true` under mem2reg + SCCP + InstCombine after
  // the cross-predecessor phi join, so SimplifyCFG collapses the
  // cascade to a direct branch on each predecessor-specialised
  // path. (An earlier revision stored
  // `ptrtoint(blockaddress(@kernel, %BB_<resolvedReturnAddr>))`
  // here -- that form survived unfolded in irreducible CFGs and
  // crashed AMDGPU ISel with `Cannot select: BlockAddress`; see
  // `emitEnumeratedDispatch` in handle-sop1.cpp for the full
  // rationale.)
  llvm::DenseMap<uint64_t, SetPcCallSiteInfo> ChainTerminators;

  // Block-start offsets newly discovered by this analysis: Pattern A
  // direct targets + Pattern B return targets. Caller merges these
  // into the overall blockStarts BEFORE basic-block layout.
  llvm::DenseSet<uint64_t> ExtraBlockStarts;
};

// Run the analysis. `blockStarts` is the set of leaders the decoder
// already discovered (used to reset the per-block symbolic-PC table
// at every BB boundary). `mc` provides MCRegisterInfo for SGPR
// classification.
SetPcAnalysis analyseSetPC(llvm::ArrayRef<DecodedInst> Insts,
                           const std::set<uint64_t> &BlockStarts,
                           const MCState &Mc);

} // namespace COMGR::hotswap

#endif
