//===- setpc-analysis.cpp - Hotswap transpiler ----------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Static analysis pass that classifies every `s_set_pc_i64` and
// `s_swap_pc_i64` site in a decoded kernel into one of three principled
// shapes:
//
//   * DirectA: source SGPR pair is produced by a complete
//     `s_get_pc_i64 + s_add_co_u32 + s_add_co_ci_u32` chain reachable
//     in the same basic block. Lowers to `br label %BB_target`.
//
//   * IndirectB: subroutine-return shape -- the source SGPR pair is the
//     ret-pair populated by a caller's chain (Pattern B). Lowers to a
//     `cmp eq + br` cascade (emitted by `emitEnumeratedDispatch` in
//     handle-sop1.cpp) over the resolved return targets, terminating
//     in an `unreachable` trap BB. Each call site in the kernel that
//     wrote that ret-pair contributes one target via the
//     `chainTerminators` rewrite hook.
//
//   * DispatchSet: multi-target dispatch -- the source SGPR pair holds
//     one of N statically-known absolute targets reaching the use site
//     through distinct CFG paths (e.g. a tensilelite "activation
//     function dispatcher" -- each predecessor block writes a different
//     chain target into the same pair, then a join block consumes it).
//     Lowers to the same enumerated-dispatch cascade as IndirectB. For
//     `s_swap_pc_i64` it ALSO writes the return-PC `blockaddress` into
//     sdst before the cascade (mirroring the DirectA dst-write).
//
// The cascade shape replaces an earlier `indirectbr` lowering; see the
// rationale block on `emitEnumeratedDispatch` in handle-sop1.cpp for
// why (FixIrreducible pass compatibility under the irreducible CFGs
// the call/return pattern produces).
//
// See canonical-op.h's `S_SET_PC_I64` and `S_SWAP_PC_I64` doc for the full
// lowering contracts. The handler in `handle-sop1.cpp` consumes the
// classification.
//
// The pass runs in five phases. The last three together implement the
// inter-block PC-chain dataflow that distinguishes DispatchSet from
// "still Unresolvable". Without that dataflow, dispatcher-shaped
// kernels (the common tensilelite case where the call target is
// computed in a predecessor of the swap_pc's block) fall into
// Unresolvable and the handler refuses, blocking large slices of the
// corpus.
//
//   Phase 1 -- pre-pass. Walk insts once to enumerate every swap/set_pc
//             instruction's fallthrough offset and pre-add it to the
//             working block-leader set. This guarantees the per-block
//             walk in Phase 2 sees every swap/set_pc as the LAST inst
//             of its block (so its fallthrough is in a separate block,
//             and so the block-exit transfer correctly summarises the
//             pair state up to and including the swap/set_pc).
//
//   Phase 2 -- per-block intra-block walk. Mirrors the original
//             single-pass analysis: build per-instruction symbolic-PC
//             state (chains, scalar imms), classify swap/set_pc sites
//             that resolve in-block as DirectA (or, if the source pair
//             was dirtied without a complete chain, as Unresolvable
//             with the intra-block-specific reason). Sites whose
//             source pair is pristine through the block are deferred
//             to Phase 4 with a `PendingDataflow` record. Also collect
//             every chain terminator (Phase 5 prunes), every PendingB
//             site (Phase 5 classifies as IndirectB), and per-block
//             transfer summaries: per-pair {SET(value, terminator),
//             KILL, PASS}.
//
//   Phase 3 -- CFG + forward dataflow. Build per-block successor lists
//             from the last instruction (S_BRANCH / S_CBRANCH_* /
//             S_SWAP_PC fallthrough / fallthrough-to-leader). Run a
//             worklist forward dataflow on a finite lattice
//             (`(SGPR pair) -> (sorted set<uint64_t>, incomplete)`)
//             where the join is set-union with an incomplete bit OR'd
//             across paths and a hard cap of `kMaxDispatchTargets`
//             values per pair (over the cap -> mark incomplete and
//             refuse the use site). Convergence is guaranteed by the
//             bounded-height lattice.
//
//   Phase 4 -- re-classify deferred sites. For each PendingDataflow
//             site, look up the entry facts at its block for the
//             source pair. Empty / incomplete -> Unresolvable with the
//             dataflow-specific reason. One value -> DirectA. Two or
//             more values (within the cap) -> DispatchSet with that
//             enumerated set as `indirectTargets`. Add every target
//             to `extraBlockStarts` so the BB-layout phase promotes
//             it to a leader.
//
//   Phase 5 -- chain terminator retention. Keep every chain terminator
//             that feeds either (a) an IndirectB ret-pair (existing
//             logic, identifies it by `retPairLowReg` membership) OR
//             (b) a DispatchSet site, where retention is conditional
//             on BOTH `retPairLowReg == DispatchSet.indirectRetPairLowReg`
//             AND `resolvedReturnAddr ∈ DispatchSet.indirectTargets`
//             (so dead chain terminators that don't match a target
//             are still pruned). Build per-pair return-target lists
//             for IndirectB. Drop unused terminators so the raiser's
//             S_ADDC_U32 hook does not gratuitously rewrite chains
//             that don't feed any classified site.
//
// Key invariants this pass enforces:
//   * Per-block reset: the symbolic-PC table is wiped at every
//     decoder-known basic-block leader. PC chains do not survive
//     control flow (within a block). Inter-block survival is encoded
//     entirely through the dataflow lattice; the per-block intra walk
//     only sees its own writes.
//   * Conservative cleanup: any SGPR write the pass cannot model
//     (e.g. an s_mov, an unrelated s_add) clears the destination's
//     entry. DirectA is only claimed when the entire chain
//     (s_get_pc_i64 -> s_add_co_u32 -> s_add_co_ci_u32) is observable
//     intra-block; DispatchSet is only claimed when the dataflow
//     joins exclusively over complete chains (any path that kills the
//     pair sets `incomplete` and refuses).
//   * Loud refusal: unresolvable sites are recorded with a
//     human-readable reason; the handler turns this into an
//     unsupportedInstructionForm failure rather than a silent stub branch.
//
// IndirectB call-site handling (Phase 5) is unchanged from the original
// design: a complete getpc+add chain terminating in s_add_co_ci_u32 to
// a ret-pair SGPR, immediately followed by an unconditional s_branch
// into the subroutine region. The post-branch instruction's offset is
// the return target the call site committed to; we record it as a
// chain-terminator hook so the raiser rewrites the chain's effect to
// materialise a `blockaddress(@kernel, %BB_returnAddr)` into the
// ret-pair (rather than the binary PC the chain would otherwise yield).
// The same rewrite hook fires for DispatchSet retained terminators --
// the value materialised is the dispatch target, not a return address,
// but the mechanism is identical (the field's name remains
// `resolvedReturnAddr` for backward compatibility with the existing
// SetPcCallSiteInfo struct).

#include "setpc-analysis.h"

#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIDefines.h"
#include "mc-state.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <deque>
#include <optional>
#include <set>

using namespace llvm;

namespace COMGR::hotswap {

namespace {

// Cap on the number of distinct PC-chain values we will enumerate per
// SGPR pair across CFG edges. Beyond this cap we mark the lattice
// element `incomplete` and the use site falls into Unresolvable. A
// practical dispatcher (e.g. an N-way activation-function switch) tops
// out far below this; any larger fan-out is much more likely to be a
// runtime-derived chain we cannot enumerate, and refusing loudly is
// the principled response.
constexpr size_t kMaxDispatchTargets = 16;

// Classify a register operand as an SGPR and return its hardware
// index. Returns nullopt for non-SGPR regs (VCC/EXEC/SCC/M0/...).
// Mirrors the SGPR branch of RaiseContext::parseReg but stays scoped
// to this analysis (no shared state, so it can run pre-IR).
std::optional<unsigned> sgprIdx(const MCRegisterInfo &MRI, MCRegister Reg) {
  if (!Reg)
    return std::nullopt;
  MCRegister Lane = MRI.getSubReg(Reg, AMDGPU::sub0);
  if (!Lane)
    Lane = Reg;
  Lane = AMDGPU::mc2PseudoReg(Lane);
  // Filter the special-purpose registers parseReg handles before
  // falling through to SGPR_32. None of them participate in PC chains.
  switch (Lane) {
  case AMDGPU::VCC_LO:
  case AMDGPU::VCC_HI:
  case AMDGPU::EXEC_LO:
  case AMDGPU::EXEC_HI:
  case AMDGPU::SCC:
  case AMDGPU::MODE:
  case AMDGPU::M0:
  case AMDGPU::FLAT_SCR_LO:
  case AMDGPU::FLAT_SCR_HI:
  case AMDGPU::SGPR_NULL:
  case AMDGPU::SGPR_NULL_HI:
  case AMDGPU::XNACK_MASK_LO:
  case AMDGPU::XNACK_MASK_HI:
  case AMDGPU::LDS_DIRECT:
    return std::nullopt;
  default:
    break;
  }
  unsigned Enc = MRI.getEncodingValue(Reg);
  if (Enc & (AMDGPU::HWEncoding::IS_VGPR | AMDGPU::HWEncoding::IS_AGPR))
    return std::nullopt;
  if (!MRI.getRegClass(AMDGPU::SGPR_32RegClassID).contains(Lane))
    return std::nullopt;
  return Enc & AMDGPU::HWEncoding::REG_IDX_MASK;
}

// Width of the as-decoded register in 32-bit lanes. A pair (s[X:X+1])
// reports width 2; a scalar reports width 1.
unsigned regWidth32(const MCRegisterInfo &MRI, MCRegister Reg) {
  const unsigned MaxSubIdx = MRI.getNumSubRegIndices();
  unsigned W = 0;
  for (unsigned SubIdx = AMDGPU::sub0; SubIdx < MaxSubIdx; ++SubIdx) {
    if (!MRI.getSubReg(Reg, SubIdx))
      break;
    ++W;
  }
  return W ? W : 1;
}

// Convenience: extract a 32-bit unsigned immediate from an MCOperand.
// Returns nullopt if the operand is not an integer immediate.
std::optional<uint32_t> imm32(const MCInst &Inst, unsigned OpIdx) {
  if (OpIdx >= Inst.getNumOperands())
    return std::nullopt;
  const MCOperand &Op = Inst.getOperand(OpIdx);
  if (!Op.isImm())
    return std::nullopt;
  return static_cast<uint32_t>(Op.getImm() & 0xFFFFFFFFu);
}

// Per-pair PC-chain state. We track the symbolic absolute kernel
// offset stored in an SGPR pair sX:X+1, plus the offset of the chain
// terminator (the s_add_co_ci_u32 high-half add) so the raiser knows
// which instruction to attach the blockaddress-materialisation hook
// to. `lowAddDone` records whether the low-half s_add_co_u32 has
// already fired (a complete chain follows the strict order
// getpc -> low-add -> high-add).
struct PcChain {
  uint64_t Value = 0;       // symbolic absolute kernel offset
  uint64_t Terminator = 0;  // offset of the high-half s_add_co_ci_u32
  bool LowAddDone = false;
};

// Per-scalar immediate state: tracks SGPR_32 values that are known
// constants (from `s_add_co_i32 sZ, IMM, IMM`-style folding). Used to
// resolve the offset operand of the low-half PC add.
struct ScalarImm {
  uint32_t Value = 0;
};

// Intra-block analysis state. Tracks per-pair PC chains and per-half
// scalar immediates. `intraDirtyHalf_` records every SGPR-half index
// the block has WRITTEN (chain ops + generic SGPR writes). It is the
// signal Phase 2 uses to decide whether a swap/set_pc site whose
// chain didn't resolve intra-block is allowed to fall back on dataflow
// entry facts (pristine half -> defer; dirtied half -> refuse loudly).
class State {
public:
  State(const MCRegisterInfo &MRI) : Mri(MRI) {}

  // Wipe everything at a basic-block leader. Chain values do not
  // survive a control-flow boundary at the intra-block layer; inter-
  // block survival is encoded entirely through the dataflow lattice.
  void resetBlock() {
    PcChains.clear();
    Scalars.clear();
    IntraDirtyHalf.clear();
  }

  // Record a known absolute PC in the SGPR pair starting at `lowIdx`.
  void recordPc(unsigned LowIdx, uint64_t Value) {
    PcChain C;
    C.Value = Value;
    PcChains[LowIdx] = C;
    // The pair's high half can no longer be a tracked scalar imm.
    Scalars.erase(LowIdx);
    Scalars.erase(LowIdx + 1);
    IntraDirtyHalf.insert(LowIdx);
    IntraDirtyHalf.insert(LowIdx + 1);
  }

  // Look up a tracked PC chain by SGPR low index.
  PcChain *findPc(unsigned LowIdx) {
    auto It = PcChains.find(LowIdx);
    return It == PcChains.end() ? nullptr : &It->second;
  }

  void recordScalar(unsigned Idx, uint32_t Value) {
    Scalars[Idx].Value = Value;
    IntraDirtyHalf.insert(Idx);
    invalidatePcAt(Idx);
  }

  std::optional<uint32_t> findScalar(unsigned Idx) const {
    auto It = Scalars.find(Idx);
    if (It == Scalars.end())
      return std::nullopt;
    return It->second.Value;
  }

  // Drop any tracked PC pair whose low or high half is `idx`. Marks
  // `idx` (and the affected pair's other half) as dirty so the
  // dataflow transfer correctly KILLS pass-through entry facts.
  void invalidatePcAt(unsigned Idx) {
    IntraDirtyHalf.insert(Idx);
    auto It = PcChains.find(Idx);
    if (It != PcChains.end()) {
      IntraDirtyHalf.insert(It->first + 1);
      PcChains.erase(It);
      return;
    }
    if (Idx == 0)
      return;
    It = PcChains.find(Idx - 1);
    if (It != PcChains.end()) {
      IntraDirtyHalf.insert(It->first);
      PcChains.erase(It);
    }
  }

  void invalidateScalarAt(unsigned Idx) {
    IntraDirtyHalf.insert(Idx);
    Scalars.erase(Idx);
  }

  // Promote an in-progress chain to "low add done": the next
  // s_add_co_ci_u32 to the matching high-half register completes the
  // chain.
  void markLowAddDone(unsigned LowIdx, uint64_t NewValue) {
    auto It = PcChains.find(LowIdx);
    if (It == PcChains.end())
      return;
    It->second.Value = NewValue;
    It->second.LowAddDone = true;
    IntraDirtyHalf.insert(LowIdx);
  }

  // Finalise the chain at `lowIdx`, returning the resolved value and
  // the terminator offset. Caller must have already verified the
  // chain reached lowAddDone.
  void finishHighAdd(unsigned LowIdx, uint64_t TerminatorOff,
                     uint64_t AddedHi) {
    auto It = PcChains.find(LowIdx);
    if (It == PcChains.end())
      return;
    It->second.Value += AddedHi << 32;
    It->second.Terminator = TerminatorOff;
    IntraDirtyHalf.insert(LowIdx + 1);
  }

  // Drop everything that is not a finalised PC pair (i.e. invalidate
  // any chain whose lowAddDone is false). Called when the analysis
  // observes any SGPR write that breaks the strict chain order.
  void dropInProgressChains() {
    for (auto &Kv : llvm::make_early_inc_range(PcChains)) {
      if (!Kv.second.LowAddDone) {
        IntraDirtyHalf.insert(Kv.first);
        IntraDirtyHalf.insert(Kv.first + 1);
        PcChains.erase(Kv.first);
      }
    }
  }

  // Whether the block has performed any SGPR write to either half of
  // pair `lowIdx`. Used by Phase 2 to decide whether a swap/set_pc
  // site may fall back on dataflow entry facts.
  bool isPairDirty(unsigned LowIdx) const {
    return IntraDirtyHalf.count(LowIdx) ||
           IntraDirtyHalf.count(LowIdx + 1);
  }

  // Accessors used by Phase 2 to construct the per-block transfer
  // summary at end-of-block.
  const llvm::DenseMap<unsigned, PcChain> &pcChains() const {
    return PcChains;
  }
  const llvm::DenseSet<unsigned> &dirtyHalves() const {
    return IntraDirtyHalf;
  }

private:
  const MCRegisterInfo &Mri;
  llvm::DenseMap<unsigned, PcChain> PcChains;
  llvm::DenseMap<unsigned, ScalarImm> Scalars;
  llvm::DenseSet<unsigned> IntraDirtyHalf;
};

// Mark every SGPR-half written by `di` as dirty in `state` and drop
// any tracked PC pair whose halves overlap. This is the generic
// fallthrough for instructions whose semantics we did not model.
void invalidateGeneralSgprDefs(const DecodedInst &Di,
                               const MCRegisterInfo &MRI, State &State) {
  for (unsigned I = 0; I < Di.NumDefs && I < Di.numOps(); ++I) {
    if (!Di.isReg(I))
      continue;
    auto Idx = sgprIdx(MRI, Di.getReg(I));
    if (!Idx)
      continue;
    unsigned W = regWidth32(MRI, Di.getReg(I));
    for (unsigned K = 0; K < W; ++K) {
      State.invalidatePcAt(*Idx + K);
      State.invalidateScalarAt(*Idx + K);
    }
  }
}

// Per-pair transfer summary computed at end of each block in Phase 2,
// consumed by the Phase 3 dataflow.
struct PairTransfer {
  enum class Kind {
    // Block did not write either half of this pair. Entry facts pass
    // through unchanged.
    Pass,
    // Block ended with this pair holding a complete PC chain whose
    // value is `value` and whose chain terminator is at offset
    // `terminator`. Overrides any incoming entry fact.
    Set,
    // Block wrote one or both halves of this pair but did not end
    // with a complete chain (or wrote a non-chain value). Pass-
    // through entry facts are killed; downstream sees a constraint-
    // less pair.
    Kill,
  };
  Kind TransferKind = Kind::Pass;
  uint64_t Value = 0;       // when transferKind == Set
  uint64_t Terminator = 0;  // when transferKind == Set
};

// Per-block descriptor used by Phase 2 / 3 / 4. `lastIdx` is inclusive.
struct BlockData {
  uint64_t Offset = 0;
  size_t FirstIdx = 0;
  size_t LastIdx = 0;
  llvm::DenseMap<unsigned, PairTransfer> Transfers;
  SmallVector<uint64_t, 2> Successors;
};

// A swap/set_pc site whose source pair was pristine through its block
// and so was deferred to the Phase 4 dataflow re-classification.
struct PendingDataflowSite {
  uint64_t BlockOffset = 0;
  uint64_t SiteOffset = 0;
  unsigned SrcPair = 0;
  bool IsSwap = false;
};

// Pattern B set_pc site whose source ret-pair could not be resolved
// to a single chain in Phase 2; Phase 5 enumerates matching chain
// terminators and either classifies as IndirectB or refuses with the
// pair-no-call-site reason.
struct PendingB {
  uint64_t SetpcOffset = 0;
  unsigned RetPairLowReg = 0;
};

// Inter-block lattice value for one SGPR pair.
//   - `values` is the enumerated set of statically-known absolute
//     kernel offsets the pair can hold at the start of a block, sorted
//     ascending and capped at kMaxDispatchTargets.
//   - `incomplete` records that at least one CFG predecessor leaves
//     the pair in an unmodeled state (overwritten by a non-chain
//     operation, never constrained at all -- pristine kernel entry --
//     or omitted from a predecessor's exit facts entirely).
//
// Use sites consume the lattice: incomplete OR cap-exceeded => refuse
// loudly; otherwise the value count picks DirectA (1) or DispatchSet
// (>1) classification.
//
// Lattice element absent from a block's entry-fact map is interpreted
// by use sites as the unconstrained default (incomplete=true, no
// values). The JOIN-over-predecessors formulation in Phase 3 only
// inserts an entry into the map when at least one predecessor's exit
// facts mention the pair; it then OR's in incomplete from any
// predecessor that DIDN'T mention the pair.
struct PcLatticeValue {
  llvm::SmallVector<uint64_t, 8> Values;  // sorted, deduped
  bool Incomplete = false;
};

bool operator==(const PcLatticeValue &A, const PcLatticeValue &B) {
  return A.Incomplete == B.Incomplete && A.Values == B.Values;
}
bool operator!=(const PcLatticeValue &A, const PcLatticeValue &B) {
  return !(A == B);
}

// Merge `src` into `dst` (set-union of values, OR of incomplete bits,
// hard cap at kMaxDispatchTargets values -- over-cap promotes to
// incomplete and stops growing the value list).
void joinValue(PcLatticeValue &Dst, const PcLatticeValue &Src) {
  if (Src.Incomplete)
    Dst.Incomplete = true;
  for (uint64_t V : Src.Values) {
    auto It = llvm::lower_bound(Dst.Values, V);
    if (It != Dst.Values.end() && *It == V)
      continue;
    if (Dst.Values.size() >= kMaxDispatchTargets) {
      Dst.Incomplete = true;
      break;
    }
    Dst.Values.insert(It, V);
  }
}

// Compute the set of CFG successor block-offsets for the block whose
// last instruction is `lastInst`. `nextBlockOffset` is the offset of
// the next block in linear layout (used for fallthrough); pass
// `nextBlockExists = false` if `lastInst` is the kernel's final inst.
//
// The successor model is intentionally conservative for analysis
// safety:
//   * S_BRANCH: 1 successor (decoded target).
//   * S_CBRANCH_*: 2 successors (target + linear fallthrough).
//   * S_ENDPGM: 0 successors.
//   * S_SET_PC_I64: 0 successors. The classification table tells the
//     raiser where this jumps; for dataflow purposes any survivors
//     would have to flow through the destination's other predecessors
//     anyway.
//   * S_SWAP_PC_I64: 1 successor (linear fallthrough). The swap is a
//     branch-and-link; control eventually returns into the
//     fallthrough, which Phase 1 added to extraBlockStarts so it is a
//     known leader. Source-pair facts survive across the swap (the
//     swap reads but does not write the source pair); the dst pair
//     is killed by the swap's transfer.
//   * Anything else (block ended only because the next inst was an
//     external BB leader): linear fallthrough.
SmallVector<uint64_t, 2>
computeSuccessors(const DecodedInst &LastInst, uint64_t NextBlockOffset,
                  bool NextBlockExists) {
  SmallVector<uint64_t, 2> Result;
  auto BranchTargetFromImm =
      [&](unsigned OpIdx) -> std::optional<uint64_t> {
    if (OpIdx >= LastInst.Inst.getNumOperands())
      return std::nullopt;
    const MCOperand &Op = LastInst.Inst.getOperand(OpIdx);
    if (!Op.isImm())
      return std::nullopt;
    int64_t Raw = Op.getImm();
    int64_t BrOff = static_cast<int64_t>(
        static_cast<int16_t>(static_cast<uint16_t>(Raw & 0xFFFF)));
    return LastInst.Offset + 4 + BrOff * 4;
  };
  switch (LastInst.CanonOp) {
  case CanonicalOp::S_BRANCH: {
    auto T = BranchTargetFromImm(0);
    if (T)
      Result.push_back(*T);
    break;
  }
  case CanonicalOp::S_CBRANCH_SCC0:
  case CanonicalOp::S_CBRANCH_SCC1:
  case CanonicalOp::S_CBRANCH_VCCZ:
  case CanonicalOp::S_CBRANCH_VCCNZ:
  case CanonicalOp::S_CBRANCH_EXECZ:
  case CanonicalOp::S_CBRANCH_EXECNZ: {
    auto T = BranchTargetFromImm(0);
    if (T)
      Result.push_back(*T);
    if (NextBlockExists)
      Result.push_back(NextBlockOffset);
    break;
  }
  case CanonicalOp::S_ENDPGM:
  case CanonicalOp::S_SET_PC_I64:
    break;
  case CanonicalOp::S_SWAP_PC_I64:
  default:
    if (NextBlockExists)
      Result.push_back(NextBlockOffset);
    break;
  }
  return Result;
}

} // namespace

SetPcAnalysis analyseSetPC(ArrayRef<DecodedInst> Insts,
                           const std::set<uint64_t> &BlockStarts,
                           const MCState &Mc) {
  SetPcAnalysis Result;
  if (Insts.empty())
    return Result;

  const MCRegisterInfo &MRI = *Mc.RegInfo;

  // ---------------------------------------------------------------
  // Phase 1 -- pre-pass: enumerate every swap/set_pc fallthrough as a
  // block leader. This guarantees the per-block walk sees each
  // swap/set_pc as the LAST inst of its block (so its block-exit
  // transfer correctly summarises the pair state up to the
  // swap/set_pc, not past it).
  // ---------------------------------------------------------------
  llvm::DenseSet<uint64_t> MergedBlockStarts(BlockStarts.begin(),
                                             BlockStarts.end());
  for (const DecodedInst &Di : Insts) {
    if (Di.CanonOp == CanonicalOp::S_SWAP_PC_I64 ||
        Di.CanonOp == CanonicalOp::S_SET_PC_I64) {
      uint64_t Fallthrough = Di.Offset + Di.Size;
      if (MergedBlockStarts.insert(Fallthrough).second)
        Result.ExtraBlockStarts.insert(Fallthrough);
    }
  }

  // ---------------------------------------------------------------
  // Build BlockData skeletons. A "block" is identified by the offset
  // of an instruction that appears in `insts` AND is in
  // `mergedBlockStarts`. We skip leader offsets that have no inst
  // (e.g. fallthroughs past the last decoded inst, or branch targets
  // outside the decoded range).
  // ---------------------------------------------------------------
  llvm::SmallVector<BlockData> Blocks;
  llvm::DenseMap<uint64_t, size_t> OffsetToBlockIdx;
  Blocks.reserve(MergedBlockStarts.size());
  for (size_t I = 0; I < Insts.size(); ++I) {
    if (MergedBlockStarts.count(Insts[I].Offset)) {
      BlockData Bd;
      Bd.Offset = Insts[I].Offset;
      Bd.FirstIdx = I;
      Bd.LastIdx = I; // fixed up below
      OffsetToBlockIdx[Bd.Offset] = Blocks.size();
      Blocks.push_back(Bd);
    }
  }
  for (size_t Bi = 0; Bi < Blocks.size(); ++Bi) {
    size_t End = (Bi + 1 < Blocks.size()) ? Blocks[Bi + 1].FirstIdx
                                          : Insts.size();
    Blocks[Bi].LastIdx = End - 1;
  }

  // Always treat the first instruction's block as kernel entry. Phase
  // 3 seeds that block's entry facts as "unconstrained" (the default
  // PcLatticeValue), which correctly models pristine register state.

  // ---------------------------------------------------------------
  // Phase 2 -- per-block intra-block walk. For each block, run the
  // existing chain analysis and collect:
  //   * setpcSites entries for sites that resolve in-block (DirectA
  //     direct-branch, or Unresolvable-with-intra-block-reason when
  //     the source pair was dirtied without a complete chain).
  //   * pendingDataflow entries for sites whose source pair was
  //     pristine through the block -- Phase 4 reclassifies these.
  //   * pendingB entries for s_set_pc sites whose source pair could
  //     not be resolved by a chain (intra OR dataflow) -- Phase 5
  //     classifies these as IndirectB by matching against
  //     chainTerminators.
  //   * chainTerminators[high_add_offset] = {chain_value, lowReg} for
  //     every completed chain. Phase 5 prunes those that don't feed
  //     any classified site.
  //   * extraBlockStarts entries for DirectA chain targets and for
  //     swap_pc fallthroughs (the fallthrough leader was added in
  //     Phase 1 for analysis correctness; we re-record it here for
  //     the caller's BB-layout merge).
  //   * the per-block transfer summary (transfers map).
  //   * the per-block successor list (computeSuccessors on lastInst).
  // ---------------------------------------------------------------
  llvm::SmallVector<PendingDataflowSite> PendingDataflow;
  llvm::SmallVector<PendingB> PendingBs;

  for (size_t Bi = 0; Bi < Blocks.size(); ++Bi) {
    BlockData &Bd = Blocks[Bi];
    State State(MRI);
    State.resetBlock();

    for (size_t I = Bd.FirstIdx; I <= Bd.LastIdx; ++I) {
      const DecodedInst &Di = Insts[I];

      switch (Di.CanonOp) {
      case CanonicalOp::S_GETPC_B64: {
        if (Di.NumDefs >= 1 && Di.isReg(0)) {
          auto Idx = sgprIdx(MRI, Di.getReg(0));
          if (Idx) {
            State.recordPc(*Idx, Di.Offset + Di.Size);
            continue;
          }
        }
        break;
      }

      case CanonicalOp::S_ADD_U32: {
        if (Di.NumDefs < 1 || !Di.isReg(0))
          break;
        auto DstIdx = sgprIdx(MRI, Di.getReg(0));
        if (!DstIdx)
          break;
        unsigned S0 = Di.FirstSrcIdx;
        unsigned S1 = S0 + 1;
        auto Src0Imm = imm32(Di.Inst, S0);
        auto Src1Imm = imm32(Di.Inst, S1);
        if (Src0Imm && Src1Imm) {
          State.recordScalar(*DstIdx, *Src0Imm + *Src1Imm);
          continue;
        }
        std::optional<unsigned> Src0Idx;
        if (Di.isReg(S0))
          Src0Idx = sgprIdx(MRI, Di.getReg(S0));
        if (!Src0Idx || *Src0Idx != *DstIdx)
          break;
        PcChain *Chain = State.findPc(*DstIdx);
        if (!Chain || Chain->LowAddDone)
          break;
        uint32_t Addend = 0;
        if (Src1Imm) {
          Addend = *Src1Imm;
        } else if (Di.isReg(S1)) {
          auto Src1Reg = sgprIdx(MRI, Di.getReg(S1));
          if (!Src1Reg)
            break;
          auto Sc = State.findScalar(*Src1Reg);
          if (!Sc)
            break;
          Addend = *Sc;
        } else {
          break;
        }
        uint64_t NewVal = Chain->Value + static_cast<uint64_t>(Addend);
        State.markLowAddDone(*DstIdx, NewVal);
        continue;
      }

      case CanonicalOp::S_ADDC_U32: {
        if (Di.NumDefs < 1 || !Di.isReg(0))
          break;
        auto DstIdx = sgprIdx(MRI, Di.getReg(0));
        if (!DstIdx || *DstIdx == 0)
          break;
        unsigned LowIdx = *DstIdx - 1;
        PcChain *Chain = State.findPc(LowIdx);
        if (!Chain || !Chain->LowAddDone)
          break;
        unsigned S0 = Di.FirstSrcIdx;
        unsigned S1 = S0 + 1;
        std::optional<unsigned> Src0Idx;
        if (Di.isReg(S0))
          Src0Idx = sgprIdx(MRI, Di.getReg(S0));
        if (!Src0Idx || *Src0Idx != *DstIdx)
          break;
        auto Src1Imm = imm32(Di.Inst, S1);
        if (!Src1Imm)
          break;
        State.finishHighAdd(LowIdx, Di.Offset,
                            static_cast<uint64_t>(*Src1Imm));
        Result.ChainTerminators[Di.Offset] =
            SetPcCallSiteInfo{State.findPc(LowIdx)->Value, LowIdx};
        continue;
      }

      case CanonicalOp::S_SWAP_PC_I64: {
        // Phase 1 already added the fallthrough to mergedBlockStarts;
        // re-record for the caller's BB-layout merge.
        Result.ExtraBlockStarts.insert(Di.Offset + Di.Size);

        std::optional<unsigned> DstLow;
        if (Di.NumDefs >= 1 && Di.isReg(0))
          DstLow = sgprIdx(MRI, Di.getReg(0));

        // Synthetic chain-terminator registration. We always record
        // it when dstLow is known; Phase 5 drops it if no downstream
        // s_set_pc_i64 reads `sdst`. The key (di.Offset) is unique
        // because LLVM never lays two instructions at the same
        // offset; the value carries (retPair=sdstLow, returnAddr=
        // swap.end). The raiser's S_ADDC_U32 post-hook does not fire
        // on S_SWAP_PC_I64 (gated by CanonicalOp), so the equivalent
        // blockaddress materialisation happens inline in handleSOP1.
        if (DstLow)
          Result.ChainTerminators[Di.Offset] =
              SetPcCallSiteInfo{Di.Offset + Di.Size, *DstLow};

        // Classify the call-target (source pair).
        unsigned SrcOpIdx = Di.FirstSrcIdx;
        if (!Di.isReg(SrcOpIdx)) {
          SetPcSiteInfo Info;
          Info.SiteKind = SetPcSiteInfo::Kind::Unresolvable;
          Info.RefusalReason =
              "s_swap_pc_i64 source operand is not a register";
          Result.SetpcSites[Di.Offset] = std::move(Info);
          if (DstLow) {
            State.invalidatePcAt(*DstLow);
            State.invalidatePcAt(*DstLow + 1);
          }
          continue;
        }
        auto SrcLow = sgprIdx(MRI, Di.getReg(SrcOpIdx));
        if (!SrcLow) {
          SetPcSiteInfo Info;
          Info.SiteKind = SetPcSiteInfo::Kind::Unresolvable;
          Info.RefusalReason =
              "s_swap_pc_i64 source register is not an SGPR pair";
          Result.SetpcSites[Di.Offset] = std::move(Info);
          if (DstLow) {
            State.invalidatePcAt(*DstLow);
            State.invalidatePcAt(*DstLow + 1);
          }
          continue;
        }
        PcChain *Chain = State.findPc(*SrcLow);
        if (Chain && Chain->LowAddDone) {
          // DirectA: chain resolves the absolute callee target intra-block.
          SetPcSiteInfo Info;
          Info.SiteKind = SetPcSiteInfo::Kind::DirectA;
          Info.DirectTarget = Chain->Value;
          Result.SetpcSites[Di.Offset] = std::move(Info);
          Result.ExtraBlockStarts.insert(Chain->Value);
          if (Chain->Terminator)
            Result.ChainTerminators.erase(Chain->Terminator);
          // The src pair is consumed inline. Mark it dirty so the
          // block transfer KILLs entry facts (a downstream block on
          // a back edge would otherwise see the chain value, but the
          // intra-block consumption invalidates further reasoning).
          State.invalidatePcAt(*SrcLow);
        } else if (State.isPairDirty(*SrcLow)) {
          // The block wrote to srcPair (chain-or-otherwise) but it
          // didn't end with a complete chain. Dataflow entry facts
          // are dead. Refuse loudly.
          SetPcSiteInfo Info;
          Info.SiteKind = SetPcSiteInfo::Kind::Unresolvable;
          Info.RefusalReason =
              (llvm::Twine("s_swap_pc_i64 source SGPR pair s[") +
               llvm::Twine(*SrcLow) + ":" + llvm::Twine(*SrcLow + 1) +
               "] was modified intra-block without producing a "
               "statically resolvable getpc+add chain (the block "
               "either started a chain that did not complete or "
               "overwrote the pair with a non-chain value); inter-"
               "block dataflow facts cannot recover this")
                  .str();
          Result.SetpcSites[Di.Offset] = std::move(Info);
        } else {
          // SrcPair is pristine through the block. Defer to Phase 4
          // dataflow re-classification.
          PendingDataflowSite Pds;
          Pds.BlockOffset = Bd.Offset;
          Pds.SiteOffset = Di.Offset;
          Pds.SrcPair = *SrcLow;
          Pds.IsSwap = true;
          PendingDataflow.push_back(Pds);
        }
        // Dst pair now holds an opaque (return-PC) value; remove from
        // PC tracking so a downstream s_set_pc_i64 reading dst falls
        // into Pattern B (enumerated-dispatch cascade) rather than
        // DirectA.
        if (DstLow) {
          State.invalidatePcAt(*DstLow);
          State.invalidatePcAt(*DstLow + 1);
        }
        continue;
      }

      case CanonicalOp::S_SET_PC_I64: {
        unsigned SrcOpIdx = Di.FirstSrcIdx;
        if (!Di.isReg(SrcOpIdx)) {
          SetPcSiteInfo Info;
          Info.SiteKind = SetPcSiteInfo::Kind::Unresolvable;
          Info.RefusalReason =
              "s_set_pc_i64 source operand is not a register";
          Result.SetpcSites[Di.Offset] = std::move(Info);
          continue;
        }
        auto SrcIdx = sgprIdx(MRI, Di.getReg(SrcOpIdx));
        if (!SrcIdx) {
          SetPcSiteInfo Info;
          Info.SiteKind = SetPcSiteInfo::Kind::Unresolvable;
          Info.RefusalReason =
              "s_set_pc_i64 source register is not an SGPR pair";
          Result.SetpcSites[Di.Offset] = std::move(Info);
          continue;
        }
        Result.ExtraBlockStarts.insert(Di.Offset + Di.Size);
        PcChain *Chain = State.findPc(*SrcIdx);
        if (Chain && Chain->LowAddDone) {
          // DirectA intra-block.
          SetPcSiteInfo Info;
          Info.SiteKind = SetPcSiteInfo::Kind::DirectA;
          Info.DirectTarget = Chain->Value;
          Result.SetpcSites[Di.Offset] = std::move(Info);
          Result.ExtraBlockStarts.insert(Chain->Value);
          if (Chain->Terminator)
            Result.ChainTerminators.erase(Chain->Terminator);
          State.invalidatePcAt(*SrcIdx);
          continue;
        }
        if (State.isPairDirty(*SrcIdx)) {
          // Pair was dirtied intra-block without a complete chain;
          // dataflow facts are dead. Defer to Phase 5 PendingB
          // classification (matches against chainTerminators) which
          // correctly handles the subroutine-return shape regardless
          // of intra-block dirtiness -- the IndirectB pattern relies
          // on the source pair being populated by a CALLER's chain
          // terminator, not by anything in this block. If no chain
          // terminator matches this source pair, Phase 5 emits the
          // pair-no-call-site Unresolvable diagnostic.
          struct PendingB Pb;
          Pb.SetpcOffset = Di.Offset;
          Pb.RetPairLowReg = *SrcIdx;
          PendingBs.push_back(Pb);
          continue;
        }
        // SrcPair pristine through the block. Defer to Phase 4
        // dataflow re-classification. If dataflow leaves it
        // unconstrained, Phase 4 falls through to a PendingB-style
        // resolution attempt (so subroutine-return shapes still get
        // classified as IndirectB, not refused).
        PendingDataflowSite Pds;
        Pds.BlockOffset = Bd.Offset;
        Pds.SiteOffset = Di.Offset;
        Pds.SrcPair = *SrcIdx;
        Pds.IsSwap = false;
        PendingDataflow.push_back(Pds);
        continue;
      }

      default:
        break;
      }

      // Generic fallthrough: invalidate every SGPR (and overlapping
      // PC pair) the instruction writes to. This prevents stale
      // chain values from leaking past instructions whose semantics
      // we did not model. Also drop in-progress chains so a stray
      // write between getpc and the low add cannot be silently
      // absorbed into the chain by a later matching add.
      invalidateGeneralSgprDefs(Di, MRI, State);
      State.dropInProgressChains();
    }

    // Compute block-exit transfers from the final state.
    //   * Every pair in `state.pcChains()` with lowAddDone=true ->
    //     SET(value, terminator). Pass-through is overridden.
    //   * Every dirty half whose pair is not in pcChains-with-
    //     lowAddDone -> KILL of that pair. Cover BOTH the "low" view
    //     (half index treated as the pair's low) and the "high" view
    //     (half index - 1 treated as the pair's low). A dirty half
    //     can mean two distinct pairs were partially touched; we
    //     conservatively KILL both. (In the common case the second
    //     pair is unused and the KILL is harmless.)
    //   * Pairs with no entry in transfers default to PASS.
    auto SetKill = [&](unsigned LowIdx) {
      auto &T = Bd.Transfers[LowIdx];
      if (T.TransferKind != PairTransfer::Kind::Set)
        T.TransferKind = PairTransfer::Kind::Kill;
    };
    for (const auto &Kv : State.pcChains()) {
      if (Kv.second.LowAddDone) {
        PairTransfer &T = Bd.Transfers[Kv.first];
        T.TransferKind = PairTransfer::Kind::Set;
        T.Value = Kv.second.Value;
        T.Terminator = Kv.second.Terminator;
      } else {
        SetKill(Kv.first);
        if (Kv.first > 0)
          SetKill(Kv.first - 1);
      }
    }
    for (unsigned Half : State.dirtyHalves()) {
      // Don't downgrade a SET pair to KILL.
      auto It = Bd.Transfers.find(Half);
      if (It == Bd.Transfers.end() ||
          It->second.TransferKind != PairTransfer::Kind::Set)
        SetKill(Half);
      if (Half > 0) {
        auto It2 = Bd.Transfers.find(Half - 1);
        if (It2 == Bd.Transfers.end() ||
            It2->second.TransferKind != PairTransfer::Kind::Set)
          SetKill(Half - 1);
      }
    }

    // Compute CFG successors.
    bool HasNext = (Bi + 1) < Blocks.size();
    uint64_t NextOff = HasNext ? Blocks[Bi + 1].Offset : 0;
    Bd.Successors = computeSuccessors(Insts[Bd.LastIdx], NextOff, HasNext);
  }

  // ---------------------------------------------------------------
  // Phase 3 -- forward dataflow to fixpoint.
  //
  //   entryFacts[blockIdx][pairLow] = PcLatticeValue
  //
  // Formulation: at each block B,
  //   entryFacts[B] = JOIN over P ∈ preds(B) of exitFacts(P)
  //
  // where exitFacts(P) = transfer(entryFacts[P], P.transfers):
  //   - SET overrides any incoming entry with {value, !incomplete}
  //   - KILL overrides with {∅, incomplete}
  //   - PASS leaves the entry unchanged
  //
  // and JOIN is set-union of `values` + OR of `incomplete` bits, with
  // an additional rule: any pair P that appears in SOME predecessor
  // P_i's exit but is MISSING from P_j's exit gets incomplete=true
  // (the missing predecessor leaves P at its kernel-entry-pristine
  // default = unconstrained).
  //
  // Pairs absent from entryFacts[B] are interpreted by use sites as
  // the unconstrained default, so we only insert entries into the map
  // when at least one predecessor's exit mentions the pair.
  //
  // Convergence is guaranteed by the bounded-height lattice: per
  // pair, at most kMaxDispatchTargets values + 1 incomplete bit.
  // The lattice is also monotone (values only grow, incomplete only
  // 0->1), so JOIN-over-preds-from-scratch with re-entry-on-change is
  // a sound fixpoint algorithm.
  // ---------------------------------------------------------------

  // Build predecessor map.
  llvm::SmallVector<llvm::SmallVector<size_t, 4>> Predecessors(Blocks.size());
  for (size_t Bi = 0; Bi < Blocks.size(); ++Bi) {
    for (uint64_t SuccOff : Blocks[Bi].Successors) {
      auto Sit = OffsetToBlockIdx.find(SuccOff);
      if (Sit != OffsetToBlockIdx.end())
        Predecessors[Sit->second].push_back(Bi);
    }
  }

  // computeExit applies the per-block transfer to a given entry-fact
  // map and returns the exit-fact map. PASS pairs flow through; SET
  // and KILL pairs override.
  auto ComputeExit =
      [&](const llvm::DenseMap<unsigned, PcLatticeValue> &Entry,
          const BlockData &Bd) {
        llvm::DenseMap<unsigned, PcLatticeValue> Exit = Entry;
        for (const auto &Kv : Bd.Transfers) {
          if (Kv.second.TransferKind == PairTransfer::Kind::Set) {
            PcLatticeValue V;
            V.Values.push_back(Kv.second.Value);
            V.Incomplete = false;
            Exit[Kv.first] = std::move(V);
          } else if (Kv.second.TransferKind == PairTransfer::Kind::Kill) {
            PcLatticeValue V;
            V.Incomplete = true;
            Exit[Kv.first] = std::move(V);
          }
        }
        return Exit;
      };

  llvm::SmallVector<llvm::DenseMap<unsigned, PcLatticeValue>> EntryFacts(
      Blocks.size());

  std::deque<size_t> Worklist;
  llvm::SmallVector<bool> OnWorklist(Blocks.size(), false);
  for (size_t Bi = 0; Bi < Blocks.size(); ++Bi) {
    Worklist.push_back(Bi);
    OnWorklist[Bi] = true;
  }

  while (!Worklist.empty()) {
    size_t Bi = Worklist.front();
    Worklist.pop_front();
    OnWorklist[Bi] = false;

    // Recompute entry from JOIN over predecessors.
    llvm::DenseMap<unsigned, PcLatticeValue> NewEntry;
    if (!Predecessors[Bi].empty()) {
      // Collect predecessor exits.
      llvm::SmallVector<llvm::DenseMap<unsigned, PcLatticeValue>> PredExits;
      PredExits.reserve(Predecessors[Bi].size());
      for (size_t Pi : Predecessors[Bi])
        PredExits.push_back(ComputeExit(EntryFacts[Pi], Blocks[Pi]));

      // Determine the union of pairs mentioned by any predecessor
      // exit. These are the only pairs whose entry fact differs from
      // the unconstrained default at this block.
      llvm::DenseSet<unsigned> Mentioned;
      for (const auto &Pe : PredExits)
        for (const auto &Kv : Pe)
          Mentioned.insert(Kv.first);

      // For each mentioned pair, JOIN every predecessor's
      // contribution. Predecessors whose exit doesn't mention the
      // pair contribute incomplete=true (the kernel-entry-pristine
      // default).
      for (unsigned Pair : Mentioned) {
        PcLatticeValue Acc;
        for (const auto &Pe : PredExits) {
          auto It = Pe.find(Pair);
          if (It == Pe.end()) {
            Acc.Incomplete = true;
          } else {
            joinValue(Acc, It->second);
          }
        }
        NewEntry[Pair] = std::move(Acc);
      }
    }
    // Block 0 (kernel entry) has no predecessors -> newEntry is empty,
    // which correctly represents "every pair is unconstrained".

    if (NewEntry != EntryFacts[Bi]) {
      EntryFacts[Bi] = std::move(NewEntry);
      for (uint64_t SuccOff : Blocks[Bi].Successors) {
        auto Sit = OffsetToBlockIdx.find(SuccOff);
        if (Sit != OffsetToBlockIdx.end() &&
            !OnWorklist[Sit->second]) {
          Worklist.push_back(Sit->second);
          OnWorklist[Sit->second] = true;
        }
      }
    }
  }

  // ---------------------------------------------------------------
  // Phase 4 -- re-classify deferred sites using dataflow facts.
  // ---------------------------------------------------------------
  for (const PendingDataflowSite &Pds : PendingDataflow) {
    auto Bit = OffsetToBlockIdx.find(Pds.BlockOffset);
    if (Bit == OffsetToBlockIdx.end())
      continue;
    const auto &Facts = EntryFacts[Bit->second];
    auto It = Facts.find(Pds.SrcPair);
    bool Resolved =
        (It != Facts.end()) && !It->second.Incomplete &&
        !It->second.Values.empty() &&
        It->second.Values.size() <= kMaxDispatchTargets;

    if (!Resolved) {
      if (Pds.IsSwap) {
        SetPcSiteInfo Info;
        Info.SiteKind = SetPcSiteInfo::Kind::Unresolvable;
        Info.RefusalReason =
            (llvm::Twine("s_swap_pc_i64 source SGPR pair s[") +
             llvm::Twine(Pds.SrcPair) + ":" + llvm::Twine(Pds.SrcPair + 1) +
             "] does not have a statically resolvable getpc+add "
             "chain reaching this site (intra-block analysis found "
             "no chain; inter-block dataflow could not enumerate a "
             "bounded set of targets -- the value comes from a "
             "kernarg/runtime source, an unbounded fan-in, or a "
             "control-flow path that overwrites the pair with an "
             "unmodelled value)")
                .str();
        Result.SetpcSites[Pds.SiteOffset] = std::move(Info);
      } else {
        // For s_set_pc_i64, fall through to PendingB -- a subroutine-
        // return shape relies on caller-side chain terminators, not
        // on the source pair being constrained by intra-kernel
        // dataflow.
        struct PendingB Pb;
        Pb.SetpcOffset = Pds.SiteOffset;
        Pb.RetPairLowReg = Pds.SrcPair;
        PendingBs.push_back(Pb);
      }
      continue;
    }

    SmallVector<uint64_t, 4> Targets(It->second.Values.begin(),
                                     It->second.Values.end());
    SetPcSiteInfo Info;
    if (Targets.size() == 1) {
      Info.SiteKind = SetPcSiteInfo::Kind::DirectA;
      Info.DirectTarget = Targets[0];
    } else {
      Info.SiteKind = SetPcSiteInfo::Kind::DispatchSet;
      Info.IndirectTargets = Targets;
      Info.IndirectRetPairLowReg = Pds.SrcPair;
    }
    for (uint64_t T : Targets)
      Result.ExtraBlockStarts.insert(T);
    Result.SetpcSites[Pds.SiteOffset] = std::move(Info);
  }

  // ---------------------------------------------------------------
  // Phase 5 -- chain terminator retention + IndirectB classification.
  //
  // Retention rule: keep a chain terminator iff it feeds either
  //   (a) an IndirectB ret-pair (pair-low-index match), OR
  //   (b) a DispatchSet site (pair-low-index AND value match).
  // Drop unused terminators so the raiser's S_ADDC_U32 hook does not
  // gratuitously rewrite chains that don't feed any classified site.
  // ---------------------------------------------------------------

  // Collect Pattern B consumers.
  llvm::DenseSet<unsigned> RetPairsConsumedByB;
  for (const struct PendingB &Pb : PendingBs)
    RetPairsConsumedByB.insert(Pb.RetPairLowReg);

  // Collect DispatchSet consumers (pair -> set of allowed values).
  llvm::DenseMap<unsigned, llvm::DenseSet<uint64_t>> DispatchSetTargets;
  for (const auto &Kv : Result.SetpcSites) {
    if (Kv.second.SiteKind == SetPcSiteInfo::Kind::DispatchSet) {
      auto &Set = DispatchSetTargets[Kv.second.IndirectRetPairLowReg];
      for (uint64_t T : Kv.second.IndirectTargets)
        Set.insert(T);
    }
  }

  // Prune chain terminators in place.
  for (auto &Kv : llvm::make_early_inc_range(Result.ChainTerminators)) {
    bool KeepForB = RetPairsConsumedByB.contains(Kv.second.RetPairLowReg);
    bool KeepForDispatch = false;
    auto Dt = DispatchSetTargets.find(Kv.second.RetPairLowReg);
    if (Dt != DispatchSetTargets.end() &&
        Dt->second.count(Kv.second.ResolvedReturnAddr))
      KeepForDispatch = true;
    if (!KeepForB && !KeepForDispatch)
      Result.ChainTerminators.erase(Kv.first);
  }

  // Build per-pair return-target lists for IndirectB from the
  // surviving terminators.
  llvm::DenseMap<unsigned, SmallVector<uint64_t, 4>> TargetsByPair;
  for (const auto &Kv : Result.ChainTerminators) {
    if (!RetPairsConsumedByB.count(Kv.second.RetPairLowReg))
      continue;
    TargetsByPair[Kv.second.RetPairLowReg].push_back(
        Kv.second.ResolvedReturnAddr);
    Result.ExtraBlockStarts.insert(Kv.second.ResolvedReturnAddr);
  }
  for (auto &Kv : TargetsByPair) {
    auto &V = Kv.second;
    llvm::sort(V);
    V.erase(std::unique(V.begin(), V.end()), V.end());
  }

  // Classify PendingB sites.
  for (const struct PendingB &Pb : PendingBs) {
    if (Result.SetpcSites.count(Pb.SetpcOffset))
      continue;  // already classified by Phase 4 (e.g. DispatchSet)
    auto It = TargetsByPair.find(Pb.RetPairLowReg);
    if (It == TargetsByPair.end() || It->second.empty()) {
      SetPcSiteInfo Info;
      Info.SiteKind = SetPcSiteInfo::Kind::Unresolvable;
      Info.RefusalReason =
          (llvm::Twine("s_set_pc_i64 reads SGPR pair s[") +
           llvm::Twine(Pb.RetPairLowReg) + ":" +
           llvm::Twine(Pb.RetPairLowReg + 1) +
           "] but no statically resolvable call-site getpc+add chain "
           "targets that pair (and inter-block dataflow could not "
           "enumerate a bounded target set either)")
              .str();
      Result.SetpcSites[Pb.SetpcOffset] = std::move(Info);
      continue;
    }
    SetPcSiteInfo Info;
    Info.SiteKind = SetPcSiteInfo::Kind::IndirectB;
    Info.IndirectTargets.assign(It->second.begin(), It->second.end());
    Info.IndirectRetPairLowReg = Pb.RetPairLowReg;
    Result.SetpcSites[Pb.SetpcOffset] = std::move(Info);
  }

  return Result;
}

} // namespace COMGR::hotswap
