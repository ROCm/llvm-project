//===- wave-size-obstruction.cpp - Hotswap transpiler ---------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "wave-size-obstruction.h"
#include "wave-projection.h"

#include <cstdlib>

#include "canonical-op.h"
#include "decoded-inst.h"
#include "isa-profile.h"
#include "mc-state.h"
#include "wave-projection.h"

#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

// SIInstrFlags::DPP / SDWA bits live in the AMDGPU target's SIDefines
// header. We only use the flag constant; no other target dependency.
#include "SIDefines.h"

namespace COMGR::hotswap {

using namespace llvm;

// ----------------------------------------------------------------------------
// Taxonomy rendering. The text in each branch is the label that
// surfaces in the classifier trace and that lit tests assert on. The
// obstruction-class number (Class 1..4, see hotswap/docs/wave-size-
// translation.md sec. 6) is included parenthetically so operators reading
// the trace can cross-reference the spec without mental translation.
// ----------------------------------------------------------------------------

const char *obstructionKindName(ObstructionKind K) {
  switch (K) {
  case ObstructionKind::None:
    return "None";
  case ObstructionKind::MbcntHiLaneIdLeak:
    return "MbcntHiLaneIdLeak (sec. 3 Class 1: absolute lane-ID leak via "
           "v_mbcnt_hi)";
  case ObstructionKind::OutOfRangeLaneOperand:
    return "OutOfRangeLaneOperand (sec. 3 Class 1: readlane/writelane operand "
           ">= W_s)";
  case ObstructionKind::TtmpWaveIdLeak:
    return "TtmpWaveIdLeak (sec. 3 Class 1: source read of ttmp8 under "
           "cross-widening -- wave_id_in_wg field)";
  case ObstructionKind::WaveIdLiftScalarized:
    return "WaveIdLiftScalarized (sec. 3 Class 1: canonical wave_id BFE lift "
           "+ v_writelane/v_readlane + WMMA -- cross-lane primitive scalarises "
           "the divergent lift, collapsing per-source-wave distinction)";
  case ObstructionKind::WorkitemIdPredicateChain:
    // see hotswap/docs/modrep-predicate-chain.md sec. 5 (narrow-O1 classifier)
    return "WorkitemIdPredicateChain (sec. 3 Class 5: workitem.id.x() feeds a "
           "lane-position-scoped icmp against compile-time constant K <= "
           "W_s-1, gating a side effect -- wave-size-sensitive predicate "
           "chain under modulo-replication)";
  case ObstructionKind::FullWaveRotate:
    return "FullWaveRotate (sec. 3 Class 2: unrewritable v_permlane64)";
  case ObstructionKind::LaneGroupShuffle:
    return "LaneGroupShuffle (sec. 3 Class 2: permlane16 / permlanex16 / "
           "permlane*_swap)";
  case ObstructionKind::DsSwizzle:
    return "DsSwizzle (sec. 3 Class 2: ds_swizzle_b32)";
  case ObstructionKind::DppCrossLane:
    return "DppCrossLane (sec. 3 Class 2: DPP modifier)";
  case ObstructionKind::DsBpermuteGather:
    return "DsBpermuteGather (sec. 3 Class 2: ds_bpermute_b32)";
  case ObstructionKind::NonCommutativeAtomic:
    return "NonCommutativeAtomic (sec. 3 Class 3: cmpswap/swap/xchg, replica "
           "race)";
  case ObstructionKind::CmpxFromLaneId:
    return "CmpxFromLaneId (sec. 3 Class 4: lane-predicated v_cmpx)";
  case ObstructionKind::SaveExecFromLaneId:
    return "SaveExecFromLaneId (sec. 3 Class 4: lane-predicated "
           "s_*_saveexec_b32)";
  }
  return "UnknownObstructionKind";
}

const char *rewriteIdName(RewriteId R) {
  switch (R) {
  case RewriteId::None:
    return "none";
  case RewriteId::P1_DsBpermute:
    return "P1 (llvm.amdgcn.ds.bpermute)";
  case RewriteId::P2_PermLane16:
    return "P2 (llvm.amdgcn.permlane16)";
  case RewriteId::P3_PermLane64:
    return "P3 (reserved: v_permlane64 has no rewrite)";
  case RewriteId::P4_PermLaneSwap:
    return "P4 (permlane*_swap via LDS round-trip or permlane16 pair)";
  case RewriteId::P5_DppModifier:
    return "P5 (llvm.amdgcn.update.dpp)";
  case RewriteId::P6_DsSwizzle:
    return "P6 (llvm.amdgcn.ds.swizzle)";
  case RewriteId::LaneOpBoundsValidator:
    return "raise-time readlane/writelane bounds validator";
  case RewriteId::SaveExecLaneRelative:
    return "saveexec source-wave-relative";
  case RewriteId::AtomicOneReplica:
    return "store-only atomic gated to one MODREP replica";
  case RewriteId::PostRaiseCrossLaneRewrite:
    return "post-raise cross-lane rewrite (writelane -> select, "
           "readlane -> ds.bpermute)";
  case RewriteId::WaveNativeMbcntCmpx:
    return "WaveNative source-wave mbcnt -> V_CMPX EXEC projection";
  case RewriteId::WaveNativeMbcntSaveExec:
    return "WaveNative source-wave mbcnt -> s_*_saveexec_b32 EXEC projection";
  }
  return "UnknownRewriteId";
}

// ----------------------------------------------------------------------------
// ObstructionReport queries.
// ----------------------------------------------------------------------------

bool ObstructionReport::hasUnrewritable() const {
  for (const auto &S : Sites)
    if (S.Rewrite == RewriteId::None)
      return true;
  return false;
}

bool ObstructionReport::hasPendingRewrite() const {
  for (const auto &S : Sites)
    if (S.Rewrite != RewriteId::None && !S.RewriteImplemented)
      return true;
  return false;
}

bool ObstructionReport::isOblivious() const {
  for (const auto &S : Sites)
    if (!S.RewriteImplemented)
      return false;
  return true;
}

const ObstructionSite *ObstructionReport::firstUnrewritable() const {
  for (const auto &S : Sites)
    if (S.Rewrite == RewriteId::None)
      return &S;
  return nullptr;
}

const ObstructionSite *ObstructionReport::firstPending() const {
  for (const auto &S : Sites)
    if (S.Rewrite != RewriteId::None && !S.RewriteImplemented)
      return &S;
  return nullptr;
}

// ----------------------------------------------------------------------------
// Classification primitives.
// ----------------------------------------------------------------------------

namespace {

// Extract the lane operand of a v_readlane / v_writelane instruction.
// Returns std::nullopt when the operand is register-typed (dynamic) or
// when LLVM's named-operand index lookup fails (e.g. on a future LLVM
// version that renames `src1`).
//
// We use `AMDGPU::getNamedOperandIdx(opcode, AMDGPU::OpName::src1)`
// which is the authoritative way to find a named MCInst operand and
// is robust to operand-layout reordering between LLVM versions.
// Implementation note: both v_readlane_b32 (dst, src0, src1) and
// v_writelane_b32 (vdst, src0, src1, vdst_in) name the lane operand
// `src1`, so a single getNamedOperandIdx call covers both.
std::optional<int64_t> extractLaneOperandImm(const DecodedInst &Di) {
  const MCInst &Inst = Di.Inst;
  int Idx = AMDGPU::getNamedOperandIdx(Inst.getOpcode(), AMDGPU::OpName::src1);
  if (Idx < 0 || static_cast<unsigned>(Idx) >= Inst.getNumOperands())
    return std::nullopt;
  const MCOperand &Op = Inst.getOperand(Idx);
  if (Op.isImm())
    return Op.getImm();
  return std::nullopt; // dynamic SGPR operand -- cannot statically prove range
}

// Decide whether a `ds_swizzle_b32` immediate encodes a swizzle mode
// that is *structurally* wave-size-oblivious under modulo-replication
// (the projection-ladder's first rung, see hotswap/docs/wave-size-
// translation.md sec. 2.2).
//
// The 16-bit imm encodes one of seven modes (SIDefines.h
// `Swizzle::Id`). Per AMDGPU SIDefines.h `Swizzle::EncBits`:
//
//   QUAD_PERM_ENC    == 0x8000, QUAD_PERM_ENC_MASK    == 0xFF00
//   BITMASK_PERM_ENC == 0x0000, BITMASK_PERM_ENC_MASK == 0x8000
//   FFT_MODE_ENC     == 0xE000  (FFT_ROTATE_MODE_MASK  == 0xF000)
//   ROTATE_MODE_ENC  == 0xC000  (FFT_ROTATE_MODE_MASK  == 0xF000)
//
// Within each envelope, only specific sub-encodings are *valid*:
//
//   QUAD_PERM     -- bits 0..7  encode four 2-bit lane selectors.
//                   All 256 imms in [0x8000, 0x80FF] are valid.
//
//   BITMASK_PERM  -- bits 0..4  AND mask (0..31)
//                   bits 5..9  OR mask  (0..31)
//                   bits 10..14 XOR mask (0..31)
//                   bit 15      = 0 (envelope discriminator)
//                   All 32K imms in [0x0000, 0x7FFF] are valid.
//
//   FFT_MODE      -- bits 0..4   FFT_SWIZZLE_MASK (0..31)
//                   bits 5..11  reserved, MUST be 0
//                   bits 12..15 = 0xE (envelope discriminator)
//                   Valid imms: exactly the 32 in [0xE000, 0xE01F].
//
//   ROTATE_MODE   -- bits 0..4   reserved, MUST be 0
//                   bits 5..9   ROTATE_SIZE_MASK (0..31)
//                   bit  10     ROTATE_DIR_MASK  (0|1)
//                   bit  11     reserved, MUST be 0
//                   bits 12..15 = 0xC (envelope discriminator)
//                   Valid imms: exactly the 64 with bits 0..4=0,
//                   bit 11=0, top nibble=0xC.
//
// All four valid-encoding sub-spaces are wave-size-oblivious under
// modulo-replication. The argument is the same for all four: wave64
// ds_swizzle hardware preserves bit 5 of the lane id, so each
// 32-lane half independently performs the same permutation.
// Preservation verified empirically on gfx942 (CDNA3) wave64 across
// the four envelopes (see the MODREP block in handle-ds.cpp for the
// per-envelope probe results and the LLVM-upstream-test cross-
// citation that lifts the property to the wave64 GPU family).
//
// REFUSED imms (any of):
//   * RESERVED top-nibble envelopes (top nibble in
//     {0x9, 0xA, 0xB, 0xD, 0xF}) -- `Swizzle::EncBits` assigns no
//     semantics; hardware behavior is undefined.
//   * FFT_MODE imms with reserved bits 5..11 set -- within the FFT
//     envelope but outside the valid sub-encoding space; hardware
//     behavior likewise undefined.
//   * ROTATE_MODE imms with reserved bits 0..4 or 11 set -- same
//     argument.
//
// Refusing these (rather than silently passing them through to
// `llvm.amdgcn.ds.swizzle`, which would emit `ds_swizzle_b32
// offset:<imm>` and let the wave64 hardware do whatever it does
// for an undefined imm) matches the no-fallback contract.
//
// Check ordering note: the four valid-encoding checks are pairwise
// disjoint on the discriminator bits (bit 15 separates QUAD_PERM /
// BITMASK_PERM; the top nibble separates FFT_MODE / ROTATE_MODE
// from QUAD_PERM and from each other), so the order between them
// is irrelevant for correctness. The implicit "no match -> return
// false" branch correctly catches all the REFUSED categories.
bool dsSwizzleSafeForModRep(uint16_t Imm) {
  using namespace AMDGPU::Swizzle;
  if ((Imm & QUAD_PERM_ENC_MASK) == QUAD_PERM_ENC)
    return true;
  if ((Imm & BITMASK_PERM_ENC_MASK) == BITMASK_PERM_ENC)
    return true;
  // FFT_MODE: discriminator (top nibble = 0xE) AND every reserved
  // bit (5..11) clear. Equivalent to: imm & ~FFT_SWIZZLE_MASK ==
  // FFT_MODE_ENC, with the mask cast to uint16_t to keep the
  // bitwise-not within the 16-bit envelope (otherwise `~uint16_t`
  // promotes to int and would set bits 16..31 of the comparison
  // operand).
  if ((Imm & static_cast<uint16_t>(~FFT_SWIZZLE_MASK)) == FFT_MODE_ENC)
    return true;
  // ROTATE_MODE: discriminator (top nibble = 0xC) AND only the size
  // (bits 5..9) and direction (bit 10) bits set. Variable-bit mask
  // is (ROTATE_SIZE_MASK << ROTATE_SIZE_SHIFT) | (ROTATE_DIR_MASK
  // << ROTATE_DIR_SHIFT) = 0x7E0; everything else (including
  // reserved bits 0..4 and 11) must match ROTATE_MODE_ENC exactly.
  constexpr uint16_t ROTATE_VAR_MASK = (ROTATE_SIZE_MASK << ROTATE_SIZE_SHIFT) |
                                       (ROTATE_DIR_MASK << ROTATE_DIR_SHIFT);
  if ((Imm & static_cast<uint16_t>(~ROTATE_VAR_MASK)) == ROTATE_MODE_ENC)
    return true;
  // RESERVED top-nibble envelope, FFT/ROTATE with reserved bits
  // set, or any other non-valid encoding: refuse.
  return false;
}

struct LanePredicatedExecSite {
  const DecodedInst *Inst;
  ObstructionKind Kind; // CmpxFromLaneId or SaveExecFromLaneId.
  RewriteId Rewrite = RewriteId::None;
  bool RewriteImplemented = false;
  std::string Detail;
};

class LaneIdProvenanceTracker {
public:
  explicit LaneIdProvenanceTracker(const MCRegisterInfo &MRI) : MRI(MRI) {}

  bool anySourceTainted(const DecodedInst &Di) const {
    for (unsigned I = 0; I < Di.NumSrcs; ++I) {
      unsigned OpIdx = Di.SrcMap[I];
      if (OpIdx >= Di.Inst.getNumOperands())
        continue;
      const MCOperand &Op = Di.Inst.getOperand(OpIdx);
      if (Op.isReg() && isRegTainted(Op.getReg()))
        return true;
    }
    return false;
  }

  bool execTainted() const {
    return isRegTainted(AMDGPU::EXEC_LO) || isRegTainted(AMDGPU::EXEC_HI) ||
           isRegTainted(AMDGPU::EXEC);
  }

  bool anyVopdHalfSourceTainted(const DecodedInst::VopdHalf &Half) const {
    for (unsigned I = 0; I < Half.NumSrcs; ++I)
      if (vopdSourceTainted(Half.Src[I]))
        return true;
    return false;
  }

  void updateAfterVopd(const DecodedInst &Di) {
    for (const DecodedInst::VopdHalf &Half : Di.Vopd) {
      if (Half.CanonOp == CanonicalOp::Unknown || !Half.DstReg)
        continue;
      setRegTaint(Half.DstReg, anyVopdHalfSourceTainted(Half));
    }
  }

  void updateAfterInstruction(const DecodedInst &Di, bool ExplicitDefsTainted,
                              bool VccTainted, bool ExecTainted,
                              bool SccTainted) {
    for (unsigned I = 0; I < Di.NumDefs; ++I) {
      if (I >= Di.Inst.getNumOperands())
        continue;
      const MCOperand &Op = Di.Inst.getOperand(I);
      if (Op.isReg())
        setRegTaint(Op.getReg(), ExplicitDefsTainted);
    }
    if (Di.DefsVcc) {
      setRegTaint(AMDGPU::VCC_LO, VccTainted);
      setRegTaint(AMDGPU::VCC_HI, VccTainted);
      setRegTaint(AMDGPU::VCC, VccTainted);
    }
    if (Di.DefsExec) {
      setRegTaint(AMDGPU::EXEC_LO, ExecTainted);
      setRegTaint(AMDGPU::EXEC_HI, ExecTainted);
      setRegTaint(AMDGPU::EXEC, ExecTainted);
    }
    if (Di.DefsScc)
      setRegTaint(AMDGPU::SCC, SccTainted);
  }

private:
  void appendCanonicalRegLanes(MCRegister Reg,
                               SmallVectorImpl<unsigned> &Out) const {
    if (!Reg)
      return;
    MCRegister Lane = MRI.getSubReg(Reg, AMDGPU::sub0);
    if (!Lane)
      Lane = Reg;
    Out.push_back(static_cast<unsigned>(AMDGPU::mc2PseudoReg(Lane)));

    const unsigned MaxSubIdx = MRI.getNumSubRegIndices();
    for (unsigned SubIdx = AMDGPU::sub1; SubIdx < MaxSubIdx; ++SubIdx) {
      MCRegister Sub = MRI.getSubReg(Reg, SubIdx);
      if (!Sub)
        break;
      Out.push_back(static_cast<unsigned>(AMDGPU::mc2PseudoReg(Sub)));
    }
  }

  bool isRegTainted(MCRegister Reg) const {
    SmallVector<unsigned> Lanes;
    appendCanonicalRegLanes(Reg, Lanes);
    for (unsigned Lane : Lanes)
      if (TaintedRegs.contains(Lane))
        return true;
    return false;
  }

  bool vopdSourceTainted(const DecodedInst::VopdSource &Src) const {
    using Kind = DecodedInst::VopdSource::Kind;
    switch (Src.SrcKind) {
    case Kind::VGPR:
    case Kind::AGPR:
    case Kind::SGPR:
    case Kind::TTMP:
    case Kind::VCC:
    case Kind::EXEC:
    case Kind::SCC:
    case Kind::M0:
      return isRegTainted(Src.Reg);
    case Kind::Imm:
    case Kind::None:
      return false;
    }
    return false;
  }

  void setRegTaint(MCRegister Reg, bool Tainted) {
    SmallVector<unsigned> Lanes;
    appendCanonicalRegLanes(Reg, Lanes);
    for (unsigned Lane : Lanes) {
      if (Tainted)
        TaintedRegs.insert(Lane);
      else
        TaintedRegs.erase(Lane);
    }
  }

  const MCRegisterInfo &MRI;
  DenseSet<unsigned> TaintedRegs;
};

bool isSaveExecB32(CanonicalOp Sop) {
  return Sop == CanonicalOp::S_AND_SAVEEXEC_B32 ||
         Sop == CanonicalOp::S_OR_SAVEEXEC_B32 ||
         Sop == CanonicalOp::S_XOR_SAVEEXEC_B32 ||
         Sop == CanonicalOp::S_ANDN2_SAVEEXEC_B32 ||
         Sop == CanonicalOp::S_ORN2_SAVEEXEC_B32;
}

SmallVector<LanePredicatedExecSite>
findLanePredicatedExecSites(ArrayRef<DecodedInst> Insts,
                            const MCRegisterInfo &MRI,
                            const WaveProjection &Projection) {
  LaneIdProvenanceTracker Tracker(MRI);
  SmallVector<LanePredicatedExecSite> Sites;

  for (const DecodedInst &Di : Insts) {
    const CanonicalOp Sop = Di.CanonOp;
    if (Di.HasVopd) {
      Tracker.updateAfterVopd(Di);
      continue;
    }
    const bool SourceTainted = Tracker.anySourceTainted(Di);
    const bool OldExecTainted = Tracker.execTainted();

    bool ExplicitDefsTainted = SourceTainted;
    bool VccTainted = SourceTainted;
    bool ExecTainted = SourceTainted || OldExecTainted;
    bool SccTainted = SourceTainted;

    if (Sop == CanonicalOp::V_MBCNT_LO_U32_B32 ||
        Sop == CanonicalOp::V_MBCNT_HI_U32_B32) {
      ExplicitDefsTainted = true;
      VccTainted = false;
      ExecTainted = OldExecTainted;
      SccTainted = false;
    } else if (Sop == CanonicalOp::V_CMPX) {
      if (SourceTainted) {
        if (Projection.preservesMbcntDerivedVcmpxExec()) {
          Sites.push_back(
              {&Di, ObstructionKind::CmpxFromLaneId,
               RewriteId::WaveNativeMbcntCmpx, /*RewriteImplemented=*/true,
               "v_cmpx operand dataflow is derived from v_mbcnt_*; "
               "WaveNative lowers v_mbcnt_lo with a source-wave-local mask "
               "and ballots the compare into target-width EXEC storage"});
        } else {
          Sites.push_back(
              {&Di, ObstructionKind::CmpxFromLaneId, RewriteId::None,
               /*RewriteImplemented=*/false,
               "v_cmpx operand dataflow is derived from v_mbcnt_*; EXEC "
               "would need independent source-wave masks, but the selected "
               "projection aliases target lanes L and L+W_s through one "
               "source-width EXEC mask"});
        }
      }
      ExplicitDefsTainted = false;
      VccTainted = false;
      ExecTainted = OldExecTainted || SourceTainted;
      SccTainted = false;
    } else if (isSaveExecB32(Sop)) {
      if (SourceTainted) {
        if (Projection.preservesMbcntDerivedSaveExec()) {
          // WaveNative packs two DISTINCT source waves into one target wave
          // (lanes L and L+W_s carry different work), so the saveexec mask
          // needs an independent target-width EXEC mask. Project the
          // preceding v_cmp's ballot into target-width EXEC storage.
          Sites.push_back(
              {&Di, ObstructionKind::SaveExecFromLaneId,
               RewriteId::WaveNativeMbcntSaveExec, /*RewriteImplemented=*/true,
               "s_*_saveexec_b32 source mask dataflow is derived from "
               "v_mbcnt_*; WaveNative projects the preceding v_cmp's ballot "
               "into target-width EXEC storage (per-lane shadow), so the "
               "saveexec combine and old-EXEC save run at target width"});
        } else if (Projection.numSourceWavesPerTarget() == 1 ||
                   Projection.sourceWaveScopedLaneOps()) {
          // A single source wave per target wave (MODREP redundant-replica /
          // phantom-lane regime) or a source-wave-scoped iteration
          // (ThreadLoop, one source wave per pass). mbcnt lifting already
          // makes the lane id source-wave-relative (handle-valu-cross-lane:
          // mbcnt_hi pass-through + mbcnt_lo mod W_s), so a lane-derived
          // saveexec mask is already correct per source wave and needs no
          // target-width projection.
          Sites.push_back(
              {&Di, ObstructionKind::SaveExecFromLaneId,
               RewriteId::SaveExecLaneRelative, /*RewriteImplemented=*/true,
               "s_*_saveexec_b32 source mask dataflow is derived from "
               "v_mbcnt_*; the lane id is already source-wave-relative and "
               "one source wave maps to each target wave, so the mask is "
               "correct per source wave without a target-width projection"});
        } else {
          // A projection that packs DISTINCT source waves into lanes L and
          // L+W_s but offers no target-width saveexec projection: one
          // source-width mask cannot represent both waves' EXEC independently.
          Sites.push_back(
              {&Di, ObstructionKind::SaveExecFromLaneId,
               RewriteId::None, /*RewriteImplemented=*/false,
               "s_*_saveexec_b32 source mask dataflow is derived from "
               "v_mbcnt_*; the selected projection aliases target lanes L and "
               "L+W_s through one source-width EXEC mask, so no independent "
               "source-wave mask projection is representable"});
        }
      }
      ExplicitDefsTainted = OldExecTainted;
      VccTainted = false;
      ExecTainted = OldExecTainted || SourceTainted;
      SccTainted = ExecTainted;
    }

    Tracker.updateAfterInstruction(Di, ExplicitDefsTainted, VccTainted,
                                   ExecTainted, SccTainted);
  }

  return Sites;
}

// Return true iff any source-operand register of `di` covers the 32-bit
// TTMP8 lane -- either as a bare `ttmp8` or as part of a larger tuple
// (e.g. `ttmp[8:9]`). Defs are skipped: only reads of TTMP8 constitute a
// wave_id leak. The surrounding `(src.waveSize != tgt.waveSize)` check
// in the caller gates this to cross-widening only.
//
// Detection strategy (TableGen-authoritative, no enum arithmetic):
//   1. Find the TTMP_32 register class -- its register at position 8 IS
//      the generation-agnostic pseudo for `ttmp8` (the class is declared
//      `(add (sequence "TTMP%u", 0, 15))` in SIRegisterInfo.td so
//      position == index).
//   2. For every source reg operand (index >= di.numDefs), normalise
//      each 32-bit sub-register via `mc2PseudoReg` (strips subtarget
//      suffixes such as `TTMP8_gfx9plus` -> `TTMP8`) and compare against
//      the pseudo from step 1. A tuple like `ttmp[8:9]` contributes
//      sub0 = TTMP8, sub1 = TTMP9 -- we match on sub0.
bool readsTtmp8Source(const DecodedInst &Di, const MCRegisterInfo &MRI) {
  const MCRegisterClass &TTMP32 = MRI.getRegClass(AMDGPU::TTMP_32RegClassID);
  MCRegister Ttmp8Pseudo = TTMP32.getRegister(8);
  const llvm::MCInst &Inst = Di.Inst;
  for (unsigned I = Di.NumDefs, E = Inst.getNumOperands(); I < E; ++I) {
    const MCOperand &Op = Inst.getOperand(I);
    if (!Op.isReg())
      continue;
    MCRegister Reg = Op.getReg();
    if (!Reg)
      continue;
    // Walk 32-bit sub-registers. A 32-bit TTMP lane has no sub0 and
    // mc2PseudoReg normalises it directly; a TTMP pair (`ttmp[8:9]`)
    // has sub0 = TTMP8_aliased / sub1 = TTMP9_aliased and we match on
    // the sub0 lane.
    MCRegister Lane = MRI.getSubReg(Reg, AMDGPU::sub0);
    if (!Lane)
      Lane = Reg;
    if (AMDGPU::mc2PseudoReg(Lane) == Ttmp8Pseudo)
      return true;
    // Also check sub1..subN in case TTMP8 appears in the upper half of a
    // pair that starts earlier (unusual but possible in tuple-aligned
    // encodings).
    const unsigned MaxSubIdx = MRI.getNumSubRegIndices();
    for (unsigned SubIdx = AMDGPU::sub1; SubIdx < MaxSubIdx; ++SubIdx) {
      MCRegister S = MRI.getSubReg(Reg, SubIdx);
      if (!S)
        break;
      if (AMDGPU::mc2PseudoReg(S) == Ttmp8Pseudo)
        return true;
    }
  }
  return false;
}

// Return true iff `di` is the canonical gfx1250 HIP-emitted wave_id
// extraction pattern: `s_bfe_u32 sDST, ttmp8, 0x50019`. The immediate
// encodes (offset=25, width=5), which extracts bits [29:25] of ttmp8
// -- the command processor's `wave_id_in_workgroup` field.
//
// Pairs with the handle-sop2.cpp `S_BFE_U32` pattern-lift that emits
// `dst = (workitem.id.x >> log2(W_s)) & 0x1F` for this exact shape.
// The lift gives the BFE a principled meaning under cross-widening
// (each target lane gets its source-wave rank as a divergent VGPR),
// so we must NOT refuse on this shape here. Any OTHER ttmp8 read
// falls through to the `ttmp8ReadSites` path below.
bool isCanonicalWaveIdBfe(const DecodedInst &Di, const MCRegisterInfo &MRI) {
  if (Di.CanonOp != CanonicalOp::S_BFE_U32)
    return false;
  // S_BFE_U32 canonically has one destination (sDST), one source reg
  // (sSRC or ttmp), and one immediate control. Defs come first in the
  // MCInst operand list, sources follow. We want src0 = ttmp8 and
  // src1 = imm 0x50019.
  if (Di.NumSrcs < 2)
    return false;
  unsigned Src0Idx = Di.SrcMap[0];
  unsigned Src1Idx = Di.SrcMap[1];
  if (!Di.isReg(Src0Idx) || !Di.isImm(Src1Idx))
    return false;
  if (Di.getImm(Src1Idx) != 0x50019)
    return false;
  const MCRegisterClass &TTMP32 = MRI.getRegClass(AMDGPU::TTMP_32RegClassID);
  MCRegister Ttmp8Pseudo = TTMP32.getRegister(8);
  MCRegister Src0Reg = Di.Inst.getOperand(Src0Idx).getReg();
  if (!Src0Reg)
    return false;
  // Canonical pattern is a single `ttmp8` (not a tuple), so the
  // operand register itself should normalise to the TTMP8 pseudo.
  if (AMDGPU::mc2PseudoReg(Src0Reg) == Ttmp8Pseudo)
    return true;
  return false;
}

} // namespace

// ----------------------------------------------------------------------------
// buildObstructionReport -- the main walk.
// ----------------------------------------------------------------------------

ObstructionReport buildObstructionReport(ArrayRef<DecodedInst> Insts,
                                         const MCState &Mc,
                                         const WaveProjection &Projection,
                                         bool EnableWritelaneRewrite) {
  ObstructionReport Report;
  const ISAProfile &Src = Projection.sourceIsa();
  const ISAProfile &Tgt = Projection.targetIsa();
  if (Src.WaveSize == Tgt.WaveSize)
    return Report;
  const MCRegisterInfo &MRI = *Mc.RegInfo;

  // First pass: tag the self-contained obstruction kinds (lane-id
  // leaks, cross-lane shuffles, replica races). Class-4
  // lane-predicated EXEC sites are found by the decoded-register
  // provenance pre-walk below and emitted after the main walk.
  //
  // The walk matches purely on `CanonicalOp` and `MCInstrDesc` TSFlags
  // bits; there is no string matching on `rawMnemonic`. New
  // obstruction triggers should be added by extending canonical-op.h +
  // opcode-map.cpp (so the lookup is a single enum compare here),
  // not by adding `raw.contains(...)` substring tests.
  bool HaveWmma = false;
  SmallVector<LanePredicatedExecSite> LanePredicatedExecSites =
      findLanePredicatedExecSites(Insts, MRI, Projection);
  // Deferred TtmpWaveIdLeak site emission. The canonical shape --
  // `s_bfe_u32 sDST, ttmp8, 0x50019` -- has a principled rescue in
  // `handle-sop2.cpp`'s `S_BFE_U32` pattern-lift, which emits
  // `dst = (workitem.id.x >> log2(W_s)) & 0x1F` directly from the
  // divergent-leaf intrinsic and sidesteps the backend's implicit
  // scalarisation of the formally-scalar BFE -> SGPR path. The lift
  // preserves per-source-wave semantics across cross-widening, so
  // the canonical shape is NOT recorded here (filtered via
  // `isCanonicalWaveIdBfe`).
  //
  // Any OTHER ttmp8 source read (non-canonical immediates, AND /
  // LSHR / s_load offsets, trap-handler prologues, etc.) is still a
  // Class 1 leak: the raiser's ttmp8 init (`raiser.cpp` phase-4)
  // only models the `bits [29:25] = wave_id` field, so a consumer
  // that reads other bits or uses a different bitfield extract
  // semantics would silently miscompile. Those sites are collected
  // here; non-WMMA kernels have a future escape hatch through
  // `ThreadLoopProjection` (sec. 2.2 -- iterate the body R = W_t / W_s
  // times with a synthetic per-source-wave wave_id in ttmp8), and
  // WMMA kernels refuse because the sec. 5.2 lane layout requires the
  // full target wave simultaneously and cannot be TLP-split.
  llvm::SmallVector<const DecodedInst *> Ttmp8ReadSites;

  // Co-occurrence tracking for the WaveIdLiftScalarized refusal below.
  //
  //   - `canonicalWaveIdBfeSites`: every occurrence of the canonical
  //     `s_bfe_u32 sDST, ttmp8, 0x50019` pattern that the handle-sop2.cpp
  //     lift rescues by making sDST a per-lane divergent VGPR value
  //     (workitem.id.x >> log2(W_s)). That rescue is only semantically
  //     valid if the per-lane divergent value never feeds a source-ISA
  //     construct that enforces scalar-in-source semantics at the
  //     hardware level. `v_writelane_b32` / `v_readlane_b32` enforce
  //     exactly that scalar-in rule (the `src0` operand is an SGPR in
  //     the encoding), and the backend materialises that by inserting
  //     a readfirstlane on a divergent input -- which collapses target
  //     lanes 0..31 (source_wave[0]) with 32..63 (source_wave[1]) back
  //     to a single value. That collapse silently miscompiles every
  //     wave_id-dependent tile address the matmul encoded.
  //
  //   - `crossLaneScalarSites`: every `v_writelane_b32` / `v_readlane_b32`
  //     in the kernel. The refusal fires once per site so the diagnostic
  //     trace points at the precise instructions where the collapse
  //     happens, not just at the BFE where the divergent value was
  //     manufactured.
  //
  // Both buffers are emptied into `ObstructionReport::sites` after the
  // walk completes, gated on `haveWMMA` -- the non-WMMA case has a
  // future ThreadLoopProjection escape hatch (sec. 2.2; iterate the body
  // R = W_t / W_s times with a synthetic per-source-wave wave_id in
  // ttmp8) and must not be refused preemptively here. WMMA kernels
  // cannot use TLP because sec. 5.2 WMMA lane layout requires the full
  // target wave simultaneously, so the refusal is terminal.
  llvm::SmallVector<const DecodedInst *> CanonicalWaveIdBfeSites;
  llvm::SmallVector<const DecodedInst *> CrossLaneScalarSites;

  for (const DecodedInst &Di : Insts) {
    const CanonicalOp Sop = Di.CanonOp;

    // --- sec. 3 Class 1: wave_id leak via ttmp8 source read --------------
    // Under cross-widening, raiser.cpp seeds the transpiler's ttmp8
    // alloca from `workitem.id.x >> 5` shifted into bits [29:25] so
    // the per-lane value encodes the source's `wave_id_in_workgroup`.
    //
    // The canonical shape `s_bfe_u32 sDST, ttmp8, 0x50019` is rescued
    // inline by `handle-sop2.cpp`'s pattern-lift: it emits
    // `dst = (workitem.id.x >> log2(W_s)) & 0x1F` directly from the
    // divergent-leaf intrinsic, bypassing the backend's implicit
    // SGPR-class scalarisation. That shape is therefore NOT a
    // Class 1 refusal surface and is filtered out here.
    //
    // Any other source reference to ttmp8 (non-canonical BFE
    // immediates, `s_and_b32` / `s_lshr_b32` operating on ttmp8,
    // `s_load_dword` using ttmp8 as offset, trap-handler prologues
    // touching ttmp8..ttmp15, etc.) is still a leak: the raiser's
    // init only models the `[29:25] = wave_id` field, so consumers
    // of other bits read either zero or a garbage pattern. We defer
    // the site emission until after the loop has established whether
    // the kernel also contains WMMA (see below). Without WMMA the
    // leak is handled by ThreadLoopProjection; with WMMA it is
    // unrewritable (TLP and WMMA are mutually exclusive -- sec. 5.2 WMMA
    // lane layout requires the full target wave) and we refuse.
    if (readsTtmp8Source(Di, MRI) && !isCanonicalWaveIdBfe(Di, MRI))
      Ttmp8ReadSites.push_back(&Di);

    // Track canonical wave_id BFE sites for the WaveIdLiftScalarized
    // post-loop check. The lift in handle-sop2.cpp makes this BFE's
    // destination SGPR carry a per-lane divergent value (wave_id mod W_s)
    // instead of the backend's scalar BFE result; that rescue is
    // correct in isolation but collapses back to uniform when the
    // divergent value is consumed by any construct the source-ISA
    // encodes with scalar-in semantics (writelane src, readlane src).
    // The post-loop join below pairs these sites with the
    // crossLaneScalarSites + haveWMMA co-occurrence to decide the
    // refusal.
    if (isCanonicalWaveIdBfe(Di, MRI))
      CanonicalWaveIdBfeSites.push_back(&Di);

    // WMMA-family detection. If any of these show up in the kernel,
    // the WMMA -> MFMA lowering (matrix-translation.md) is going to be
    // invoked and the TLP escape hatch is not available -- every
    // deferred ttmp8 site in this kernel becomes an unrewritable
    // refusal surface.
    switch (Sop) {
    case CanonicalOp::V_WMMA_F32_16x16x32_F16:
    case CanonicalOp::V_WMMA_F32_16x16x32_BF16:
    case CanonicalOp::V_WMMA_F32_16x16x4_F32:
    case CanonicalOp::V_WMMA_F32_16x16x64_FP8_FP8:
    case CanonicalOp::V_WMMA_F32_16x16x64_FP8_BF8:
    case CanonicalOp::V_WMMA_F32_16x16x64_BF8_FP8:
    case CanonicalOp::V_WMMA_F32_16x16x64_BF8_BF8:
    case CanonicalOp::V_WMMA_I32_16x16x64_IU8:
    case CanonicalOp::V_WMMA_SCALE_F32_16x16x128_F8F6F4:
      HaveWmma = true;
      break;
    default:
      break;
    }

    // --- sec. 3 Class 1: absolute lane-ID leaks --------------------------
    if (Sop == CanonicalOp::V_MBCNT_HI_U32_B32) {
      // For wave32 source widened to wave64, mbcnt_hi is a pass-through
      // of src1: the hi-half mask `(1 << max(0, L - 32)) - 1` is empty
      // for every source lane L in [0, 31], so the popcount is 0 and
      // the result equals src1. The cross-widen handler emits src1
      // directly, so there is no lane-ID leak. Other directions
      // (e.g. wave64 source narrowed to wave32 target) do still leak
      // because exec_hi has no equivalent on the narrower target.
      if (Src.isWave32() && Tgt.WaveSize > Src.WaveSize)
        continue;
      ObstructionSite Site;
      Site.Inst = &Di;
      Site.Kind = ObstructionKind::MbcntHiLaneIdLeak;
      Site.Rewrite = RewriteId::None;
      Site.RewriteImplemented = false;
      Site.Detail = "v_mbcnt_hi reads target exec_hi -- no wave32 semantics "
                    "to preserve under modulo-replication";
      Report.Sites.push_back(std::move(Site));
      continue;
    }
    if (Sop == CanonicalOp::V_MBCNT_LO_U32_B32) {
      // v_mbcnt_lo alone is not a leak by itself (wave32 sources use
      // it as the canonical lane-id probe and it's lane-position-
      // independent inside its wave). The C4 provenance pre-walk above
      // tracks whether its result actually reaches an EXEC writer; a
      // standalone lane-index probe is not an obstruction.
      continue;
    }
    if (Sop == CanonicalOp::V_READLANE_B32 ||
        Sop == CanonicalOp::V_WRITELANE_B32) {
      // Track every readlane/writelane -- in-bounds or otherwise -- for
      // the WaveIdLiftScalarized post-loop check. Out-of-range static
      // lane operands additionally emit an OutOfRangeLaneOperand site
      // in the block below; the two conditions are independent (a
      // kernel could have a well-formed writelane whose `val` operand
      // carries a wave_id-derived divergent value, and that refusal
      // must fire even when every lane operand is in-bounds).
      CrossLaneScalarSites.push_back(&Di);

      auto Imm = extractLaneOperandImm(Di);
      // Negative-value guard: an int64_t imm cast to uint64_t for the
      // bounds compare wraps around to a value > waveSize for any
      // negative value, which is the correct logical answer (negative
      // lane indices are never in [0, W_s)) but is implicit in the
      // cast. Make it explicit so the intent survives a refactor.
      if (Imm.has_value() &&
          (*Imm < 0 || static_cast<uint64_t>(*Imm) >= Src.WaveSize)) {
        // Static constant operand provably out of source wave range.
        // No rewrite preserves the semantics on a wider target wave.
        ObstructionSite Site;
        Site.Inst = &Di;
        Site.Kind = ObstructionKind::OutOfRangeLaneOperand;
        Site.Rewrite = RewriteId::None;
        Site.RewriteImplemented = false;
        std::string Det;
        raw_string_ostream Os(Det);
        Os << "operand value " << *Imm << " out of [0, W_s=" << Src.WaveSize
           << ")";
        Site.Detail = Os.str();
        Report.Sites.push_back(std::move(Site));
      }
      // Otherwise (static imm < W_s, or dynamic operand): do not
      // emit a site.
      //
      // - Static imm < W_s: provably in-bounds, safe by construction.
      // - Dynamic operand (SGPR): we cannot statically prove the
      //   runtime value is < W_s, BUT we also cannot prove it is out
      //   of bounds. Triton softmax / matmul patterns use
      //   `v_writelane_b32` with
      //   dynamic lane operands that are in-bounds at runtime but not
      //   statically provable. Flagging those as refusal would
      //   collapse coverage on every cross-target kernel that uses
      //   them.
      //
      // TODO(dataflow-upgrade): graduate dynamic operands from "not
      // flagged" to "proved via LLVM uniformity / value-range
      // analysis on the raised IR" once the post-raise dataflow
      // analysis lands. Today this is a sound-not-complete choice
      // toward false negatives on readlane/writelane specifically --
      // tracked in wave-size-obstruction.h's TODO block.
      continue;
    }

    // --- sec. 3 Class 2: wave-width-specific cross-lane shuffles --------
    if (Sop == CanonicalOp::V_PERMLANE64_B32) {
      ObstructionSite Site;
      Site.Inst = &Di;
      Site.Kind = ObstructionKind::FullWaveRotate;
      Site.Rewrite = RewriteId::None;
      Site.RewriteImplemented = false;
      Site.Detail = "v_permlane64 has no wave32 analogue";
      Report.Sites.push_back(std::move(Site));
      continue;
    }
    if (Sop == CanonicalOp::V_PERMLANE16_B32 ||
        Sop == CanonicalOp::V_PERMLANEX16_B32 ||
        Sop == CanonicalOp::V_PERMLANE16_SWAP_B32 ||
        Sop == CanonicalOp::V_PERMLANE32_SWAP_B32) {
      ObstructionSite Site;
      Site.Inst = &Di;
      Site.Kind = ObstructionKind::LaneGroupShuffle;
      // P2 covers base permlane16/permlanex16; P4 covers the swap
      // variants. `rewriteImplemented` is set per-CanonicalOp so the
      // decider emits a precise "cross-wave-shuffle-rewrite-pending"
      // diagnostic for the specific P-item still missing.
      //
      // P2 (base permlane16/permlanex16) landed:
      // `handle-valu-cross-lane.cpp` emits ds_bpermute-emulated
      // permlane16/permlanex16 with fi / bc extracted from the
      // VOP3 src{0,1}_modifiers' OP_SEL_0 bit.
      //
      // P4 (permlane16_swap_b32) landed: ds_bpermute-emulated
      // partner = lane_id XOR 16, two bpermutes for the two-VGPR
      // exchange. The wider permlane32_swap_b32 variant stays
      // unrewritable -- its XOR-32 partner spans wave64's two
      // 32-lane halves, which has no wave32 analogue, so a wave32
      // source kernel cannot meaningfully encode it. Seeing it in
      // a wave32 source binary indicates either a corrupted
      // disassembly or a wave64 source mis-classified as wave32.
      if (Sop == CanonicalOp::V_PERMLANE32_SWAP_B32) {
        Site.Rewrite = RewriteId::P4_PermLaneSwap;
        Site.RewriteImplemented = false;
        Site.Detail = "v_permlane32_swap_b32: XOR-32 partner spans "
                      "wave64 32-lane halves; no wave32 analogue";
      } else if (Sop == CanonicalOp::V_PERMLANE16_SWAP_B32) {
        Site.Rewrite = RewriteId::P4_PermLaneSwap;
        Site.RewriteImplemented = true;
      } else {
        Site.Rewrite = RewriteId::P2_PermLane16;
        Site.RewriteImplemented = true;
      }
      Report.Sites.push_back(std::move(Site));
      continue;
    }
    if (Sop == CanonicalOp::DS_SWIZZLE_B32) {
      ObstructionSite Site;
      Site.Inst = &Di;
      Site.Kind = ObstructionKind::DsSwizzle;
      Site.Rewrite = RewriteId::P6_DsSwizzle;
      // P6 landed (see the DS_SWIZZLE_B32 row of hotswap/docs/wave-
      // size-translation.md sec. 5.3): the handler in handle-ds.cpp emits
      // `llvm.amdgcn.ds.swizzle(value, offset)`
      // with the 16-bit immediate plumbed through. The lift is only
      // wave-size-oblivious for the QUAD_PERM and BITMASK_PERM
      // sub-modes (see `dsSwizzleSafeForModRep` above for the
      // structural argument); FFT_MODE / ROTATE_MODE / unknown-high-
      // nibble imms remain pending and refuse loudly via the same
      // CrossWaveShuffleRewritePending channel as before.
      //
      // The 16-bit imm is extracted at decode time into
      // `di.dsSwizzleImm` (see `decode.cpp::decodeDsSwizzleImm`).
      // `!di.hasDsSwizzleImm` here means the decoder rejected the
      // operand (missing, non-immediate, or out of 16-bit range);
      // we mirror that rejection as a P6-pending refusal with a
      // malformed-disassembly diagnostic so the kernel fails loudly
      // rather than the handler inventing a value.
      if (!Di.HasDsSwizzleImm) {
        Site.RewriteImplemented = false;
        Site.Detail = "ds_swizzle_b32 missing/invalid OpName::offset "
                      "immediate operand -- disassembly malformed or "
                      "outside 16-bit range";
      } else {
        Site.RewriteImplemented = dsSwizzleSafeForModRep(Di.DsSwizzleImm);
        if (!Site.RewriteImplemented) {
          std::string Det;
          raw_string_ostream Os(Det);
          Os << "ds_swizzle_b32 imm 0x"
             << format_hex_no_prefix(Di.DsSwizzleImm, 4)
             << " is not a valid swizzle encoding (not QUAD_PERM, "
                "BITMASK_PERM, valid FFT_MODE, or valid ROTATE_MODE) "
                "-- RESERVED top-nibble or FFT/ROTATE reserved bits "
                "set; AMDGPU hardware semantics undefined";
          Site.Detail = Os.str();
        }
      }
      Report.Sites.push_back(std::move(Site));
      continue;
    }
    // DPP detection via the MCInstrDesc TSFlags bit. opcode-map.cpp
    // canonicalises DPP variants down to their base CanonicalOp, so the
    // CanonicalOp alone cannot identify them, but `di.tsFlags` is captured
    // from the *original* MCInstrDesc (see decode.cpp) so the DPP
    // bit is still visible. Same for SDWA -- though SDWA is same-lane
    // and not a cross-wave concern, so we don't flag it.
    if (Di.TsFlags & SIInstrFlags::DPP) {
      ObstructionSite Site;
      Site.Inst = &Di;
      Site.Kind = ObstructionKind::DppCrossLane;
      Site.Rewrite = RewriteId::P5_DppModifier;
      // P5 (DPP16 lift via `llvm.amdgcn.update.dpp`) landed for the
      // DPP16 encoding family when FI is not set. `decodeDppModifiers`
      // sets `di.hasDpp = true` only after successfully extracting every
      // required DPP16 modifier operand (dpp_ctrl / row_mask / bank_mask /
      // bound_ctrl). DPP8 (and any future DPP layout the decoder does not
      // recognise yet) leaves `hasDpp` false by design so the classifier
      // refuses rather than partially decoding. DPP16 FI is decoded but not
      // representable by `llvm.amdgcn.update.dpp`, so it is treated as pending
      // too. The `tsFlags & SIInstrFlags::DPP` check still fires for all forms
      // -- all are Class-2 cross-lane sites by the hotswap/docs/wave-size-
      // translation.md sec. 6 taxonomy; the flipped-by-form
      // `rewriteImplemented` bit separates "handled" from "pending" without
      // changing the taxonomy.
      Site.RewriteImplemented = Di.HasDpp && !Di.DppFi;
      if (!Di.HasDpp)
        Site.Detail =
            "DPP cross-lane site (TSFlags::DPP) but `decodeDppModifiers` did "
            "not populate the DPP16 bundle (`hasDpp == false`) -- includes "
            "DPP8 encodings and any newer AMDGPU DPP operand layout this "
            "decoder does not yet handle; pending P5 / decode extension.";
      else if (Di.DppFi)
        Site.Detail =
            "DPP16 FI fetch-inactive form -- llvm.amdgcn.update.dpp has no "
            "FI operand; extending P5 requires modelling Table 57 "
            "fetch-inactive semantics rather than silently dropping FI.";
      Report.Sites.push_back(std::move(Site));
      continue;
    }
    if (Sop == CanonicalOp::DS_BPERMUTE_B32) {
      // P1 is IMPLEMENTED in handle-ds.cpp (see lit_tests/ds_bpermute_b32).
      // Record the site so the trace shows it, but mark as
      // `rewriteImplemented = true` so the decider treats it as
      // outcome (a)/(b) rather than refusal.
      ObstructionSite Site;
      Site.Inst = &Di;
      Site.Kind = ObstructionKind::DsBpermuteGather;
      Site.Rewrite = RewriteId::P1_DsBpermute;
      Site.RewriteImplemented = true;
      Report.Sites.push_back(std::move(Site));
      continue;
    }

    // --- sec. 3 Class 3: replica races on shared state ------------------
    // The CanonicalOp set here is the complete enumeration of
    // non-commutative atomics modeled in canonical-op.h today. New
    // non-commutative atomic encodings (e.g. SCRATCH_ATOMIC_SWAP if we
    // ever model it) should be added by extending the enum +
    // opcode-map.cpp + semop.cpp, not by adding a substring check
    // here. Atomics not yet modeled refuse via the existing Phase 5
    // unsupportedOpcode path.
    //
    // Vector atomics (GLOBAL / FLAT / BUFFER _SWAP / _CMPSWAP): the
    // race is *lane-level* -- under modulo-replication the wave64
    // source is projected onto two wave32 sub-waves, so lanes `i` and
    // `i + W_s` issue concurrently against the same target slot. For
    // non-commutative binops the two possible orderings produce
    // different terminal values, and the source program has no way to
    // restore the intended single-wave ordering.
    if (Sop == CanonicalOp::GLOBAL_ATOMIC_SWAP ||
        Sop == CanonicalOp::GLOBAL_ATOMIC_CMPSWAP ||
        Sop == CanonicalOp::FLAT_ATOMIC_SWAP ||
        Sop == CanonicalOp::FLAT_ATOMIC_CMPSWAP ||
        Sop == CanonicalOp::FLAT_ATOMIC_SWAP_X2 ||
        Sop == CanonicalOp::FLAT_ATOMIC_CMPSWAP_X2 ||
        Sop == CanonicalOp::BUFFER_ATOMIC_SWAP ||
        Sop == CanonicalOp::BUFFER_ATOMIC_CMPSWAP) {
      ObstructionSite Site;
      Site.Inst = &Di;
      Site.Kind = ObstructionKind::NonCommutativeAtomic;
      Site.Rewrite = RewriteId::None;
      Site.RewriteImplemented = false;
      Site.Detail = "non-commutative vector atomic races target lanes "
                    "i and i+W_s under modulo-replication";
      // Rescue: a *store-only* (non-returning, numDefs==0) SWAP under a
      // one-source-wave-per-target MODREP projection has no cross-lane
      // race that survives predication -- lanes i and i+W_s are redundant
      // copies of the same source lane, so gating the atomic to replica-0
      // (`lane_id < W_s`, emitted in handle-flat.cpp) leaves
      // exactly one issue per source lane, identical to native wave32.
      // A *returning* swap (numDefs>0) is NOT rescued here: replica-1
      // lanes would lose the returned `old` value, which needs an
      // explicit replica-0 -> replica-1 broadcast we do not emit.
      // CMPSWAP is likewise left to refuse (returning form / compare
      // semantics). The projection must be MODREP (numSourceWavesPerTarget
      // ==1); WaveNative (==2) keeps lanes 32..63 as distinct real work
      // and must not be gated.
      const bool StoreOnly = Di.NumDefs == 0;
      const bool IsSwap = Sop == CanonicalOp::GLOBAL_ATOMIC_SWAP ||
                          Sop == CanonicalOp::FLAT_ATOMIC_SWAP ||
                          Sop == CanonicalOp::BUFFER_ATOMIC_SWAP;
      const bool ModRepOneWave = Projection.numSourceWavesPerTarget() == 1 &&
                                 !Projection.providesFullWaveExecInvariant();
      if (StoreOnly && IsSwap && ModRepOneWave) {
        Site.Rewrite = RewriteId::AtomicOneReplica;
        Site.RewriteImplemented = true;
        Site.Detail = "store-only vector atomic swap gated to MODREP "
                      "replica-0 (lane_id < W_s); one issue per source lane";
      }
      Report.Sites.push_back(std::move(Site));
      continue;
    }
    // Scalar atomics (S_ATOMIC_*): the race is *wave-level*, not
    // lane-level. The scalar unit executes the atomic exactly once per
    // wave, so there is no lane-i/lane-i+W_s race by construction; the
    // failure mode is that each source wave becomes two target
    // sub-waves under wave64 -> 2 x wave32 projection and each
    // sub-wave's scalar unit issues the atomic independently, so the
    // counter receives two updates per source wave instead of one.
    //
    //   * S_ATOMIC_SWAP under double-issue writes the intended value
    //     twice (idempotent in value, but the returned `old` the
    //     second sub-wave sees is the data the first wrote, not the
    //     pre-instruction memory value).
    //   * S_ATOMIC_DEC under double-issue decrements twice, and both
    //     sub-waves see pre-decrement values that are off-by-one from
    //     what the source program computed -- fatal for the AITER
    //     split-k "am I the last workgroup?" barrier idiom keyed on
    //     `old == 1`, which is the dominant corpus consumer today.
    //
    // A genuine rewrite would need to mask the atomic to one of the
    // two sub-waves (EXEC-gated single issuance); we haven't
    // implemented that, so refuse.
    if (Sop == CanonicalOp::S_ATOMIC_SWAP || Sop == CanonicalOp::S_ATOMIC_DEC) {
      ObstructionSite Site;
      Site.Inst = &Di;
      Site.Kind = ObstructionKind::NonCommutativeAtomic;
      Site.Rewrite = RewriteId::None;
      Site.RewriteImplemented = false;
      Site.Detail = "non-commutative scalar atomic double-issues under "
                    "wave64 -> 2 x wave32 modulo-replication (each "
                    "target sub-wave's scalar unit fires the atomic "
                    "independently)";
      Report.Sites.push_back(std::move(Site));
      continue;
    }

    // --- sec. 3 Class 4: lane-predicated EXEC writers -------------------
    // Handled by `findLanePredicatedExecSites` above.  The main walk
    // intentionally does not emit C4 sites from mere opcode presence:
    // kernels often contain `v_mbcnt_*` for ds_bpermute selectors and
    // unrelated bounds/saveexec masks in the same instruction stream.
    // Only actual decoded-register dataflow from mbcnt into an EXEC
    // writer is an obstruction.
  }

  // Deferred TtmpWaveIdLeak emission. See the pre-loop comment on
  // `ttmp8ReadSites` for the rationale: the `s_bfe_u32 ttmp8, 0x50019`
  // wave_id extraction is clang/hip boilerplate in every non-trivial
  // gfx1250 kernel, so unconditionally refusing on it would collapse
  // coverage. We only refuse when the kernel also contains WMMA --
  // in which case ThreadLoopProjection (the sec. 2.2 escape hatch for
  // class-4 wave_id leaks) cannot be applied because the sec. 5.2 WMMA
  // lane layout requires the full target wave simultaneously. In the
  // non-WMMA case, fall through silently; the caller's projection
  // selector will pick TLP in raiser.cpp.
  if (HaveWmma) {
    for (const DecodedInst *Di : Ttmp8ReadSites) {
      ObstructionSite Site;
      Site.Inst = Di;
      Site.Kind = ObstructionKind::TtmpWaveIdLeak;
      Site.Rewrite = RewriteId::None;
      Site.RewriteImplemented = false;
      Site.Detail =
          "source reads ttmp8 under cross-widening -- bits [29:25] carry "
          "wave_id_in_workgroup, which is a function of the target's "
          "absolute lane position (not of lane_id mod W_s). Kernel also "
          "contains WMMA, so ThreadLoopProjection is not available -- refuse.";
      Report.Sites.push_back(std::move(Site));
    }
  }

  // WaveIdLiftScalarized -- the canonical-BFE rescue collapses inside a
  // cross-lane scalar primitive under WMMA.
  //
  // This is the Matmul128x128 / `matmul_f16_large_gfx1250` pattern:
  //
  //     s_bfe_u32 s2, ttmp8, 0x50019          ; lifted -> divergent wave_id
  //     s_and_b32 s73, s2, 3                  ; tainted (SGPR but divergent)
  //     s_lshl_b32 s18, s2, 5                 ; tainted
  //     v_writelane_b32 vgpr256, s18, 4       ; backend readfirstlane(s18)
  //                                           ; collapses target lanes
  //                                           ; 0..31 with 32..63 to a
  //                                           ; single value -> per-
  //                                           ; source-wave tile offset
  //                                           ; LOST.
  //     ...
  //     v_wmma_f32_16x16x32_f16 ...             ; WMMA -> TLP not available.
  //     ...
  //     v_readlane_b32 sDST, vgpr256, 4       ; reads the collapsed value.
  //     v_or_b32 v_col, sDST, v_col_within    ; per-source-wave column
  //                                           ; base is now uniform,
  //                                           ; writing the wrong tile.
  //
  // The refusal is syntactic (three-way co-occurrence over the kernel)
  // and therefore a sound-not-complete over-approximation.  Class-4
  // CmpxFromLaneId / SaveExecFromLaneId uses the provenance pre-walk
  // above; this separate WaveIdLiftScalarized shape remains intentionally
  // co-occurrence-based until a post-raise SSA query owns that path too.
  // Kernels that happen to contain all three constructs but do NOT
  // route the wave_id value into the cross-lane scalar source would be
  // over-refused -- benign (false positive). Kernels that are missing
  // any of the three would not be refused here but ALSO cannot express
  // the matmul-shaped wave_id-dependent tile column bug (no canonical
  // BFE means ttmp8 is only read via other shapes, which fall to the
  // ttmp8ReadSites + WMMA refusal above; no cross-lane scalar means
  // the divergent value has no scalar-semantics consumer to collapse
  // into; no WMMA means TLP is available for the class-4 escape). The
  // implications chain is safe by construction.
  //
  // TODO(dataflow-upgrade): replace the syntactic co-occurrence with a
  // precise check that the BFE's destination SGPR flows (through the
  // raised IR's SSA uses) into a v_writelane / v_readlane scalar
  // source operand. The LLVM Uniformity Analysis on the raised IR
  // (post-Phase-2) is the natural place to land this -- see
  // wave-size-obstruction.h's TODO block.
  if (HaveWmma && !CanonicalWaveIdBfeSites.empty() &&
      !CrossLaneScalarSites.empty()) {
    for (const DecodedInst *Di : CrossLaneScalarSites) {
      ObstructionSite Site;
      Site.Inst = Di;
      Site.Kind = ObstructionKind::WaveIdLiftScalarized;
      // When `enableWritelaneRewrite` is on, the site has an
      // implemented rewrite (the post-mem2reg pass in
      // `rewrite_cross_lane_divergent.{hpp,cpp}` replaces the
      // collapsing cross-lane primitive with a per-source-wave
      // `select` / `ds.bpermute`). Tag it accordingly so the
      // pre-translation abort below does NOT fire -- the rewrite
      // discharges the obstruction during Phase 6.5 of raiser.cpp.
      // Paired with a post-raise safety net in raiser.cpp that
      // verifies the rewrite pass actually rewrote at least one
      // site (guards against an oracle false-negative disagreeing
      // with this syntactic co-occurrence classifier).
      if (EnableWritelaneRewrite) {
        Site.Rewrite = RewriteId::PostRaiseCrossLaneRewrite;
        Site.RewriteImplemented = true;
      } else {
        Site.Rewrite = RewriteId::None;
        Site.RewriteImplemented = false;
      }
      Site.Detail =
          "kernel also contains the canonical `s_bfe_u32 sDST, ttmp8, "
          "0x50019` wave_id lift and v_wmma_* -- the lift's per-lane "
          "divergent result is scalarised by the backend on entry to "
          "this cross-lane primitive's scalar source operand, "
          "collapsing source_wave[0]'s and source_wave[1]'s distinct "
          "values into a single uniform. WMMA forecloses the "
          "ThreadLoopProjection escape hatch (sec. 5.2 requires the full "
          "target wave simultaneously), so no correct projection is "
          "available.";
      Report.Sites.push_back(std::move(Site));
    }
  }

  // Second pass: emit Class-4 EXEC writers whose predicate/mask was actually
  // proven to depend on a v_mbcnt_* result by the decoded-register provenance
  // pre-walk. findLanePredicatedExecSites already picks the projection-aware
  // rewrite for each site (WaveNative target-width projection, lane-relative
  // lift for single-source-wave / source-wave-scoped projections, or refuse
  // when neither applies), so the decision is copied verbatim here.
  for (const auto &Pw : LanePredicatedExecSites) {
    ObstructionSite Site;
    Site.Inst = Pw.Inst;
    Site.Kind = Pw.Kind;
    Site.Rewrite = Pw.Rewrite;
    Site.RewriteImplemented = Pw.RewriteImplemented;
    Site.Detail = Pw.Detail;
    Report.Sites.push_back(std::move(Site));
  }

  return Report;
}

// ----------------------------------------------------------------------------
// Rendering -- stable-enough-for-lit trace format.
// ----------------------------------------------------------------------------

std::string renderObstructionTrace(const ObstructionReport &Report,
                                   StringRef KernelName, StringRef SrcIsa,
                                   StringRef TgtIsa, unsigned SrcWaveSize,
                                   unsigned TgtWaveSize) {
  std::string Out;
  raw_string_ostream Os(Out);

  Os << "transpiler: projection decision for kernel '" << KernelName << "':\n";
  Os << "  source: " << SrcIsa << " (wave" << SrcWaveSize
     << ") -> target: " << TgtIsa << " (wave" << TgtWaveSize
     << "), R=" << (SrcWaveSize > 0 ? TgtWaveSize / SrcWaveSize : 0) << "\n";

  if (Report.Sites.empty()) {
    Os << "  obstructions found: none\n"
       << "  outcome: (a) wave-size-oblivious -- emit selected projection\n";
    return Out;
  }

  Os << "  obstructions found:\n";
  for (const ObstructionSite &S : Report.Sites) {
    Os << "    " << obstructionKindName(S.Kind);
    if (S.Inst) {
      Os << " @ 0x" << format_hex_no_prefix(S.Inst->Offset, 4) << ": "
         << S.Inst->RawMnemonic;
    }
    Os << "\n      rewrite: " << rewriteIdName(S.Rewrite);
    if (S.Rewrite != RewriteId::None)
      Os << " [" << (S.RewriteImplemented ? "implemented" : "pending") << "]";
    if (!S.Detail.empty())
      Os << "\n      detail: " << S.Detail;
    Os << "\n";
  }

  if (Report.hasUnrewritable()) {
    Os << "  outcome: (c) refuse -- at least one obstruction has no rewrite "
          "in wave-size-translation.md sec. 7's unrewritable table\n";
  } else if (Report.hasPendingRewrite()) {
    Os << "  outcome: (c) refuse -- rewrite(s) exist on paper but the "
          "matching handler(s) have not yet landed "
          "(wave-size-translation.md sec. 7's pending-rewrite table)\n";
  } else {
    Os << "  outcome: (b) rewrite-then-emit -- all obstruction sites have "
          "an implemented rewrite/projection; emit selected projection\n";
  }
  return Out;
}

// ----------------------------------------------------------------------------
// Failure selection -- pick the first refusal-worthy site and package
// it as a RaiseFailure for raiser.cpp to propagate.
// ----------------------------------------------------------------------------

llvm::Error selectFailureFromReport(const ObstructionReport &Report) {
  // Prefer unrewritable over pending -- the caller should see the
  // strongest refusal reason first. Ties broken by decoded order (the
  // `sites` vector is in decoded order, so `firstUnrewritable` /
  // `firstPending` both return the earliest match).
  //
  // Twine lifetime: each `Twine(...) + ... + ...` chain is built and
  // consumed in the SAME full-expression as the factory call below.
  // This is the LLVM-supported lifetime contract -- Twine concat
  // results are temporaries that hold references into their operands
  // and *must not* be bound to a named variable (`const Twine x = a +
  // b + c` would leave `x` referencing temporaries that are destroyed
  // at the end of that statement). See the LLVM Programmer's Manual
  // on Twine.
  if (const ObstructionSite *Site = Report.firstUnrewritable()) {
    switch (Site->Kind) {
    case ObstructionKind::MbcntHiLaneIdLeak:
    case ObstructionKind::OutOfRangeLaneOperand:
    case ObstructionKind::TtmpWaveIdLeak:
    case ObstructionKind::WaveIdLiftScalarized:
      return RaiseFailure::crossWaveLaneIdLeak(
          *Site->Inst,
          Twine(obstructionKindName(Site->Kind)) + " [" + Site->Detail + "]");
    case ObstructionKind::FullWaveRotate:
      return RaiseFailure::crossWaveUnrewritableShuffle(
          *Site->Inst,
          Twine(obstructionKindName(Site->Kind)) + " [" + Site->Detail + "]");
    case ObstructionKind::NonCommutativeAtomic:
      return RaiseFailure::crossWaveReplicaRace(
          *Site->Inst,
          Twine(obstructionKindName(Site->Kind)) + " [" + Site->Detail + "]");
    case ObstructionKind::CmpxFromLaneId:
    case ObstructionKind::SaveExecFromLaneId:
      return RaiseFailure::crossWaveLanePredicatedExec(
          *Site->Inst,
          Twine(obstructionKindName(Site->Kind)) + " [" + Site->Detail + "]");
    // The kinds below never set `rewrite = None` under
    // buildObstructionReport, so they cannot reach firstUnrewritable().
    // If they do, our state is inconsistent -- fail loudly rather than
    // silently fall through to the empty-RaiseFailure return below.
    case ObstructionKind::LaneGroupShuffle:
    case ObstructionKind::DsSwizzle:
    case ObstructionKind::DppCrossLane:
    case ObstructionKind::DsBpermuteGather:
    case ObstructionKind::None:
      llvm_unreachable("ObstructionKind classified as unrewritable but "
                       "buildObstructionReport never tags it that way");
    case ObstructionKind::WorkitemIdPredicateChain:
      // Produced only by the post-mem2reg IR-level classifier in
      // `c5-predicate-chain-classifier.cpp`; that classifier surfaces
      // its own `RaiseFailure::crossWavePredicateChain` directly from
      // raiser.cpp Phase 6.6. buildObstructionReport's MC-level walk
      // cannot see `workitem.id.x` emission (it's an IR-level
      // intrinsic call, not a source-side CanonicalOp), so tagging a
      // DecodedInst with this kind is a contract violation.
      // See hotswap/docs/modrep-predicate-chain.md sec. 5.
      llvm_unreachable(
          "WorkitemIdPredicateChain is produced only by the IR-level "
          "classifier (c5-predicate-chain-classifier.cpp); "
          "buildObstructionReport must not tag a DecodedInst with it");
    }
    llvm_unreachable("unhandled ObstructionKind in selectFailureFromReport "
                     "(unrewritable branch)");
  }
  if (const ObstructionSite *Site = Report.firstPending()) {
    return RaiseFailure::crossWaveShuffleRewritePending(
        *Site->Inst, Twine(obstructionKindName(Site->Kind)) + " [rewrite " +
                         rewriteIdName(Site->Rewrite) + " pending]");
  }
  // Oblivious / fully-rewritten: no failure.
  return llvm::Error::success();
}

} // namespace COMGR::hotswap
