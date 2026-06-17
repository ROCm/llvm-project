//===- raise-context.cpp - Hotswap transpiler -----------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "raise-context.h"

#include "MCTargetDesc/AMDGPUMCTargetDesc.h" // AMDGPU::VCC, AMDGPU::EXEC, ...
#include "SIDefines.h"                        // AMDGPU::HWEncoding::*
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>

using namespace llvm;

namespace COMGR::hotswap {

BasicBlock *RaiseContext::lookupBB(uint64_t Addr) {
  auto It = OffsetToBb.find(Addr);
  if (It != OffsetToBb.end())
    return It->second;
  errs() << "transpiler: missing basic block for offset 0x" << utohexstr(Addr)
         << " -- creating fallback\n";
  BasicBlock *Bb =
      BasicBlock::Create(C, "bb_fallback_0x" + utohexstr(Addr), Kernel);
  OffsetToBb[Addr] = Bb;
  return Bb;
}

void RaiseContext::computeVGPRAdjust(const DecodedInst &Di) {
  std::fill_n(CurrentVgprAdjust, KMaxOps, 0u);
  if (VgprMsBs == 0)
    return;

  // vgprMSBs is an 8-bit state shared by single-issue instructions and both
  // halves of a VOPD pair.  Layout: src0[1:0], src1[3:2], src2[5:4], dst[7:6].
  unsigned DstMsb = (static_cast<unsigned>(VgprMsBs >> 6) & 0x3u) * 256u;
  unsigned SrcMsb[3] = {
      (static_cast<unsigned>(VgprMsBs >> 0) & 0x3u) * 256u,
      (static_cast<unsigned>(VgprMsBs >> 2) & 0x3u) * 256u,
      (static_cast<unsigned>(VgprMsBs >> 4) & 0x3u) * 256u,
  };

  for (unsigned I = 0; I < Di.NumDefs && I < KMaxOps; I++)
    CurrentVgprAdjust[I] = DstMsb;

  for (unsigned I = 0; I < Di.NumSrcs && I < 3; I++) {
    unsigned OpIdx = Di.SrcMap[I];
    if (OpIdx < KMaxOps)
      CurrentVgprAdjust[OpIdx] = SrcMsb[I];
  }
}

// Count how many 32-bit sub-registers make up `reg`. A 32-bit register has
// no sub0 (getSubReg returns 0) and is reported as width 1. Tuples
// (SGPR0_SGPR1, VReg_128, ...) walk sub0, sub1, ... until exhausted.
// The loop is bounded by the target's declared sub-reg index count so it
// terminates even if a future TableGen change introduced a cycle in the
// sub-reg graph.
static int computeRegWidth32(const MCRegisterInfo &MRI, MCRegister Reg) {
  const unsigned MaxSubIdx = MRI.getNumSubRegIndices();
  int W = 0;
  for (unsigned SubIdx = AMDGPU::sub0; SubIdx < MaxSubIdx; ++SubIdx) {
    if (!MRI.getSubReg(Reg, SubIdx))
      break;
    ++W;
  }
  return W ? W : 1;
}

// Locate `reg` inside `RC` and return its 0-based position. Used where the
// hardware encoding is not what we want (TTMPs live at generation-specific
// HW slots 108+ or 112+, but downstream we index a logical `ttmp[16]`
// array). Relies only on TableGen's declared class membership -- no enum
// arithmetic assumptions.
static int findIndexInClass(const MCRegisterClass &RC, MCRegister Reg) {
  for (unsigned I = 0, E = RC.getNumRegs(); I != E; ++I)
    if (RC.getRegister(I) == Reg)
      return static_cast<int>(I);
  return -1;
}

ParsedReg RaiseContext::parseReg(MCRegister Reg, int MciOpIdx) const {
  ParsedReg Pr;
  if (!Reg) {
    Pr.RegKind = ParsedReg::NOREG;
    return Pr;
  }

  const MCRegisterInfo &MRI = *Mc.RegInfo;

  // Width is computed on the as-decoded register: only the subtarget-
  // specific aliases (TTMPx_gfx9plus, FLAT_SCR_vi, ...) carry the correct
  // sub0/sub1/... chain from the disassembler.
  const int Width = computeRegWidth32(MRI, Reg);

  // Reduce everything to a canonical 32-bit pseudo for class/enum lookups:
  //   * sub0 on the as-decoded register picks the first 32-bit lane out
  //     of a tuple (sub-reg graph is authoritative on the real MC reg).
  //   * mc2PseudoReg then strips any subtarget suffix:
  //       TTMP8_gfx9plus         -> TTMP8
  //       FLAT_SCR_LO_vi         -> FLAT_SCR_LO
  //       SGPR_NULL64_gfx11plus  -> SGPR_NULL
  //       M0_gfx11plus           -> M0
  MCRegister Lane = MRI.getSubReg(Reg, AMDGPU::sub0);
  if (!Lane)
    Lane = Reg;
  Lane = AMDGPU::mc2PseudoReg(Lane);

  switch (Lane) {
  // Wave-mask registers. The ``_LO`` / ``_HI`` halves get the same
  // classification as the full pair; downstream VCC/EXEC handling routes
  // through loadVCC/storeVCC (which already respects wave size), so the
  // ``width`` field is informational here rather than load-bearing.
  case AMDGPU::VCC_HI:
    // On a wave32 source, hardware VCC is 32 bits (== VCC_LO); VCC_HI is a
    // free general-purpose scratch scalar. Route it to its own slot so the
    // (wave64-widened) VCC mask written by `v_cmp` does not clobber it.
    if (Isa.isWave32()) {
      Pr.RegKind = ParsedReg::VCC_HI_SCRATCH;
      Pr.Width = 1;
      return Pr;
    }
    [[fallthrough]];
  case AMDGPU::VCC_LO:
    Pr.RegKind = ParsedReg::VCC;
    Pr.Width = Isa.isWave32() ? 1 : 2;
    return Pr;
  case AMDGPU::EXEC_HI:
    // On a WAVE32 source, hardware EXEC is 32 bits (== EXEC_LO); EXEC_HI is a
    // free general-purpose scratch scalar (symmetric with VCC_HI above). Route
    // it to its own slot so the (wave64-widened) EXEC mask does not clobber it.
    // The full 64-bit EXEC pair resolves through EXEC_LO (sub0), so this only
    // intercepts an explicitly-named standalone exec_hi: always scratch on
    // wave32, never the mask.
    if (Isa.isWave32()) {
      Pr.RegKind = ParsedReg::EXEC_HI_SCRATCH;
      Pr.Width = 1;
      return Pr;
    }
    [[fallthrough]];
  case AMDGPU::EXEC_LO:
    Pr.RegKind = ParsedReg::EXEC;
    // baseIdx discriminates between the two 32-bit halves of wave64 EXEC
    // (0 = EXEC_LO, 1 = EXEC_HI). The full 64-bit pair also resolves here
    // via `sub0(EXEC) = EXEC_LO`, but `width = 2` tags it distinctly so
    // storeExec partial-write logic can route correctly.
    Pr.BaseIdx = (Lane == AMDGPU::EXEC_HI) ? 1 : 0;
    Pr.Width = Width;
    return Pr;
  case AMDGPU::SCC:
    Pr.RegKind = ParsedReg::SCC;
    Pr.Width = 1;
    return Pr;
  case AMDGPU::MODE:
    Pr.RegKind = ParsedReg::MODE;
    Pr.Width = 1;
    return Pr;
  case AMDGPU::M0:
    Pr.RegKind = ParsedReg::M0;
    Pr.Width = 1;
    return Pr;
  case AMDGPU::FLAT_SCR_LO:
  case AMDGPU::FLAT_SCR_HI:
    Pr.RegKind = ParsedReg::FLAT_SCR;
    Pr.Width = Width;
    return Pr;
  // GFX11+ uses SGPR_NULL / SGPR_NULL_HI (and the 64-bit pair SGPR_NULL64)
  // as carry-discard sinks, e.g. `v_mad_co_u64_u32 ..., null, ...`. They
  // have no backing slot -- treat writes to them as no-ops.
  case AMDGPU::SGPR_NULL:
  case AMDGPU::SGPR_NULL_HI:
    Pr.RegKind = ParsedReg::NOREG;
    return Pr;
  // XNACK_MASK controls page-fault retry masking per lane. On data-center
  // GPUs (MI300/MI350) XNACK is typically disabled and the register has no
  // effect on compute semantics. We treat it as NOREG (reads->zero,
  // writes->nop). If a kernel compiled with XNACK enabled relies on the
  // mask for correctness, this approximation is wrong -- but such kernels
  // are not expected in practice.
  case AMDGPU::XNACK_MASK_LO:
  case AMDGPU::XNACK_MASK_HI:
    Pr.RegKind = ParsedReg::NOREG;
    return Pr;
  // LDS_DIRECT (src_lds_direct, enc 254): reads a dword from LDS at the
  // byte offset held in M0. Used as a VALU source after buffer_load_*_lds.
  case AMDGPU::LDS_DIRECT:
    Pr.RegKind = ParsedReg::LDS_DIRECT;
    Pr.Width = 1;
    return Pr;
  // Source-only "compact predicate" registers
  // (SIRegisterInfo.td:198-200). They have no backing storage; their
  // value at use-time is a single i1 derived from VCC / EXEC / SCC.
  // Mark them here so readOp32 / readOp64 can materialise the boolean
  // (zext to the requested width). Encountered as VOP src operands in
  // gfx1250 Tensile kernels (e.g. `v_sub_f16 v64, src_vccz, v48`).
  case AMDGPU::SRC_VCCZ:
    Pr.RegKind = ParsedReg::SRC_VCCZ;
    Pr.Width = 1;
    return Pr;
  case AMDGPU::SRC_EXECZ:
    Pr.RegKind = ParsedReg::SRC_EXECZ;
    Pr.Width = 1;
    return Pr;
  case AMDGPU::SRC_SCC:
    Pr.RegKind = ParsedReg::SRC_SCC;
    Pr.Width = 1;
    return Pr;
  // Aperture / runtime-defined source registers: SRC_SHARED_BASE /
  // _LIMIT, SRC_PRIVATE_BASE / _LIMIT, SRC_FLAT_SCRATCH_BASE_LO /
  // _HI, SRC_POPS_EXITING_WAVE_ID. Their values are set per-queue by
  // the firmware and have no compile-time-knowable IR encoding, so
  // we cannot lower them principledly. Classify as OTHER so parseReg
  // does not crash; readOp32 / readOp64 will route OTHER through
  // `recordReadFailure(unsupportedInstructionForm)` and the per-instruction
  // dispatch loop in raiser.cpp will surface it as a clean
  // unsupported-instruction-form failure rather than a SIGABRT.
  case AMDGPU::SRC_SHARED_BASE_LO:
  case AMDGPU::SRC_SHARED_LIMIT_LO:
  case AMDGPU::SRC_PRIVATE_BASE_LO:
  case AMDGPU::SRC_PRIVATE_LIMIT_LO:
  case AMDGPU::SRC_POPS_EXITING_WAVE_ID:
  case AMDGPU::SRC_FLAT_SCRATCH_BASE_LO:
  case AMDGPU::SRC_FLAT_SCRATCH_BASE_HI:
    Pr.RegKind = ParsedReg::OTHER;
    Pr.Width = Width;
    return Pr;
  default:
    break;
  }

  // Family classification via the HW encoding flag bits. getEncodingValue
  // returns the correct HWEncoding payload for both pseudos and subtarget-
  // specific aliases. IS_VGPR (bit 10) and IS_AGPR (bit 11) are defined as
  // disjoint in SIRegisterInfo.td, so checking either first is correct;
  // AGPR goes first only because it is the more specific case.
  unsigned Enc = MRI.getEncodingValue(Reg);
  unsigned HwIdx = Enc & AMDGPU::HWEncoding::REG_IDX_MASK;

  if (Enc & AMDGPU::HWEncoding::IS_AGPR) {
    Pr.RegKind = ParsedReg::AGPR;
    Pr.BaseIdx = HwIdx;
    Pr.Width = Width;
    if (MciOpIdx >= 0 && static_cast<unsigned>(MciOpIdx) < KMaxOps)
      Pr.BaseIdx += CurrentVgprAdjust[MciOpIdx];
    return Pr;
  }
  if (Enc & AMDGPU::HWEncoding::IS_VGPR) {
    Pr.RegKind = ParsedReg::VGPR;
    Pr.BaseIdx = HwIdx;
    Pr.Width = Width;
    if (MciOpIdx >= 0 && static_cast<unsigned>(MciOpIdx) < KMaxOps)
      Pr.BaseIdx += CurrentVgprAdjust[MciOpIdx];
    return Pr;
  }

  // TTMPs live at a generation-specific HW encoding (108+ on gfx9+ vs 112+
  // on gfx8), so we cannot use the raw encoding as the logical 0..15
  // index. Locate the lane inside TTMP_32RegClass instead; the class is
  // defined as `(add (sequence "TTMP%u", 0, 15))` so position == index.
  const MCRegisterClass &TTMP32 =
      MRI.getRegClass(AMDGPU::TTMP_32RegClassID);
  if (int Idx = findIndexInClass(TTMP32, Lane); Idx >= 0) {
    Pr.RegKind = ParsedReg::TTMP;
    Pr.BaseIdx = Idx;
    Pr.Width = Width;
    return Pr;
  }

  // SGPR_32 is the narrow class for `SGPR0..SGPR105`; SReg_32 would also
  // include VCC_LO, EXEC_LO, FLAT_SCR_LO, M0, TTMP_32, SGPR_NULL, and the
  // SRC_* inline-value registers, which we have already ruled out above.
  if (MRI.getRegClass(AMDGPU::SGPR_32RegClassID).contains(Lane)) {
    Pr.RegKind = ParsedReg::SGPR;
    Pr.BaseIdx = HwIdx;
    Pr.Width = Width;
    return Pr;
  }

  report_fatal_error(Twine("transpiler: parseReg could not classify '") +
                     MRI.getName(Reg) + "' (enc=0x" +
                     Twine::utohexstr(Enc) + ")");
}

Value *RaiseContext::readOp32(const DecodedInst &Di, unsigned OpIdx) {
  if (Di.isReg(OpIdx)) {
    ParsedReg Pr = parseReg(Di.getReg(OpIdx), OpIdx);
    if (Pr.RegKind == ParsedReg::VCC) {
      if (Projection.sourceWaveScopedLaneOps()) {
        Value *Mask = Regs.readVCCAsWaveMask(B, Regs.ExecTy);
        Value *Lo = B.CreateTrunc(Mask, I32Ty, "vcc_src_wave_lo");
        Value *Hi = B.CreateTrunc(B.CreateLShr(Mask, Isa.WaveSize),
                                  I32Ty, "vcc_src_wave_hi");
        Value *Lane = Projection.emitLaneIdx(B);
        Value *Upper =
            B.CreateICmpUGE(Lane, ConstantInt::get(I32Ty, Isa.WaveSize),
                            "vcc_src_wave_upper");
        return B.CreateSelect(Upper, Hi, Lo, "vcc_src_wave_mask");
      }
      // Reading VCC as an i32 (wave32 wave-mask, or low 32 bits on
      // wave64) is a cross-lane collection: emit amdgcn.ballot so each
      // lane gets the same bit-mask assembled from all lanes' per-lane
      // VCC bits. On wave64 this is the low 32 lanes; upper-half reads
      // are separately materialised via readOp64.
      return Regs.readVCCAsWaveMask(B, I32Ty);
    }
    if (Pr.RegKind == ParsedReg::EXEC) {
      Value *V = Regs.loadExec(B);
      if (V->getType() == I32Ty)
        return V;
      if (Pr.Width < 2 && Pr.BaseIdx == 1)
        V = B.CreateLShr(V, 32, "exec_hi_shr");
      return B.CreateTrunc(V, I32Ty,
                            (Pr.Width < 2 && Pr.BaseIdx == 1) ? "exec_hi"
                                                               : "exec_lo");
    }
    if (Pr.RegKind == ParsedReg::SCC)
      return B.CreateZExt(Regs.loadSCC(B), I32Ty);
    if (Pr.RegKind == ParsedReg::SRC_SCC)
      return B.CreateZExt(Regs.loadSCC(B), I32Ty);
    if (Pr.RegKind == ParsedReg::SRC_VCCZ) {
      Value *Vcc = Regs.readVCCAsWaveMask(B, Regs.ExecTy);
      Value *Zero = ConstantInt::get(Regs.ExecTy, 0);
      return B.CreateZExt(B.CreateICmpEQ(Vcc, Zero, "vccz"), I32Ty);
    }
    if (Pr.RegKind == ParsedReg::SRC_EXECZ) {
      Value *Exec = Regs.loadExec(B);
      Value *Zero = ConstantInt::get(Exec->getType(), 0);
      return B.CreateZExt(B.CreateICmpEQ(Exec, Zero, "execz"), I32Ty);
    }
    if (Pr.RegKind == ParsedReg::NOREG)
      return ConstantInt::get(I32Ty, 0);
    if (Pr.RegKind == ParsedReg::MODE)
      return ConstantInt::get(I32Ty, 0);
    // OTHER is the parser's "I recognised the register but cannot
    // model it" channel, used today for runtime-defined aperture
    // registers (SRC_SHARED_BASE / SRC_FLAT_SCRATCH_BASE_LO etc.,
    // see parseReg's switch). Surface a clean unsupported-instruction-form
    // failure on the dispatch loop and return undef so we don't
    // crash mid-handler -- the next instruction-boundary check in
    // raiser.cpp will abort the kernel raise.
    if (Pr.RegKind == ParsedReg::OTHER) {
      recordReadFailure(RaiseFailure::unsupportedInstructionForm(
          Di, "operand-read",
          (Twine("readOp32 saw unmodeled register '") +
           Mc.RegInfo->getName(Di.getReg(OpIdx)) + "' in " + Di.Mnemonic)
              .str()));
      return UndefValue::get(I32Ty);
    }
    Value *V = Regs.readReg32(B, Pr);
    if (!V) {
      errs() << "transpiler: unreadable register '"
             << Mc.RegInfo->getName(Di.getReg(OpIdx)) << "' in " << Di.Mnemonic
             << "\n";
      return UndefValue::get(I32Ty);
    }
    return V;
  }
  if (std::optional<int64_t> Val = evalOperandAsConst(Di.Inst, OpIdx)) {
    return ConstantInt::get(I32Ty, static_cast<uint32_t>(*Val));
  }
  errs() << "transpiler: readOp32 unresolvable operand " << OpIdx << " in "
         << Di.Mnemonic << "\n";
  return UndefValue::get(I32Ty);
}

Value *RaiseContext::readOp64(const DecodedInst &Di, unsigned OpIdx) {
  if (Di.isReg(OpIdx)) {
    ParsedReg Pr = parseReg(Di.getReg(OpIdx), OpIdx);
    if (Pr.RegKind == ParsedReg::VCC)
      return Regs.readVCCAsWaveMask(B, I64Ty);
    if (Pr.RegKind == ParsedReg::EXEC) {
      Value *V = Regs.loadExec(B);
      if (V->getType() != I64Ty)
        V = B.CreateZExt(V, I64Ty, "exec_ext");
      return V;
    }
    // Mirror the readOp32 NOREG / MODE handling: SGPR_NULL64 (carry
    // sink), XNACK_MASK pairs, and the architectural MODE register
    // have no backing slot in our reg-file model. The 32-bit path
    // already returns the zero constant; the 64-bit path crashed
    // because readReg64 had no branch for these kinds. Returning
    // i64 0 matches hardware (SGPR_NULL reads as 0; XNACK_MASK and
    // MODE behave as 0 in compute kernels, see parseReg and the
    // SHORTCUTS_AND_LIMITATIONS XNACK note for rationale).
    if (Pr.RegKind == ParsedReg::NOREG || Pr.RegKind == ParsedReg::MODE)
      return ConstantInt::get(I64Ty, 0);
    if (Pr.RegKind == ParsedReg::SRC_SCC)
      return B.CreateZExt(Regs.loadSCC(B), I64Ty);
    if (Pr.RegKind == ParsedReg::SRC_VCCZ) {
      Value *Vcc = Regs.readVCCAsWaveMask(B, Regs.ExecTy);
      Value *Zero = ConstantInt::get(Regs.ExecTy, 0);
      return B.CreateZExt(B.CreateICmpEQ(Vcc, Zero, "vccz"), I64Ty);
    }
    if (Pr.RegKind == ParsedReg::SRC_EXECZ) {
      Value *Exec = Regs.loadExec(B);
      Value *Zero = ConstantInt::get(Exec->getType(), 0);
      return B.CreateZExt(B.CreateICmpEQ(Exec, Zero, "execz"), I64Ty);
    }
    if (Pr.RegKind == ParsedReg::OTHER) {
      recordReadFailure(RaiseFailure::unsupportedInstructionForm(
          Di, "operand-read",
          (Twine("readOp64 saw unmodeled register '") +
           Mc.RegInfo->getName(Di.getReg(OpIdx)) + "' in " + Di.Mnemonic)
              .str()));
      return UndefValue::get(I64Ty);
    }
    Value *V = Regs.readReg64(B, Pr);
    if (!V) {
      errs() << "transpiler: unreadable register64 '"
             << Mc.RegInfo->getName(Di.getReg(OpIdx)) << "' in " << Di.Mnemonic
             << "\n";
      return UndefValue::get(I64Ty);
    }
    return V;
  }
  if (std::optional<int64_t> Val = evalOperandAsConst(Di.Inst, OpIdx)) {
    return ConstantInt::getSigned(I64Ty, *Val);
  }
  errs() << "transpiler: readOp64 unresolvable operand " << OpIdx << " in "
         << Di.Mnemonic << "\n";
  return UndefValue::get(I64Ty);
}

Value *RaiseContext::emitUpdateDpp(Value *OldVal, Value *Src, uint16_t Ctrl,
                                    uint8_t RowMask, uint8_t BankMask,
                                    bool BoundCtrl) {
  // P5 lowering -- see the DPP row of hotswap/docs/wave-size-
  // translation.md §5.3: lift the DPP src-pathway modifier through
  // `llvm.amdgcn.update.dpp`. The intrinsic is type-overloaded
  // (`llvm_any_ty`). We route through integer overloads sized to match
  // the input's bit-width and bitcast through when the input is a
  // same-width non-integer type (f32 / <2 x i16> / etc.); codegen
  // picks the same DPP lowering for all same-width overloads.
  //
  // Widths supported today: 32-bit and 64-bit, matching the hardware
  // DPP encoding families (VOP_DPP and VOP_DPP_64). Other widths
  // `report_fatal_error` -- the tsFlags DPP bit can only be set on
  // VOP1/VOP2/VOP3 classes whose operands are always 32-bit or 64-bit
  // in AMDGPU's ISA, so any other width is a decoder/tblgen drift
  // situation worth surfacing loudly rather than silently downgrading.
  //
  // Historical regression gate: a wave32 DPP quad-permute
  // source kernel using `v_mov_b32_dpp ... quad_perm:[1,0,3,2]`
  // and runs it on gfx942 wave64, verifying the per-lane XOR-1
  // quad swap pattern across all 64 lanes. A future change that
  // breaks the dpp_ctrl/row_mask/bank_mask/bound_ctrl plumbing
  // through this helper, or the OpResolver `wrapDppIfNeeded` hook
  // that calls it, would fail this test.
  assert(OldVal->getType() == Src->getType() &&
         "emitUpdateDpp: old and src must have matching types");
  Type *OrigTy = Src->getType();
  const unsigned Bits = OrigTy->getPrimitiveSizeInBits();
  Type *IntTy = nullptr;
  if (Bits == 32)
    IntTy = I32Ty;
  else if (Bits == 64)
    IntTy = I64Ty;
  else
    report_fatal_error(
        "emitUpdateDpp: unsupported DPP operand width (expected 32 or 64 "
        "bits). Extend the bit-width dispatch below when a new DPP "
        "operand width lands in AMDGPU's ISA.");
  auto ToIntTy = [&](Value *V) {
    return V->getType() == IntTy ? V : B.CreateBitCast(V, IntTy);
  };
  Value *OldInt = ToIntTy(OldVal);
  Value *SrcInt = ToIntTy(Src);
  Function *Fn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::amdgcn_update_dpp, {IntTy});
  Value *Result =
      B.CreateCall(Fn, {OldInt, SrcInt, B.getInt32(Ctrl),
                         B.getInt32(RowMask), B.getInt32(BankMask),
                         B.getInt1(BoundCtrl)},
                   "dpp");
  if (Result->getType() != OrigTy)
    Result = B.CreateBitCast(Result, OrigTy);
  return Result;
}

Value *RaiseContext::emitLaneIdx() {
  // Per-BB memoisation, mirroring `emitLaneActiveBit` below -- the
  // `mbcnt_lo` (and on wave64, `mbcnt_hi`) pair WaveProjection emits
  // is invariant within a basic block for a given EXEC value, and
  // the cached SSA value dominates anything emitted later in the
  // same BB.
  //
  // Invalidation: shared with the lane_active cache via
  // `resetLaneActiveCache`, which the main raiser loop fires at
  // every source-instruction boundary. See `emitLaneActiveBit`'s
  // comment block below for the dominance argument that keeps reuse
  // safe across `emitUnderExec` diamonds within a single source
  // instruction's emission.
  if (CachedLaneIdx && B.GetInsertBlock() == CachedLaneIdxBb)
    return CachedLaneIdx;
  CachedLaneIdx = Projection.emitLaneIdx(B);
  CachedLaneIdxBb = B.GetInsertBlock();
  return CachedLaneIdx;
}

Value *RaiseContext::emitLaneActiveBit() {
  // Memoisation (see RaiseContext::resetLaneActiveCache docs).
  //
  // Dominance argument for reusing a cached i1 across blocks within a
  // single source instruction's emission:
  //
  //   Each emitUnderExec diamond is structurally linear:
  //
  //     preBB ──┬─▶ doBB ──▶ skipBB
  //             └────────────▶ skipBB   (the conditional skip edge)
  //
  //   Chaining N emitUnderExecs yields
  //     preBB -> (doBB1 -> skipBB1) -> (doBB2 -> skipBB2) -> … -> skipBBN
  //
  //   where every successor BB has preBB on its dominator path. So an
  //   i1 defined in preBB dominates every subsequent doBB/skipBB emitted
  //   by the same source-instruction handler.
  //
  // The reuse invariant is therefore maintained by the invalidation
  // contract alone:
  //   * The raiser main loop calls `resetLaneActiveCache` at every
  //     source-instruction boundary, covering (a) possible EXEC writes
  //     by the prior instruction and (b) jumps into offset-keyed
  //     named BBs where the prior `active` no longer dominates.
  //   * `ctx.storeExec` resets on EXEC mutation.
  //   * Any handler that writes EXEC via a lower-level path is
  //     responsible for calling `resetLaneActiveCache` itself (the
  //     allow-list audit in raiser.cpp documents that all such sites
  //     route through `ctx.storeExec` or `writeReg*`->`storeExec`).
  //
  // Consequently, a cache *hit* is valid regardless of whether the
  // current BB equals `cachedLaneActiveBB`.
  if (CachedLaneActive)
    return CachedLaneActive;

  // The projection owns the modulo-replication math; this context only
  // handles the cache + EXEC load.
  Value *Active = Projection.emitLaneActiveBit(B, Regs.loadExec(B));
  CachedLaneActive = Active;
  CachedLaneActiveBb = B.GetInsertBlock();
  return Active;
}

void RaiseContext::writeReg32(ParsedReg Pr, Value *V) {
  if (Pr.RegKind == ParsedReg::VGPR || Pr.RegKind == ParsedReg::AGPR) {
    emitUnderExec([&] { Regs.writeReg32(B, Pr, V); });
  } else {
    Regs.writeReg32(B, Pr, V);
    // regs.writeReg32 dispatches to storeExec when pr.kind == EXEC. The
    // lane_active memo must be invalidated so subsequent emitUnderExec
    // calls recompute against the new EXEC value rather than the pre-
    // write snapshot. See resetLaneActiveCache docs in raise-context.h.
    if (Pr.RegKind == ParsedReg::EXEC)
      resetLaneActiveCache();
  }
}

void RaiseContext::writeReg64(ParsedReg Pr, Value *V) {
  if (Pr.RegKind == ParsedReg::VGPR || Pr.RegKind == ParsedReg::AGPR) {
    emitUnderExec([&] { Regs.writeReg64(B, Pr, V); });
  } else {
    Regs.writeReg64(B, Pr, V);
    if (Pr.RegKind == ParsedReg::EXEC)
      resetLaneActiveCache();
  }
}

void RaiseContext::writeRegVec(ParsedReg Pr, Value *V) {
  if (Pr.RegKind == ParsedReg::VGPR || Pr.RegKind == ParsedReg::AGPR) {
    emitUnderExec([&] { Regs.writeRegVec(B, Pr, V); });
  } else {
    // Vector SGPR writes can't target EXEC (EXEC is scalar/pair, never
    // vector), so no cache invalidation is needed.
    Regs.writeRegVec(B, Pr, V);
  }
}

void RaiseContext::writeRegExecWidth(ParsedReg Pr, Value *V) {
  // Wave-level commit. SGPR-pair / VCC / EXEC writes carry the wave mask
  // itself and are computed cross-lane (ballot / sext-i1 today), so they
  // must not be predicated on the per-lane EXEC bit.
  Regs.writeRegExecWidth(B, Pr, V);
  if (Pr.RegKind == ParsedReg::EXEC)
    resetLaneActiveCache();
}

void RaiseContext::storeVGPR32(int Idx, Value *V) {
  emitUnderExec([&] { Regs.storeVGPR32(B, Idx, V); });
}

void RaiseContext::storeVGPR64(int Idx, Value *V) {
  emitUnderExec([&] { Regs.storeVGPR64(B, Idx, V); });
}

void RaiseContext::storeAGPR32(int Idx, Value *V) {
  emitUnderExec([&] { Regs.storeAGPR32(B, Idx, V); });
}

void RaiseContext::emitUnderExec(llvm::function_ref<void()> Body) {
  Value *Active = emitLaneActiveBit();
  BasicBlock *PreBb = B.GetInsertBlock();
  Function *F = PreBb->getParent();
  BasicBlock *DoBb = BasicBlock::Create(C, "spe_do", F);
  BasicBlock *SkipBb = BasicBlock::Create(C, "spe_skip", F);
  B.CreateCondBr(Active, DoBb, SkipBb);

  B.SetInsertPoint(DoBb);
  Body();
  // `body()` normally falls through without terminating. If a handler ever
  // ends its emission with an unconditional control-flow op (shouldn't
  // happen for the side-effectful ops we wrap, but defensively handled),
  // don't double-terminate doBB.
  if (!B.GetInsertBlock()->hasTerminator())
    B.CreateBr(SkipBb);

  B.SetInsertPoint(SkipBb);
}


Value *RaiseContext::readOpExecWidth(const DecodedInst &Di, unsigned OpIdx) {
  // All callers expect the returned value at `regs.execTy` (the EXEC
  // alloca storage width). Under modulo-replication `execTy` matches
  // the source wave-mask width and reads of source-width SGPR / imm
  // operands are already at the right width. Under wave-native cross-
  // widening `execTy` is wider than the source-named SGPR (i64 vs
  // i32 on wave32 source -> wave64 target), so we widen with the same
  // symmetric replication that `writeReg32(EXEC_LO)` uses on the
  // write side: `(v << W_src) | v` lifts a wave32 scalar wave mask
  // to a wave64 scalar wave mask where target lane K and K+W_src
  // agree. This keeps the save/restore round trip `s_mov_b32 sN,
  // exec_lo; …; s_mov_b32 exec_lo, sN` behaving as the wave32
  // author expected, and matches the replication done inside
  // `WaveNativeProjection::extractLaneBitFromWaveMask` on the VCC
  // consumer side.
  auto WidenToExec = [&](Value *Narrow) -> Value * {
    if (Narrow->getType() == Regs.ExecTy)
      return Narrow;
    unsigned Have = Narrow->getType()->getPrimitiveSizeInBits();
    unsigned Want = Regs.ExecTy->getPrimitiveSizeInBits();
    if (Have >= Want)
      return B.CreateZExtOrTrunc(Narrow, Regs.ExecTy);
    Value *Zext = B.CreateZExt(Narrow, Regs.ExecTy, "wn_src_to_exec_zext");
    Value *Hi = B.CreateShl(Zext, Have);
    return B.CreateOr(Zext, Hi, "wn_src_to_exec_mask");
  };

  if (Di.isReg(OpIdx)) {
    ParsedReg Pr = parseReg(Di.getReg(OpIdx), OpIdx);
    if (Pr.RegKind == ParsedReg::VCC)
      return Regs.readVCCAsWaveMask(B, Regs.ExecTy);
    if (Pr.RegKind == ParsedReg::EXEC)
      return Regs.loadExec(B);
    if (Pr.RegKind == ParsedReg::SGPR) {
      Value *Narrow =
          (Projection.sourceWaveScopedLaneOps() && Pr.Width >= 2)
              ? Regs.loadSGPR64(B, Pr.BaseIdx)
              : (Isa.isWave32() ? Regs.loadSGPR32(B, Pr.BaseIdx)
                                : Regs.loadSGPR64(B, Pr.BaseIdx));
      Value *Fallback = WidenToExec(Narrow);
      if (Value *ShadowValid = loadSgprWaveMaskValid(Pr.BaseIdx)) {
        Value *ShadowExec = loadSgprWaveMaskExec(Pr.BaseIdx);
        if (ShadowExec->getType() != Regs.ExecTy)
          ShadowExec = B.CreateZExtOrTrunc(ShadowExec, Regs.ExecTy,
                                           "wm_shadow_exec_cast");
        return B.CreateSelect(ShadowValid, ShadowExec, Fallback,
                              "exec_width_sgpr_shadow_sel");
      }
      return Fallback;
    }
    errs() << "transpiler: readOpExecWidth unresolvable register '"
           << Mc.RegInfo->getName(Di.getReg(OpIdx)) << "' in " << Di.Mnemonic
           << "\n";
    return UndefValue::get(Regs.ExecTy);
  }
  // Immediate and relocation-expression operands are always encoded at
  // the source wave-mask width (32 bits on wave32 source). Materialise
  // the narrow constant first and then widen through the same
  // replication path so an author's `s_mov_b32 exec_lo, 0xFFFF0000`
  // composes the same wave64 EXEC pattern as a save/restore of that
  // mask through an SGPR would.
  //
  // ISA-immediate-as-bit-pattern. `di.getImm` returns an `int64_t`
  // container that holds the raw literal bits the MC decoder read
  // out of the instruction's immediate field. Wave-mask idioms
  // routinely set the high bit of the source-width word --
  // `0xFFFF0000` (Triton's high-half upper-short mask, flagged in
  // the example above), `0xFFFFFFFF` (`-1` = all-lanes), `0x80000000`
  // (lane-31 bit), etc. An earlier `ConstantInt::getSigned(srcTy,
  // di.getImm(opIdx))` misinterpreted the container as a SIGNED
  // value and, on a wave32 source whose immediate reads as
  // `uint32_t ≥ 0x80000000`, tripped APInt's signed-range assertion
  // (`isIntN(BitWidth, val) && "Value is not an N-bit signed
  // value"`) -- because `(int64_t)0xFFFF0000 == +4294901760` is
  // outside `[-2^31, 2^31 - 1]`.
  //
  // Principled fix: match `readOp32`'s bit-pattern contract -- treat the
  // immediate as an unsigned bit
  // pattern and pack it into the source-width integer via
  // `ConstantInt::get(..., IsSigned=false)`. Masking to the source
  // width before the call is a defensive invariant: on wave32
  // sources, any MC-surfaced 64-bit literal is either the zero-
  // extended i32 the hardware carries or a bug upstream; masking
  // lets the invariant hold without relying on `ImplicitTrunc`
  // (kept off so the call still asserts on a truly malformed
  // literal rather than silently dropping bits). On wave64
  // sources the mask is the identity.
  //
  // Callers ordered through `readOpExecWidth` today (post-topk
  // SOP2 immediate-shadow propagation, landed in the
  // `s_xor_b32 ..., -1` idiom) now reach this immediate path; the
  // classifier-shielded path that previously never evaluated this
  // codepath on wave32 sources with high-bit-set literals
  // (GPT-OSS `_bitmatrix_metadata_compute_stage2`'s `s_and_b32
  // sN, sM, 0xFFFF0000` sites) no longer traps here.
  Type *SrcTy = Isa.isWave32() ? I32Ty : I64Ty;
  uint64_t SrcMask =
      Isa.isWave32() ? 0xFFFFFFFFull : 0xFFFFFFFFFFFFFFFFull;
  if (std::optional<int64_t> Val = evalOperandAsConst(Di.Inst, OpIdx)) {
    uint64_t Bits = static_cast<uint64_t>(*Val) & SrcMask;
    Value *Narrow = ConstantInt::get(SrcTy, Bits, /*IsSigned=*/false);
    return WidenToExec(Narrow);
  }
  errs() << "transpiler: readOpExecWidth unresolvable operand " << OpIdx
         << " in " << Di.Mnemonic << "\n";
  return UndefValue::get(Regs.ExecTy);
}

} // namespace COMGR::hotswap
