//===- raise-context.cpp - Hotswap transpiler -----------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "raise-context.h"

#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIDefines.h"
#include "Utils/AMDGPUBaseInfo.h"
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

RaiseContext::RaiseContext(
    LLVMContext &C, Module &M, IRBuilder<> &B, AllocaRegFile &Regs,
    const WaveProjection &Projection, const MCState &MC, const ISAProfile &Isa,
    ISAProfile TargetIsa, unsigned TargetCodeObjectVersion,
    KernargLayout &Kernargs, const UserSgprLayout *Layout, Function *Kernel,
    BasicBlock *ThreadLoopLatch, DenseMap<uint64_t, BasicBlock *> &OffsetToBb,
    ArrayRef<uint8_t> SourceTextBytes, uint64_t SourceTextBaseAddress,
    ArrayRef<TextSection::ImageSection> SourceImageSections,
    uint64_t KernelStartOffset, uint64_t KernelEndOffset)
    : C(C), M(M), B(B), Regs(Regs), Projection(Projection), MC(MC), Isa(Isa),
      TargetIsa(TargetIsa), TargetCodeObjectVersion(TargetCodeObjectVersion),
      Kernargs(Kernargs), Layout(Layout), Kernel(Kernel),
      ThreadLoopLatch(ThreadLoopLatch), OffsetToBb(OffsetToBb),
      SourceTextBytes(SourceTextBytes),
      SourceTextBaseAddress(SourceTextBaseAddress),
      SourceImageSections(SourceImageSections),
      KernelStartOffset(KernelStartOffset), KernelEndOffset(KernelEndOffset) {
  I1Ty = Type::getInt1Ty(C);
  I8Ty = Type::getInt8Ty(C);
  I16Ty = Type::getInt16Ty(C);
  I32Ty = Type::getInt32Ty(C);
  I64Ty = Type::getInt64Ty(C);
  F32Ty = Type::getFloatTy(C);
  F16Ty = Type::getHalfTy(C);
  F64Ty = Type::getDoubleTy(C);
  PtrGlobalTy = PointerType::get(C, 1);
}

BasicBlock *RaiseContext::lookupBB(uint64_t Addr) {
  DenseMap<uint64_t, BasicBlock *>::iterator It = OffsetToBb.find(Addr);
  if (It != OffsetToBb.end())
    return It->second;
  // Every branch target is a block leader recorded during CFG layout, so a
  // miss is a raiser bug, not a recoverable case.
  report_fatal_error(Twine("transpiler: missing basic block for offset 0x") +
                     utohexstr(Addr));
}

// Returns true for source opcodes whose vector operands ignore active
// S_SET_VGPR_MSB state, so a missing operand-role table is intentional
// rather than a decode gap.
static bool ignoresVGPRMsb(unsigned Opc) {
  switch (Opc) {
  case AMDGPU::V_WMMA_LD_SCALE_PAIRED_B32:
  case AMDGPU::V_WMMA_LD_SCALE_PAIRED_B32_gfx1250:
  case AMDGPU::V_WMMA_LD_SCALE16_PAIRED_B64:
  case AMDGPU::V_WMMA_LD_SCALE16_PAIRED_B64_gfx1250:
    return true;
  default:
    return false;
  }
}

// Returns true iff the decoded MC operands include a real VGPR or AGPR
// register. Non-register operands and no-register sentinels are ignored.
static bool hasVectorRegOperand(const DecodedInst &Di,
                                const MCRegisterInfo &MRI) {
  for (unsigned I = 0, E = Di.Inst.getNumOperands(); I != E; ++I) {
    const MCOperand &Op = Di.Inst.getOperand(I);
    if (!Op.isReg() || !Op.getReg())
      continue;
    unsigned Enc = MRI.getEncodingValue(Op.getReg());
    if (Enc & (AMDGPU::HWEncoding::IS_VGPR | AMDGPU::HWEncoding::IS_AGPR))
      return true;
  }
  return false;
}

Error RaiseContext::computeVGPRAdjust(const DecodedInst &Di) {
  std::fill_n(CurrentVgprAdjust, KMaxOps, 0u);
  if (VgprMsBs == 0)
    return Error::success();

  // The low byte of the S_SET_VGPR_MSB immediate holds four 2-bit MSB fields,
  // one per slot, that form bits [9:8] of the VGPR address (i.e. extend the
  // index by field * 256) -- the mechanism gfx1250 uses to reach all 1024
  // VGPRs. VgprMsBs holds the active state, which persists until the next
  // S_SET_VGPR_MSB.
  //
  // The slot->operand mapping is instruction-format-specific, not VALU's
  // positional src0/src1/src2/vdst order (VBUFFER maps slot 0 to vaddr and
  // slot 3 to vdata, VDS maps slots 0/1/2 to addr/data0/data1), so resolve it
  // through getVGPRLoweringOperandTables. Only the single-issue (X) table is
  // consulted; VOPD dual-issue packets carry their MSBs separately.
  unsigned Opc = Di.Inst.getOpcode();
  const MCInstrDesc &Desc = MC.InstrInfo->get(Opc);
  const AMDGPU::OpName *Ops = AMDGPU::getVGPRLoweringOperandTables(Desc).first;
  if (!Ops) {
    if (ignoresVGPRMsb(Opc) || !hasVectorRegOperand(Di, *MC.RegInfo) ||
        Desc.isPseudo() || Desc.isMetaInstruction())
      return Error::success();
    return createStringError(
        Twine("transpiler: S_SET_VGPR_MSB has no "
              "operand-role table for vector instruction ") +
        strippedMnemonic(MC, Di.Inst));
  }

  for (unsigned Slot = 0; Slot != 4; ++Slot) {
    // NUM_OPERAND_NAMES marks a slot this format does not use (e.g. VBUFFER
    // leaves slots 1 and 2 empty).
    if (Ops[Slot] == AMDGPU::OpName::NUM_OPERAND_NAMES)
      continue;
    // Slot N is the 2-bit field at bits [2N+1:2N]; its value is the high VGPR
    // bank, so the operand's index offset is bank * 256.
    unsigned Adjust =
        ((static_cast<unsigned>(VgprMsBs) >> (Slot * 2)) & 0x3u) * 256u;
    if (Adjust == 0)
      continue;
    // Resolve the slot's role to this instruction's operand index and record
    // the offset parseReg() will apply. getNamedOperandIdx returns -1 if the
    // operand is absent; KMaxOps bounds the CurrentVgprAdjust table.
    int OpIdx = AMDGPU::getNamedOperandIdx(Opc, Ops[Slot]);
    if (OpIdx < 0)
      continue;
    if (static_cast<unsigned>(OpIdx) >= KMaxOps)
      return createStringError(
          "transpiler: S_SET_VGPR_MSB operand index " + Twine(OpIdx) +
          " exceeds CurrentVgprAdjust capacity " + Twine(KMaxOps) + " for " +
          strippedMnemonic(MC, Di.Inst));
    CurrentVgprAdjust[OpIdx] = Adjust;
  }
  return Error::success();
}

// Return the register width in 32-bit subregisters. Scalar registers have no
// sub0 and therefore have width one.
static unsigned computeRegWidth32(const MCRegisterInfo &MRI, MCRegister Reg) {
  const unsigned MaxSubIdx = MRI.getNumSubRegIndices();
  unsigned W = 0;
  for (unsigned SubIdx = AMDGPU::sub0; SubIdx < MaxSubIdx; ++SubIdx) {
    if (!MRI.getSubReg(Reg, SubIdx))
      break;
    ++W;
  }
  return W ? W : 1;
}

// Return Reg's position in RC, or std::nullopt if it is not a member.
static std::optional<unsigned> findIndexInClass(const MCRegisterClass &RC,
                                                MCRegister Reg) {
  for (unsigned I = 0, E = RC.getNumRegs(); I != E; ++I)
    if (RC.getRegister(I) == Reg)
      return I;
  return std::nullopt;
}

ParsedReg RaiseContext::parseReg(MCRegister Reg, int MciOpIdx) const {
  ParsedReg Pr;
  if (!Reg) {
    Pr.RegKind = ParsedReg::NOREG;
    return Pr;
  }

  const MCRegisterInfo &MRI = *MC.RegInfo;

  // Width is computed on the as-decoded register: only the subtarget-
  // specific aliases (TTMPx_gfx9plus, FLAT_SCR_vi, ...) carry the correct
  // sub0/sub1/... chain from the disassembler.
  const unsigned Width = computeRegWidth32(MRI, Reg);

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
      Pr.WidthInDwords = 1;
      return Pr;
    }
    [[fallthrough]];
  case AMDGPU::VCC_LO:
    Pr.RegKind = ParsedReg::VCC;
    Pr.WidthInDwords = Isa.isWave32() ? 1 : 2;
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
      Pr.WidthInDwords = 1;
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
    Pr.WidthInDwords = Width;
    return Pr;
  case AMDGPU::SCC:
    Pr.RegKind = ParsedReg::SCC;
    Pr.WidthInDwords = 1;
    return Pr;
  case AMDGPU::MODE:
    Pr.RegKind = ParsedReg::MODE;
    Pr.WidthInDwords = 1;
    return Pr;
  case AMDGPU::M0:
    Pr.RegKind = ParsedReg::M0;
    Pr.WidthInDwords = 1;
    return Pr;
  case AMDGPU::FLAT_SCR_LO:
  case AMDGPU::FLAT_SCR_HI:
    Pr.RegKind = ParsedReg::FLAT_SCR;
    Pr.WidthInDwords = Width;
    return Pr;
  // GFX11+ uses SGPR_NULL / SGPR_NULL_HI (and the 64-bit pair SGPR_NULL64)
  // as carry-discard sinks, e.g. `v_mad_co_u64_u32 ..., null, ...`. They
  // have no backing slot -- treat writes to them as no-ops.
  case AMDGPU::SGPR_NULL:
  case AMDGPU::SGPR_NULL_HI:
    Pr.RegKind = ParsedReg::NOREG;
    return Pr;
  // XNACK_MASK controls per-lane page-fault retry masking, which is disabled
  // and has no effect on compute semantics on the targeted GPUs, so treat it
  // as NOREG (reads -> zero, writes -> nop).
  case AMDGPU::XNACK_MASK_LO:
  case AMDGPU::XNACK_MASK_HI:
    Pr.RegKind = ParsedReg::NOREG;
    return Pr;
  // LDS_DIRECT (src_lds_direct, enc 254): reads a dword from LDS at the
  // byte offset held in M0. Used as a VALU source after buffer_load_*_lds.
  case AMDGPU::LDS_DIRECT:
    Pr.RegKind = ParsedReg::LDS_DIRECT;
    Pr.WidthInDwords = 1;
    return Pr;
  // Source-only registers with no backing storage; their value at use time is
  // a single i1 derived from VCC / EXEC / SCC, materialised (zext to width) by
  // the read paths.
  case AMDGPU::SRC_VCCZ:
    Pr.RegKind = ParsedReg::SRC_VCCZ;
    Pr.WidthInDwords = 1;
    return Pr;
  case AMDGPU::SRC_EXECZ:
    Pr.RegKind = ParsedReg::SRC_EXECZ;
    Pr.WidthInDwords = 1;
    return Pr;
  case AMDGPU::SRC_SCC:
    Pr.RegKind = ParsedReg::SRC_SCC;
    Pr.WidthInDwords = 1;
    return Pr;
  // Aperture / runtime-defined source registers. Their values are set
  // per-queue by the firmware and have no compile-time-knowable IR encoding,
  // so classify as OTHER; the read paths surface a clean unsupported-form
  // failure rather than crashing.
  case AMDGPU::SRC_SHARED_BASE_LO:
  case AMDGPU::SRC_SHARED_LIMIT_LO:
  case AMDGPU::SRC_PRIVATE_BASE_LO:
  case AMDGPU::SRC_PRIVATE_LIMIT_LO:
  case AMDGPU::SRC_POPS_EXITING_WAVE_ID:
  case AMDGPU::SRC_FLAT_SCRATCH_BASE_LO:
  case AMDGPU::SRC_FLAT_SCRATCH_BASE_HI:
    Pr.RegKind = ParsedReg::OTHER;
    Pr.WidthInDwords = Width;
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
    Pr.WidthInDwords = Width;
    if (MciOpIdx >= 0 && static_cast<unsigned>(MciOpIdx) < KMaxOps)
      HwIdx += CurrentVgprAdjust[MciOpIdx];
    Pr.BaseIdx = HwIdx;
    return Pr;
  }
  if (Enc & AMDGPU::HWEncoding::IS_VGPR) {
    Pr.RegKind = ParsedReg::VGPR;
    Pr.WidthInDwords = Width;
    if (MciOpIdx >= 0 && static_cast<unsigned>(MciOpIdx) < KMaxOps)
      HwIdx += CurrentVgprAdjust[MciOpIdx];
    Pr.BaseIdx = HwIdx;
    return Pr;
  }

  // TTMPs live at a generation-specific HW encoding (108+ on gfx9+ vs 112+
  // on gfx8), so we cannot use the raw encoding as the logical 0..15
  // index. Locate the lane inside TTMP_32RegClass instead; the class is
  // defined as `(add (sequence "TTMP%u", 0, 15))` so position == index.
  const MCRegisterClass &TTMP32 = MRI.getRegClass(AMDGPU::TTMP_32RegClassID);
  if (std::optional<unsigned> Index = findIndexInClass(TTMP32, Lane)) {
    Pr.RegKind = ParsedReg::TTMP;
    Pr.BaseIdx = *Index;
    Pr.WidthInDwords = Width;
    return Pr;
  }

  // SGPR_32 is the narrow class for `SGPR0..SGPR105`; SReg_32 would also
  // include VCC_LO, EXEC_LO, FLAT_SCR_LO, M0, TTMP_32, SGPR_NULL, and the
  // SRC_* inline-value registers, which we have already ruled out above.
  if (MRI.getRegClass(AMDGPU::SGPR_32RegClassID).contains(Lane)) {
    Pr.RegKind = ParsedReg::SGPR;
    Pr.BaseIdx = HwIdx;
    Pr.WidthInDwords = Width;
    return Pr;
  }

  report_fatal_error(Twine("transpiler: parseReg could not classify '") +
                     MRI.getName(Reg) + "' (enc=0x" + Twine::utohexstr(Enc) +
                     ")");
}

Value *RaiseContext::readOp32(const DecodedInst &Di, unsigned OpIdx) {
  if (Di.isReg(OpIdx)) {
    ParsedReg Pr = parseReg(Di.getReg(OpIdx), OpIdx);
    if (Pr.RegKind == ParsedReg::VCC) {
      if (Projection.sourceWaveScopedLaneOps()) {
        Value *Mask = Regs.readVCCAsWaveMask(B, Projection.execStorageTy());
        Value *Lo = B.CreateTrunc(Mask, I32Ty, "vcc_src_wave_lo");
        Value *Hi = B.CreateTrunc(B.CreateLShr(Mask, Isa.waveSize()), I32Ty,
                                  "vcc_src_wave_hi");
        Value *Lane = Projection.emitLaneIdx(B);
        Value *Upper = B.CreateICmpUGE(
            Lane, ConstantInt::get(I32Ty, Isa.waveSize()), "vcc_src_wave_upper");
        return B.CreateSelect(Upper, Hi, Lo, "vcc_src_wave_mask");
      }
      // Reading VCC as an i32 (wave32 wave-mask, or low 32 bits on wave64) is
      // a cross-lane collection: emit amdgcn.ballot so each lane gets the same
      // bit-mask assembled from all lanes' per-lane VCC bits.
      return Regs.readVCCAsWaveMask(B, I32Ty);
    }
    if (Pr.RegKind == ParsedReg::EXEC) {
      Value *V = Regs.loadExec(B);
      if (V->getType() == I32Ty)
        return V;
      if (Pr.WidthInDwords < 2 && Pr.BaseIdx == 1)
        V = B.CreateLShr(V, 32, "exec_hi_shr");
      return B.CreateTrunc(
          V, I32Ty,
          (Pr.WidthInDwords < 2 && Pr.BaseIdx == 1) ? "exec_hi" : "exec_lo");
    }
    if (Pr.RegKind == ParsedReg::SCC)
      return B.CreateZExt(Regs.loadSCC(B), I32Ty);
    if (Pr.RegKind == ParsedReg::SRC_SCC)
      return B.CreateZExt(Regs.loadSCC(B), I32Ty);
    if (Pr.RegKind == ParsedReg::SRC_VCCZ) {
      Value *Vcc = Regs.readVCCAsWaveMask(B, Projection.execStorageTy());
      Value *Zero = ConstantInt::get(Projection.execStorageTy(), 0);
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
    // OTHER means parseReg recognised the register but cannot model it (the
    // runtime-defined aperture registers). Record a structured failure and
    // return undef so we do not crash mid-handler.
    if (Pr.RegKind == ParsedReg::OTHER) {
      recordReadFailure(RaiseFailure::atInstruction(
          RaiseFailureReason::UnsupportedInstructionForm,
          strippedMnemonic(MC, Di.Inst), Di.Offset, "operand-read",
          Twine("readOp32 saw unmodeled register '") +
              MC.RegInfo->getName(Di.getReg(OpIdx)) + "' in " +
              strippedMnemonic(MC, Di.Inst)));
      return UndefValue::get(I32Ty);
    }
    Value *V = Regs.readReg32(B, Pr);
    if (!V) {
      recordReadFailure(RaiseFailure::atInstruction(
          RaiseFailureReason::UnsupportedInstructionForm,
          strippedMnemonic(MC, Di.Inst), Di.Offset, "operand-read",
          Twine("readOp32 could not read register '") +
              MC.RegInfo->getName(Di.getReg(OpIdx)) + "' in " +
              strippedMnemonic(MC, Di.Inst)));
      return UndefValue::get(I32Ty);
    }
    return V;
  }
  if (std::optional<int64_t> Val = evalOperandAsConst(Di.Inst, OpIdx)) {
    return ConstantInt::get(I32Ty, static_cast<uint32_t>(*Val));
  }
  recordReadFailure(RaiseFailure::atInstruction(
      RaiseFailureReason::UnsupportedInstructionForm,
      strippedMnemonic(MC, Di.Inst), Di.Offset, "operand-read",
      Twine("readOp32 could not resolve operand ") + Twine(OpIdx) + " in " +
          strippedMnemonic(MC, Di.Inst)));
  return UndefValue::get(I32Ty);
}

Value *RaiseContext::readOpSourceWaveMask32(const DecodedInst &Di,
                                            unsigned OpIdx) {
  if (!Di.isReg(OpIdx))
    return readOp32(Di, OpIdx);

  ParsedReg Pr = parseReg(Di.getReg(OpIdx), OpIdx);
  if (Pr.RegKind == ParsedReg::EXEC)
    return Projection.emitCurrentSourceWaveMask(B, Regs.loadExec(B),
                                                "exec_srcwave_mask");
  if (Pr.RegKind == ParsedReg::VCC)
    return Projection.emitCurrentSourceWaveMask(
        B, Regs.readVCCAsWaveMask(B, Projection.execStorageTy()),
        "vcc_srcwave_mask");
  if (Pr.RegKind == ParsedReg::SGPR && Pr.BaseIdx) {
    Value *Fallback = readOp32(Di, OpIdx);
    if (Value *ShadowValid = loadSgprWaveMaskValid(*Pr.BaseIdx)) {
      Value *ShadowExec = loadSgprWaveMaskExec(*Pr.BaseIdx);
      if (ShadowExec->getType() != Projection.execStorageTy())
        ShadowExec = B.CreateZExtOrTrunc(ShadowExec, Projection.execStorageTy(),
                                         "sgpr_mask_exec_cast");
      Value *ShadowMask = Projection.emitCurrentSourceWaveMask(
          B, ShadowExec, "sgpr_srcwave_mask_shadow");
      return B.CreateSelect(ShadowValid, ShadowMask, Fallback,
                            "sgpr_srcwave_mask");
    }
    return Fallback;
  }

  return readOp32(Di, OpIdx);
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
    // SGPR_NULL64 (carry sink), XNACK_MASK pairs, and the architectural MODE
    // register have no backing slot in the reg-file model. Reading i64 0
    // matches hardware: SGPR_NULL reads as 0, and XNACK_MASK and MODE behave
    // as 0 in compute kernels.
    if (Pr.RegKind == ParsedReg::NOREG || Pr.RegKind == ParsedReg::MODE)
      return ConstantInt::get(I64Ty, 0);
    if (Pr.RegKind == ParsedReg::SRC_SCC)
      return B.CreateZExt(Regs.loadSCC(B), I64Ty);
    if (Pr.RegKind == ParsedReg::SRC_VCCZ) {
      Value *Vcc = Regs.readVCCAsWaveMask(B, Projection.execStorageTy());
      Value *Zero = ConstantInt::get(Projection.execStorageTy(), 0);
      return B.CreateZExt(B.CreateICmpEQ(Vcc, Zero, "vccz"), I64Ty);
    }
    if (Pr.RegKind == ParsedReg::SRC_EXECZ) {
      Value *Exec = Regs.loadExec(B);
      Value *Zero = ConstantInt::get(Exec->getType(), 0);
      return B.CreateZExt(B.CreateICmpEQ(Exec, Zero, "execz"), I64Ty);
    }
    if (Pr.RegKind == ParsedReg::OTHER) {
      recordReadFailure(RaiseFailure::atInstruction(
          RaiseFailureReason::UnsupportedInstructionForm,
          strippedMnemonic(MC, Di.Inst), Di.Offset, "operand-read",
          Twine("readOp64 saw unmodeled register '") +
              MC.RegInfo->getName(Di.getReg(OpIdx)) + "' in " +
              strippedMnemonic(MC, Di.Inst)));
      return UndefValue::get(I64Ty);
    }
    Value *V = Regs.readReg64(B, Pr);
    if (!V) {
      recordReadFailure(RaiseFailure::atInstruction(
          RaiseFailureReason::UnsupportedInstructionForm,
          strippedMnemonic(MC, Di.Inst), Di.Offset, "operand-read",
          Twine("readOp64 could not read register '") +
              MC.RegInfo->getName(Di.getReg(OpIdx)) + "' in " +
              strippedMnemonic(MC, Di.Inst)));
      return UndefValue::get(I64Ty);
    }
    return V;
  }
  if (std::optional<int64_t> Val = evalOperandAsConst(Di.Inst, OpIdx)) {
    return ConstantInt::getSigned(I64Ty, *Val);
  }
  recordReadFailure(RaiseFailure::atInstruction(
      RaiseFailureReason::UnsupportedInstructionForm,
      strippedMnemonic(MC, Di.Inst), Di.Offset, "operand-read",
      Twine("readOp64 could not resolve operand ") + Twine(OpIdx) + " in " +
          strippedMnemonic(MC, Di.Inst)));
  return UndefValue::get(I64Ty);
}

Value *RaiseContext::emitLaneIdx() {
  // Lane id is function-invariant; the projection emits it once and caches.
  return Projection.emitLaneIdx(B);
}

Value *RaiseContext::freezeMemAddr(Value *Addr) {
  // See the header for the correctness argument. Only widening
  // wave32 -> wave64 lifts can leak an undef address into a memory op via
  // the reg-file first-def phi; other directions keep byte-identical IR.
  if (!Isa.isWave32() || TargetIsa.isWave32())
    return Addr;
  return B.CreateFreeze(Addr, "mem_addr_frozen");
}

Value *RaiseContext::emitLaneActiveBit() {
  // A cache hit is valid across blocks within one source instruction's
  // emission: each emitUnderExec diamond is structurally linear, so the i1
  // defined in the entry block dominates every later do/skip block, and the
  // cache is invalidated at every instruction boundary and on every EXEC
  // write. So the current BB need not equal CachedLaneActiveBb.
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
    // A write to EXEC changes the lane-active mask, so invalidate the memo.
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

void RaiseContext::storeVGPR32(unsigned Idx, Value *V) {
  emitUnderExec([&] { Regs.storeVGPR32(B, Idx, V); });
}

void RaiseContext::storeVGPR64(unsigned Idx, Value *V) {
  emitUnderExec([&] { Regs.storeVGPR64(B, Idx, V); });
}

void RaiseContext::storeAGPR32(unsigned Idx, Value *V) {
  emitUnderExec([&] { Regs.storeAGPR32(B, Idx, V); });
}

void RaiseContext::emitUnderExec(llvm::function_ref<void()> Body) {
  Value *Active = emitLaneActiveBit();
  BasicBlock *PreBb = B.GetInsertBlock();
  Function *F = PreBb->getParent();
  BasicBlock *DoBb = BasicBlock::Create(C, "spe_do", F);
  BasicBlock *SkipBb = BasicBlock::Create(C, "spe_skip", F);
  KernargPtrProvenance PreProvenance = CurrentKernargPtrProvenance;
  B.CreateCondBr(Active, DoBb, SkipBb);

  B.SetInsertPoint(DoBb);
  Body();
  KernargPtrProvenance DoProvenance = CurrentKernargPtrProvenance;
  // `body()` normally falls through without terminating. If a handler ever
  // ends its emission with an unconditional control-flow op (shouldn't
  // happen for the side-effectful ops we wrap, but defensively handled),
  // don't double-terminate doBB.
  if (!B.GetInsertBlock()->hasTerminator()) {
    B.CreateBr(SkipBb);
    CurrentKernargPtrProvenance =
        joinKernargPtrProvenance(PreProvenance, DoProvenance);
  } else {
    CurrentKernargPtrProvenance = PreProvenance;
  }

  B.SetInsertPoint(SkipBb);
}

Value *RaiseContext::readOpExecWidth(const DecodedInst &Di, unsigned OpIdx) {
  // Return the operand at the EXEC alloca storage width. When widening wave32
  // -> wave64, a source-named SGPR is narrower than that width, so widen a
  // scalar wave mask by symmetric replication
  // `(v << W_src) | v` where target lane K and K+W_src agree; this keeps an
  // `s_mov exec_lo, sN` save/restore round trip intact.
  auto WidenToExec = [&](Value *Narrow) -> Value * {
    Type *ExecTy = Projection.execStorageTy();
    if (Narrow->getType() == ExecTy)
      return Narrow;
    unsigned Have = Narrow->getType()->getPrimitiveSizeInBits();
    unsigned Want = ExecTy->getPrimitiveSizeInBits();
    if (Have >= Want)
      return B.CreateZExtOrTrunc(Narrow, ExecTy);
    Value *Zext = B.CreateZExt(Narrow, ExecTy, "wn_src_to_exec_zext");
    Value *Hi = B.CreateShl(Zext, Have);
    return B.CreateOr(Zext, Hi, "wn_src_to_exec_mask");
  };

  if (Di.isReg(OpIdx)) {
    ParsedReg Pr = parseReg(Di.getReg(OpIdx), OpIdx);
    if (Pr.RegKind == ParsedReg::VCC)
      return Regs.readVCCAsWaveMask(B, Projection.execStorageTy());
    if (Pr.RegKind == ParsedReg::EXEC)
      return Regs.loadExec(B);
    if (Pr.RegKind == ParsedReg::VCC_HI_SCRATCH ||
        Pr.RegKind == ParsedReg::EXEC_HI_SCRATCH)
      // Wave32 vcc_hi / exec_hi are scratch scalars, not the wave mask.
      return WidenToExec(Regs.readReg32(B, Pr));
    if (Pr.RegKind == ParsedReg::SGPR) {
      assert(Pr.BaseIdx && "SGPR must have a base register index");
      unsigned BaseIdx = *Pr.BaseIdx;
      Value *Narrow =
          (Projection.sourceWaveScopedLaneOps() && Pr.WidthInDwords >= 2)
              ? Regs.loadSGPR64(B, BaseIdx)
              : (Isa.isWave32() ? Regs.loadSGPR32(B, BaseIdx)
                                : Regs.loadSGPR64(B, BaseIdx));
      Value *Fallback = WidenToExec(Narrow);
      if (Value *ShadowValid = loadSgprWaveMaskValid(BaseIdx)) {
        Value *ShadowExec = loadSgprWaveMaskExec(BaseIdx);
        if (ShadowExec->getType() != Projection.execStorageTy())
          ShadowExec = B.CreateZExtOrTrunc(
              ShadowExec, Projection.execStorageTy(), "wm_shadow_exec_cast");
        return B.CreateSelect(ShadowValid, ShadowExec, Fallback,
                              "exec_width_sgpr_shadow_sel");
      }
      return Fallback;
    }
    recordReadFailure(RaiseFailure::atInstruction(
        RaiseFailureReason::UnsupportedInstructionForm,
        strippedMnemonic(MC, Di.Inst), Di.Offset, "operand-read",
        Twine("readOpExecWidth could not read register '") +
            MC.RegInfo->getName(Di.getReg(OpIdx)) + "' in " +
            strippedMnemonic(MC, Di.Inst)));
    return UndefValue::get(Projection.execStorageTy());
  }
  // Immediate and relocation-expression operands are always encoded at
  // the source wave-mask width (32 bits on wave32 source). Materialise
  // the narrow constant first and then widen through the same
  // replication path so an author's `s_mov_b32 exec_lo, 0xFFFF0000`
  // composes the same wave64 EXEC pattern as a save/restore of that
  // mask through an SGPR would.
  //
  // Treat the immediate as an unsigned bit pattern (matching readOp32) rather
  // than a signed value: wave-mask idioms routinely set the high bit of the
  // source-width word (0xFFFF0000, 0xFFFFFFFF, 0x80000000), which a signed
  // interpretation would place outside the source-width signed range. Mask to
  // the source width first so ConstantInt::get with IsSigned=false still
  // asserts on a truly malformed literal instead of silently truncating.
  Type *SrcTy = Isa.isWave32() ? I32Ty : I64Ty;
  uint64_t SrcMask = Isa.isWave32() ? 0xFFFFFFFFull : 0xFFFFFFFFFFFFFFFFull;
  if (std::optional<int64_t> Val = evalOperandAsConst(Di.Inst, OpIdx)) {
    uint64_t Bits = static_cast<uint64_t>(*Val) & SrcMask;
    Value *Narrow = ConstantInt::get(SrcTy, Bits, /*IsSigned=*/false);
    return WidenToExec(Narrow);
  }
  recordReadFailure(RaiseFailure::atInstruction(
      RaiseFailureReason::UnsupportedInstructionForm,
      strippedMnemonic(MC, Di.Inst), Di.Offset, "operand-read",
      Twine("readOpExecWidth could not resolve operand ") + Twine(OpIdx) +
          " in " + strippedMnemonic(MC, Di.Inst)));
  return UndefValue::get(Projection.execStorageTy());
}

} // namespace COMGR::hotswap
