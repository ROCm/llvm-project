//===- raise-context.cpp - Hotswap transpiler -----------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "raise-context.h"

#include "hotswap/decoder/amdgpu-formats.h"

#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIDefines.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>

using namespace llvm;

namespace COMGR::hotswap {

RaiseContext::RaiseContext(
    IRBuilder<> &B, AllocaRegFile &Regs, const WaveProjection &Projection,
    const MCState &MC, unsigned TargetCodeObjectVersion,
    KernargLayout &Kernargs, const UserSgprLayout &Layout,
    BasicBlock *ThreadLoopLatch, DenseMap<uint64_t, BasicBlock *> &OffsetToBb,
    ArrayRef<uint8_t> SourceTextBytes, uint64_t SourceTextBaseAddress,
    ArrayRef<TextSection::ImageSection> SourceImageSections,
    uint64_t KernelStartOffset, uint64_t KernelEndOffset)
    : B(B), Regs(Regs), Projection(Projection), MC(MC),
      TargetCodeObjectVersion(TargetCodeObjectVersion), Kernargs(Kernargs),
      Layout(Layout), ThreadLoopLatch(ThreadLoopLatch), OffsetToBb(OffsetToBb),
      SourceTextBytes(SourceTextBytes),
      SourceTextBaseAddress(SourceTextBaseAddress),
      SourceImageSections(SourceImageSections),
      KernelStartOffset(KernelStartOffset), KernelEndOffset(KernelEndOffset) {
  SgprShadows.reserve(Regs.Sgpr.size());
  for (unsigned I = 0, E = Regs.Sgpr.size(); I != E; ++I) {
    AllocaInst *WaveMask = B.CreateAlloca(Projection.execStorageTy(), nullptr,
                                          "sgpr_mask_shadow_" + Twine(I));
    AllocaInst *WaveMaskValid =
        B.CreateAlloca(B.getInt1Ty(), nullptr, "sgpr_mask_valid_" + Twine(I));
    AllocaInst *SourceWavePair = B.CreateAlloca(
        B.getInt64Ty(), nullptr, "source_wave_sgpr_pair_" + Twine(I));
    AllocaInst *SourceWavePairValid = B.CreateAlloca(
        B.getInt1Ty(), nullptr, "source_wave_sgpr_pair_valid_" + Twine(I));
    B.CreateStore(ConstantInt::get(Projection.execStorageTy(), 0), WaveMask);
    B.CreateStore(B.getFalse(), WaveMaskValid);
    B.CreateStore(B.getInt64(0), SourceWavePair);
    B.CreateStore(B.getFalse(), SourceWavePairValid);
    SgprShadows.push_back(
        {WaveMask, WaveMaskValid, SourceWavePair, SourceWavePairValid});
  }
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

void RaiseContext::computeVGPRAdjust(const DecodedInst &Di) {
  unsigned Opc = Di.Inst.getOpcode();
  const MCInstrDesc &Desc = MC.InstrInfo->get(Opc);
  CurrentVgprAdjust.assign(std::max(Di.numOperands(), Desc.getNumOperands()),
                           0u);
  if (VgprMsBs == 0)
    return;

  // Operand slots are format-specific rather than positional, so use the
  // backend's operand-role table to apply each two-bit VGPR bank field.
  auto [XOps, YOps] = AMDGPU::getVGPRLoweringOperandTables(Desc);
  if (!XOps && !YOps)
    return;

  for (unsigned Slot = 0; Slot != 4; ++Slot) {
    unsigned Adjust =
        ((static_cast<unsigned>(VgprMsBs) >> (Slot * 2)) & 0x3u) * 256u;
    if (Adjust == 0)
      continue;
    auto RecordAdjustment = [&](const AMDGPU::OpName *Ops) {
      if (!Ops || Ops[Slot] == AMDGPU::OpName::NUM_OPERAND_NAMES)
        return;
      int OpIdx = AMDGPU::getNamedOperandIdx(Opc, Ops[Slot]);
      if (OpIdx < 0)
        llvm_unreachable("VGPR operand table names a missing operand");
      if (static_cast<unsigned>(OpIdx) >= CurrentVgprAdjust.size())
        llvm_unreachable("VGPR operand index exceeds instruction operands");
      CurrentVgprAdjust[OpIdx] = Adjust;
    };
    RecordAdjustment(XOps);
    RecordAdjustment(YOps);
  }
}

// Return Reg's position in RC, or std::nullopt if it is not a member.
static std::optional<unsigned> findIndexInClass(const MCRegisterClass &RC,
                                                MCRegister Reg) {
  for (unsigned I = 0, E = RC.getNumRegs(); I != E; ++I)
    if (RC.getRegister(I) == Reg)
      return I;
  return std::nullopt;
}

Expected<ParsedReg> RaiseContext::parseReg(const DecodedInst &Di,
                                           unsigned OperandIndex) const {
  assert(OperandIndex < Di.numOperands() && "operand index out of range");
  assert(Di.isReg(OperandIndex) && "operand must be a register");
  MCRegister Reg = Di.getReg(OperandIndex);
  ParsedReg Pr;
  if (!Reg) {
    Pr.RegKind = ParsedReg::NOREG;
    return Pr;
  }

  const MCRegisterInfo &MRI = *MC.RegInfo;
  const MCRegister CanonicalReg = AMDGPU::mc2PseudoReg(Reg);
  const auto RegisterFailure = [&](const Twine &Detail) -> Error {
    return RaiseFailure::atInstruction(
        RaiseFailureReason::UnsupportedInstructionForm,
        strippedMnemonic(MC, Di.Inst), Di.Offset,
        formatName(Di.TargetSpecificFlags),
        Twine("register-decode: ") + Detail);
  };

  const MCInstrDesc &Descriptor = MC.InstrInfo->get(Di.Inst.getOpcode());
  if (OperandIndex >= Descriptor.getNumOperands())
    llvm_unreachable("register operand has no instruction descriptor");
  const MCOperandInfo &OperandInfo = Descriptor.operands()[OperandIndex];
  if (OperandInfo.RegClass == -1)
    llvm_unreachable("register operand has no register class");
  const int16_t RegisterClassID = MC.InstrInfo->getOpRegClassID(
      OperandInfo,
      MC.SubtargetInfo->getHwMode(MCSubtargetInfo::HwMode_RegInfo));
  if (RegisterClassID < 0)
    llvm_unreachable("register class lookup failed");
  const MCRegisterClass &RegisterClass =
      MRI.getRegClass(static_cast<unsigned>(RegisterClassID));
  if (!RegisterClass.contains(CanonicalReg) &&
      !AMDGPU::isInlineValue(CanonicalReg))
    return RegisterFailure(Twine("register '") + MRI.getName(Reg) +
                           "' is not in operand register class '" +
                           MRI.getRegClassName(&RegisterClass) + "'");
  const unsigned WidthInDwords =
      divideCeil(AMDGPU::getRegBitWidth(RegisterClass), 32u);

  MCRegister Lane = MRI.getSubReg(Reg, AMDGPU::sub0);
  if (!Lane)
    Lane = Reg;
  Lane = AMDGPU::mc2PseudoReg(Lane);

  switch (Lane) {
  case AMDGPU::VCC_HI:
    // VCC_HI is a scratch scalar, not part of VCC, on wave32.
    if (Projection.sourceIsa().isWave32()) {
      Pr.RegKind = ParsedReg::VCC_HI_SCRATCH;
      Pr.WidthInDwords = 1;
      return Pr;
    }
    [[fallthrough]];
  case AMDGPU::VCC_LO:
    Pr.RegKind = ParsedReg::VCC;
    Pr.BaseIdx = (Lane == AMDGPU::VCC_HI) ? 1 : 0;
    Pr.WidthInDwords =
        CanonicalReg == AMDGPU::VCC
            ? static_cast<uint8_t>(Projection.sourceIsa().waveSize() / 32)
            : 1;
    return Pr;
  case AMDGPU::EXEC_HI:
    // EXEC_HI is a scratch scalar, not part of EXEC, on wave32.
    if (Projection.sourceIsa().isWave32()) {
      Pr.RegKind = ParsedReg::EXEC_HI_SCRATCH;
      Pr.WidthInDwords = 1;
      return Pr;
    }
    [[fallthrough]];
  case AMDGPU::EXEC_LO:
    Pr.RegKind = ParsedReg::EXEC;
    Pr.BaseIdx = (Lane == AMDGPU::EXEC_HI) ? 1 : 0;
    Pr.WidthInDwords = WidthInDwords;
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
    Pr.BaseIdx = (Lane == AMDGPU::FLAT_SCR_HI) ? 1 : 0;
    Pr.WidthInDwords = CanonicalReg == AMDGPU::FLAT_SCR ? 2 : 1;
    return Pr;
  // GFX11+ uses SGPR_NULL / SGPR_NULL_HI (and the 64-bit pair SGPR_NULL64)
  // as carry-discard sinks, e.g. `v_mad_co_u64_u32 ..., null, ...`. They
  // have no backing slot -- treat writes to them as no-ops.
  case AMDGPU::SGPR_NULL:
  case AMDGPU::SGPR_NULL_HI:
    Pr.RegKind = ParsedReg::NOREG;
    return Pr;
  case AMDGPU::XNACK_MASK_LO:
  case AMDGPU::XNACK_MASK_HI:
    return RegisterFailure(Twine("unsupported register '") + MRI.getName(Reg) +
                           "'");
  // LDS_DIRECT (src_lds_direct, enc 254): reads a dword from LDS at the
  // byte offset held in M0. Used as a VALU source after buffer_load_*_lds.
  case AMDGPU::LDS_DIRECT:
    Pr.RegKind = ParsedReg::LDS_DIRECT;
    Pr.WidthInDwords = 1;
    return Pr;
  // Source-only predicates have no backing register-file slot.
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
  // Runtime-defined aperture registers have no static IR representation.
  case AMDGPU::SRC_SHARED_BASE_LO:
  case AMDGPU::SRC_SHARED_LIMIT_LO:
  case AMDGPU::SRC_PRIVATE_BASE_LO:
  case AMDGPU::SRC_PRIVATE_LIMIT_LO:
  case AMDGPU::SRC_POPS_EXITING_WAVE_ID:
  case AMDGPU::SRC_FLAT_SCRATCH_BASE_LO:
  case AMDGPU::SRC_FLAT_SCRATCH_BASE_HI:
    return RegisterFailure(Twine("unsupported register '") + MRI.getName(Reg) +
                           "'");
  default:
    break;
  }

  // The hardware encoding identifies vector and accumulator register families.
  unsigned Enc = MRI.getEncodingValue(Reg);
  unsigned HwIdx = Enc & AMDGPU::HWEncoding::REG_IDX_MASK;

  if (Enc & AMDGPU::HWEncoding::IS_AGPR) {
    Pr.RegKind = ParsedReg::AGPR;
    Pr.WidthInDwords = WidthInDwords;
    if (OperandIndex < CurrentVgprAdjust.size())
      HwIdx += CurrentVgprAdjust[OperandIndex];
    Pr.BaseIdx = HwIdx;
    return Pr;
  }
  if (Enc & AMDGPU::HWEncoding::IS_VGPR) {
    Pr.RegKind = ParsedReg::VGPR;
    Pr.WidthInDwords = WidthInDwords;
    if (OperandIndex < CurrentVgprAdjust.size())
      HwIdx += CurrentVgprAdjust[OperandIndex];
    Pr.BaseIdx = HwIdx;
    return Pr;
  }

  // TTMP encodings vary by generation; class position is the stable index.
  const MCRegisterClass &TTMP32 = MRI.getRegClass(AMDGPU::TTMP_32RegClassID);
  if (std::optional<unsigned> Index = findIndexInClass(TTMP32, Lane)) {
    Pr.RegKind = ParsedReg::TTMP;
    Pr.BaseIdx = *Index;
    Pr.WidthInDwords = WidthInDwords;
    return Pr;
  }

  // The broader SReg_32 class also contains architectural registers handled
  // above, so classify general SGPRs through SGPR_32.
  if (MRI.getRegClass(AMDGPU::SGPR_32RegClassID).contains(Lane)) {
    Pr.RegKind = ParsedReg::SGPR;
    Pr.BaseIdx = HwIdx;
    Pr.WidthInDwords = WidthInDwords;
    return Pr;
  }

  return RaiseFailure::atInstruction(
      RaiseFailureReason::UnsupportedInstructionForm,
      strippedMnemonic(MC, Di.Inst), Di.Offset,
      formatName(Di.TargetSpecificFlags),
      Twine("register-decode: could not classify register '") +
          MRI.getName(Reg) + "' (enc=0x" + Twine::utohexstr(Enc) + ")");
}

Expected<Value *> RaiseContext::readOp32(const DecodedInst &Di,
                                         unsigned OpIdx) {
  IntegerType *I32Ty = B.getInt32Ty();
  if (Di.isReg(OpIdx)) {
    Expected<ParsedReg> Reg = parseReg(Di, OpIdx);
    if (!Reg)
      return Reg.takeError();
    ParsedReg Pr = *Reg;
    if (Pr.RegKind == ParsedReg::VCC) {
      if (Projection.sourceWaveScopedLaneOps()) {
        Value *Mask = Regs.readVCCAsWaveMask(B, Projection.execStorageTy());
        Value *Lo = B.CreateTrunc(Mask, I32Ty, "vcc_src_wave_lo");
        Value *Hi =
            B.CreateTrunc(B.CreateLShr(Mask, Projection.sourceIsa().waveSize()),
                          I32Ty, "vcc_src_wave_hi");
        Value *Lane = Projection.emitLaneIdx(B);
        Value *Upper = B.CreateICmpUGE(
            Lane, ConstantInt::get(I32Ty, Projection.sourceIsa().waveSize()),
            "vcc_src_wave_upper");
        return B.CreateSelect(Upper, Hi, Lo, "vcc_src_wave_mask");
      }
      return Regs.readReg32(B, Pr);
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
    Value *V = Regs.readReg32(B, Pr);
    if (!V)
      return RaiseFailure::atInstruction(
          RaiseFailureReason::UnsupportedInstructionForm,
          strippedMnemonic(MC, Di.Inst), Di.Offset,
          formatName(Di.TargetSpecificFlags),
          Twine("operand-read: could not read 32-bit register '") +
              MC.RegInfo->getName(Di.getReg(OpIdx)) + "' in " +
              strippedMnemonic(MC, Di.Inst));
    return V;
  }
  if (std::optional<int64_t> Val = evalOperandAsConst(Di.Inst, OpIdx)) {
    return ConstantInt::get(I32Ty, static_cast<uint32_t>(*Val));
  }
  return RaiseFailure::atInstruction(
      RaiseFailureReason::UnsupportedInstructionForm,
      strippedMnemonic(MC, Di.Inst), Di.Offset,
      formatName(Di.TargetSpecificFlags),
      Twine("operand-read: could not resolve 32-bit operand ") + Twine(OpIdx) +
          " in " + strippedMnemonic(MC, Di.Inst));
}

Expected<Value *> RaiseContext::readOpSourceWaveMask32(const DecodedInst &Di,
                                                       unsigned OpIdx) {
  if (!Di.isReg(OpIdx))
    return readOp32(Di, OpIdx);

  Expected<ParsedReg> Reg = parseReg(Di, OpIdx);
  if (!Reg)
    return Reg.takeError();
  ParsedReg Pr = *Reg;
  if (Pr.RegKind == ParsedReg::EXEC)
    return Projection.emitCurrentSourceWaveMask(B, Regs.loadExec(B),
                                                "exec_srcwave_mask");
  if (Pr.RegKind == ParsedReg::VCC)
    return Projection.emitCurrentSourceWaveMask(
        B, Regs.readVCCAsWaveMask(B, Projection.execStorageTy()),
        "vcc_srcwave_mask");
  if (Pr.RegKind == ParsedReg::SGPR && Pr.BaseIdx) {
    Expected<Value *> Fallback = readOp32(Di, OpIdx);
    if (!Fallback)
      return Fallback.takeError();
    if (Value *ShadowValid = loadSgprWaveMaskValid(*Pr.BaseIdx)) {
      Value *ShadowExec = loadSgprWaveMaskExec(*Pr.BaseIdx);
      if (ShadowExec->getType() != Projection.execStorageTy())
        ShadowExec = B.CreateZExtOrTrunc(ShadowExec, Projection.execStorageTy(),
                                         "sgpr_mask_exec_cast");
      Value *ShadowMask = Projection.emitCurrentSourceWaveMask(
          B, ShadowExec, "sgpr_srcwave_mask_shadow");
      return B.CreateSelect(ShadowValid, ShadowMask, *Fallback,
                            "sgpr_srcwave_mask");
    }
    return *Fallback;
  }

  return readOp32(Di, OpIdx);
}

Expected<Value *> RaiseContext::readOp64(const DecodedInst &Di,
                                         unsigned OpIdx) {
  IntegerType *I64Ty = B.getInt64Ty();
  if (Di.isReg(OpIdx)) {
    Expected<ParsedReg> Reg = parseReg(Di, OpIdx);
    if (!Reg)
      return Reg.takeError();
    ParsedReg Pr = *Reg;
    if (Pr.RegKind == ParsedReg::VCC)
      return Regs.readVCCAsWaveMask(B, I64Ty);
    if (Pr.RegKind == ParsedReg::EXEC) {
      Value *V = Regs.loadExec(B);
      if (V->getType() != I64Ty)
        V = B.CreateZExt(V, I64Ty, "exec_ext");
      return V;
    }
    // These unbacked architectural registers read as zero for compute kernels.
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
    Value *V = Regs.readReg64(B, Pr);
    if (!V)
      return RaiseFailure::atInstruction(
          RaiseFailureReason::UnsupportedInstructionForm,
          strippedMnemonic(MC, Di.Inst), Di.Offset,
          formatName(Di.TargetSpecificFlags),
          Twine("operand-read: could not read 64-bit register '") +
              MC.RegInfo->getName(Di.getReg(OpIdx)) + "' in " +
              strippedMnemonic(MC, Di.Inst));
    return V;
  }
  if (std::optional<int64_t> Val = evalOperandAsConst(Di.Inst, OpIdx)) {
    return ConstantInt::getSigned(I64Ty, *Val);
  }
  return RaiseFailure::atInstruction(
      RaiseFailureReason::UnsupportedInstructionForm,
      strippedMnemonic(MC, Di.Inst), Di.Offset,
      formatName(Di.TargetSpecificFlags),
      Twine("operand-read: could not resolve 64-bit operand ") + Twine(OpIdx) +
          " in " + strippedMnemonic(MC, Di.Inst));
}

Value *RaiseContext::emitLaneIdx() { return Projection.emitLaneIdx(B); }

Value *RaiseContext::freezeMemAddr(Value *Addr) {
  if (!Projection.sourceIsa().isWave32() || Projection.targetIsa().isWave32())
    return Addr;
  return B.CreateFreeze(Addr, "mem_addr_frozen");
}

Value *RaiseContext::emitLaneActiveBit() {
  // Linear lane-active diamonds remain dominated by the first value emitted
  // for an instruction. Instruction boundaries and EXEC writes reset it.
  if (CachedLaneActive)
    return CachedLaneActive;

  Value *Active = Projection.emitLaneActiveBit(B, Regs.loadExec(B));
  CachedLaneActive = Active;
  return Active;
}

void RaiseContext::writeReg32(ParsedReg Pr, Value *V) {
  if (Pr.RegKind == ParsedReg::NOREG)
    return;
  if (Pr.RegKind == ParsedReg::VGPR || Pr.RegKind == ParsedReg::AGPR) {
    emitUnderExec([&] { Regs.writeReg32(B, Pr, V); });
  } else {
    Regs.writeReg32(B, Pr, V);
    if (Pr.RegKind == ParsedReg::EXEC)
      resetLaneActiveCache();
    else if (Pr.RegKind == ParsedReg::SGPR) {
      assert(Pr.BaseIdx && "SGPR must have a base register index");
      invalidateSgprWaveMaskI1(*Pr.BaseIdx);
    } else if (Pr.RegKind == ParsedReg::M0) {
      updateM0Const(V);
    }
  }
}

void RaiseContext::writeReg64(ParsedReg Pr, Value *V) {
  if (Pr.RegKind == ParsedReg::NOREG)
    return;
  if (Pr.RegKind == ParsedReg::VGPR || Pr.RegKind == ParsedReg::AGPR) {
    emitUnderExec([&] { Regs.writeReg64(B, Pr, V); });
  } else {
    Regs.writeReg64(B, Pr, V);
    if (Pr.RegKind == ParsedReg::EXEC)
      resetLaneActiveCache();
    else if (Pr.RegKind == ParsedReg::SGPR) {
      assert(Pr.BaseIdx && "SGPR must have a base register index");
      invalidateSgprWaveMaskI1(*Pr.BaseIdx);
      invalidateSgprWaveMaskI1(*Pr.BaseIdx + 1);
    }
  }
}

void RaiseContext::writeRegVec(ParsedReg Pr, Value *V) {
  if (Pr.RegKind == ParsedReg::NOREG)
    return;
  if (Pr.RegKind == ParsedReg::VGPR || Pr.RegKind == ParsedReg::AGPR) {
    emitUnderExec([&] { Regs.writeRegVec(B, Pr, V); });
  } else {
    // Vector-valued scalar writes cannot target EXEC.
    Regs.writeRegVec(B, Pr, V);
    if (Pr.RegKind == ParsedReg::SGPR) {
      assert(Pr.BaseIdx && "SGPR must have a base register index");
      for (unsigned I = 0; I != Pr.WidthInDwords; ++I)
        invalidateSgprWaveMaskI1(*Pr.BaseIdx + I);
    }
  }
}

void RaiseContext::writeRegExecWidth(ParsedReg Pr, Value *V) {
  if (Pr.RegKind == ParsedReg::NOREG)
    return;
  // Wave-mask writes are wave-level effects and must not be EXEC-predicated.
  Regs.writeRegExecWidth(B, Pr, V);
  if (Pr.RegKind == ParsedReg::EXEC)
    resetLaneActiveCache();
  else if (Pr.RegKind == ParsedReg::SGPR) {
    assert(Pr.BaseIdx && "SGPR must have a base register index");
    unsigned WidthInDwords =
        Projection.sourceWaveScopedLaneOps() && Pr.WidthInDwords >= 2
            ? 2
            : Projection.sourceWaveMaskTy()
                      ->getPrimitiveSizeInBits()
                      .getFixedValue() /
                  32;
    for (unsigned I = 0; I != WidthInDwords; ++I)
      invalidateSgprWaveMaskI1(*Pr.BaseIdx + I);
  }
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
  BasicBlock *DoBb = BasicBlock::Create(B.getContext(), "spe_do", F);
  BasicBlock *SkipBb = BasicBlock::Create(B.getContext(), "spe_skip", F);
  KernargPtrProvenance PreProvenance = CurrentKernargPtrProvenance;
  B.CreateCondBr(Active, DoBb, SkipBb);

  B.SetInsertPoint(DoBb);
  Body();
  KernargPtrProvenance DoProvenance = CurrentKernargPtrProvenance;
  // Body may terminate its block; do not add a second terminator.
  if (!B.GetInsertBlock()->hasTerminator()) {
    B.CreateBr(SkipBb);
    CurrentKernargPtrProvenance =
        joinKernargPtrProvenance(PreProvenance, DoProvenance);
  } else {
    CurrentKernargPtrProvenance = PreProvenance;
  }

  B.SetInsertPoint(SkipBb);
}

Expected<Value *> RaiseContext::readOpExecWidth(const DecodedInst &Di,
                                                unsigned OpIdx) {
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
    Expected<ParsedReg> Reg = parseReg(Di, OpIdx);
    if (!Reg)
      return Reg.takeError();
    ParsedReg Pr = *Reg;
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
              : (Projection.sourceIsa().isWave32()
                     ? Regs.loadSGPR32(B, BaseIdx)
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
    return RaiseFailure::atInstruction(
        RaiseFailureReason::UnsupportedInstructionForm,
        strippedMnemonic(MC, Di.Inst), Di.Offset,
        formatName(Di.TargetSpecificFlags),
        Twine("operand-read: could not read EXEC-width register '") +
            MC.RegInfo->getName(Di.getReg(OpIdx)) + "' in " +
            strippedMnemonic(MC, Di.Inst));
  }
  // Interpret immediate masks at source width and replicate them like SGPR
  // operands when widening.
  Type *SrcTy =
      Projection.sourceIsa().isWave32() ? B.getInt32Ty() : B.getInt64Ty();
  uint64_t SrcMask =
      Projection.sourceIsa().isWave32() ? 0xFFFFFFFFull : 0xFFFFFFFFFFFFFFFFull;
  if (std::optional<int64_t> Val = evalOperandAsConst(Di.Inst, OpIdx)) {
    uint64_t Bits = static_cast<uint64_t>(*Val) & SrcMask;
    Value *Narrow = ConstantInt::get(SrcTy, Bits, /*IsSigned=*/false);
    return WidenToExec(Narrow);
  }
  return RaiseFailure::atInstruction(
      RaiseFailureReason::UnsupportedInstructionForm,
      strippedMnemonic(MC, Di.Inst), Di.Offset,
      formatName(Di.TargetSpecificFlags),
      Twine("operand-read: could not resolve EXEC-width operand ") +
          Twine(OpIdx) + " in " + strippedMnemonic(MC, Di.Inst));
}

} // namespace COMGR::hotswap
