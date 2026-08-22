//===- register-state.cpp - Hotswap transpiler ----------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap/raiser/register-state.h"

#include "hotswap/decoder/amdgpu-formats.h"
#include "hotswap/decoder/amdgpu-mc-tables.h"
#include "hotswap/raiser/raise_failure.h"

#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIDefines.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <utility>

using namespace llvm;

namespace COMGR::hotswap {

Expected<RegisterState> RegisterState::create(IRBuilder<> &B,
                                              const WaveProjection &Projection,
                                              const MCState &MC,
                                              const KernelMeta &Meta) {
  UserSgprLayout Layout;
  if (Error Err = UserSgprLayout::tryFromKernelMeta(
          Meta, Projection.sourceIsa(), MC.SubtargetInfo->getCPU(), Layout))
    return std::move(Err);
  return RegisterState(B, Projection, MC, std::move(Layout));
}

RegisterState::RegisterState(IRBuilder<> &B, const WaveProjection &Projection,
                             const MCState &MC, UserSgprLayout Layout)
    : B(B), Projection(Projection), MC(MC), Layout(std::move(Layout)) {
  Regs.init(B, B.getInt32Ty(), B.getInt1Ty(), Projection.sourceIsa(),
            *MC.RegInfo, Projection);

  SgprShadows.reserve(Regs.Sgpr.size());
  for (unsigned I = 0, E = Regs.Sgpr.size(); I != E; ++I) {
    AllocaInst *WaveMask = B.CreateAlloca(Projection.execStorageTy(), nullptr,
                                          "sgpr_mask_shadow_" + Twine(I));
    AllocaInst *WaveMaskValid =
        B.CreateAlloca(B.getInt1Ty(), nullptr, "sgpr_mask_valid_" + Twine(I));
    AllocaInst *WaveMaskIsPair =
        B.CreateAlloca(B.getInt1Ty(), nullptr, "sgpr_mask_is_pair_" + Twine(I));
    AllocaInst *SourceWavePair = B.CreateAlloca(
        B.getInt64Ty(), nullptr, "source_wave_sgpr_pair_" + Twine(I));
    AllocaInst *SourceWavePairValid = B.CreateAlloca(
        B.getInt1Ty(), nullptr, "source_wave_sgpr_pair_valid_" + Twine(I));
    B.CreateStore(ConstantInt::get(Projection.execStorageTy(), 0), WaveMask);
    B.CreateStore(B.getFalse(), WaveMaskValid);
    B.CreateStore(B.getFalse(), WaveMaskIsPair);
    B.CreateStore(B.getInt64(0), SourceWavePair);
    B.CreateStore(B.getFalse(), SourceWavePairValid);
    SgprShadows.push_back({WaveMask, WaveMaskValid, WaveMaskIsPair,
                           SourceWavePair, SourceWavePairValid});
  }
}

void RegisterState::computeVGPRAdjust(const DecodedInst &Di) {
  unsigned Opc = Di.Inst.getOpcode();
  const MCInstrDesc &Desc = MC.InstrInfo->get(Opc);
  CurrentVgprAdjust.assign(std::max(Di.numOperands(), Desc.getNumOperands()),
                           0u);
  if (VgprMsBs == 0)
    return;

  // Operand slots are format-specific rather than positional.
  const VGPRMSBOperandIndices OperandIndices = getVGPRMSBOperandIndices(Desc);
  for (unsigned Slot = 0; Slot != OperandIndices.size(); ++Slot) {
    unsigned Adjust =
        ((static_cast<unsigned>(VgprMsBs) >> (Slot * 2)) & 0x3u) * 256u;
    if (Adjust == 0)
      continue;
    auto [XOperandIndex, YOperandIndex] = OperandIndices[Slot];
    auto RecordAdjustment = [&](std::optional<unsigned> OperandIndex) {
      if (!OperandIndex)
        return;
      if (*OperandIndex >= CurrentVgprAdjust.size())
        llvm_unreachable("VGPR operand index exceeds instruction operands");
      CurrentVgprAdjust[*OperandIndex] = Adjust;
    };
    RecordAdjustment(XOperandIndex);
    RecordAdjustment(YOperandIndex);
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

// Return the selected 32-bit register half from the current source-wave mask.
static Value *emitSourceWaveMask32(IRBuilder<> &B,
                                   const WaveProjection &Projection,
                                   Value *Mask, ParsedReg Reg,
                                   const Twine &Name) {
  IntegerType *I32Ty = B.getInt32Ty();
  Mask = Projection.emitCurrentSourceWaveMask(B, Mask, Name);
  if (Mask->getType() == I32Ty) {
    return Mask;
  }
  if (Reg.WidthInDwords < 2 && Reg.BaseIdx == 1) {
    Mask = B.CreateLShr(Mask, 32, Name + "_hi_shr");
  }
  return B.CreateTrunc(
      Mask, I32Ty,
      Reg.WidthInDwords < 2 && Reg.BaseIdx == 1 ? Name + "_hi" : Name + "_lo");
}

// Return whether the current source wave's mask is empty.
static Value *emitSourceWaveMaskIsZero(IRBuilder<> &B,
                                       const WaveProjection &Projection,
                                       Value *Mask, const Twine &Name) {
  Mask = Projection.emitCurrentSourceWaveMask(B, Mask, Name + "_mask");
  return B.CreateICmpEQ(Mask, ConstantInt::get(Mask->getType(), 0), Name);
}

Expected<ParsedReg> RegisterState::parseReg(const DecodedInst &Di,
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
  const MCRegister CanonicalReg = stripRegEncoding(Reg);
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
  if (!RegisterClass.contains(CanonicalReg) && !isInlineValue(CanonicalReg))
    return RegisterFailure(Twine("register '") + MRI.getName(Reg) +
                           "' is not in operand register class '" +
                           MRI.getRegClassName(&RegisterClass) + "'");
  const unsigned WidthInDwords = divideCeil(RegisterClass.getSizeInBits(), 32u);

  MCRegister Lane = MRI.getSubReg(Reg, AMDGPU::sub0);
  if (!Lane)
    Lane = Reg;
  Lane = stripRegEncoding(Lane);

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

Expected<Value *> RegisterState::readOp32(const DecodedInst &Di,
                                          unsigned OpIdx) {
  IntegerType *I32Ty = B.getInt32Ty();
  if (Di.isReg(OpIdx)) {
    Expected<ParsedReg> Reg = parseReg(Di, OpIdx);
    if (!Reg)
      return Reg.takeError();
    ParsedReg Pr = *Reg;
    if (Pr.RegKind == ParsedReg::VCC) {
      Value *Mask = Regs.readVCCAsWaveMask(B, Projection.execStorageTy());
      return emitSourceWaveMask32(B, Projection, Mask, Pr, "vcc_src_wave");
    }
    if (Pr.RegKind == ParsedReg::EXEC) {
      return emitSourceWaveMask32(B, Projection, Regs.loadExec(B), Pr,
                                  "exec_src_wave");
    }
    if (Pr.RegKind == ParsedReg::SCC)
      return B.CreateZExt(Regs.loadSCC(B), I32Ty);
    if (Pr.RegKind == ParsedReg::SRC_SCC)
      return B.CreateZExt(Regs.loadSCC(B), I32Ty);
    if (Pr.RegKind == ParsedReg::SRC_VCCZ) {
      Value *Vcc = Regs.readVCCAsWaveMask(B, Projection.execStorageTy());
      return B.CreateZExt(emitSourceWaveMaskIsZero(B, Projection, Vcc, "vccz"),
                          I32Ty);
    }
    if (Pr.RegKind == ParsedReg::SRC_EXECZ) {
      Value *Exec = Regs.loadExec(B);
      return B.CreateZExt(
          emitSourceWaveMaskIsZero(B, Projection, Exec, "execz"), I32Ty);
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

Expected<Value *> RegisterState::readOpSourceWaveMask32(const DecodedInst &Di,
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

Expected<Value *> RegisterState::readOp64(const DecodedInst &Di,
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
      return B.CreateZExt(emitSourceWaveMaskIsZero(B, Projection, Vcc, "vccz"),
                          I64Ty);
    }
    if (Pr.RegKind == ParsedReg::SRC_EXECZ) {
      Value *Exec = Regs.loadExec(B);
      return B.CreateZExt(
          emitSourceWaveMaskIsZero(B, Projection, Exec, "execz"), I64Ty);
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

Value *RegisterState::emitLaneActiveBit() {
  // Linear lane-active diamonds remain dominated by the first value emitted
  // for an instruction. Instruction boundaries and EXEC writes reset it.
  if (CachedLaneActive)
    return CachedLaneActive;

  Value *Active = Projection.emitLaneActiveBit(B, Regs.loadExec(B));
  CachedLaneActive = Active;
  return Active;
}

void RegisterState::writeReg32(ParsedReg Pr, Value *V) {
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

void RegisterState::writeReg64(ParsedReg Pr, Value *V) {
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

void RegisterState::writeRegVec(ParsedReg Pr, Value *V) {
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

void RegisterState::writeRegExecWidth(ParsedReg Pr, Value *V) {
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

void RegisterState::storeVGPR32(unsigned Idx, Value *V) {
  emitUnderExec([&] { Regs.storeVGPR32(B, Idx, V); });
}

void RegisterState::storeVGPR64(unsigned Idx, Value *V) {
  emitUnderExec([&] { Regs.storeVGPR64(B, Idx, V); });
}

void RegisterState::storeAGPR32(unsigned Idx, Value *V) {
  emitUnderExec([&] { Regs.storeAGPR32(B, Idx, V); });
}

void RegisterState::emitUnderExec(llvm::function_ref<void()> Body) {
  Value *Active = emitLaneActiveBit();
  BasicBlock *PreBb = B.GetInsertBlock();
  Function *F = PreBb->getParent();
  BasicBlock *DoBb = BasicBlock::Create(B.getContext(), "spe_do", F);
  BasicBlock *SkipBb = BasicBlock::Create(B.getContext(), "spe_skip", F);
  B.CreateCondBr(Active, DoBb, SkipBb);

  B.SetInsertPoint(DoBb);
  Body();
  // Body may terminate its block; do not add a second terminator.
  if (!B.GetInsertBlock()->hasTerminator())
    B.CreateBr(SkipBb);

  B.SetInsertPoint(SkipBb);
}

Expected<Value *> RegisterState::readOpExecWidth(const DecodedInst &Di,
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

void RegisterState::recordSgprWaveMaskI1(unsigned BaseIdx, Value *CmpI1,
                                         bool IsPair) {
  LastSgprWaveMaskI1[BaseIdx] = WaveMaskEntry{CmpI1, IsPair};
  if (BaseIdx < SgprShadows.size()) {
    Value *ExecMask = Projection.ballotI1ToWidth(
        B, CmpI1, Projection.execStorageTy(), "wm_shadow_exec");
    B.CreateStore(ExecMask, SgprShadows[BaseIdx].WaveMask);
    B.CreateStore(B.getTrue(), SgprShadows[BaseIdx].WaveMaskValid);
    B.CreateStore(B.getInt1(IsPair), SgprShadows[BaseIdx].WaveMaskIsPair);
  }
}

Value *RegisterState::emitCurrentSourceWaveHasActiveLane() {
  Value *Exec = Regs.loadExec(B);
  if (!Projection.providesFullWaveExecInvariant())
    return emitLaneActiveBit();
  const ISAProfile &SourceIsa = Projection.sourceIsa();
  unsigned SourceBits = SourceIsa.waveSize();
  assert(SourceIsa.hasValidWaveSize() && "source wave size must be 32 or 64");
  if (SourceBits >= 64)
    return B.CreateICmpNE(Exec, ConstantInt::get(Exec->getType(), 0),
                          "source_wave_active");
  Type *ExecTy = Exec->getType();
  Value *Lane = B.CreateZExtOrTrunc(Projection.emitLaneIdx(B), ExecTy,
                                    "source_wave_lane");
  Value *Group = B.CreateUDiv(Lane, ConstantInt::get(ExecTy, SourceBits),
                              "source_wave_group");
  Value *Shift = B.CreateMul(Group, ConstantInt::get(ExecTy, SourceBits),
                             "source_wave_shift");
  Value *Shifted = B.CreateLShr(Exec, Shift, "source_wave_exec");
  uint64_t Mask = (uint64_t{1} << SourceBits) - 1;
  Value *GroupMask =
      B.CreateAnd(Shifted, ConstantInt::get(ExecTy, Mask), "source_wave_mask");
  return B.CreateICmpNE(GroupMask, ConstantInt::get(ExecTy, 0),
                        "source_wave_active");
}

void RegisterState::recordSourceWaveSgprPair(unsigned BaseIdx, Value *V) {
  if (!Projection.providesFullWaveExecInvariant()) {
    return;
  }
  if (BaseIdx >= SgprShadows.size()) {
    return;
  }
  const SgprShadow &Shadow = SgprShadows[BaseIdx];
  Value *Old = B.CreateLoad(B.getInt64Ty(), Shadow.SourceWavePair,
                            "source_wave_sgpr_pair_old");
  Value *OldValid = B.CreateLoad(B.getInt1Ty(), Shadow.SourceWavePairValid,
                                 "source_wave_sgpr_pair_valid_old");
  Value *Active = emitCurrentSourceWaveHasActiveLane();
  Value *Merged = B.CreateSelect(Active, V, Old, "source_wave_sgpr_pair");
  Value *Valid = B.CreateSelect(Active, B.getTrue(), OldValid,
                                "source_wave_sgpr_pair_valid");
  B.CreateStore(Merged, Shadow.SourceWavePair);
  B.CreateStore(Valid, Shadow.SourceWavePairValid);
}

Value *RegisterState::materializeSourceWaveSgprPair(unsigned BaseIdx,
                                                    Value *Fallback) {
  if (!Projection.providesFullWaveExecInvariant() ||
      BaseIdx >= SgprShadows.size()) {
    return Fallback;
  }
  const SgprShadow &Shadow = SgprShadows[BaseIdx];
  Value *Recorded = B.CreateLoad(B.getInt64Ty(), Shadow.SourceWavePair,
                                 "source_wave_sgpr_pair");
  Value *Valid = B.CreateLoad(B.getInt1Ty(), Shadow.SourceWavePairValid,
                              "source_wave_sgpr_pair_valid");
  return B.CreateSelect(Valid, Recorded, Fallback, "source_wave_sgpr_pair_sel");
}

Value *RegisterState::loadSgprWaveMaskExec(unsigned BaseIdx) const {
  if (BaseIdx >= SgprShadows.size()) {
    return nullptr;
  }
  return B.CreateLoad(Projection.execStorageTy(), SgprShadows[BaseIdx].WaveMask,
                      "sgpr_mask_exec");
}

Value *RegisterState::loadSgprWaveMaskValid(unsigned BaseIdx) const {
  if (BaseIdx >= SgprShadows.size()) {
    return nullptr;
  }
  return B.CreateLoad(B.getInt1Ty(), SgprShadows[BaseIdx].WaveMaskValid,
                      "sgpr_mask_valid");
}

void RegisterState::invalidateSgprWaveMaskI1(unsigned BaseIdx) {
  LastSgprWaveMaskI1.erase(BaseIdx);
  SourceImageSgprPairAddrShadow.erase(BaseIdx);
  if (BaseIdx < SgprShadows.size()) {
    B.CreateStore(B.getFalse(), SgprShadows[BaseIdx].WaveMaskValid);
    B.CreateStore(B.getFalse(), SgprShadows[BaseIdx].SourceWavePairValid);
  }
  if (BaseIdx > 0) {
    DenseMap<unsigned, WaveMaskEntry>::iterator Prev =
        LastSgprWaveMaskI1.find(BaseIdx - 1);
    if (Prev != LastSgprWaveMaskI1.end() && Prev->second.IsPair) {
      LastSgprWaveMaskI1.erase(Prev);
    }
    if (BaseIdx - 1 < SgprShadows.size()) {
      const SgprShadow &Previous = SgprShadows[BaseIdx - 1];
      Value *PreviousValid = B.CreateLoad(B.getInt1Ty(), Previous.WaveMaskValid,
                                          "sgpr_mask_previous_valid");
      Value *PreviousIsPair = B.CreateLoad(
          B.getInt1Ty(), Previous.WaveMaskIsPair, "sgpr_mask_previous_is_pair");
      Value *KeepPrevious =
          B.CreateAnd(PreviousValid, B.CreateNot(PreviousIsPair),
                      "sgpr_mask_keep_previous");
      B.CreateStore(KeepPrevious, Previous.WaveMaskValid);
      B.CreateStore(B.getFalse(), SgprShadows[BaseIdx - 1].SourceWavePairValid);
    }
    SourceImageSgprPairAddrShadow.erase(BaseIdx - 1);
  }
}

std::optional<uint64_t>
RegisterState::lookupSourceImageSgprPairAddr(unsigned BaseIdx) const {
  DenseMap<unsigned, uint64_t>::const_iterator It =
      SourceImageSgprPairAddrShadow.find(BaseIdx);
  if (It == SourceImageSgprPairAddrShadow.end())
    return std::nullopt;
  return It->second;
}

void RegisterState::updateM0Const(Value *V) {
  if (ConstantInt *CI = dyn_cast<ConstantInt>(V))
    M0Const = CI->getZExtValue();
  else
    M0Const = std::nullopt;
}

void RegisterState::enterBlock() {
  LastSgprWaveMaskI1.clear();
  SourceImageSgprPairAddrShadow.clear();
  M0Const = std::nullopt;
  // The MSB mode is architectural rather than a raise-time fact: LLVM's
  // VGPR-encoding lowering resets it at every block boundary, so a raised block
  // must not inherit the mode a predecessor left set.
  VgprMsBs = 0;
  resetLaneActiveCache();
}

void RegisterState::invalidateSgprShadows() {
  for (const SgprShadow &Shadow : SgprShadows) {
    B.CreateStore(B.getFalse(), Shadow.WaveMaskValid);
    B.CreateStore(B.getFalse(), Shadow.SourceWavePairValid);
  }
}

void RegisterState::collectAllocas(SmallVectorImpl<AllocaInst *> &Out) const {
  Regs.collectAllocas(Out);
  for (const SgprShadow &Shadow : SgprShadows) {
    Out.push_back(Shadow.WaveMask);
    Out.push_back(Shadow.WaveMaskValid);
    Out.push_back(Shadow.WaveMaskIsPair);
    Out.push_back(Shadow.SourceWavePair);
    Out.push_back(Shadow.SourceWavePairValid);
  }
}

} // namespace COMGR::hotswap
