//===- AMDGPUSim/MCAdapter.cpp - MCInst Adapter ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Implementation of the MCInstInfo class for MCInst property queries.
//
//===----------------------------------------------------------------------===//

#include "MCAdapter.h"
#include "HWModel.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIDefines.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"

namespace llvm {
namespace AMDGPUSim {

//===----------------------------------------------------------------------===//
// Constructor
//===----------------------------------------------------------------------===//

MCInstInfo::MCInstInfo(const MCInstrInfo &MCII, const MCRegisterInfo &MRI)
    : MCII(MCII), MRI(MRI) {}

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

static bool isMCInstTRANS(uint64_t TSFlags) {
  return TSFlags & SIInstrFlags::TRANS;
}

static bool isMCInstWMMA(uint64_t TSFlags) {
  return (TSFlags & SIInstrFlags::IsWMMA) || (TSFlags & SIInstrFlags::IsSWMMAC);
}

static bool isMCInstWaitcnt(unsigned Opc) {
  switch (Opc) {
  case AMDGPU::S_WAIT_DSCNT:
  case AMDGPU::S_WAIT_LOADCNT:
  case AMDGPU::S_WAIT_STORECNT:
  case AMDGPU::S_WAIT_KMCNT:
  case AMDGPU::S_WAIT_TENSORCNT:
  case AMDGPU::S_WAIT_XCNT:
  case AMDGPU::S_WAITCNT_DEPCTR:
    return true;
  default:
    return false;
  }
}

//===----------------------------------------------------------------------===//
// Instruction Classification
//===----------------------------------------------------------------------===//

InstClass MCInstInfo::classifyInst(const MCInst &MI) const {
  unsigned Opc = MI.getOpcode();
  const MCInstrDesc &Desc = MCII.get(Opc);
  uint64_t TSFlags = Desc.TSFlags;

  // Get opcode name for pattern matching (handles encoding variants)
  StringRef Name = MCII.getName(Opc);

  // Check specific opcodes/names first
  if (Opc == AMDGPU::S_DELAY_ALU || Name.starts_with("S_DELAY_ALU"))
    return InstClass::DELAY_ALU;

  if (Opc == AMDGPU::S_SET_VGPR_MSB || Name.starts_with("S_SET_VGPR_MSB"))
    return InstClass::MSB_SET;

  // Check opcode name for V_NOP
  if (Name.starts_with("V_NOP"))
    return InstClass::VALU;

  if (Opc == AMDGPU::S_NOP || Name.starts_with("S_NOP") ||
      Name.starts_with("S_CLAUSE"))
    return InstClass::SALU;

  // Barrier instructions (use name matching for encoding variants)
  if (Name.starts_with("S_BARRIER"))
    return InstClass::BARRIER;

  // Wait instructions
  if (isMCInstWaitcnt(Opc) || Name.starts_with("S_WAIT"))
    return InstClass::WAITCNT;

  // Branch instructions
  if (Desc.isBranch())
    return InstClass::BRANCH;

  // WMMA instructions (check before VALU since WMMA has VALU flag too)
  if (isMCInstWMMA(TSFlags))
    return InstClass::WMMA;

  // TDM instructions
  if (Name.starts_with("TENSOR_LOAD_TO_LDS"))
    return InstClass::TDM;

  // DS (LDS) instructions
  if (TSFlags & SIInstrFlags::DS) {
    if (Desc.mayLoad())
      return InstClass::DS_READ;
    if (Desc.mayStore())
      return InstClass::DS_WRITE;
    return InstClass::OTHER;
  }

  // VMEM instructions
  if ((TSFlags & SIInstrFlags::MUBUF) || (TSFlags & SIInstrFlags::MTBUF) ||
      (TSFlags & SIInstrFlags::MIMG) || (TSFlags & SIInstrFlags::VIMAGE) ||
      (TSFlags & SIInstrFlags::VSAMPLE) ||
      ((TSFlags & SIInstrFlags::FLAT) &&
       ((TSFlags & SIInstrFlags::FlatGlobal) ||
        !(TSFlags & SIInstrFlags::FlatScratch)))) {
    if (Desc.mayLoad())
      return InstClass::VMEM_READ;
    if (Desc.mayStore())
      return InstClass::VMEM_WRITE;
    return InstClass::OTHER;
  }

  // SMEM instructions
  if (TSFlags & SIInstrFlags::SMRD)
    return InstClass::SMEM;

  // SALU instructions
  if (TSFlags & SIInstrFlags::SALU)
    return InstClass::SALU;

  // TRANS instructions (check before generic VALU)
  if (isMCInstTRANS(TSFlags))
    return InstClass::TRANS;

  // VALU instructions
  if (TSFlags & SIInstrFlags::VALU)
    return InstClass::VALU;

  return InstClass::OTHER;
}

//===----------------------------------------------------------------------===//
// Latency Computation
//===----------------------------------------------------------------------===//

unsigned MCInstInfo::getLatency(InstClass IC) const {
  // At MC layer, we don't have access to SchedModel, so use defaults
  return getLatencyForClass(IC);
}

//===----------------------------------------------------------------------===//
// SimInst Creation
//===----------------------------------------------------------------------===//

SimInst MCInstInfo::createSimInst(const MCInst &MI) const {
  InstClass IC = classifyInst(MI);
  unsigned Lat = getLatency(IC);
  FunctionalUnit Unit = getUnitForClass(IC);

  return SimInst(const_cast<MCInst *>(&MI), IC, Lat, Unit);
}

//===----------------------------------------------------------------------===//
// SimInstInfo Interface Implementation
//===----------------------------------------------------------------------===//

unsigned MCInstInfo::getRepeatRate(const SimInst &SI) const {
  return getResourceCycles(SI);
}

bool MCInstInfo::isLOLVALU(const SimInst &SI) const {
  // Long-latency VALU if it's a VALU instruction with repeat rate > 1
  if (SI.Class != InstClass::VALU)
    return false;
  return getResourceCycles(SI) > 1;
}

bool MCInstInfo::isTRANS(const SimInst &SI) const {
  return SI.Class == InstClass::TRANS;
}

unsigned MCInstInfo::getResourceCycles(const SimInst &SI) const {
  const auto *MI = SI.getAs<MCInst>();
  StringRef Name = MCII.getName(MI->getOpcode());

  // Check specific opcodes/instructions
  if (Name.starts_with("V_CVT_SCALEF32"))
    return 4;

  // Fall back to simplified resource cycles
  InstClass IC = SI.Class;
  if (IC == InstClass::WMMA)
    return 8;
  if (IC == InstClass::TRANS)
    return 2;
  return 1;
}

unsigned MCInstInfo::getDelayAluImm(const SimInst &SI) const {
  const auto *MI = SI.getAs<MCInst>();
  if (MI->getOpcode() == AMDGPU::S_DELAY_ALU && MI->getNumOperands() > 0 &&
      MI->getOperand(0).isImm())
    return MI->getOperand(0).getImm();
  return 0;
}

std::pair<WaitType, unsigned> MCInstInfo::getWaitInfo(const SimInst &SI) const {
  const auto *MI = SI.getAs<MCInst>();
  unsigned Opc = MI->getOpcode();
  unsigned WaitCount = 0;
  if (MI->getNumOperands() > 0 && MI->getOperand(0).isImm())
    WaitCount = MI->getOperand(0).getImm();

  // Use name-based matching for encoding variants (e.g., S_WAIT_DSCNT_gfx12)
  StringRef Name = MCII.getName(Opc);

  if (Opc == AMDGPU::S_WAIT_DSCNT || Name.starts_with("S_WAIT_DSCNT"))
    return {WaitType::DS, WaitCount};
  if (Opc == AMDGPU::S_WAIT_LOADCNT || Name.starts_with("S_WAIT_LOADCNT"))
    return {WaitType::VMEMLoad, WaitCount};
  if (Opc == AMDGPU::S_WAIT_STORECNT || Name.starts_with("S_WAIT_STORECNT"))
    return {WaitType::VMEMStore, WaitCount};
  if (Opc == AMDGPU::S_WAIT_KMCNT || Name.starts_with("S_WAIT_KMCNT"))
    return {WaitType::SMEM, WaitCount};
  if (Opc == AMDGPU::S_WAIT_TENSORCNT || Name.starts_with("S_WAIT_TENSORCNT"))
    return {WaitType::Tensor, WaitCount};
  if (Opc == AMDGPU::S_WAIT_XCNT || Name.starts_with("S_WAIT_XCNT"))
    return {WaitType::XCnt, WaitCount};
  if (Opc == AMDGPU::S_WAITCNT_DEPCTR || Name.starts_with("S_WAITCNT_DEPCTR"))
    return {WaitType::DepCtr, WaitCount};

  return {WaitType::None, 0};
}

unsigned MCInstInfo::getVaVdstTarget(const SimInst &SI) const {
  const auto *MI = SI.getAs<MCInst>();
  unsigned Opc = MI->getOpcode();
  StringRef Name = MCII.getName(Opc);

  // Use name-based matching for encoding variants (e.g., S_WAITCNT_DEPCTR_gfx12)
  if ((Opc == AMDGPU::S_WAITCNT_DEPCTR || Name.starts_with("S_WAITCNT_DEPCTR")) &&
      MI->getNumOperands() > 0 && MI->getOperand(0).isImm())
    return AMDGPU::DepCtr::decodeFieldVaVdst(MI->getOperand(0).getImm());
  return 15; // Default: don't wait
}

std::pair<unsigned, unsigned> MCInstInfo::getDestRegInfo(const SimInst &SI,
                                                         bool IsVGPR) const {
  // At MC layer, we have limited register info compared to MIR
  // Return conservative defaults
  (void)SI;
  (void)IsVGPR;
  return {0, 0};
}

WMMAVariant MCInstInfo::getWMMAVariant(const SimInst &SI) const {
  const auto *MI = SI.getAs<MCInst>();
  StringRef Name = MCII.getName(MI->getOpcode());

  // Check for specific patterns in the opcode name
  if (Name.contains("IU8") && Name.contains("16X16X64"))
    return WMMAVariant::IU8_16x16x64;
  if (Name.contains("F4") && Name.contains("32X16X128"))
    return WMMAVariant::F4_32x16x128;
  if (Name.contains("16X16X128")) {
    if (Name.contains("FP8") || Name.contains("BF8"))
      return WMMAVariant::FP8_16x16x128;
    if (Name.contains("F8") || Name.contains("F6") || Name.contains("F4"))
      return WMMAVariant::F8F6F4_16x16x128;
  }
  if (Name.contains("16X16X64")) {
    if (Name.contains("FP8"))
      return WMMAVariant::FP8_16x16x64;
    if (Name.contains("BF8"))
      return WMMAVariant::BF8_16x16x64;
  }
  if (Name.contains("16X16X32")) {
    if (Name.contains("F16"))
      return WMMAVariant::F16_16x16x32;
    if (Name.contains("BF16"))
      return WMMAVariant::BF16_16x16x32;
  }

  return WMMAVariant::Default;
}

bool MCInstInfo::hasScaling(const SimInst &SI) const {
  const auto *MI = SI.getAs<MCInst>();
  StringRef Name = MCII.getName(MI->getOpcode());
  return Name.contains_insensitive("scale");
}

bool MCInstInfo::hasSGPROperands(const SimInst &SI) const {
  SmallVector<RegOperand, 8> SrcRegs;
  getSrcRegs(SI, SrcRegs);
  for (const RegOperand &RO : SrcRegs) {
    if (RO.RegType == RegOperand::Type::SGPR)
      return true;
  }
  return false;
}

void MCInstInfo::getSrcRegs(const SimInst &SI,
                            SmallVectorImpl<RegOperand> &Regs) const {
  const auto *MI = SI.getAs<MCInst>();
  const MCInstrDesc &Desc = MCII.get(MI->getOpcode());

  for (unsigned i = Desc.getNumDefs(); i < MI->getNumOperands(); ++i) {
    const MCOperand &Op = MI->getOperand(i);
    if (!Op.isReg())
      continue;

    MCRegister Reg = Op.getReg();
    if (Reg == 0)
      continue;

    // Determine register type and size using AMDGPU utilities
    RegOperand::Type Type = RegOperand::Type::Other;
    unsigned NumComponents = 1;

    if (const MCRegisterClass *RC = AMDGPU::getVGPRPhysRegClass(Reg, MRI)) {
      Type = RegOperand::Type::VGPR;
      NumComponents = RC->getSizeInBits() / 32;
    } else if (AMDGPU::isSGPR(Reg, &MRI)) {
      Type = RegOperand::Type::SGPR;
      // For SGPRs, get size from operand info if available
      if (i < Desc.getNumOperands()) {
        int16_t RCID = Desc.operands()[i].RegClass;
        if (RCID >= 0) {
          const MCRegisterClass &RC = MRI.getRegClass(RCID);
          NumComponents = RC.getSizeInBits() / 32;
        }
      }
    }

    // Get HW register index
    unsigned HWIndex = MRI.getEncodingValue(Reg) & AMDGPU::HWEncoding::REG_IDX_MASK;

    if (NumComponents == 0)
      NumComponents = 1;

    Regs.push_back(RegOperand(Type, HWIndex, NumComponents));
  }
}

void MCInstInfo::getWMMASrcRegs(const SimInst &SI,
                                SmallVectorImpl<RegOperand> &Regs) const {
  // MC layer: fall back to getSrcRegs (no named operand support)
  getSrcRegs(SI, Regs);
}

void MCInstInfo::getDstRegs(const SimInst &SI,
                            SmallVectorImpl<RegOperand> &Regs) const {
  const auto *MI = SI.getAs<MCInst>();
  const MCInstrDesc &Desc = MCII.get(MI->getOpcode());

  for (unsigned i = 0; i < Desc.getNumDefs() && i < MI->getNumOperands(); ++i) {
    const MCOperand &Op = MI->getOperand(i);
    if (!Op.isReg())
      continue;

    MCRegister Reg = Op.getReg();
    if (Reg == 0)
      continue;

    // Determine register type and size using AMDGPU utilities
    RegOperand::Type Type = RegOperand::Type::Other;
    unsigned NumComponents = 1;

    if (const MCRegisterClass *RC = AMDGPU::getVGPRPhysRegClass(Reg, MRI)) {
      Type = RegOperand::Type::VGPR;
      NumComponents = RC->getSizeInBits() / 32;
    } else if (AMDGPU::isSGPR(Reg, &MRI)) {
      Type = RegOperand::Type::SGPR;
      // For SGPRs, get size from operand info if available
      if (i < Desc.getNumOperands()) {
        int16_t RCID = Desc.operands()[i].RegClass;
        if (RCID >= 0) {
          const MCRegisterClass &RC = MRI.getRegClass(RCID);
          NumComponents = RC.getSizeInBits() / 32;
        }
      }
    }

    // Get HW register index
    unsigned HWIndex = MRI.getEncodingValue(Reg) & AMDGPU::HWEncoding::REG_IDX_MASK;
    if (NumComponents == 0)
      NumComponents = 1;

    Regs.push_back(RegOperand(Type, HWIndex, NumComponents));
  }
}

bool MCInstInfo::waitsForVALU(const SimInst &SI) const {
  // Conservative default based on InstClass at MC layer.
  // Memory instructions implicitly wait for all VALU (VA_VDST==0).
  switch (SI.Class) {
  case InstClass::DS_READ:
  case InstClass::DS_WRITE:
  case InstClass::VMEM_READ:
  case InstClass::VMEM_WRITE:
  case InstClass::SMEM:
  case InstClass::TDM:
    return true;
  default:
    return false;
  }
}

bool MCInstInfo::isVOPD(const SimInst &SI) const {
  const auto *MI = SI.getAs<MCInst>();
  return AMDGPU::isVOPD(MI->getOpcode());
}

bool MCInstInfo::isPacked(const SimInst &SI) const {
  const auto *MI = SI.getAs<MCInst>();
  const MCInstrDesc &Desc = MCII.get(MI->getOpcode());
  uint64_t TSFlags = Desc.TSFlags;
  return (TSFlags & SIInstrFlags::IsPacked) != 0;
}

unsigned MCInstInfo::getInstBytes(const SimInst &SI) const {
  const auto *MI = SI.getAs<MCInst>();
  const MCInstrDesc &Desc = MCII.get(MI->getOpcode());
  return Desc.getSize();
}

} // namespace AMDGPUSim
} // namespace llvm
