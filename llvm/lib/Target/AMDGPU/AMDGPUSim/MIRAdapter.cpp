//===- AMDGPUSim/MIRAdapter.cpp - MachineInstr Adapter --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Implementation of the MachineInstrInfo class for MachineInstr property
/// queries.
//
//===----------------------------------------------------------------------===//

#include "MIRAdapter.h"
#include "HWModel.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIDefines.h"
#include "SIInstrInfo.h"
#include "SIRegisterInfo.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetSchedule.h"
#include <cmath>

namespace llvm {
namespace AMDGPUSim {

//===----------------------------------------------------------------------===//
// Constructor
//===----------------------------------------------------------------------===//

MachineInstrInfo::MachineInstrInfo(const SIInstrInfo &TII,
                                   const SIRegisterInfo &TRI)
    : TII(TII), TRI(TRI) {}

//===----------------------------------------------------------------------===//
// Instruction Classification
//===----------------------------------------------------------------------===//

InstClass MachineInstrInfo::classifyInst(const MachineInstr &MI) const {
  unsigned Opc = MI.getOpcode();

  if (Opc == AMDGPU::S_DELAY_ALU)
    return InstClass::DELAY_ALU;

  if (Opc == AMDGPU::S_SET_VGPR_MSB)
    return InstClass::MSB_SET;

  StringRef Name = TII.getName(Opc);
  if (Name.starts_with("V_NOP"))
    return InstClass::VALU;

  if (Opc == AMDGPU::S_NOP || Name.starts_with("S_CLAUSE"))
    return InstClass::SALU;

  if (Opc == AMDGPU::S_BARRIER || Opc == AMDGPU::S_BARRIER_SIGNAL_M0 ||
      Opc == AMDGPU::S_BARRIER_SIGNAL_ISFIRST_M0 ||
      Opc == AMDGPU::S_BARRIER_WAIT)
    return InstClass::BARRIER;

  if (TII.isWaitcnt(Opc) || Opc == AMDGPU::S_WAIT_XCNT ||
      Opc == AMDGPU::S_WAIT_TENSORCNT)
    return InstClass::WAITCNT;

  if (MI.isBranch())
    return InstClass::BRANCH;

  if (TII.isXDLWMMA(MI))
    return InstClass::WMMA;

  if (Opc == AMDGPU::TENSOR_LOAD_TO_LDS || Opc == AMDGPU::TENSOR_LOAD_TO_LDS_D2)
    return InstClass::TDM;

  uint64_t TSFlags = MI.getDesc().TSFlags;

  if (TSFlags & SIInstrFlags::DS) {
    if (MI.mayLoad())
      return InstClass::DS_READ;
    if (MI.mayStore())
      return InstClass::DS_WRITE;
    return InstClass::OTHER;
  }

  if (TII.isVMEM(MI)) {
    if (MI.mayLoad())
      return InstClass::VMEM_READ;
    if (MI.mayStore())
      return InstClass::VMEM_WRITE;
    return InstClass::OTHER;
  }

  if (TII.isSMRD(MI))
    return InstClass::SMEM;

  if (TII.isSALU(MI))
    return InstClass::SALU;

  if (SIInstrInfo::isTRANS(MI))
    return InstClass::TRANS;

  if (TII.isVALU(MI))
    return InstClass::VALU;

  return InstClass::OTHER;
}

//===----------------------------------------------------------------------===//
// Latency Computation
//===----------------------------------------------------------------------===//

unsigned MachineInstrInfo::getLatency(const MachineInstr &MI,
                                      InstClass IC) const {
  switch (IC) {
  case InstClass::DS_READ:
    return DefaultLatency::DS_READ;
  case InstClass::DS_WRITE:
    return DefaultLatency::DS_WRITE;
  case InstClass::VMEM_READ:
  case InstClass::VMEM_WRITE:
    return DefaultLatency::VMEM;
  case InstClass::SMEM:
    return DefaultLatency::SMEM;
  case InstClass::BARRIER:
    return DefaultLatency::BARRIER;
  case InstClass::NOP:
  case InstClass::DELAY_ALU:
  case InstClass::WAITCNT:
  case InstClass::BRANCH:
  case InstClass::MSB_SET:
    return 1;
  default:
    break;
  }

  const TargetSchedModel &SchedModel = TII.getSchedModel();
  if (SchedModel.hasInstrSchedModel()) {
    unsigned Lat = SchedModel.computeInstrLatency(&MI);
    if (Lat > 0)
      return Lat;
  }

  return getLatencyForClass(IC);
}

//===----------------------------------------------------------------------===//
// SimInst Creation
//===----------------------------------------------------------------------===//

SimInst MachineInstrInfo::createSimInst(const MachineInstr &MI) const {
  InstClass IC = classifyInst(MI);
  unsigned Lat = getLatency(MI, IC);
  FunctionalUnit Unit = getUnitForClass(IC);

  return SimInst(const_cast<MachineInstr *>(&MI), IC, Lat, Unit);
}

//===----------------------------------------------------------------------===//
// SimInstInfo Interface Implementation
//===----------------------------------------------------------------------===//

unsigned MachineInstrInfo::getRepeatRate(const SimInst &SI) const {
  const auto *MI = SI.getAs<MachineInstr>();
  return TII.getRepeatRate(*MI);
}

bool MachineInstrInfo::isLOLVALU(const SimInst &SI) const {
  if (SI.Class != InstClass::VALU)
    return false;
  return getRepeatRate(SI) > 1;
}

bool MachineInstrInfo::isTRANS(const SimInst &SI) const {
  return SI.Class == InstClass::TRANS;
}

unsigned MachineInstrInfo::getResourceCycles(const SimInst &SI) const {
  const auto *MI = SI.getAs<MachineInstr>();
  InstClass IC = SI.Class;

  // Use getRepeatRate() for VALU/TRANS to get canonical long-lat VALU resource
  // cycles. getRepeatRate returns 1 for regular VALU, >1 for long-lat VALU
  // (PK8=4, PK16=8, F64=32, etc.)
  if (IC == InstClass::VALU || IC == InstClass::TRANS) {
    unsigned RepeatRate = TII.getRepeatRate(*MI);
    if (RepeatRate > 1)
      return RepeatRate;
  }

  if (AMDGPU::isVOPD(MI->getOpcode()))
    return 1;

  if (IC == InstClass::DS_READ || IC == InstClass::DS_WRITE)
    return 1;

  const TargetSchedModel &SchedModel = TII.getSchedModel();
  if (SchedModel.hasInstrSchedModel()) {
    double RecipThroughput = SchedModel.computeReciprocalThroughput(MI);
    if (RecipThroughput > 0.0) {
      unsigned Cycles =
          std::max(1u, static_cast<unsigned>(std::ceil(RecipThroughput)));
      if (IC == InstClass::TRANS && Cycles < 2)
        return 2;
      return Cycles;
    }
  }

  if (IC == InstClass::WMMA)
    return 8;
  if (IC == InstClass::TRANS)
    return 2;

  return 1;
}

unsigned MachineInstrInfo::getDelayAluImm(const SimInst &SI) const {
  const auto *MI = SI.getAs<MachineInstr>();
  if (MI->getOpcode() == AMDGPU::S_DELAY_ALU && MI->getNumOperands() > 0 &&
      MI->getOperand(0).isImm())
    return MI->getOperand(0).getImm();
  return 0;
}

std::pair<WaitType, unsigned>
MachineInstrInfo::getWaitInfo(const SimInst &SI) const {
  const auto *MI = SI.getAs<MachineInstr>();
  unsigned Opc = MI->getOpcode();
  unsigned WaitCount = 0;
  if (MI->getNumOperands() > 0 && MI->getOperand(0).isImm())
    WaitCount = MI->getOperand(0).getImm();

  switch (Opc) {
  case AMDGPU::S_WAIT_DSCNT:
    return {WaitType::DS, WaitCount};
  case AMDGPU::S_WAIT_LOADCNT:
    return {WaitType::VMEMLoad, WaitCount};
  case AMDGPU::S_WAIT_STORECNT:
    return {WaitType::VMEMStore, WaitCount};
  case AMDGPU::S_WAIT_KMCNT:
    return {WaitType::SMEM, WaitCount};
  case AMDGPU::S_WAIT_TENSORCNT:
    return {WaitType::Tensor, WaitCount};
  case AMDGPU::S_WAIT_XCNT:
    return {WaitType::XCnt, WaitCount};
  case AMDGPU::S_WAITCNT_DEPCTR:
    return {WaitType::DepCtr, WaitCount};
  default:
    return {WaitType::None, 0};
  }
}

unsigned MachineInstrInfo::getVaVdstTarget(const SimInst &SI) const {
  const auto *MI = SI.getAs<MachineInstr>();
  if (MI->getOpcode() == AMDGPU::S_WAITCNT_DEPCTR && MI->getNumOperands() > 0 &&
      MI->getOperand(0).isImm())
    return AMDGPU::DepCtr::decodeFieldVaVdst(MI->getOperand(0).getImm());
  return 15; // Default: don't wait
}

std::pair<unsigned, unsigned>
MachineInstrInfo::getDestRegInfo(const SimInst &SI, bool IsVGPR) const {
  const auto *MI = SI.getAs<MachineInstr>();
  if (MI->getNumOperands() == 0 || !MI->getOperand(0).isReg())
    return {0, 0};

  Register Reg = MI->getOperand(0).getReg();
  if (!Reg.isPhysical())
    return {0, 0};

  const TargetRegisterClass *RC = TRI.getPhysRegBaseClass(Reg);

  if (IsVGPR) {
    if (!TRI.hasVGPRs(RC))
      return {0, 0};
  } else {
    if (TRI.hasVGPRs(RC) || TRI.hasAGPRs(RC))
      return {0, 0};
  }

  unsigned BaseIdx = TRI.getHWRegIndex(Reg);
  unsigned SizeInBits = TRI.getRegSizeInBits(*RC);
  unsigned NumRegs = SizeInBits / 32;

  return {BaseIdx, NumRegs};
}

WMMAVariant MachineInstrInfo::getWMMAVariant(const SimInst &SI) const {
  const auto *MI = SI.getAs<MachineInstr>();
  StringRef Name = TII.getName(MI->getOpcode());

  // LLVM opcode names use uppercase X in dimension patterns (e.g., 16X16X32).
  // Check for specific patterns in the opcode name.
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

bool MachineInstrInfo::hasScaling(const SimInst &SI) const {
  const auto *MI = SI.getAs<MachineInstr>();
  StringRef Name = TII.getName(MI->getOpcode());
  return Name.contains_insensitive("scale");
}

bool MachineInstrInfo::hasSGPROperands(const SimInst &SI) const {
  const auto *MI = SI.getAs<MachineInstr>();
  const MachineRegisterInfo &MRI = MI->getMF()->getRegInfo();

  for (const MachineOperand &MO : MI->explicit_operands()) {
    if (!MO.isReg() || !MO.getReg().isPhysical())
      continue;
    if (TRI.isSGPRReg(MRI, MO.getReg()))
      return true;
  }
  return false;
}

void MachineInstrInfo::getSrcRegs(const SimInst &SI,
                                  SmallVectorImpl<RegOperand> &Regs) const {
  const auto *MI = SI.getAs<MachineInstr>();
  const MachineRegisterInfo &MRI = MI->getMF()->getRegInfo();

  // Include all explicit_uses (even non-register ones as Type::Other
  // placeholders) to preserve the port index mapping used by the VGPR source
  // cache. The cache assigns Port = OperandIndex % 3, so non-register operands
  // must be included to maintain correct port numbering.
  for (const MachineOperand &MO : MI->explicit_uses()) {
    if (!MO.isReg() || !MO.getReg().isPhysical()) {
      Regs.push_back(RegOperand(RegOperand::Type::Other, 0, 0));
      continue;
    }

    Register Reg = MO.getReg();
    const TargetRegisterClass *RC = TRI.getMinimalPhysRegClass(Reg);
    unsigned NumComponents = (RC ? TRI.getRegSizeInBits(*RC) : 32) / 32;
    unsigned BaseHWReg = TRI.getHWRegIndex(Reg);

    RegOperand::Type Type;
    if (TRI.isVGPR(MRI, Reg))
      Type = RegOperand::Type::VGPR;
    else if (TRI.isSGPRReg(MRI, Reg))
      Type = RegOperand::Type::SGPR;
    else
      Type = RegOperand::Type::Other;

    Regs.push_back(RegOperand(Type, BaseHWReg, NumComponents));
  }
}

void MachineInstrInfo::getWMMASrcRegs(const SimInst &SI,
                                      SmallVectorImpl<RegOperand> &Regs) const {
  const auto *MI = SI.getAs<MachineInstr>();
  const MachineRegisterInfo &MRI = MI->getMF()->getRegInfo();

  // Only process src0 (A) and src1 (B) matrix operands for WMMA cache
  // tracking. Skip C (tied-def) and scale registers.
  auto ProcessOperand = [&](AMDGPU::OpName OpName) {
    int Idx = AMDGPU::getNamedOperandIdx(MI->getOpcode(), OpName);
    if (Idx < 0)
      return;
    const MachineOperand &MO = MI->getOperand(Idx);
    if (!MO.isReg() || !MO.getReg().isPhysical())
      return;
    Register Reg = MO.getReg();
    if (!TRI.isVGPR(MRI, Reg))
      return;
    const TargetRegisterClass *RC = TRI.getMinimalPhysRegClass(Reg);
    unsigned NumComponents = (RC ? TRI.getRegSizeInBits(*RC) : 32) / 32;
    unsigned BaseHWReg = TRI.getHWRegIndex(Reg);
    Regs.push_back(
        RegOperand(RegOperand::Type::VGPR, BaseHWReg, NumComponents));
  };

  ProcessOperand(AMDGPU::OpName::src0);
  ProcessOperand(AMDGPU::OpName::src1);
}

void MachineInstrInfo::getDstRegs(const SimInst &SI,
                                  SmallVectorImpl<RegOperand> &Regs) const {
  const auto *MI = SI.getAs<MachineInstr>();
  const MachineRegisterInfo &MRI = MI->getMF()->getRegInfo();

  for (const MachineOperand &MO : MI->defs()) {
    if (!MO.isReg() || !MO.getReg().isPhysical())
      continue;

    Register Reg = MO.getReg();
    const TargetRegisterClass *RC = TRI.getMinimalPhysRegClass(Reg);
    unsigned NumComponents = (RC ? TRI.getRegSizeInBits(*RC) : 32) / 32;
    unsigned BaseHWReg = TRI.getHWRegIndex(Reg);

    RegOperand::Type Type;
    if (TRI.isVGPR(MRI, Reg))
      Type = RegOperand::Type::VGPR;
    else if (TRI.isSGPRReg(MRI, Reg))
      Type = RegOperand::Type::SGPR;
    else
      Type = RegOperand::Type::Other;

    Regs.push_back(RegOperand(Type, BaseHWReg, NumComponents));
  }
}

bool MachineInstrInfo::waitsForVALU(const SimInst &SI) const {
  const auto *MI = SI.getAs<MachineInstr>();
  // Same logic as AMDGPUInsertDelayAlu::instructionWaitsForVALU
  const uint64_t VA_VDST_0 = SIInstrFlags::DS | SIInstrFlags::EXP |
                             SIInstrFlags::FLAT | SIInstrFlags::MIMG |
                             SIInstrFlags::MTBUF | SIInstrFlags::MUBUF;
  if (MI->getDesc().TSFlags & VA_VDST_0)
    return true;
  if (MI->getOpcode() == AMDGPU::S_SENDMSG_RTN_B32 ||
      MI->getOpcode() == AMDGPU::S_SENDMSG_RTN_B64)
    return true;
  if (MI->getOpcode() == AMDGPU::S_WAITCNT_DEPCTR &&
      AMDGPU::DepCtr::decodeFieldVaVdst(MI->getOperand(0).getImm()) == 0)
    return true;
  return false;
}

bool MachineInstrInfo::isVOPD(const SimInst &SI) const {
  const auto *MI = SI.getAs<MachineInstr>();
  return AMDGPU::isVOPD(MI->getOpcode());
}

bool MachineInstrInfo::isPacked(const SimInst &SI) const {
  const auto *MI = SI.getAs<MachineInstr>();
  uint64_t TSFlags = MI->getDesc().TSFlags;
  return (TSFlags & SIInstrFlags::IsPacked) != 0;
}

unsigned MachineInstrInfo::getInstBytes(const SimInst &SI) const {
  const auto *MI = SI.getAs<MachineInstr>();
  return TII.getInstSizeInBytes(*MI);
}

} // namespace AMDGPUSim
} // namespace llvm
