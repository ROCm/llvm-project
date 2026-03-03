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
#include "llvm/MC/MCSchedule.h"
#include "llvm/MC/MCSubtargetInfo.h"

namespace llvm {
namespace AMDGPUSim {

//===----------------------------------------------------------------------===//
// Constructor
//===----------------------------------------------------------------------===//

MCInstInfo::MCInstInfo(const MCInstrInfo &MCII, const MCRegisterInfo &MRI,
                       const MCSubtargetInfo *STI)
    : MCII(MCII), MRI(MRI), STI(STI) {}

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
    // Note: S_WAITCNT_DEPCTR is classified as SALU, not WAITCNT
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

  // S_WAITCNT_DEPCTR is SALU (not WAITCNT) - it has latency 2 and computes VaVdst
  if (Name.starts_with("S_WAITCNT_DEPCTR"))
    return InstClass::SALU;

  // Wait instructions
  if (isMCInstWaitcnt(Opc) || Name.starts_with("S_WAIT"))
    return InstClass::WAITCNT;

  // Branch instructions
  if (Desc.isBranch())
    return InstClass::BRANCH;

  // WMMA instructions (check before VALU since WMMA has VALU flag too)
  if (isMCInstWMMA(TSFlags))
    return InstClass::WMMA;

  // TDM instructions (use name matching for encoding variants)
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
  // Fall back to default latencies when no SchedModel
  return getLatencyForClass(IC);
}

/// Get latency from the scheduling model for an MCInst.
/// Since MC-layer scheduling classes often don't match MIR-level resolved classes
/// (which can account for operand properties), we use opcode-based lookup for
/// known instructions and fall back to the scheduling model when available.
static unsigned getSchedModelLatency(const MCInst &MI, const MCInstrInfo &MCII,
                                     const MCSubtargetInfo &STI) {
  const MCInstrDesc &Desc = MCII.get(MI.getOpcode());
  StringRef Name = MCII.getName(MI.getOpcode());

  // S_WAITCNT_DEPCTR has latency 2 (not 1 like other wait instructions)
  if (Name.starts_with("S_WAITCNT_DEPCTR"))
    return 2;

  // Wait/barrier/control flow instructions - latency 1
  if (Name.starts_with("S_WAIT") || Name.starts_with("S_BARRIER") ||
      Name.starts_with("S_NOP") || Name.starts_with("S_DELAY_ALU") ||
      Name.starts_with("S_BRANCH") || Name.starts_with("S_CBRANCH") ||
      Name.starts_with("S_SETPC") || Name.starts_with("S_SWAPPC") ||
      Name.starts_with("S_ENDPGM"))
    return 1;

  // GFX1250 VALU latencies based on scheduling model and observed behavior.
  // These hardcoded values provide more accurate results than the size heuristic.
  // Latencies from SISchedule.td GFX1250SpeedModel:
  //   Write32Bit = 5, Write64Bit = 7, WriteTrans32 = 7, WriteQuarterRate32 = 6,
  //   WriteFloatCvt = 5, WriteFloatFMA = 5, WriteIntMul = 11

  // Lane operations (READLANE/WRITELANE) - latency 6
  // Note: Using 6 to match MIR adapter's scheduling model behavior
  if (Name.starts_with("V_READLANE") || Name.starts_with("V_WRITELANE"))
    return 6;

  // Transcendental instructions - latency 7
  if (Name.starts_with("V_EXP_") || Name.starts_with("V_LOG_") ||
      Name.starts_with("V_RCP_") || Name.starts_with("V_RSQ_") ||
      Name.starts_with("V_SQRT_") || Name.starts_with("V_SIN_") ||
      Name.starts_with("V_COS_"))
    return 7;

  // Integer multiply - latency 11
  if (Name.starts_with("V_MUL_HI") || Name.starts_with("V_MUL_LO") ||
      Name.starts_with("V_MAD_U64") || Name.starts_with("V_MAD_I64"))
    return 11;

  // Double precision - latency 32
  if (Name.contains("_F64"))
    return 32;

  // Specific VALU instruction overrides - latency 7
  // Add instructions here as needed when discrepancies are found
  if (Name.starts_with("V_MOV_B32") || Name.starts_with("V_ADD_U32") ||
      Name.starts_with("V_ADD_NC_U32") || Name.starts_with("V_AND_B32") ||
      Name.starts_with("V_OR_B32") || Name.starts_with("V_XOR_B32") ||
      Name.starts_with("V_LSHL_OR_B32") || Name.starts_with("V_LSHLREV_B32") ||
      Name.starts_with("V_CVT_SCALEF32") || Name.starts_with("V_DUAL_"))
    return 7;

  // VMEM instructions - use default 300 (scheduling model returns 320)
  if (Name.starts_with("BUFFER_LOAD") || Name.starts_with("BUFFER_STORE") ||
      Name.starts_with("GLOBAL_LOAD") || Name.starts_with("GLOBAL_STORE") ||
      Name.starts_with("FLAT_LOAD") || Name.starts_with("FLAT_STORE") ||
      Name.starts_with("SCRATCH_LOAD") || Name.starts_with("SCRATCH_STORE"))
    return 0; // Fall back to class-based default (300)

  // DS (LDS) instructions - use default 50/8 (scheduling model returns 20)
  if (Name.starts_with("DS_READ") || Name.starts_with("DS_WRITE") ||
      Name.starts_with("DS_LOAD") || Name.starts_with("DS_STORE"))
    return 0; // Fall back to class-based default

  // TDM (TENSOR_LOAD_TO_LDS) - latency 60
  if (Name.starts_with("TENSOR_LOAD_TO_LDS"))
    return 60;

  // For non-VALU instructions, try the scheduling model
  const MCSchedModel &SM = STI.getSchedModel();
  if (SM.hasInstrSchedModel()) {
    unsigned SchedClass = Desc.getSchedClass();
    const MCSchedClassDesc *SCDesc = SM.getSchedClassDesc(SchedClass);
    if (SCDesc && SCDesc->isValid() && !SCDesc->isVariant())
      return MCSchedModel::computeInstrLatency(STI, *SCDesc);
  }

  // Default: return 0 to fall back to class-based defaults
  return 0;
}

/// Get resource cycles (ReleaseAtCycle) from the scheduling model.
static unsigned getSchedModelResourceCycles(const MCInst &MI,
                                            const MCInstrInfo &MCII,
                                            const MCSubtargetInfo &STI) {
  const MCSchedModel &SM = STI.getSchedModel();
  if (!SM.hasInstrSchedModel())
    return 0;

  unsigned SchedClass = MCII.get(MI.getOpcode()).getSchedClass();
  const MCSchedClassDesc *SCDesc = SM.getSchedClassDesc(SchedClass);
  if (!SCDesc || SCDesc->isVariant())
    return 0;

  // Return the maximum resource cycles (ReleaseAtCycle)
  unsigned MaxCycles = 0;
  for (const MCWriteProcResEntry *PRE = STI.getWriteProcResBegin(SCDesc),
                                 *E = STI.getWriteProcResEnd(SCDesc);
       PRE != E; ++PRE) {
    if (PRE->ReleaseAtCycle > MaxCycles)
      MaxCycles = PRE->ReleaseAtCycle;
  }
  return MaxCycles;
}

//===----------------------------------------------------------------------===//
// SimInst Creation
//===----------------------------------------------------------------------===//

SimInst MCInstInfo::createSimInst(const MCInst &MI) const {
  InstClass IC = classifyInst(MI);
  FunctionalUnit Unit = getUnitForClass(IC);

  // Try to get latency from scheduling model first
  unsigned Lat = 0;
  if (STI)
    Lat = getSchedModelLatency(MI, MCII, *STI);
  // Fall back to default latencies
  if (Lat == 0)
    Lat = getLatency(IC);

  return SimInst(const_cast<MCInst *>(&MI), IC, Lat, Unit);
}

//===----------------------------------------------------------------------===//
// SimInstInfo Interface Implementation
//===----------------------------------------------------------------------===//

unsigned MCInstInfo::getRepeatRate(const SimInst &SI) const {
  // Use resource cycles from scheduling model as repeat rate
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

  // VOPD (dual) instructions - resource cycles 1
  if (Name.starts_with("V_DUAL_"))
    return 1;

  // Long-latency VALU (LOLVALU) instructions - resource cycles 4
  if (Name.starts_with("V_CVT_SCALEF32"))
    return 4;

  // Try scheduling model first
  if (STI) {
    unsigned Cycles = getSchedModelResourceCycles(*MI, MCII, *STI);
    if (Cycles > 0)
      return Cycles;
  }

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
  // Note: Tablegen names use uppercase X in dimensions (e.g., 16X16X128)
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
  // Match MIR adapter logic from AMDGPUInsertDelayAlu::instructionWaitsForVALU
  // Only DS, FLAT, MIMG, MTBUF, MUBUF instructions implicitly wait (VA_VDST==0).
  // Note: TDM (TENSOR_LOAD_TO_LDS) does NOT wait for VALU.
  switch (SI.Class) {
  case InstClass::DS_READ:
  case InstClass::DS_WRITE:
  case InstClass::VMEM_READ:
  case InstClass::VMEM_WRITE:
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
