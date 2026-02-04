//===- AMDGPUSim/MIRAdapter.h - MachineInstr Adapter -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Adapter to convert MachineInstr to SimInst and provide instruction property
/// queries for use with the AMDGPUSim library in MachineFunction passes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUSIM_MIRADAPTER_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUSIM_MIRADAPTER_H

#include "SimInstInfo.h"

namespace llvm {

class MachineInstr;
class SIInstrInfo;
class SIRegisterInfo;

namespace AMDGPUSim {

/// Concrete implementation of SimInstInfo for MachineInstr.
/// Provides instruction property queries using SIInstrInfo and SIRegisterInfo.
class MachineInstrInfo : public SimInstInfo {
  const SIInstrInfo &TII;
  const SIRegisterInfo &TRI;

public:
  MachineInstrInfo(const SIInstrInfo &TII, const SIRegisterInfo &TRI);

  /// Create a SimInst from a MachineInstr.
  /// Populates the cached basic properties (Class, Latency, Unit).
  SimInst createSimInst(const MachineInstr &MI) const;

  //===--------------------------------------------------------------------===//
  // SimInstInfo Interface Implementation
  //===--------------------------------------------------------------------===//

  unsigned getRepeatRate(const SimInst &SI) const override;
  bool isLOLVALU(const SimInst &SI) const override;
  bool isTRANS(const SimInst &SI) const override;
  unsigned getResourceCycles(const SimInst &SI) const override;
  unsigned getDelayAluImm(const SimInst &SI) const override;
  std::pair<WaitType, unsigned> getWaitInfo(const SimInst &SI) const override;
  unsigned getVaVdstTarget(const SimInst &SI) const override;
  std::pair<unsigned, unsigned> getDestRegInfo(const SimInst &SI,
                                               bool IsVGPR) const override;
  WMMAVariant getWMMAVariant(const SimInst &SI) const override;
  bool hasScaling(const SimInst &SI) const override;
  bool hasSGPROperands(const SimInst &SI) const override;
  void getSrcRegs(const SimInst &SI,
                  SmallVectorImpl<RegOperand> &Regs) const override;
  void getWMMASrcRegs(const SimInst &SI,
                      SmallVectorImpl<RegOperand> &Regs) const override;
  void getDstRegs(const SimInst &SI,
                  SmallVectorImpl<RegOperand> &Regs) const override;
  bool waitsForVALU(const SimInst &SI) const override;
  bool isVOPD(const SimInst &SI) const override;
  bool isPacked(const SimInst &SI) const override;
  unsigned getInstBytes(const SimInst &SI) const override;

private:
  /// Classify a MachineInstr into an InstClass.
  InstClass classifyInst(const MachineInstr &MI) const;

  /// Get latency for a MachineInstr.
  unsigned getLatency(const MachineInstr &MI, InstClass IC) const;
};

} // namespace AMDGPUSim
} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUSIM_MIRADAPTER_H
