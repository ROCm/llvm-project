//===- AMDGPUSim/MCAdapter.h - MCInst Adapter ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Adapter to convert MCInst to SimInst and provide instruction property
/// queries for use with the AMDGPUSim library in MC layer passes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUSIM_MCADAPTER_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUSIM_MCADAPTER_H

#include "SimInstInfo.h"

namespace llvm {

class MCInst;
class MCInstrInfo;
class MCRegisterInfo;
class MCSubtargetInfo;

namespace AMDGPUSim {

/// Concrete implementation of SimInstInfo for MCInst.
/// Provides instruction property queries using MCInstrInfo, MCRegisterInfo,
/// and MCSubtargetInfo (for scheduling model access).
class MCInstInfo : public SimInstInfo {
  const MCInstrInfo &MCII;
  const MCRegisterInfo &MRI;
  const MCSubtargetInfo *STI;

public:
  MCInstInfo(const MCInstrInfo &MCII, const MCRegisterInfo &MRI,
             const MCSubtargetInfo *STI = nullptr);

  /// Create a SimInst from an MCInst.
  /// Populates the cached basic properties (Class, Latency, Unit).
  SimInst createSimInst(const MCInst &MI) const;

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
  /// Classify an MCInst into an InstClass.
  InstClass classifyInst(const MCInst &MI) const;

  /// Get latency for an instruction class.
  unsigned getLatency(InstClass IC) const;
};

} // namespace AMDGPUSim
} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUSIM_MCADAPTER_H
