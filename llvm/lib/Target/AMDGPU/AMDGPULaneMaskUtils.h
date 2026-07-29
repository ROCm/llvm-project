//===- AMDGPULaneMaskUtils.h - Exec/lane mask helper functions -*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Various utility functions for dealing with lane masks during code
/// generation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_UTILS_AMDGPULANEMASKUTILS_H
#define LLVM_LIB_TARGET_AMDGPU_UTILS_AMDGPULANEMASKUTILS_H

#include "GCNSubtarget.h"
#include "SIRegisterInfo.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineSSAUpdater.h"
#include "llvm/CodeGen/Register.h"

namespace llvm {

class AMDGPULaneMaskAnalysis;
class GCNSubtarget;
class MachineFunction;

namespace AMDGPU {

class LaneMaskConstants {
public:
  const Register ExecReg;
  const Register VccReg;
  const unsigned AndOpc;
  const unsigned AndTermOpc;
  const unsigned AndN2Opc;
  const unsigned AndN2SaveExecOpc;
  const unsigned AndN2TermOpc;
  const unsigned AndN2WrExecOpc; // GFX10+ (HasNoSdstCMPX) only
  const unsigned AndSaveExecOpc;
  const unsigned AndSaveExecTermOpc;
  const unsigned BfmOpc;
  const unsigned CMovOpc;
  const unsigned CSelectOpc;
  const unsigned MovOpc;
  const unsigned MovTermOpc;
  const unsigned OrOpc;
  const unsigned OrN2Opc;
  const unsigned OrTermOpc;
  const unsigned OrSaveExecOpc;
  const unsigned XorOpc;
  const unsigned XorTermOpc;
  const unsigned WQMOpc;
  const TargetRegisterClass *LaneMaskRC;

  constexpr LaneMaskConstants(bool IsWave32)
      : ExecReg(IsWave32 ? AMDGPU::EXEC_LO : AMDGPU::EXEC),
        VccReg(IsWave32 ? AMDGPU::VCC_LO : AMDGPU::VCC),
        AndOpc(IsWave32 ? AMDGPU::S_AND_B32 : AMDGPU::S_AND_B64),
        AndTermOpc(IsWave32 ? AMDGPU::S_AND_B32_term : AMDGPU::S_AND_B64_term),
        AndN2Opc(IsWave32 ? AMDGPU::S_ANDN2_B32 : AMDGPU::S_ANDN2_B64),
        AndN2SaveExecOpc(IsWave32 ? AMDGPU::S_ANDN2_SAVEEXEC_B32
                                  : AMDGPU::S_ANDN2_SAVEEXEC_B64),
        AndN2TermOpc(IsWave32 ? AMDGPU::S_ANDN2_B32_term
                              : AMDGPU::S_ANDN2_B64_term),
        AndN2WrExecOpc(IsWave32 ? AMDGPU::S_ANDN2_WREXEC_B32
                                : AMDGPU::S_ANDN2_WREXEC_B64),
        AndSaveExecOpc(IsWave32 ? AMDGPU::S_AND_SAVEEXEC_B32
                                : AMDGPU::S_AND_SAVEEXEC_B64),
        AndSaveExecTermOpc(IsWave32 ? AMDGPU::S_AND_SAVEEXEC_B32_term
                                    : AMDGPU::S_AND_SAVEEXEC_B64_term),
        BfmOpc(IsWave32 ? AMDGPU::S_BFM_B32 : AMDGPU::S_BFM_B64),
        CMovOpc(IsWave32 ? AMDGPU::S_CMOV_B32 : AMDGPU::S_CMOV_B64),
        CSelectOpc(IsWave32 ? AMDGPU::S_CSELECT_B32 : AMDGPU::S_CSELECT_B64),
        MovOpc(IsWave32 ? AMDGPU::S_MOV_B32 : AMDGPU::S_MOV_B64),
        MovTermOpc(IsWave32 ? AMDGPU::S_MOV_B32_term : AMDGPU::S_MOV_B64_term),
        OrOpc(IsWave32 ? AMDGPU::S_OR_B32 : AMDGPU::S_OR_B64),
        OrN2Opc(IsWave32 ? AMDGPU::S_ORN2_B32 : AMDGPU::S_ORN2_B64),
        OrTermOpc(IsWave32 ? AMDGPU::S_OR_B32_term : AMDGPU::S_OR_B64_term),
        OrSaveExecOpc(IsWave32 ? AMDGPU::S_OR_SAVEEXEC_B32
                               : AMDGPU::S_OR_SAVEEXEC_B64),
        XorOpc(IsWave32 ? AMDGPU::S_XOR_B32 : AMDGPU::S_XOR_B64),
        XorTermOpc(IsWave32 ? AMDGPU::S_XOR_B32_term : AMDGPU::S_XOR_B64_term),
        WQMOpc(IsWave32 ? AMDGPU::S_WQM_B32 : AMDGPU::S_WQM_B64),
        LaneMaskRC(IsWave32 ? &AMDGPU::SReg_32RegClass
                            : &AMDGPU::SReg_64RegClass) {}

  static inline const LaneMaskConstants &get(const GCNSubtarget &ST);
};

static constexpr LaneMaskConstants LaneMaskConstants32 =
    LaneMaskConstants(/*IsWave32=*/true);
static constexpr LaneMaskConstants LaneMaskConstants64 =
    LaneMaskConstants(/*IsWave32=*/false);

inline const LaneMaskConstants &LaneMaskConstants::get(const GCNSubtarget &ST) {
  unsigned WavefrontSize = ST.getWavefrontSize();
  assert(WavefrontSize == 32 || WavefrontSize == 64);
  return WavefrontSize == 32 ? LaneMaskConstants32 : LaneMaskConstants64;
}

} // end namespace AMDGPU

/// \brief Helper class for lane-mask related tasks.
class AMDGPULaneMaskUtils {
private:
  MachineFunction &MF;
  const AMDGPU::LaneMaskConstants &LMC;

public:
  AMDGPULaneMaskUtils() = delete;
  explicit AMDGPULaneMaskUtils(MachineFunction &MF)
      : MF(MF),
        LMC(AMDGPU::LaneMaskConstants::get(MF.getSubtarget<GCNSubtarget>())) {}

  MachineFunction *function() const { return &MF; }
  const AMDGPU::LaneMaskConstants &getLaneMaskConsts() const { return LMC; }

  const SIRegisterInfo &getRegisterInfo() const {
    return *MF.getSubtarget<GCNSubtarget>().getRegisterInfo();
  }

  bool maybeLaneMask(Register Reg) const;
  bool isConstantLaneMask(Register Reg, bool &Val, MachineBasicBlock &MBB,
                          MachineBasicBlock::iterator I) const;

  Register createLaneMaskReg() const;
  void buildMergeLaneMasks(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator I, const DebugLoc &DL,
                           Register DstReg, Register PrevReg, Register CurReg,
                           AMDGPULaneMaskAnalysis *LMA = nullptr,
                           bool isPrevZeroReg = false) const;
};

/// Lazy analyses of lane masks.
class AMDGPULaneMaskAnalysis {
private:
  AMDGPULaneMaskUtils LMU;

public:
  AMDGPULaneMaskAnalysis(MachineFunction &MF) : LMU(MF) {}

  bool isSubsetOfExec(Register Reg, MachineBasicBlock &UseBlock,
                      MachineBasicBlock::iterator I,
                      unsigned RemainingDepth = 5);
};

/// \brief SSA-updater for lane masks.
///
/// Each lane is assumed to provide a "true" available value only
/// once, and to never attempt to change the value back to "false" -- except
/// that all lanes are reset to false in "reset blocks" as explained below.
/// The bits for lanes that never contributed with an available value are 0.
///
/// All lanes are reset to 0 at certain points in "reset blocks"
///  which are added via \ref addReset. The reset happens in one or both
/// of two modes:
///  - ResetInMiddle: Reset logically happens after the point queried by
///    \ref getValueInMiddleOfBlock and before the contribution of the block's
///    available value ("merge").
///  - ResetAtEnd: Reset logically happens after the contribution of the
///    block's available value, but before the point queried by
///    \ref getValueAtEndOfBlock. Use \ref getValueAfterMerge to query the
///    value just after contribution of the reset block's available value.
///
class AMDGPULaneMaskUpdater {
public:
  enum ResetFlags {
    ResetInMiddle = (1 << 0),
    ResetAtEnd = (1 << 1),
  };

private:
  AMDGPULaneMaskUtils LMU;
  AMDGPULaneMaskAnalysis *LMA = nullptr;

  struct BlockInfo {
    MachineBasicBlock *Block;
    unsigned Flags = 0; // ResetFlags
    Register Value;

    explicit BlockInfo(MachineBasicBlock *Block) : Block(Block) {}

    void dump() {
      dbgs() << "BlockInfo{";
      dbgs() << " Block:" << printMBBReference(*Block) << ",";
      dbgs() << " Flags:";
      if (Flags & ResetAtEnd)
        dbgs() << "ResetAtEnd,";
      if (Flags & ResetInMiddle)
        dbgs() << "ResetInMiddle,";
      dbgs() << "}\n";
    }
  };

  SmallVector<BlockInfo, 4> Blocks;

  Register ZeroReg;
  Register Accumulator;
  DenseMap<MachineBasicBlock *, SmallVector<std::pair<Register, unsigned>, 2>>
      AccumulatorResetBlocks;
  SmallDenseSet<Register, 4> AllAccumulators;

public:
  SmallDenseSet<Register, 4> &getAllAccumulators() { return AllAccumulators; }

  AMDGPULaneMaskUpdater(MachineFunction &MF) : LMU(MF) {}

  void setLaneMaskAnalysis(AMDGPULaneMaskAnalysis *Analysis) { LMA = Analysis; }

  void init();
  void cleanup();

  void addReset(MachineBasicBlock &Block, ResetFlags Flags);
  void addAvailable(MachineBasicBlock &Block, Register Value);

  Register getMergedMask(MachineBasicBlock &Block);
  void insertAccumulatorResets();

private:
  SmallVectorImpl<BlockInfo>::iterator findBlockInfo(MachineBasicBlock &Block);
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_UTILS_AMDGPULANEMASKUTILS_H
