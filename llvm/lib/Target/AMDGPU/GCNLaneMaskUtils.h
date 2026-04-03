//===- GCNLaneMaskUtils.h ----------------------------------------*- C++ -*-==//
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

#ifndef LLVM_LIB_TARGET_AMDGPU_GCNLANEMASKUTILS_H
#define LLVM_LIB_TARGET_AMDGPU_GCNLANEMASKUTILS_H

#include "AMDGPULaneMaskUtils.h"

#include "SIRegisterInfo.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineSSAUpdater.h"

namespace llvm {

class GCNLaneMaskAnalysis;
class MachineFunction;

/// \brief Helper class for lane-mask related tasks.
class GCNLaneMaskUtils {
private:
  MachineFunction &MF;
  const AMDGPU::LaneMaskConstants &LMC;

public:
  GCNLaneMaskUtils() = delete;
  explicit GCNLaneMaskUtils(MachineFunction &MF)
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
                           GCNLaneMaskAnalysis *LMA = nullptr,
                           bool isPrevZeroReg = false) const;
};

/// Lazy analyses of lane masks.
class GCNLaneMaskAnalysis {
private:
  GCNLaneMaskUtils LMU;

public:
  GCNLaneMaskAnalysis(MachineFunction &MF) : LMU(MF) {}

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
class GCNLaneMaskUpdater {
public:
  enum ResetFlags {
    ResetInMiddle = (1 << 0),
    ResetAtEnd = (1 << 1),
  };

private:
  GCNLaneMaskUtils LMU;
  GCNLaneMaskAnalysis *LMA = nullptr;

  bool Processed = false;

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
  DenseSet<MachineInstr *> PotentiallyDead;
  DenseMap<MachineBasicBlock *, SmallVector<std::pair<Register, unsigned>, 2>>
      AccumulatorResetBlocks;
  SmallDenseSet<Register, 4> AllAccumulators;

public:
  Register Accumulator;

  const SmallDenseSet<Register, 4> &getAllAccumulators() const {
    return AllAccumulators;
  }

  GCNLaneMaskUpdater(MachineFunction &MF) : LMU(MF) {}

  void setLaneMaskAnalysis(GCNLaneMaskAnalysis *Analysis) { LMA = Analysis; }

  void init();
  void cleanup();

  void addReset(MachineBasicBlock &Block, ResetFlags Flags);
  void addAvailable(MachineBasicBlock &Block, Register Value);

  Register getValueInMiddleOfBlock(MachineBasicBlock &Block);
  Register getValueAtEndOfBlock(MachineBasicBlock &Block);
  Register getValueAfterMerge(MachineBasicBlock &Block);
  void insertAccumulatorResets();

private:
  void process();
  SmallVectorImpl<BlockInfo>::iterator findBlockInfo(MachineBasicBlock &Block);
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_GCNLANEMASKUTILS_H
