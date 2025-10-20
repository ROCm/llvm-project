//===-- AMDGPUHotBlockRegisterRenaming.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief Reduces value density in hot basic blocks by remapping local values
/// from overused physical registers to free physical registers.
///
/// This gives the Post-RA scheduler more flexibility to reorder instructions
/// by reducing false dependencies created by register reuse.
///
/// Algorithm:
/// 1. Sort basic blocks by frequency (hottest first)
/// 2. For each BB:
///    a. Calculate value density (count of local values per PhysReg)
///    b. Identify free PhysRegs (completely unused in this BB)
///    c. Iteratively move local values from dense to free registers
/// 3. VirtRegRewriter applies the updated VirtRegMap
///
/// Constraints (conservative):
/// - Only move 32-bit VGPRs
/// - Only move local values (single segment, entirely within BB)
/// - Only move to completely free registers
/// - Skip values with allocation hints
/// - Skip reserved registers
///
//===----------------------------------------------------------------------===//

#include "AMDGPUHotBlockRegisterRenaming.h"
#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "SIMachineFunctionInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveRegMatrix.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include <queue>

using namespace llvm;

#define DEBUG_TYPE "amdgpu-hot-block-reg-renaming"

STATISTIC(NumBlocksProcessed, "Number of hot blocks processed");
STATISTIC(NumValuesRemapped, "Number of values remapped to reduce density");
STATISTIC(NumBlocksSkipped,
          "Number of blocks skipped (no dense regs or no free regs)");

namespace {

class AMDGPUHotBlockRegisterRenamingImpl {
public:
  AMDGPUHotBlockRegisterRenamingImpl(VirtRegMap *VRM, LiveRegMatrix *LRM,
                                     LiveIntervals *LIS,
                                     MachineBlockFrequencyInfo *MBFI,
                                     const GCNSubtarget *ST,
                                     const SIMachineFunctionInfo &MFI)
      : VRM(VRM), LRM(LRM), LIS(LIS), MBFI(MBFI), ST(ST), MFI(MFI) {}

  bool run(MachineFunction &MF);

private:
  VirtRegMap *VRM;
  LiveRegMatrix *LRM;
  LiveIntervals *LIS;
  MachineBlockFrequencyInfo *MBFI;
  const GCNSubtarget *ST;
  const SIMachineFunctionInfo &MFI;
  const SIRegisterInfo *TRI = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  unsigned VGPRLimit = 0; // Register limit based on occupancy

  /// Cache of VirtRegs that cannot be moved (e.g. tied operands)
  DenseSet<Register> UnmovableVRegs;

  /// Process a single basic block
  bool processBasicBlock(MachineBasicBlock *MBB);

  /// Calculate value density map for a basic block
  void calculateValueDensity(MachineBasicBlock *MBB,
                             DenseMap<MCRegister, unsigned> &ValueDensity);

  /// Find free physical registers in a basic block
  void findFreeRegisters(MachineBasicBlock *MBB,
                         SmallVectorImpl<MCRegister> &FreeRegs);

  /// Check if a segment is local to a basic block
  bool isLocalSegment(const LiveInterval::Segment &Seg, SlotIndex BBStart,
                      SlotIndex BBEnd) const;

  /// Check if a register is suitable for our optimization
  bool isSuitableRegister(MCRegister PhysReg) const;

  /// Check if a virtual register can be safely moved
  bool isVirtRegMovable(Register VirtReg, MCRegister CurrentPhysReg,
                        MCRegister TargetPhysReg, SlotIndex BBStart,
                        SlotIndex BBEnd);

  /// Try to move a value from DenseReg to FreeReg
  bool tryMoveValue(MCRegister DenseReg, MCRegister FreeReg,
                    MachineBasicBlock *MBB, SlotIndex BBStart, SlotIndex BBEnd);
};

class AMDGPUHotBlockRegisterRenamingLegacy : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUHotBlockRegisterRenamingLegacy() : MachineFunctionPass(ID) {
    initializeAMDGPUHotBlockRegisterRenamingLegacyPass(
        *PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "AMDGPU Hot Block Register Renaming";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LiveIntervalsWrapperPass>();
    AU.addRequired<VirtRegMapWrapperLegacy>();
    AU.addRequired<LiveRegMatrixWrapperLegacy>();
    AU.addRequired<MachineBlockFrequencyInfoWrapperPass>();
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // end anonymous namespace

char AMDGPUHotBlockRegisterRenamingLegacy::ID = 0;

char &llvm::AMDGPUHotBlockRegisterRenamingID =
    AMDGPUHotBlockRegisterRenamingLegacy::ID;

INITIALIZE_PASS_BEGIN(AMDGPUHotBlockRegisterRenamingLegacy, DEBUG_TYPE,
                      "AMDGPU Hot Block Register Renaming", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(VirtRegMapWrapperLegacy)
INITIALIZE_PASS_DEPENDENCY(LiveRegMatrixWrapperLegacy)
INITIALIZE_PASS_DEPENDENCY(MachineBlockFrequencyInfoWrapperPass)
INITIALIZE_PASS_END(AMDGPUHotBlockRegisterRenamingLegacy, DEBUG_TYPE,
                    "AMDGPU Hot Block Register Renaming", false, false)

bool AMDGPUHotBlockRegisterRenamingLegacy::runOnMachineFunction(
    MachineFunction &MF) {
  VirtRegMap *VRM = &getAnalysis<VirtRegMapWrapperLegacy>().getVRM();
  LiveRegMatrix *LRM = &getAnalysis<LiveRegMatrixWrapperLegacy>().getLRM();
  LiveIntervals *LIS = &getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  MachineBlockFrequencyInfo *MBFI =
      &getAnalysis<MachineBlockFrequencyInfoWrapperPass>().getMBFI();

  const GCNSubtarget *ST = &MF.getSubtarget<GCNSubtarget>();
  const SIMachineFunctionInfo &MFI = *MF.getInfo<SIMachineFunctionInfo>();

  AMDGPUHotBlockRegisterRenamingImpl Impl(VRM, LRM, LIS, MBFI, ST, MFI);
  return Impl.run(MF);
}

bool AMDGPUHotBlockRegisterRenamingImpl::run(MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "AMDGPUHotBlockRegisterRenaming: Processing "
                    << MF.getName() << "\n");

  TRI = ST->getRegisterInfo();
  MRI = &MF.getRegInfo();

  // Calculate VGPR limit based on occupancy
  unsigned Occupancy = MFI.getOccupancy();
  VGPRLimit = ST->getMaxNumVGPRs(Occupancy, MFI.getDynamicVGPRBlockSize());

  LLVM_DEBUG(dbgs() << "  Occupancy: " << Occupancy
                    << ", VGPR Limit: " << VGPRLimit << "\n");

  // Sort basic blocks by frequency (hottest first)
  SmallVector<MachineBasicBlock *, 16> SortedBBs;
  for (MachineBasicBlock &MBB : MF) {
    SortedBBs.push_back(&MBB);
  }

  llvm::sort(SortedBBs, [this](MachineBasicBlock *A, MachineBasicBlock *B) {
    return MBFI->getBlockFreq(A) > MBFI->getBlockFreq(B);
  });

  bool Changed = false;
  for (MachineBasicBlock *MBB : SortedBBs) {
    Changed |= processBasicBlock(MBB);
  }

  return Changed;
}

bool AMDGPUHotBlockRegisterRenamingImpl::processBasicBlock(
    MachineBasicBlock *MBB) {
  LLVM_DEBUG(dbgs() << "  Processing BB#" << MBB->getNumber() << " (freq="
                    << MBFI->getBlockFreq(MBB).getFrequency() << ")\n");

  // Clear the unmovable cache for each BB (tied operands are BB-specific)
  UnmovableVRegs.clear();

  SlotIndex BBStart = LIS->getMBBStartIdx(MBB);
  SlotIndex BBEnd = LIS->getMBBEndIdx(MBB);

  // Step 1: Calculate value density
  DenseMap<MCRegister, unsigned> ValueDensity;
  calculateValueDensity(MBB, ValueDensity);

  if (ValueDensity.empty()) {
    LLVM_DEBUG(dbgs() << "    No values found, skipping\n");
    ++NumBlocksSkipped;
    return false;
  }

  // Step 2: Find free registers
  SmallVector<MCRegister, 64> FreeRegs;
  findFreeRegisters(MBB, FreeRegs);

  if (FreeRegs.empty()) {
    LLVM_DEBUG(dbgs() << "    No free registers, skipping\n");
    ++NumBlocksSkipped;
    return false;
  }

  LLVM_DEBUG(dbgs() << "    Found " << ValueDensity.size()
                    << " registers with values, " << FreeRegs.size()
                    << " free registers\n");

  // Step 3: Create max heap of dense registers
  auto Comparator = [&ValueDensity](MCRegister A, MCRegister B) {
    return ValueDensity[A] < ValueDensity[B]; // max heap
  };
  std::priority_queue<MCRegister, std::vector<MCRegister>, decltype(Comparator)>
      DenseRegsHeap(Comparator);

  for (auto &Entry : ValueDensity) {
    if (Entry.second > 1) { // Only interested in registers with density > 1
      DenseRegsHeap.push(Entry.first);
    }
  }

  if (DenseRegsHeap.empty()) {
    LLVM_DEBUG(
        dbgs() << "    No dense registers (all density <= 1), skipping\n");
    ++NumBlocksSkipped;
    return false;
  }

  // Step 4: Iteratively move values
  bool Changed = false;
  size_t FreeRegIdx = 0;

  while (!DenseRegsHeap.empty() && FreeRegIdx < FreeRegs.size()) {
    MCRegister DenseReg = DenseRegsHeap.top();
    DenseRegsHeap.pop();

    MCRegister FreeReg = FreeRegs[FreeRegIdx++];

    if (tryMoveValue(DenseReg, FreeReg, MBB, BBStart, BBEnd)) {
      Changed = true;
      ++NumValuesRemapped;

      // Update density
      ValueDensity[DenseReg]--;

      // If still dense, put back in heap
      if (ValueDensity[DenseReg] > 1) {
        DenseRegsHeap.push(DenseReg);
      }
    }
  }

  if (Changed) {
    ++NumBlocksProcessed;
  } else {
    ++NumBlocksSkipped;
  }

  return Changed;
}

void AMDGPUHotBlockRegisterRenamingImpl::calculateValueDensity(
    MachineBasicBlock *MBB, DenseMap<MCRegister, unsigned> &ValueDensity) {
  SlotIndex BBStart = LIS->getMBBStartIdx(MBB);
  SlotIndex BBEnd = LIS->getMBBEndIdx(MBB);

  // Iterate over VGPR_32 register class
  const TargetRegisterClass *VGPR_32_RC =
      TRI->getRegClass(AMDGPU::VGPR_32RegClassID);

  for (MCRegister PhysReg : *VGPR_32_RC) {
    if (MRI->isReserved(PhysReg))
      continue;

    unsigned LocalValueCount = 0;

    // Access LiveIntervalUnion for this PhysReg
    for (MCRegUnit Unit : TRI->regunits(PhysReg)) {
      LiveIntervalUnion &LIU = LRM->getLiveUnions()[Unit];

      for (LiveIntervalUnion::SegmentIter SI = LIU.begin(); SI.valid(); ++SI) {
        SlotIndex SegStart = SI.start();
        SlotIndex SegEnd = SI.stop();

        // Check if segment is entirely within this BB
        if (SegStart >= BBStart && SegEnd < BBEnd) {
          LocalValueCount++;
        }
      }
    }

    if (LocalValueCount > 0) {
      ValueDensity[PhysReg] = LocalValueCount;
    }
  }
}

void AMDGPUHotBlockRegisterRenamingImpl::findFreeRegisters(
    MachineBasicBlock *MBB, SmallVectorImpl<MCRegister> &FreeRegs) {
  SlotIndex BBStart = LIS->getMBBStartIdx(MBB);
  SlotIndex BBEnd = LIS->getMBBEndIdx(MBB);

  const TargetRegisterClass *VGPR_32_RC =
      TRI->getRegClass(AMDGPU::VGPR_32RegClassID);

  unsigned RegIdx = 0;
  for (MCRegister PhysReg : *VGPR_32_RC) {
    // Only consider registers up to VGPRLimit (based on occupancy)
    if (RegIdx >= VGPRLimit)
      break;
    RegIdx++;

    if (MRI->isReserved(PhysReg))
      continue;

    bool IsFree = true;

    // Check all register units
    for (MCRegUnit Unit : TRI->regunits(PhysReg)) {
      LiveIntervalUnion &LIU = LRM->getLiveUnions()[Unit];

      // Check if anything is live in this BB
      LiveIntervalUnion::SegmentIter SI = LIU.find(BBStart);
      if (SI.valid() && SI.start() < BBEnd) {
        IsFree = false;
        break;
      }
    }

    if (IsFree) {
      FreeRegs.push_back(PhysReg);
    }
  }
}

bool AMDGPUHotBlockRegisterRenamingImpl::isVirtRegMovable(Register VirtReg,
                                                          MCRegister CurrentPhysReg,
                                                          MCRegister TargetPhysReg,
                                                          SlotIndex BBStart,
                                                          SlotIndex BBEnd) {

  LiveInterval &VirtRegLI = LIS->getInterval(VirtReg);

  // Verify precondition: single value with single segment in BB
  unsigned SegmentCount = 0;
  for (const LiveRange::Segment &S : VirtRegLI) {
    if (S.start >= BBStart && S.end <= BBEnd)
      SegmentCount++;
  }
  assert(SegmentCount == 1 &&
         "isVirtRegMovable expects VirtReg with single segment in BB");
  assert(VirtRegLI.getNumValNums() == 1 &&
         "isVirtRegMovable expects VirtReg with single value");

  // Check for tied operands
  // A tied operand means the instruction requires source and destination to be
  // the same physical register. Moving such a value would break this
  // constraint.

  for (const LiveRange::Segment &S : VirtRegLI) {
    // Only check segments within this BB
    if (S.start < BBStart || S.end > BBEnd)
      continue;

    // Check if this segment starts at a tied def point
    // (meaning it's the destination of a tied operand instruction)
    MachineInstr *DefMI = LIS->getInstructionFromIndex(S.start);
    if (!DefMI)
      continue;

    for (unsigned OpIdx = 0, E = DefMI->getNumOperands(); OpIdx < E; ++OpIdx) {
      const MachineOperand &MO = DefMI->getOperand(OpIdx);
      if (MO.isReg() && MO.getReg() == VirtReg && MO.isDef() && MO.isTied()) {
        // Found a tied def - need to check the source operand it's tied to
        unsigned TiedIdx = DefMI->findTiedOperandIdx(OpIdx);
        const MachineOperand &TiedMO = DefMI->getOperand(TiedIdx);
        
        // If the tied source is a register, verify it won't conflict
        if (TiedMO.isReg()) {
          Register TiedReg = TiedMO.getReg();
          if (TiedReg.isVirtual()) {
            MCRegister TiedPhysReg = VRM->getPhys(TiedReg);
            // Cannot move if it would violate the tied constraint
            // (source and dest must be in same physical register)
            if (TiedPhysReg != CurrentPhysReg) {
              LLVM_DEBUG(dbgs() << "        Cannot move " << printReg(VirtReg, TRI)
                                << ": tied to " << printReg(TiedReg, TRI)
                                << " which is in different PhysReg "
                                << printReg(TiedPhysReg, TRI) << " at " << S.start
                                << " in " << *DefMI);
              return false;
            }
          }
        }
        
        LLVM_DEBUG(dbgs() << "        Cannot move " << printReg(VirtReg, TRI)
                          << ": has tied def at " << S.start << " in "
                          << *DefMI);
        return false;
      }
    }
  }

  // Future checks can be added here:
  // - Register class constraints
  // - Special register restrictions
  // - Architecture-specific constraints

  return true;
}

bool AMDGPUHotBlockRegisterRenamingImpl::tryMoveValue(MCRegister DenseReg,
                                                      MCRegister FreeReg,
                                                      MachineBasicBlock *MBB,
                                                      SlotIndex BBStart,
                                                      SlotIndex BBEnd) {
  // Find a movable local value in DenseReg
  for (MCRegUnit Unit : TRI->regunits(DenseReg)) {
    LiveIntervalUnion &LIU = LRM->getLiveUnions()[Unit];

    for (LiveIntervalUnion::SegmentIter SI = LIU.begin(); SI.valid(); ++SI) {
      Register VirtReg = SI.value()->reg();

      // Check if this VirtReg is mapped to DenseReg
      // NOTE: This is NOT redundant! We iterate per register unit, and units
      // can be shared between aliased registers (e.g., VGPR0 and VGPR0_VGPR1).
      // This check filters out VirtRegs mapped to aliased registers.
      if (VRM->getPhys(VirtReg) != DenseReg)
        continue;

      // Get the proper LiveInterval from LiveIntervals
      LiveInterval &VirtRegLI = LIS->getInterval(VirtReg);

      // Check: segment is local (entirely within BB)
      SlotIndex SegStart = SI.start();
      SlotIndex SegEnd = SI.stop();
      if (SegStart < BBStart || SegEnd >= BBEnd)
        continue;

      // Check: LiveInterval has only one segment (conservative)
      if (VirtRegLI.size() != 1)
        continue;

      // Check: No subranges (conservative - avoid complex cases)
      if (VirtRegLI.hasSubRanges())
        continue;

      // Check: No allocation hints
      if (VRM->hasKnownPreference(VirtReg))
        continue;

      // Check: Cached unmovable VirtRegs
      if (UnmovableVRegs.contains(VirtReg)) {
        LLVM_DEBUG(dbgs() << "        Skipping " << printReg(VirtReg, TRI)
                          << " (cached as unmovable)\n");
        continue;
      }

      // Check: Can this value be safely moved?
      if (!isVirtRegMovable(VirtReg, DenseReg, FreeReg, BBStart, BBEnd)) {
        // Cache the result to avoid checking again
        UnmovableVRegs.insert(VirtReg);
        continue;
      }

      // This VirtReg is movable! Perform the remap
      LLVM_DEBUG(dbgs() << "      Moving " << printReg(VirtReg, TRI) << " from "
                        << printReg(DenseReg, TRI) << " to "
                        << printReg(FreeReg, TRI) << "\n");

      // Safety check: must be assigned before unassign
      if (!VRM->hasPhys(VirtReg)) {
        LLVM_DEBUG(
            dbgs() << "        WARNING: VirtReg not assigned, skipping\n");
        continue;
      }

      LRM->unassign(VirtRegLI); // Remove from LiveRegMatrix
      LRM->assign(VirtRegLI,
                  FreeReg); // Assign to new physreg (updates VirtRegMap too)

      // Sanity check: verify VirtReg is now mapped to FreeReg
      assert(VRM->getPhys(VirtReg) == FreeReg &&
             "VirtRegMap not updated correctly");

      return true; // Successfully moved one value
    }
  }

  return false; // No movable value found
}

PreservedAnalyses
AMDGPUHotBlockRegisterRenamingPass::run(MachineFunction &MF,
                                        MachineFunctionAnalysisManager &MFAM) {
  VirtRegMap *VRM = &MFAM.getResult<VirtRegMapAnalysis>(MF);
  LiveRegMatrix *LRM = &MFAM.getResult<LiveRegMatrixAnalysis>(MF);
  LiveIntervals *LIS = &MFAM.getResult<LiveIntervalsAnalysis>(MF);
  MachineBlockFrequencyInfo *MBFI =
      &MFAM.getResult<MachineBlockFrequencyAnalysis>(MF);

  const GCNSubtarget *ST = &MF.getSubtarget<GCNSubtarget>();
  const SIMachineFunctionInfo &MFI = *MF.getInfo<SIMachineFunctionInfo>();

  AMDGPUHotBlockRegisterRenamingImpl Impl(VRM, LRM, LIS, MBFI, ST, MFI);
  if (!Impl.run(MF))
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}
