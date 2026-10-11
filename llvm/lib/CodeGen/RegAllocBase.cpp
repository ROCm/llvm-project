//===- RegAllocBase.cpp - Register Allocator Base Class -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the RegAllocBase class which provides common functionality
// for LiveIntervalUnion-based register allocators.
//
//===----------------------------------------------------------------------===//

#include "RegAllocBase.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/LiveInterval.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveRegMatrix.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Spiller.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassTimingInfo.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "regalloc"

STATISTIC(NumNewQueued, "Number of new live ranges queued");

// Temporary verification option until we can put verification inside
// MachineVerifier.
static cl::opt<bool, true>
    VerifyRegAlloc("verify-regalloc", cl::location(RegAllocBase::VerifyEnabled),
                   cl::Hidden, cl::desc("Verify during register allocation"));

static cl::opt<bool> EnableVGPRFragAnalysis(
    "regalloc-frag", cl::Hidden,
    cl::desc("Emit VGPR fragmentation metrics at each selectOrSplit"));

// Debug flag for tracing foreachUnit behavior during fragmentation analysis.
bool DebugFragTrace = false;

static cl::opt<unsigned>
    FragTraceSeq("regalloc-frag-trace-seq", cl::Hidden,
                 cl::desc("Trace foreachUnit for this seq number (0=disabled)"),
                 cl::init(0));

const char RegAllocBase::TimerGroupName[] = "regalloc";
const char RegAllocBase::TimerGroupDescription[] = "Register Allocation";
bool RegAllocBase::VerifyEnabled = false;

//===----------------------------------------------------------------------===//
//                         RegAllocBase Implementation
//===----------------------------------------------------------------------===//

// Pin the vtable to this file.
void RegAllocBase::anchor() {}

void RegAllocBase::init(VirtRegMap &vrm, LiveIntervals &lis,
                        LiveRegMatrix &mat) {
  TRI = &vrm.getTargetRegInfo();
  MRI = &vrm.getRegInfo();
  VRM = &vrm;
  LIS = &lis;
  Matrix = &mat;
  MRI->freezeReservedRegs();
  RegClassInfo.runOnMachineFunction(vrm.getMachineFunction());
  FailedVRegs.clear();
}

// Visit all the live registers. If they are already assigned to a physical
// register, unify them with the corresponding LiveIntervalUnion, otherwise push
// them on the priority queue for later assignment.
void RegAllocBase::seedLiveRegs() {
  NamedRegionTimer T("seed", "Seed Live Regs", TimerGroupName,
                     TimerGroupDescription, TimePassesIsEnabled);
  for (unsigned i = 0, e = MRI->getNumVirtRegs(); i != e; ++i) {
    Register Reg = Register::index2VirtReg(i);
    if (MRI->reg_nodbg_empty(Reg))
      continue;
    enqueue(&LIS->getInterval(Reg));
  }
}

/// Find the VGPR_32 base register class by name. Returns nullptr if not found
/// (non-AMDGPU targets).
static const TargetRegisterClass *
findVGPR32Class(const TargetRegisterInfo *TRI) {
  for (const TargetRegisterClass *RC : TRI->regclasses()) {
    if (RC->isBaseClass() && RC->isAllocatable() &&
        TRI->getRegSizeInBits(*RC) == 32 &&
        StringRef(TRI->getRegClassName(RC)) == "VGPR_32")
      return RC;
  }
  return nullptr;
}

// [Num256Regs, CovBy256] =
static auto computeAvailNumAndCoverage(unsigned RegSizeIn32Bit, int startFreeI,
                                       int endFreeI) {
  assert(RegSizeIn32Bit >= 2 && (RegSizeIn32Bit & 1) == 0);
  assert(startFreeI != -1 && startFreeI < endFreeI && startFreeI >= 0 &&
         "startFreeI should be set");

  // Align 2 -- can be only at even indices.
  if (startFreeI % 2 == 1)
    startFreeI += 1;
  int LastPossibleStartI = endFreeI - RegSizeIn32Bit;
  if (LastPossibleStartI % 2 == 1)
    LastPossibleStartI -= 1;
  if (startFreeI > LastPossibleStartI)
    return std::make_tuple(0u, 0u, 0u);

  // ceiling((LastPossibleStartI + 1) - startFreeI)
  unsigned NumStartIndices = ((LastPossibleStartI + 1) - startFreeI + 1) / 2;
  unsigned CoveredByRegSizeIn32Bit =
      LastPossibleStartI - startFreeI + RegSizeIn32Bit;
  unsigned DistinctFreeRegs = (endFreeI - startFreeI + 1) / RegSizeIn32Bit;
  return std::make_tuple(NumStartIndices, CoveredByRegSizeIn32Bit,
                         DistinctFreeRegs);
}

/// Compute and emit VGPR fragmentation metrics for the current VirtReg.
/// The bitmap represents effective interference: a VGPR is "occupied" if any
/// assigned vreg on it overlaps in time with VirtReg's live range.
static void computeVGPRFragmentation(
    unsigned SeqNum, const LiveInterval &VirtReg, LiveRegMatrix *Matrix,
    const TargetRegisterInfo *TRI, const MachineRegisterInfo *MRI,
    const RegisterClassInfo &RegClassInfo, ArrayRef<MCPhysReg> VGPROrder) {
  // const TargetRegisterClass *VGPR32RC) {
  // ArrayRef<MCPhysReg> VGPROrder = RegClassInfo.getOrder(VGPR32RC);
  unsigned NumVGPRs = VGPROrder.size();
  if (NumVGPRs == 0)
    return;

  // Build occupancy bitmap: occupied[i] = true if VGPROrder[i] interferes
  // with VirtReg's live range.
  std::vector<bool> Occupied(NumVGPRs, false);
  unsigned NumOccupied = 0;
  for (unsigned I = 0; I < NumVGPRs; ++I) {
    auto IK = Matrix->checkInterference(VirtReg, VGPROrder[I]);
    if (IK != LiveRegMatrix::IK_Free) {
      Occupied[I] = true;
      ++NumOccupied;
    }
  }

  unsigned NumFree = NumVGPRs - NumOccupied;

  // Compute max contiguous free block.
  unsigned MaxBlock = 0;
  unsigned CurBlock = 0;
  for (unsigned I = 0; I < NumVGPRs; ++I) {
    if (!Occupied[I]) {
      ++CurBlock;
      MaxBlock = std::max(MaxBlock, CurBlock);
    } else {
      CurBlock = 0;
    }
  }

  // Count available slots per register class width (all with align-2).
  // avail_256: even-aligned start, 8 consecutive free
  // avail_128: even-aligned start, 4 consecutive free
  // avail_64:  even-aligned start, 2 consecutive free
  // avail_32:  any single free (= NumFree)
  unsigned Avail256 = 0, Avail128 = 0, Avail64 = 0;

  for (unsigned I = 0; I + 8 <= NumVGPRs; I += 2) {
    bool AllFree = true;
    for (unsigned J = I; J < I + 8; ++J) {
      if (Occupied[J]) {
        AllFree = false;
        break;
      }
    }
    if (AllFree)
      ++Avail256;
  }

  for (unsigned I = 0; I + 4 <= NumVGPRs; I += 2) {
    bool AllFree = true;
    for (unsigned J = I; J < I + 4; ++J) {
      if (Occupied[J]) {
        AllFree = false;
        break;
      }
    }
    if (AllFree)
      ++Avail128;
  }

  for (unsigned I = 0; I + 2 <= NumVGPRs; I += 2) {
    if (!Occupied[I] && !Occupied[I + 1])
      ++Avail64;
  }

  // Fragmentation ratio: fraction of free VGPRs that do NOT belong to any
  // valid 8-wide align-2 free window.
  // A free VGPR "belongs to" a valid VReg_256 window if there exists an
  // even-aligned 8-wide block containing it where all 8 are free.
  unsigned CoveredBy256 = 0;
  std::vector<bool> InValid256(NumVGPRs, false);
  for (unsigned I = 0; I + 8 <= NumVGPRs; I += 2) {
    bool AllFree = true;
    for (unsigned J = I; J < I + 8; ++J) {
      if (Occupied[J]) {
        AllFree = false;
        break;
      }
    }
    if (AllFree) {
      for (unsigned J = I; J < I + 8; ++J)
        InValid256[J] = true;
    }
  }
  for (unsigned I = 0; I < NumVGPRs; ++I) {
    // assert(InValid256[I] == !Occupied[I] && "InValid256 and Occupied must be
    // opposite"); ?? Why is this not true?
    if (!Occupied[I] && InValid256[I])
      ++CoveredBy256;
  }

  double FragRatio = 0.0;
  if (NumFree > 0)
    FragRatio = 1.0 - (double)CoveredBy256 / (double)NumFree;

  // My logic
  unsigned Avail_256 = 0, Avail_128 = 0, Avail_64 = 0;
  unsigned Distinct_256 = 0, Distinct_128 = 0, Distinct_64 = 0;
  unsigned TotalCovBy256 = 0, TotalCovBy128 = 0, TotalCovBy64 = 0;
  enum State { FREE, OCCUPIED };
  bool state = OCCUPIED;
  int startFreeI = 0, endFreeI = -1;
  int consecutiveFree = 0;
  unsigned I = 0;
  for (; I < NumVGPRs; I++) {
    if (Occupied[I]) {
      if (state ==
          FREE) { // Transition from FREE to OCCUPIED - compute Free region
        state = OCCUPIED;
        endFreeI = I;

        assert(startFreeI != -1 && startFreeI < endFreeI &&
               "startFreeI should be set");
        auto [Num256Regs, CovBy256, Distinct256Regs] =
            computeAvailNumAndCoverage(8 /* 256-bit reg*/, startFreeI,
                                       endFreeI);
        Avail_256 += Num256Regs;
        TotalCovBy256 += CovBy256;
        Distinct_256 += Distinct256Regs;
        auto [Num128Regs, CovBy128, Distinct128Regs] =
            computeAvailNumAndCoverage(4 /* 128-bit reg*/, startFreeI,
                                       endFreeI);
        Avail_128 += Num128Regs;
        TotalCovBy128 += CovBy128;
        Distinct_128 += Distinct128Regs;
        auto [Num64Regs, CovBy64, Distinct64Regs] =
            computeAvailNumAndCoverage(2 /* 64-bit reg*/, startFreeI, endFreeI);
        Avail_64 += Num64Regs;
        TotalCovBy64 += CovBy64;
        Distinct_64 += Distinct64Regs;
      }
      consecutiveFree = 0;
    } else { // free
      if (state ==
          OCCUPIED) { // Transition from OCCUPIED to FREE - start Free region
        state = FREE;
        startFreeI = I;
      }
      consecutiveFree++;
    }
  }
  if (state == FREE) { // Finished with Free state - End of Free region -
                       // compute Free region
    endFreeI = I;

    assert(startFreeI != -1 && startFreeI < endFreeI &&
           "startFreeI should be set");
    assert(consecutiveFree == (endFreeI - startFreeI) && "Error in epilog");
    auto [Num256Regs, CovBy256, Distinct256Regs] =
        computeAvailNumAndCoverage(8 /* 256-bit reg*/, startFreeI, endFreeI);
    Avail_256 += Num256Regs;
    TotalCovBy256 += CovBy256;
    Distinct_256 += Distinct256Regs;
    auto [Num128Regs, CovBy128, Distinct128Regs] =
        computeAvailNumAndCoverage(4 /* 128-bit reg*/, startFreeI, endFreeI);
    Avail_128 += Num128Regs;
    TotalCovBy128 += CovBy128;
    Distinct_128 += Distinct128Regs;
    auto [Num64Regs, CovBy64, Distinct64Regs] =
        computeAvailNumAndCoverage(2 /* 64-bit reg*/, startFreeI, endFreeI);
    Avail_64 += Num64Regs;
    TotalCovBy64 += CovBy64;
    Distinct_64 += Distinct64Regs;
  }

  // Why these assertions are not true?
  assert(Avail256 == Avail_256 && "Avail_256 error.");
  assert(Avail128 == Avail_128 && "Avail_128 error.");
  assert(Avail64 == Avail_64 && "Avail_64 error.");
  assert(CoveredBy256 == TotalCovBy256 &&
         "CoverBy256 and TotalCovBy256 are diff.");

  double FragRatio128 = 0.0;
  if (NumFree > 0)
    FragRatio128 = 1.0 - (double)TotalCovBy128 / (double)NumFree;

  double FragRatio64 = 0.0;
  if (NumFree > 0)
    FragRatio64 = 1.0 - (double)TotalCovBy64 / (double)NumFree;

  double DistinctRatio256 = 0.0;
  if (NumFree > 0)
    // (NumFree/8) is the number of 256-bit registers that can be allocated,
    // if NumFree VGPR32 registers are available consecutively.
    // Distinct_256 is actual number of 256-bit registers that can be allocated.
    // That number can reflect fragmentation of the VGPR32 registers.
    DistinctRatio256 = 1 - (double)Distinct_256 / (double)NumFree * 8;
  double DistinctRatio128 = 0.0;
  if (NumFree > 0)
    DistinctRatio128 = 1 - (double)Distinct_128 / (double)NumFree * 4;
  double DistinctRatio64 = 0.0;
  if (NumFree > 0)
    DistinctRatio64 = 1 - (double)Distinct_64 / (double)NumFree * 2;

  const TargetRegisterClass *RC = MRI->getRegClass(VirtReg.reg());
  dbgs() << "VGPR_FRAG:"
         << " seq=" << SeqNum << " rc=" << TRI->getRegClassName(RC)
         << " vreg=" << printReg(VirtReg.reg(), TRI) << " range=" << VirtReg
         << " total=" << NumVGPRs << " occupied=" << NumOccupied
         << " free=" << NumFree << " max_block=" << MaxBlock
         << " avail_256=" << Avail256 << " avail_128=" << Avail128
         << " avail_64=" << Avail64 << " avail_32=" << NumFree
         << " frag_256_ratio=" << format("%.4f", FragRatio)
         << " frag_128_ratio=" << format("%.4f", FragRatio128)
         << " frag_64_ratio=" << format("%.4f", FragRatio64)
         << " distinct_256=" << Distinct_256 << " distinct_128=" << Distinct_128
         << " distinct_64=" << Distinct_64
         << " distinct_256_ratio=" << format("%.4f", DistinctRatio256)
         << " distinct_128_ratio=" << format("%.4f", DistinctRatio128)
         << " distinct_64_ratio=" << format("%.4f", DistinctRatio64) << '\n';

  bool myLogicPrinted = false;
  if (Avail256 != Avail_256) {
    myLogicPrinted = true;
    dbgs() << " Avail_256:" << Avail_256;
  }
  if (Avail128 != Avail_128) {
    myLogicPrinted = true;
    dbgs() << " Avail_128:" << Avail_128;
  }
  if (Avail64 != Avail_64) {
    myLogicPrinted = true;
    dbgs() << " Avail_64:" << Avail_64;
  }
  if (CoveredBy256 != TotalCovBy256) {
    myLogicPrinted = true;
    dbgs() << " Avail_TotalCovBy256:" << TotalCovBy256;
  }
  if (myLogicPrinted)
    dbgs() << "\n";
  // dbgs() << " == My Logic == Avail_256=" << Avail_256 << " Avail_128=" <<
  // Avail_128 << " Avail_64=" << Avail_64 << " TotalCovBy256=" << TotalCovBy256
  // << " TotalCovBy128=" << TotalCovBy128 << " TotalCovBy64=" << TotalCovBy64
  // << '\n';
}

// Top-level driver to manage the queue of unassigned VirtRegs and call the
// selectOrSplit implementation.
void RegAllocBase::allocatePhysRegs() {
  seedLiveRegs();

  // Look up the VGPR_32 base class for fragmentation analysis (AMDGPU only).
  const TargetRegisterClass *VGPR32RC = nullptr;
  unsigned FragSeqNum = 0;
  if (EnableVGPRFragAnalysis)
    VGPR32RC = findVGPR32Class(TRI);
  ArrayRef<MCPhysReg> VGPROrder = RegClassInfo.getOrder(VGPR32RC);

  // Continue assigning vregs one at a time to available physical registers.
  while (const LiveInterval *VirtReg = dequeue()) {
    assert(!VRM->hasPhys(VirtReg->reg()) && "Register already assigned");

    // Unused registers can appear when the spiller coalesces snippets.
    if (MRI->reg_nodbg_empty(VirtReg->reg())) {
      LLVM_DEBUG(dbgs() << "Dropping unused " << *VirtReg << '\n');
      aboutToRemoveInterval(*VirtReg);
      LIS->removeInterval(VirtReg->reg());
      continue;
    }

    // Invalidate all interference queries, live ranges could have changed.
    Matrix->invalidateVirtRegs();

    // Emit VGPR fragmentation metrics before the allocation decision.
    if (VGPR32RC) {
      ++FragSeqNum;
      computeVGPRFragmentation(FragSeqNum, *VirtReg, Matrix, TRI, MRI,
                               RegClassInfo, VGPROrder);
    }

    // selectOrSplit requests the allocator to return an available physical
    // register if possible and populate a list of new live intervals that
    // result from splitting.
    LLVM_DEBUG(dbgs() << "\nselectOrSplit "
                      << TRI->getRegClassName(MRI->getRegClass(VirtReg->reg()))
                      << ':' << *VirtReg << '\n');

    using VirtRegVec = SmallVector<Register, 4>;

    VirtRegVec SplitVRegs;
    MCRegister AvailablePhysReg = selectOrSplit(*VirtReg, SplitVRegs);

    if (AvailablePhysReg == ~0u) {
      // selectOrSplit failed to find a register!
      // Probably caused by an inline asm.
      MachineInstr *MI = nullptr;
      for (MachineInstr &MIR : MRI->reg_instructions(VirtReg->reg())) {
        MI = &MIR;
        if (MI->isInlineAsm())
          break;
      }

      const TargetRegisterClass *RC = MRI->getRegClass(VirtReg->reg());
      AvailablePhysReg = getErrorAssignment(*RC, MI);

      // Keep going after reporting the error.
      cleanupFailedVReg(VirtReg->reg(), AvailablePhysReg, SplitVRegs);
    } else if (AvailablePhysReg)
      Matrix->assign(*VirtReg, AvailablePhysReg);

    for (Register Reg : SplitVRegs) {
      assert(LIS->hasInterval(Reg));

      LiveInterval *SplitVirtReg = &LIS->getInterval(Reg);
      assert(!VRM->hasPhys(SplitVirtReg->reg()) && "Register already assigned");
      if (MRI->reg_nodbg_empty(SplitVirtReg->reg())) {
        assert(SplitVirtReg->empty() && "Non-empty but used interval");
        LLVM_DEBUG(dbgs() << "not queueing unused  " << *SplitVirtReg << '\n');
        aboutToRemoveInterval(*SplitVirtReg);
        LIS->removeInterval(SplitVirtReg->reg());
        continue;
      }
      LLVM_DEBUG(dbgs() << "queuing new interval: " << *SplitVirtReg << "\n");
      assert(SplitVirtReg->reg().isVirtual() &&
             "expect split value in virtual register");
      enqueue(SplitVirtReg);
      ++NumNewQueued;
    }
  }
}

void RegAllocBase::postOptimization() {
  spiller().postOptimization();

  // Verify LiveRegMatrix after spilling (no dangling pointers).
  assert(Matrix->isValid() && "LiveRegMatrix validation failed");

  for (auto *DeadInst : DeadRemats) {
    LIS->RemoveMachineInstrFromMaps(*DeadInst);
    DeadInst->eraseFromParent();
  }
  DeadRemats.clear();
}

void RegAllocBase::cleanupFailedVReg(Register FailedReg, MCRegister PhysReg,
                                     SmallVectorImpl<Register> &SplitRegs) {
  // We still should produce valid IR. Kill all the uses and reduce the live
  // ranges so that we don't think it's possible to introduce kill flags later
  // which will fail the verifier.
  for (MachineOperand &MO : MRI->reg_operands(FailedReg)) {
    if (MO.readsReg())
      MO.setIsUndef(true);
  }

  if (!MRI->isReserved(PhysReg)) {
    // Physical liveness for any aliasing registers is now unreliable, so delete
    // the uses.
    for (MCRegAliasIterator Aliases(PhysReg, TRI, true); Aliases.isValid();
         ++Aliases) {
      for (MachineOperand &MO : MRI->reg_operands(*Aliases)) {
        if (MO.readsReg())
          MO.setIsUndef(true);
      }
    }
  }

  // Directly perform the rewrite, and do not leave it to VirtRegRewriter as
  // usual. This avoids trying to manage illegal overlapping assignments in
  // LiveRegMatrix.
  MRI->replaceRegWith(FailedReg, PhysReg);
  LIS->removeInterval(FailedReg);
}

void RegAllocBase::enqueue(const LiveInterval *LI) {
  const Register Reg = LI->reg();

  assert(Reg.isVirtual() && "Can only enqueue virtual registers");

  if (VRM->hasPhys(Reg))
    return;

  if (shouldAllocateRegister(Reg)) {
    LLVM_DEBUG(dbgs() << "Enqueuing " << printReg(Reg, TRI) << '\n');
    enqueueImpl(LI);
  } else {
    LLVM_DEBUG(dbgs() << "Not enqueueing " << printReg(Reg, TRI)
                      << " in skipped register class\n");
  }
}

MCPhysReg RegAllocBase::getErrorAssignment(const TargetRegisterClass &RC,
                                           const MachineInstr *CtxMI) {
  MachineFunction &MF = VRM->getMachineFunction();

  // Avoid printing the error for every single instance of the register. It
  // would be better if this were per register class.
  bool EmitError = !MF.getProperties().hasFailedRegAlloc();
  if (EmitError)
    MF.getProperties().setFailedRegAlloc();

  const Function &Fn = MF.getFunction();
  LLVMContext &Context = Fn.getContext();

  ArrayRef<MCPhysReg> AllocOrder = RegClassInfo.getOrder(&RC);
  if (AllocOrder.empty()) {
    // If the allocation order is empty, it likely means all registers in the
    // class are reserved. We still to need to pick something, so look at the
    // underlying class.
    ArrayRef<MCPhysReg> RawRegs = RC.getRegisters();

    if (EmitError) {
      Context.diagnose(DiagnosticInfoRegAllocFailure(
          "no registers from class available to allocate", Fn,
          CtxMI ? CtxMI->getDebugLoc() : DiagnosticLocation()));
    }

    assert(!RawRegs.empty() && "register classes cannot have no registers");
    return RawRegs.front();
  }

  if (EmitError) {
    if (CtxMI && CtxMI->isInlineAsm()) {
      CtxMI->emitInlineAsmError(
          "inline assembly requires more registers than available");
    } else {
      Context.diagnose(DiagnosticInfoRegAllocFailure(
          "ran out of registers during register allocation", Fn,
          CtxMI ? CtxMI->getDebugLoc() : DiagnosticLocation()));
    }
  }

  return AllocOrder.front();
}
