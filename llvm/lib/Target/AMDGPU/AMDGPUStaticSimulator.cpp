//===- AMDGPUStaticSimulator.cpp - Static Performance Simulator -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Static simulator for AMDGPU kernels that estimates performance metrics
/// without running on hardware. Currently enabled only for gfx1250.
///
/// This pass runs at the end of the pipeline before MC lowering. It walks
/// the MachineFunction, simulating instruction execution using the AMDGPUSim
/// library to produce instruction counts, stall estimates, and efficiency
/// metrics. Results are emitted as assembly comments.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUStaticSimulator.h"
#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIDefines.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineBranchProbabilityInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachinePostDominators.h"
#include "llvm/CodeGen/TargetSchedule.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include <cmath>
#include <cstdlib>

using namespace llvm;
using namespace llvm::AMDGPU;

#define DEBUG_TYPE "amdgpu-static-simulator"

static cl::opt<bool> EnableStaticSimulator(
    "amdgpu-enable-static-simulator",
    cl::desc("Enable static performance simulator for AMDGPU kernels"),
    cl::init(false), cl::Hidden);

static cl::opt<bool> VerboseSimulation(
    "amdgpu-static-sim-verbose",
    cl::desc("Enable verbose per-instruction logging in static simulator"),
    cl::init(false), cl::Hidden);

static cl::opt<bool> EnableScoreboardOpt(
    "amdgpu-static-sim-scoreboard",
    cl::desc(
        "Enable register scoreboard for RAW detection without s_delay_alu"),
    cl::init(false), cl::Hidden);

static cl::opt<unsigned> VaVdstMultiplierOpt(
    "amdgpu-static-sim-va-vdst-multiplier",
    cl::desc("Multiplier for VA_VDST latency tracking (default 4)"),
    cl::init(4), cl::Hidden);

static cl::opt<unsigned> SQCToISLatency(
    "amdgpu-static-sim-sqc-is-latency",
    cl::desc(
        "SQC to IS (Instruction Store) cache line fetch latency in cycles"),
    cl::init(26), cl::Hidden);

static cl::opt<bool> EnableISCacheModel(
    "amdgpu-static-sim-is-cache",
    cl::desc("Enable Instruction Store cache line modeling"),
    cl::init(true), cl::Hidden);

/// Check if enabled via cl::opt or AMDGPU_ENABLE_STATIC_SIM env var.
static bool isStaticSimulatorEnabled() {
  if (const char *EnvVal = std::getenv("AMDGPU_ENABLE_STATIC_SIM"))
    return StringRef(EnvVal) == "1";
  return EnableStaticSimulator;
}

namespace {

//===----------------------------------------------------------------------===//
// False Wait Detection
//===----------------------------------------------------------------------===//

static SmallSet<unsigned, 16> collectUsedVGPRs(const MachineInstr &MI,
                                                const SIInstrInfo &TII) {
  SmallSet<unsigned, 16> UsedVGPRs;
  const SIRegisterInfo &TRI = TII.getRegisterInfo();

  for (const MachineOperand &MO : MI.uses()) {
    if (!MO.isReg() || !MO.getReg().isPhysical() || MO.isImplicit())
      continue;

    Register Reg = MO.getReg();
    const TargetRegisterClass *RC = TRI.getPhysRegBaseClass(Reg);
    if (!TRI.hasVGPRs(RC))
      continue;

    unsigned BaseIdx = TRI.getHWRegIndex(Reg);
    unsigned SizeInBits = TRI.getRegSizeInBits(*RC);
    unsigned NumRegs = SizeInBits / 32;
    for (unsigned i = 0; i < NumRegs; ++i)
      UsedVGPRs.insert(BaseIdx + i);
  }

  return UsedVGPRs;
}

static const MachineInstr *
findNextConsumer(MachineBasicBlock::const_instr_iterator It,
                 MachineBasicBlock::const_instr_iterator End,
                 MachineInstrInfo &MII) {
  for (++It; It != End; ++It) {
    const MachineInstr &MI = *It;
    if (MI.isBundle() || MI.isMetaInstruction() || MI.isDebugInstr())
      continue;
    if (MI.isImplicitDef())
      continue;
    SimInst SI = MII.createSimInst(MI);
    if (SI.Class == InstClass::WAITCNT || SI.Class == InstClass::NOP ||
        SI.Class == InstClass::DELAY_ALU || SI.Class == InstClass::MSB_SET)
      continue;
    return &MI;
  }
  return nullptr;
}

struct FalseWaitResult {
  unsigned Count = 0;
  unsigned WastedCycles = 0;
};

static FalseWaitResult
analyzeFalseWaitsInQueue(const MachineInstr &WaitMI, unsigned WaitCount,
                         const std::deque<PendingMemOp> &Queue,
                         const MachineInstr *Consumer,
                         const SIInstrInfo &TII, unsigned CurrentCycle) {
  FalseWaitResult Result;
  if (!Consumer)
    return Result;
  if (Queue.size() <= WaitCount)
    return Result;

  unsigned NumWaited = Queue.size() - WaitCount;
  SmallSet<unsigned, 16> ConsumerUses = collectUsedVGPRs(*Consumer, TII);
  if (ConsumerUses.empty())
    return Result;

  unsigned MaxTrueWaitCompletion = 0;
  unsigned MaxAllWaitCompletion = 0;

  for (unsigned i = 0; i < NumWaited && i < Queue.size(); ++i) {
    const PendingMemOp &Op = Queue[i];
    MaxAllWaitCompletion = std::max(MaxAllWaitCompletion, Op.CompletionCycle);

    if (!Op.IsLoad)
      continue;

    // Convert SmallSet to SmallVector for writesToAny
    SmallVector<unsigned, 16> ConsumerUseVec(ConsumerUses.begin(),
                                             ConsumerUses.end());
    bool IsNeeded = Op.writesToAny(ConsumerUseVec);
    if (IsNeeded) {
      MaxTrueWaitCompletion =
          std::max(MaxTrueWaitCompletion, Op.CompletionCycle);
    } else {
      Result.Count++;
      if (VerboseSimulation) {
        dbgs() << "    False wait: op writes v" << Op.DestReg;
        if (Op.NumRegs > 1)
          dbgs() << "-v" << (Op.DestReg + Op.NumRegs - 1);
        dbgs() << " (completes @ " << Op.CompletionCycle
               << ") not used by consumer\n";
      }
    }
  }

  if (MaxAllWaitCompletion > MaxTrueWaitCompletion) {
    unsigned ActualStall = (MaxAllWaitCompletion > CurrentCycle)
                               ? (MaxAllWaitCompletion - CurrentCycle)
                               : 0;
    unsigned OptimalStall = (MaxTrueWaitCompletion > CurrentCycle)
                                ? (MaxTrueWaitCompletion - CurrentCycle)
                                : 0;
    Result.WastedCycles = ActualStall - OptimalStall;

    if (VerboseSimulation && Result.WastedCycles > 0) {
      dbgs() << "    Wasted cycles: " << Result.WastedCycles << " (actual stall "
             << ActualStall << ", optimal " << OptimalStall << ")\n";
    }
  }

  return Result;
}

static FalseWaitResult
analyzeFalseWaitsForWait(const MachineInstr &MI,
                         MachineBasicBlock::const_instr_iterator It,
                         MachineBasicBlock::const_instr_iterator End,
                         const GPUSimState &State, const SIInstrInfo &TII,
                         MachineInstrInfo &MII) {
  unsigned Opc = MI.getOpcode();
  if (Opc != AMDGPU::S_WAIT_DSCNT && Opc != AMDGPU::S_WAIT_LOADCNT)
    return {};

  unsigned WaitCount = 0;
  if (MI.getNumOperands() > 0 && MI.getOperand(0).isImm())
    WaitCount = MI.getOperand(0).getImm();

  const MachineInstr *Consumer = findNextConsumer(It, End, MII);
  if (VerboseSimulation && Consumer)
    dbgs() << "    Consumer: " << *Consumer;

  if (Opc == AMDGPU::S_WAIT_DSCNT) {
    return analyzeFalseWaitsInQueue(MI, WaitCount, State.PendingDS, Consumer,
                                    TII, State.CurrentCycle);
  }
  return analyzeFalseWaitsInQueue(MI, WaitCount, State.PendingVMEMLoad,
                                  Consumer, TII, State.CurrentCycle);
}

//===----------------------------------------------------------------------===//
// Stall Attribution
//===----------------------------------------------------------------------===//

static void attributeStall(const StallBreakdown &B, FunctionalUnit Unit,
                           InstClass IC, BlockMetrics &Metrics) {
  Metrics.VGPRCacheHits += 0;   // Cache info is in InstrSimInfo
  Metrics.VGPRCacheMisses += 0; // (attributed separately)

  if (B.RegBankInWMMAWindow && B.RegBank > 0)
    Metrics.RegBankConflictsInWMMAWindow += B.RegBank;

  unsigned TotalStall = B.total();
  if (TotalStall == 0)
    return;

  if (B.WaitCnt == TotalStall) {
    Metrics.StallWaitCnt += TotalStall;
  } else if (B.MemFIFO == TotalStall) {
    Metrics.StallMemFIFO += TotalStall;
  } else if (B.FU == TotalStall) {
    Metrics.StallFunctionalUnit += TotalStall;
    switch (Unit) {
    case FunctionalUnit::XDL:
      Metrics.StallXDL += TotalStall;
      break;
    case FunctionalUnit::VALU:
      Metrics.StallVALU += TotalStall;
      break;
    case FunctionalUnit::TRANS:
      Metrics.StallTRANSUnit += TotalStall;
      break;
    case FunctionalUnit::SALU:
      Metrics.StallSALU += TotalStall;
      break;
    case FunctionalUnit::LDS:
      Metrics.StallLDS += TotalStall;
      break;
    case FunctionalUnit::VMEM:
      Metrics.StallVMEMUnit += TotalStall;
      break;
    default:
      break;
    }
  } else if (B.VALUSlot == TotalStall) {
    Metrics.StallFunctionalUnit += TotalStall;
    Metrics.StallVALU += TotalStall;
  } else if (B.CoExec == TotalStall) {
    Metrics.StallCoExec += TotalStall;
    switch (IC) {
    case InstClass::VALU:
      Metrics.CoExecMissVALU += TotalStall;
      break;
    case InstClass::TRANS:
      Metrics.CoExecMissTRANS += TotalStall;
      break;
    case InstClass::DS_READ:
    case InstClass::DS_WRITE:
    case InstClass::VMEM_READ:
    case InstClass::VMEM_WRITE:
    case InstClass::SMEM:
    case InstClass::TDM:
      Metrics.CoExecMissMemory += TotalStall;
      break;
    default:
      Metrics.CoExecMissOther += TotalStall;
      break;
    }
  } else if (B.DelayAlu == TotalStall) {
    Metrics.StallDelayAlu += TotalStall;
  } else if (B.LongLatVALU == TotalStall) {
    Metrics.StallCoExec += TotalStall;
    Metrics.StallLongLatVALU += TotalStall;
  } else if (B.LOLVALUTRANSHazard == TotalStall) {
    Metrics.StallLOLVALUTRANS += TotalStall;
  } else if (B.SSRC == TotalStall) {
    Metrics.StallVaSSRC += TotalStall;
  } else if (B.VaVdst == TotalStall) {
    Metrics.StallVaVdst += TotalStall;
  } else if (B.RAW == TotalStall) {
    Metrics.StallRAW += TotalStall;
  } else if (B.RegBank == TotalStall && !B.RegBankInWMMAWindow) {
    Metrics.StallRegBankConflict += TotalStall;
  } else if (B.ISFetch == TotalStall) {
    // IS fetch stalls attributed separately
  }
}

//===----------------------------------------------------------------------===//
// WMMA Co-execution Tracking
//===----------------------------------------------------------------------===//

static void trackWMMACoExec(InstClass IC, const InstrSimInfo &Info,
                            const GPUSimState &State, BlockMetrics &Metrics) {
  bool InWMMAWindow = State.inWMMAWindow() && IC != InstClass::WMMA;
  if (!InWMMAWindow)
    return;

  if (Info.Breakdown.CoExec > 0)
    Metrics.WMMACoExecBlocked++;
  else
    Metrics.WMMACoExecUsed++;

  // Track I-slot utilization
  auto StageOpt = State.getWMMAStage();
  if (StageOpt) {
    uint8_t StageMask = State.ActiveWMMA.Info.StageMask[*StageOpt];
    bool IsISlot = (StageMask & AMDGPUSim::CoExecMask::VALU) != 0;

    if (IsISlot && Info.Breakdown.CoExec == 0) {
      Metrics.ISlotTotal++;
      if (IC == InstClass::VALU || IC == InstClass::TRANS)
        Metrics.ISlotUsedByVALU++;
      else
        Metrics.ISlotWastedOnNonVALU++;
    }
  }
}

//===----------------------------------------------------------------------===//
// Instruction Counting
//===----------------------------------------------------------------------===//

static void countInstruction(const SimInst &SI, MachineInstrInfo &MII,
                             BlockMetrics &Metrics) {
  Metrics.NumInstructions++;

  switch (SI.Class) {
  case InstClass::VALU: {
    Metrics.NumVALU++;
    if (MII.isVOPD(SI)) {
      Metrics.NumVOPD++;
      Metrics.NumVALU++; // VOPD = 2 VALU ops
    } else if (MII.isPacked(SI)) {
      Metrics.NumPacked++;
      Metrics.NumVALU++; // Packed = 2 VALU ops
    }
    break;
  }
  case InstClass::SALU:
    Metrics.NumSALU++;
    break;
  case InstClass::TRANS:
    Metrics.NumTRANS++;
    break;
  case InstClass::WMMA:
    Metrics.NumWMMA++;
    break;
  case InstClass::DS_READ:
    Metrics.NumDSRead++;
    break;
  case InstClass::DS_WRITE:
    Metrics.NumDSWrite++;
    break;
  case InstClass::VMEM_READ:
  case InstClass::VMEM_WRITE:
    Metrics.NumVMEM++;
    break;
  case InstClass::SMEM:
    Metrics.NumSMEM++;
    break;
  case InstClass::TDM:
    Metrics.NumTDM++;
    break;
  case InstClass::BRANCH:
    Metrics.NumBranch++;
    break;
  case InstClass::BARRIER:
    Metrics.NumBarrier++;
    break;
  case InstClass::WAITCNT:
    Metrics.NumWaitcnt++;
    break;
  case InstClass::DELAY_ALU:
    Metrics.NumDelayAlu++;
    break;
  case InstClass::MSB_SET:
    llvm_unreachable("MSB_SET should be handled separately");
  case InstClass::NOP:
    Metrics.NumNop++;
    break;
  default:
    break;
  }

  const auto *MI = SI.getAs<MachineInstr>();
  unsigned Opc = MI->getOpcode();
  if (Opc == AMDGPU::V_WRITELANE_B32)
    Metrics.NumSGPRToVGPR++;
  else if (Opc == AMDGPU::V_READLANE_B32)
    Metrics.NumVGPRToSGPR++;

  if (SIInstrInfo::isSpill(*MI) || SIInstrInfo::isFLATScratch(*MI)) {
    if (MI->mayStore())
      Metrics.NumSpill++;
    if (MI->mayLoad())
      Metrics.NumReload++;
  }
}

//===----------------------------------------------------------------------===//
// Verbose Logging: Instruction Header (pass-level, needs MachineInstr)
//===----------------------------------------------------------------------===//

static void logInstHeader(unsigned Cycle, const MachineInstr &MI,
                          const SimInst &SI, MachineInstrInfo &MII,
                          const GPUSimState &State,
                          const AMDGPUSim::ISCacheState *ISCache) {
  dbgs() << "\n[Cycle " << Cycle << "] ";
  MI.print(dbgs(), /*IsStandalone=*/true, /*SkipOpers=*/false,
           /*SkipDebugLoc=*/true, /*AddNewLine=*/false);
  dbgs() << "\n";

  unsigned InstBytes = MII.getInstBytes(SI);
  dbgs() << "  Class: " << getInstClassName(SI.Class)
         << " | Unit: " << getUnitName(SI.Unit) << " | Latency: " << SI.Latency
         << " | ResourceCycles: " << MII.getResourceCycles(SI)
         << " | Size: " << InstBytes << " bytes\n";

  if (ISCache) {
    dbgs() << "  IS: line " << ISCache->CurrentLine << " byte "
           << ISCache->BytesConsumed << "/" << ISCache->LineSizeBytes
           << " | lines ready @[";
    for (unsigned i = 0; i < ISCache->NumLines; ++i) {
      if (i > 0)
        dbgs() << ",";
      if (ISCache->LineReadyCycle[i] <= Cycle)
        dbgs() << "now";
      else
        dbgs() << ISCache->LineReadyCycle[i];
    }
    dbgs() << "]\n";
  }
}

//===----------------------------------------------------------------------===//
// Pass-level simulateInst wrapper
//===----------------------------------------------------------------------===//

static void simulateInst(const MachineInstr &MI, Simulator &Sim,
                         MachineInstrInfo &MII, const SIInstrInfo &TII,
                         BlockMetrics &Metrics,
                         KernelPerfReport *Report = nullptr) {
  const GPUSimState &State = Sim.getState();
  unsigned EntryCycle = State.CurrentCycle;
  SimInst SI = MII.createSimInst(MI);

  // MSB_SET handling: the library handles MSB_SET logic + IS cache,
  // the pass does counting + verbose logging.
  if (SI.Class == InstClass::MSB_SET) {
    // Build lookahead for MSB_SET masking
    SmallVector<SimInst, 1> Lookahead;
    if (MachineInstr *NextMI =
            SIInstrInfo::getNextRealInstr(const_cast<MachineInstr *>(&MI))) {
      Lookahead.push_back(MII.createSimInst(*NextMI));
    }

    // Core simulation (library handles MSB_SET logic + IS cache)
    InstrSimInfo Info = Sim.simulateInst(SI, Lookahead);

    // IS cache metric attribution
    Metrics.StallISFetch += Info.ISFetchStall;
    Metrics.ISFetchesTriggered += Info.ISFetchesTriggered;
    Metrics.ISBytesConsumed += Info.ISBytesConsumed;

    // Counting
    Metrics.NumInstructions++;
    Metrics.NumMSBSet++;
    if (Info.WasExposed) {
      if (Info.WasMasked) {
        Metrics.NumMSBSetMasked++;
      } else {
        Metrics.NumMSBSetExposed++;
        if (Sim.getState().inWMMAWindow()) {
          Metrics.StallCoExec++;
          Metrics.CoExecMissOther++;
        }
      }
    }

    // Verbose logging for MSB_SET (pass-level, after simulation)
    if (VerboseSimulation) {
      const auto &ISC = Sim.getState().ISCache;
      unsigned DisplayCycle =
          Info.WasFused ? (EntryCycle > 0 ? EntryCycle - 1 : 0) : EntryCycle;
      dbgs() << "\n[Cycle " << DisplayCycle << "] ";
      MI.print(dbgs(), /*IsStandalone=*/true, /*SkipOpers=*/false,
               /*SkipDebugLoc=*/true, /*AddNewLine=*/false);
      unsigned InstBytes = MII.getInstBytes(SI);
      dbgs() << "\n  Class: MSB_SET | Size: " << InstBytes << " bytes";
      if (Sim.getConfig().EnableISCache) {
        dbgs() << " | IS: line " << ISC.CurrentLine << " byte "
               << ISC.BytesConsumed << "/" << ISC.LineSizeBytes;
      }
      dbgs() << "\n  \xe2\x86\x92 MSB_SET ";
      if (Info.WasFused) {
        dbgs() << "fused with prev (free)";
      } else if (Info.WasMasked) {
        dbgs() << "exposed but MASKED (next instr stalls anyway)";
      } else {
        dbgs() << "EXPOSED (+1 cycle)";
        if (Sim.getState().inWMMAWindow())
          dbgs() << " [in WMMA window]";
      }
      dbgs() << "\n";
    }

    if (Report)
      Report->PerInstr[&MI] = Info;

    return;
  }

  // Regular instruction flow:
  // 1. Pass: log instruction header (uses IS cache state for display)
  if (VerboseSimulation)
    logInstHeader(EntryCycle, MI, SI, MII, State,
                  Sim.getConfig().EnableISCache ? &Sim.getState().ISCache
                                               : nullptr);

  // 2. Pass: false-wait analysis (before simulation modifies state)
  if (SI.Class == InstClass::WAITCNT) {
    const MachineBasicBlock *MBB = MI.getParent();
    MachineBasicBlock::const_instr_iterator It(&MI);
    FalseWaitResult FWR = analyzeFalseWaitsForWait(MI, It, MBB->instr_end(),
                                                   State, TII, MII);
    Metrics.NumFalseWaits += FWR.Count;
    Metrics.StallFalseWait += FWR.WastedCycles;

    if (VerboseSimulation && (FWR.Count > 0 || FWR.WastedCycles > 0))
      dbgs() << "  \xe2\x86\x92 False waits: " << FWR.Count
             << ", wasted cycles: " << FWR.WastedCycles << "\n";
  }

  // 3. Library: core simulation (includes IS cache handling when enabled)
  InstrSimInfo Info = Sim.simulateInst(SI);

  // 4. Pass: IS cache metric attribution
  Metrics.StallISFetch += Info.ISFetchStall;
  Metrics.ISFetchesTriggered += Info.ISFetchesTriggered;
  Metrics.ISBytesConsumed += Info.ISBytesConsumed;

  // 6. Pass: cache info attribution
  Metrics.VGPRCacheHits += Info.CacheHits;
  Metrics.VGPRCacheMisses += Info.CacheMisses;
  Metrics.VGPRCacheEvictions += Info.CacheEvictions;

  // 7. Pass: stall attribution to BlockMetrics
  attributeStall(Info.Breakdown, SI.Unit, SI.Class, Metrics);

  // 8. Pass: WMMA co-exec tracking
  trackWMMACoExec(SI.Class, Info, Sim.getState(), Metrics);

  // 9. Pass: WMMA window cycles
  if (Info.IsWMMA) {
    Metrics.WMMAWindowCycles += Sim.getState().ActiveWMMA.Info.Occupancy;
  }

  // 10. Pass: instruction counting
  countInstruction(SI, MII, Metrics);

  // 11. Pass: per-instruction report
  if (Report)
    Report->PerInstr[&MI] = Info;

  if (VerboseSimulation)
    dbgs() << "  \xe2\x86\x92 NextCycle: " << Sim.getState().CurrentCycle
           << "\n";
}

//===----------------------------------------------------------------------===//
// Block Analysis
//===----------------------------------------------------------------------===//

static BlockMetrics analyzeBlock(MachineBasicBlock &MBB, Simulator &Sim,
                                 MachineInstrInfo &MII, const SIInstrInfo &TII,
                                 KernelPerfReport *Report = nullptr) {
  if (VerboseSimulation) {
    dbgs() << "\n=== BB#" << MBB.getNumber();
    if (const BasicBlock *BB = MBB.getBasicBlock())
      if (BB->hasName())
        dbgs() << " (" << BB->getName() << ")";
    dbgs() << " [Cycle " << Sim.getState().CurrentCycle << "] ===\n";
  }

  BlockMetrics Metrics;
  unsigned StartCycle = Sim.getState().CurrentCycle;

  for (MachineInstr &MI : MBB.instrs()) {
    if (MI.isBundle() || MI.isMetaInstruction())
      continue;
    if (MI.isDebugInstr() || MI.isImplicitDef())
      continue;
    simulateInst(MI, Sim, MII, TII, Metrics, Report);
  }

  Metrics.TotalCycles = Sim.getState().CurrentCycle - StartCycle;

  if (VerboseSimulation) {
    dbgs() << "=== End BB#" << MBB.getNumber() << ": "
           << Metrics.NumInstructions << " insts, " << Metrics.TotalCycles
           << " cycles, " << Metrics.StallCycles() << " stalls ===\n";
  }

  return Metrics;
}

//===----------------------------------------------------------------------===//
// Block Frequency Helpers
//===----------------------------------------------------------------------===//

static float getBlockFrequency(const MachineBlockFrequencyInfo *MBFI,
                               const MachineBasicBlock *MBB) {
  if (!MBFI)
    return 1.0f;
  return static_cast<float>(MBFI->getBlockFreqRelativeToEntryBlock(MBB));
}

static void printBlockFrequencies(const MachineFunction &MF,
                                  const MachineBlockFrequencyInfo *MBFI) {
  if (!VerboseSimulation || !MBFI)
    return;

  dbgs() << "\n=== Block Frequencies ===\n";
  for (const MachineBasicBlock &MBB : MF) {
    dbgs() << "  bb." << MBB.getNumber() << ": "
           << format("%.3f", getBlockFrequency(MBFI, &MBB)) << "\n";
  }
}

//===----------------------------------------------------------------------===//
// Loop Analysis
//===----------------------------------------------------------------------===//

constexpr unsigned DefaultTripCount = 10;

static cl::opt<unsigned>
    TripCountOverride("amdgpu-static-sim-trip-count", cl::Hidden,
                      cl::desc("Override static sim trip count analysis."));

static unsigned computeSteadyStateISStall(unsigned LoopBodyBytes,
                                          unsigned LoopBodyCycles,
                                          unsigned FetchLatency,
                                          unsigned LineSizeBytes) {
  if (LoopBodyCycles == 0 || FetchLatency == 0)
    return 0;

  unsigned FetchableBytes =
      (LoopBodyCycles * LineSizeBytes) / FetchLatency;

  if (LoopBodyBytes > FetchableBytes) {
    unsigned ExcessBytes = LoopBodyBytes - FetchableBytes;
    unsigned ExcessLines =
        (ExcessBytes + LineSizeBytes - 1) / LineSizeBytes;
    return ExcessLines * FetchLatency;
  }
  return 0;
}

static unsigned computeIterationsUntilBackup(unsigned LoopBodyBytes,
                                             unsigned LoopBodyCycles,
                                             unsigned FetchLatency,
                                             unsigned NumLines,
                                             unsigned LineSizeBytes) {
  if (LoopBodyCycles == 0 || FetchLatency == 0 || LoopBodyBytes == 0)
    return UINT_MAX;

  unsigned InitialBuffer = NumLines * LineSizeBytes;
  unsigned FetchablePerIter =
      (LoopBodyCycles * LineSizeBytes) / FetchLatency;

  if (LoopBodyBytes <= FetchablePerIter)
    return UINT_MAX;

  unsigned DeficitPerIter = LoopBodyBytes - FetchablePerIter;
  return (InitialBuffer + DeficitPerIter - 1) / DeficitPerIter;
}

static unsigned
getLoopTripCount(MachineLoop *L,
                 const MachineBlockFrequencyInfo *MBFI = nullptr) {
  if (MBFI) {
    MachineBasicBlock *Header = L->getHeader();
    MachineBasicBlock *Preheader = L->getLoopPreheader();

    if (Header && Preheader) {
      float HeaderFreq = getBlockFrequency(MBFI, Header);
      float PreheaderFreq = getBlockFrequency(MBFI, Preheader);

      if (PreheaderFreq > 0.0f) {
        unsigned DerivedTC =
            static_cast<unsigned>(HeaderFreq / PreheaderFreq + 0.5f);
        if (DerivedTC >= 1) {
          if (VerboseSimulation) {
            dbgs() << "  Trip count from MBFI: " << DerivedTC
                   << " (header=" << format("%.1f", HeaderFreq)
                   << " / preheader=" << format("%.1f", PreheaderFreq)
                   << ")\n";
          }
          return DerivedTC;
        }
      }
    }
  }
  return DefaultTripCount;
}

static BlockMetrics analyzeLoop(MachineLoop *L, MachineLoopInfo &MLI,
                                Simulator &Sim, MachineInstrInfo &MII,
                                const SIInstrInfo &TII,
                                DenseSet<MachineBasicBlock *> &Visited,
                                KernelPerfReport &Report,
                                const MachineBlockFrequencyInfo *MBFI) {
  unsigned TripCount = TripCountOverride.getNumOccurrences()
                           ? TripCountOverride.getValue()
                           : getLoopTripCount(L, MBFI);
  unsigned LoopDepth = L->getLoopDepth();

  Report.NumLoops++;
  Report.MaxLoopDepth = std::max(Report.MaxLoopDepth, LoopDepth);
  Report.MaxTripCount = std::max(Report.MaxTripCount, TripCount);

  MachineBasicBlock *Header = L->getHeader();
  float HeaderFreq = getBlockFrequency(MBFI, Header);

  if (VerboseSimulation) {
    dbgs() << "\n=== Analyzing Loop (depth " << LoopDepth << ", trip count "
           << TripCount << ") ===\n";
    dbgs() << "  Header: " << Header->getName()
           << " (freq=" << format("%.3f", HeaderFreq) << ")\n";
  }

  DenseMap<MachineBasicBlock *, BlockMetrics> ColdPerBlock;
  DenseMap<MachineBasicBlock *, BlockMetrics> WarmPerBlock;
  DenseMap<MachineLoop *, BlockMetrics> InnerLoopMetrics;
  BlockMetrics DirectBlocksRaw;

  auto simulateIteration =
      [&](const char *Label,
          DenseMap<MachineBasicBlock *, BlockMetrics> &PerBlockOut,
          bool isCold) -> BlockMetrics {
    BlockMetrics IterMetrics;

    if (VerboseSimulation)
      dbgs() << "\n--- " << Label << " iteration ---\n";

    for (MachineBasicBlock *MBB : L->blocks()) {
      MachineLoop *InnerLoop = MLI.getLoopFor(MBB);

      if (InnerLoop != L && InnerLoop && InnerLoop->getHeader() == MBB &&
          InnerLoop->getParentLoop() == L) {
        BlockMetrics InnerMetrics;
        if (isCold) {
          InnerMetrics =
              analyzeLoop(InnerLoop, MLI, Sim, MII, TII, Visited,
                          Report, MBFI);
          InnerLoopMetrics[InnerLoop] = InnerMetrics;
        } else {
          InnerMetrics = InnerLoopMetrics.lookup(InnerLoop);
        }

        float InnerEntryFreq;
        if (MachineBasicBlock *InnerPreheader =
                InnerLoop->getLoopPreheader()) {
          InnerEntryFreq = getBlockFrequency(MBFI, InnerPreheader);
        } else {
          float InnerHeaderFreq = getBlockFrequency(MBFI, MBB);
          unsigned InnerTripCount = getLoopTripCount(InnerLoop, MBFI);
          InnerEntryFreq = (InnerTripCount > 0)
                               ? InnerHeaderFreq / InnerTripCount
                               : InnerHeaderFreq;
        }
        float RelativeFreq =
            (HeaderFreq > 0) ? InnerEntryFreq / HeaderFreq : 1.0f;

        if (VerboseSimulation) {
          dbgs() << "  Inner loop " << MBB->getName()
                 << " entry freq: " << format("%.3f", InnerEntryFreq)
                 << " relative: " << format("%.3f", RelativeFreq)
                 << (isCold ? "" : " (cached)") << "\n";
        }

        IterMetrics = IterMetrics + InnerMetrics * RelativeFreq;
      } else if (MLI.getLoopFor(MBB) == L) {
        BlockMetrics BM =
            analyzeBlock(*MBB, Sim, MII, TII, &Report);
        if (isCold)
          DirectBlocksRaw = DirectBlocksRaw + BM;

        float BlockFreq = getBlockFrequency(MBFI, MBB);
        float RelativeFreq =
            (HeaderFreq > 0) ? BlockFreq / HeaderFreq : 1.0f;
        IterMetrics = IterMetrics + BM * RelativeFreq;
        PerBlockOut[MBB] = BM;
      }
    }
    return IterMetrics;
  };

  BlockMetrics ColdMetrics = simulateIteration("Cold", ColdPerBlock, true);

  if (VerboseSimulation)
    dbgs() << "  Cold iteration: " << ColdMetrics.TotalCycles << " cycles, "
           << ColdMetrics.StallCycles() << " stall\n";

  BlockMetrics WarmMetrics = simulateIteration("Warm", WarmPerBlock, false);

  if (VerboseSimulation)
    dbgs() << "  Warm iteration: " << WarmMetrics.TotalCycles << " cycles, "
           << WarmMetrics.StallCycles() << " stall\n";

  for (MachineBasicBlock *MBB : L->blocks())
    Visited.insert(MBB);

  Report.ColdTotal = Report.ColdTotal + ColdMetrics;
  Report.WarmTotal = Report.WarmTotal + WarmMetrics;
  Report.Raw = Report.Raw + DirectBlocksRaw;

  for (MachineBasicBlock *MBB : L->blocks()) {
    if (MLI.getLoopFor(MBB) == L) {
      PerBlockInfo &Info = Report.PerBlock[MBB];
      Info.Cold = ColdPerBlock.lookup(MBB);
      Info.Warm = WarmPerBlock.lookup(MBB);
      Info.TripCount = TripCount;
      Info.IsLoopHeader = (MBB == L->getHeader());
      Info.InLoop = true;
    }
  }

  if (TripCount <= 1)
    return ColdMetrics;

  BlockMetrics ScaledMetrics = ColdMetrics + WarmMetrics * (TripCount - 1);

  if (VerboseSimulation)
    dbgs() << "  Scaled total: " << ScaledMetrics.TotalCycles << " cycles "
           << "(Cold + Warm * " << (TripCount - 1) << ")\n";

  if (EnableISCacheModel && TripCount > 1 && WarmMetrics.TotalCycles > 0) {
    unsigned LoopBodyBytes = WarmMetrics.ISBytesConsumed;
    unsigned LoopBodyCycles = WarmMetrics.TotalCycles;
    unsigned FetchLatency = SQCToISLatency;

    const auto &Model = Sim.getModel();
    unsigned LineSizeBytes = Model.ISCacheLineSize;
    unsigned NumLines = Model.ISCacheNumLines;

    unsigned SteadyStateStall = computeSteadyStateISStall(
        LoopBodyBytes, LoopBodyCycles, FetchLatency, LineSizeBytes);
    unsigned IterationsUntilBackup = computeIterationsUntilBackup(
        LoopBodyBytes, LoopBodyCycles, FetchLatency, NumLines, LineSizeBytes);

    if (VerboseSimulation) {
      dbgs() << "\n  IS Cache Analysis:\n";
      dbgs() << "    Loop body: " << LoopBodyBytes << " bytes / "
             << LoopBodyCycles << " cycles\n";
      dbgs() << "    Fetch rate: "
             << format("%.2f", static_cast<double>(LineSizeBytes) / FetchLatency)
             << " bytes/cycle\n";
      if (IterationsUntilBackup < UINT_MAX) {
        dbgs() << "    *** IS cache backs up after ~"
               << IterationsUntilBackup << " iterations ***\n";
        dbgs() << "    Steady-state stall: " << SteadyStateStall
               << " cycles/iter\n";
      } else {
        dbgs() << "    IS cache does NOT back up\n";
      }
    }

    if (IterationsUntilBackup < TripCount && SteadyStateStall > 0) {
      unsigned StallIterations = TripCount - IterationsUntilBackup;
      unsigned AdditionalISStall = StallIterations * SteadyStateStall;

      if (VerboseSimulation) {
        dbgs() << "    Adding " << AdditionalISStall
               << " estimated IS stall cycles\n";
      }

      ScaledMetrics.StallISFetch += AdditionalISStall;
      ScaledMetrics.TotalCycles += AdditionalISStall;
    }
  }

  return ScaledMetrics;
}

//===----------------------------------------------------------------------===//
// Function Analysis
//===----------------------------------------------------------------------===//

static KernelPerfReport
analyzeFunction(MachineFunction &MF, const SIInstrInfo &TII,
                MachineLoopInfo *MLI,
                const MachineBlockFrequencyInfo *MBFI) {
  KernelPerfReport Report;

  // Create MIR adapter and Simulator
  const SIRegisterInfo &TRI = TII.getRegisterInfo();
  MachineInstrInfo MII(TII, TRI);
  AMDGPUSim::HWModel Model = AMDGPUSim::createHWModel(AMDGPUSim::GPUTarget::GFX1250);
  Model.VaVdstMultiplier = VaVdstMultiplierOpt;

  SimulatorConfig Cfg;
  Cfg.Log = VerboseSimulation ? &dbgs() : nullptr;
  Cfg.EnableScoreboard = EnableScoreboardOpt;
  Cfg.EnableISCache = EnableISCacheModel;

  Model.SQCToISLatency = SQCToISLatency;

  Simulator Sim(MII, Model, Cfg);

  DenseSet<MachineBasicBlock *> Visited;
  printBlockFrequencies(MF, MBFI);

  ReversePostOrderTraversal<MachineFunction *> RPOT(&MF);

  for (MachineBasicBlock *MBB : RPOT) {
    if (Visited.contains(MBB))
      continue;

    MachineLoop *L = MLI ? MLI->getLoopFor(MBB) : nullptr;

    if (L && L->getHeader() == MBB) {
      BlockMetrics LoopMetrics =
          analyzeLoop(L, *MLI, Sim, MII, TII, Visited, Report, MBFI);

      float LoopEntryFreq = 1.0f;
      if (MachineBasicBlock *Preheader = L->getLoopPreheader()) {
        LoopEntryFreq = getBlockFrequency(MBFI, Preheader);
      } else {
        float HeaderFreq = getBlockFrequency(MBFI, MBB);
        unsigned TripCount = getLoopTripCount(L, MBFI);
        LoopEntryFreq = (TripCount > 0) ? HeaderFreq / TripCount : 1.0f;
      }

      if (VerboseSimulation)
        dbgs() << "  Loop entry frequency: " << format("%.3f", LoopEntryFreq)
               << "\n";

      Report.Scaled = Report.Scaled + LoopMetrics * LoopEntryFreq;
    } else {
      BlockMetrics BM =
          analyzeBlock(*MBB, Sim, MII, TII, &Report);
      float Freq = getBlockFrequency(MBFI, MBB);

      Report.Raw = Report.Raw + BM;
      Report.Scaled = Report.Scaled + BM * Freq;
      Visited.insert(MBB);

      PerBlockInfo &Info = Report.PerBlock[MBB];
      Info.Cold = BM;
      Info.Warm = BM;
      Info.TripCount = 1;
      Info.Frequency = Freq;
      Info.IsLoopHeader = false;
      Info.InLoop = false;
    }
  }

  for (auto &[MBB, Info] : Report.PerBlock) {
    if (Info.Frequency == 0.0f)
      Info.Frequency = getBlockFrequency(MBFI, MBB);
  }

  for (const MachineBasicBlock &MBB : MF) {
    if (MBB.succ_size() > 1)
      Report.NumBranches++;
  }

  Report.finalize();
  return Report;
}

//===----------------------------------------------------------------------===//
// Main Entry Point
//===----------------------------------------------------------------------===//

static bool runStaticSimulator(MachineFunction &MF, MachineLoopInfo *MLI,
                               const MachineBlockFrequencyInfo *MBFI) {
  if (!isStaticSimulatorEnabled())
    return false;

  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  if (!ST.hasGFX1250Insts())
    return false;

  const SIInstrInfo *TII = ST.getInstrInfo();
  if (!TII)
    return false;

  LLVM_DEBUG(dbgs() << "Running Static Simulator on: " << MF.getName()
                    << "\n");

  if (VerboseSimulation) {
    dbgs() << "\n=== Function: " << MF.getName() << " ===\n";
    if (MLI) {
      unsigned NumLoops = 0;
      for (MachineLoop *TopLoop : *MLI) {
        (void)TopLoop;
        NumLoops++;
      }
      dbgs() << "  MachineLoopInfo: " << NumLoops << " top-level loops\n";
    }
  }

  KernelPerfReport Report = analyzeFunction(MF, *TII, MLI, MBFI);
  LLVM_DEBUG(Report.print(dbgs(), MF.getName()));

  SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  MFI->setStaticSimReport(std::make_shared<KernelPerfReport>(std::move(Report)));

  return true;
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// BlockMetrics Implementation
//===----------------------------------------------------------------------===//

BlockMetrics BlockMetrics::operator*(float Factor) const {
  auto scale = [Factor](unsigned V) -> unsigned {
    return static_cast<unsigned>(V * Factor + 0.5f);
  };

  BlockMetrics R;
  R.NumInstructions = scale(NumInstructions);
  R.NumVALU = scale(NumVALU);
  R.NumSALU = scale(NumSALU);
  R.NumTRANS = scale(NumTRANS);
  R.NumWMMA = scale(NumWMMA);
  R.NumVOPD = scale(NumVOPD);
  R.NumPacked = scale(NumPacked);
  R.NumDSRead = scale(NumDSRead);
  R.NumDSWrite = scale(NumDSWrite);
  R.NumVMEM = scale(NumVMEM);
  R.NumSMEM = scale(NumSMEM);
  R.NumTDM = scale(NumTDM);
  R.NumBranch = scale(NumBranch);
  R.NumBarrier = scale(NumBarrier);
  R.NumNop = scale(NumNop);
  R.NumDelayAlu = scale(NumDelayAlu);
  R.NumMSBSet = scale(NumMSBSet);
  R.NumMSBSetMasked = scale(NumMSBSetMasked);
  R.NumSpill = scale(NumSpill);
  R.NumReload = scale(NumReload);
  R.NumSGPRToVGPR = scale(NumSGPRToVGPR);
  R.NumVGPRToSGPR = scale(NumVGPRToSGPR);
  R.NumWaitcnt = scale(NumWaitcnt);
  R.WaitLGKM = scale(WaitLGKM);
  R.WaitVMEM = scale(WaitVMEM);
  R.WaitEXP = scale(WaitEXP);
  R.NumFalseWaits = scale(NumFalseWaits);
  R.TotalCycles = scale(TotalCycles);
  R.StallFunctionalUnit = scale(StallFunctionalUnit);
  R.StallCoExec = scale(StallCoExec);
  R.StallDelayAlu = scale(StallDelayAlu);
  R.StallMemFIFO = scale(StallMemFIFO);
  R.StallWaitCnt = scale(StallWaitCnt);
  R.StallFalseWait = scale(StallFalseWait);
  R.StallXDL = scale(StallXDL);
  R.StallVALU = scale(StallVALU);
  R.StallSALU = scale(StallSALU);
  R.StallTRANSUnit = scale(StallTRANSUnit);
  R.StallLDS = scale(StallLDS);
  R.StallVMEMUnit = scale(StallVMEMUnit);
  R.StallRegBankConflict = scale(StallRegBankConflict);
  R.RegBankConflictsInWMMAWindow = scale(RegBankConflictsInWMMAWindow);
  R.StallLongLatVALU = scale(StallLongLatVALU);
  R.StallLOLVALUTRANS = scale(StallLOLVALUTRANS);
  R.StallVaSSRC = scale(StallVaSSRC);
  R.StallVaVdst = scale(StallVaVdst);
  R.StallRAW = scale(StallRAW);
  R.StallISFetch = scale(StallISFetch);
  R.ISFetchesTriggered = scale(ISFetchesTriggered);
  R.VGPRCacheHits = scale(VGPRCacheHits);
  R.VGPRCacheMisses = scale(VGPRCacheMisses);
  R.VGPRCacheEvictions = scale(VGPRCacheEvictions);
  R.WMMAWindowCycles = scale(WMMAWindowCycles);
  R.WMMACoExecUsed = scale(WMMACoExecUsed);
  R.WMMACoExecBlocked = scale(WMMACoExecBlocked);
  R.WMMAStarved = scale(WMMAStarved);
  R.CoExecMissVALU = scale(CoExecMissVALU);
  R.CoExecMissTRANS = scale(CoExecMissTRANS);
  R.CoExecMissMemory = scale(CoExecMissMemory);
  R.CoExecMissOther = scale(CoExecMissOther);
  R.ISlotTotal = scale(ISlotTotal);
  R.ISlotUsedByVALU = scale(ISlotUsedByVALU);
  R.ISlotWastedOnNonVALU = scale(ISlotWastedOnNonVALU);
  return R;
}

BlockMetrics BlockMetrics::operator+(const BlockMetrics &O) const {
  BlockMetrics R;
#define ADD(field) R.field = field + O.field
  ADD(NumInstructions); ADD(NumVALU); ADD(NumSALU); ADD(NumTRANS);
  ADD(NumWMMA); ADD(NumVOPD); ADD(NumPacked);
  ADD(NumDSRead); ADD(NumDSWrite); ADD(NumVMEM); ADD(NumSMEM);
  ADD(NumTDM); ADD(NumBranch); ADD(NumBarrier); ADD(NumNop);
  ADD(NumDelayAlu); ADD(NumMSBSet); ADD(NumMSBSetExposed);
  ADD(NumMSBSetMasked); ADD(NumSpill); ADD(NumReload);
  ADD(NumSGPRToVGPR); ADD(NumVGPRToSGPR);
  ADD(NumWaitcnt); ADD(WaitLGKM); ADD(WaitVMEM); ADD(WaitEXP);
  ADD(NumFalseWaits);
  ADD(TotalCycles);
  ADD(StallFunctionalUnit); ADD(StallCoExec); ADD(StallDelayAlu);
  ADD(StallMemFIFO); ADD(StallWaitCnt); ADD(StallFalseWait);
  ADD(StallXDL); ADD(StallVALU); ADD(StallSALU);
  ADD(StallTRANSUnit); ADD(StallLDS); ADD(StallVMEMUnit);
  ADD(StallRegBankConflict); ADD(RegBankConflictsInWMMAWindow);
  ADD(StallLongLatVALU); ADD(StallLOLVALUTRANS);
  ADD(StallVaSSRC); ADD(StallVaVdst); ADD(StallRAW);
  ADD(StallISFetch); ADD(ISFetchesTriggered);
  ADD(VGPRCacheHits); ADD(VGPRCacheMisses); ADD(VGPRCacheEvictions);
  ADD(WMMAWindowCycles); ADD(WMMACoExecUsed); ADD(WMMACoExecBlocked);
  ADD(WMMAStarved);
  ADD(CoExecMissVALU); ADD(CoExecMissTRANS); ADD(CoExecMissMemory);
  ADD(CoExecMissOther);
  ADD(ISlotTotal); ADD(ISlotUsedByVALU); ADD(ISlotWastedOnNonVALU);
#undef ADD
  return R;
}

void BlockMetrics::printInstBreakdown(raw_ostream &OS) const {
  bool First = true;
  auto Emit = [&](const char *Name, unsigned Val) {
    if (Val) {
      if (!First) OS << " ";
      OS << Name << ":" << Val;
      First = false;
    }
  };
  if (NumVALU) {
    if (!First) OS << " ";
    unsigned NumVALUInst = NumVALU - NumVOPD - NumPacked;
    OS << "VALU:" << NumVALUInst;
    if (NumVOPD || NumPacked) {
      OS << "(";
      bool DualFirst = true;
      if (NumVOPD) { OS << "VOPD:" << NumVOPD; DualFirst = false; }
      if (NumPacked) { if (!DualFirst) OS << "+"; OS << "PK:" << NumPacked; }
      OS << ")";
    }
    First = false;
  }
  Emit("SALU", NumSALU);
  Emit("TRANS", NumTRANS);
  Emit("WMMA", NumWMMA);
  Emit("DS", NumDSRead + NumDSWrite);
  Emit("VMEM", NumVMEM);
  Emit("SMEM", NumSMEM);
  Emit("TDM", NumTDM);
  unsigned NumCtrl = NumWaitcnt + NumBarrier + NumDelayAlu + NumMSBSet +
                     NumNop + NumBranch;
  Emit("Ctrl", NumCtrl);
  Emit("Spill", NumSpill);
  Emit("Reload", NumReload);
}

void BlockMetrics::printStallBreakdown(raw_ostream &OS) const {
  bool First = true;
  auto Emit = [&](const char *Name, unsigned Val) {
    if (Val) {
      if (!First) OS << " | ";
      OS << Name << ":" << Val;
      First = false;
    }
  };
  Emit("FU", StallFunctionalUnit);
  if (StallCoExec) {
    if (!First) OS << " | ";
    OS << "WMMACoExec:" << StallCoExec;
    if (CoExecMissVALU || CoExecMissTRANS || CoExecMissMemory ||
        CoExecMissOther) {
      OS << "(";
      bool SubFirst = true;
      auto EmitSub = [&](const char *Name, unsigned Val) {
        if (Val) {
          if (!SubFirst) OS << "+";
          OS << Name << ":" << Val;
          SubFirst = false;
        }
      };
      EmitSub("VALU", CoExecMissVALU);
      EmitSub("TRANS", CoExecMissTRANS);
      EmitSub("MEM", CoExecMissMemory);
      EmitSub("Other", CoExecMissOther);
      OS << ")";
    }
    First = false;
  }
  Emit("DelayAlu", StallDelayAlu);
  Emit("MemFIFO", StallMemFIFO);
  Emit("Wait", StallWaitCnt);
  Emit("RegBank", StallRegBankConflict);
  Emit("LongLatVALU", StallLongLatVALU);
  Emit("LOLVALUxTRANS", StallLOLVALUTRANS);
  Emit("VaSSRC", StallVaSSRC);
  Emit("VaVdst", StallVaVdst);
  Emit("RAW", StallRAW);
  if (StallISFetch) {
    if (!First) OS << " | ";
    OS << "ISFetch:" << StallISFetch;
    if (ISFetchesTriggered)
      OS << " (" << ISFetchesTriggered << " fetches)";
    First = false;
  }
  if (RegBankConflictsInWMMAWindow) {
    if (!First) OS << " | ";
    OS << "RegBankInWMMA:" << RegBankConflictsInWMMAWindow << " (not counted)";
    First = false;
  }
  if (NumMSBSetExposed || NumMSBSetMasked) {
    if (!First) OS << " | ";
    OS << "MSBExposed:" << NumMSBSetExposed;
    if (NumMSBSetMasked)
      OS << " (+" << NumMSBSetMasked << " masked)";
    First = false;
  }
  if (ISlotTotal) {
    if (!First) OS << " | ";
    OS << "ISlot:" << ISlotUsedByVALU << "/" << ISlotTotal;
    if (ISlotWastedOnNonVALU)
      OS << " (wasted:" << ISlotWastedOnNonVALU << ")";
    First = false;
  }
}

void BlockMetrics::printFUBreakdown(raw_ostream &OS) const {
  bool First = true;
  auto Emit = [&](const char *Name, unsigned Val) {
    if (Val) {
      if (!First) OS << " ";
      OS << Name << ":" << Val;
      First = false;
    }
  };
  Emit("XDL", StallXDL);
  Emit("VALU", StallVALU);
  Emit("TRANS", StallTRANSUnit);
  Emit("SALU", StallSALU);
  Emit("LDS", StallLDS);
  Emit("VMEM", StallVMEMUnit);
}

//===----------------------------------------------------------------------===//
// KernelPerfReport Printing
//===----------------------------------------------------------------------===//

static void printStallBreakdownReport(raw_ostream &OS, const BlockMetrics &M,
                                      const char *Indent = ";   ") {
  float StallPct =
      M.TotalCycles > 0 ? 100.0f * M.StallCycles() / M.TotalCycles : 0.0f;
  OS << formatv("{0}Stall: {1} cycles ({2:F1}%)\n", Indent, M.StallCycles(),
                StallPct);
  OS << Indent << "  ";
  M.printStallBreakdown(OS);
  OS << "\n";
  if (M.StallFunctionalUnit > 0) {
    OS << Indent << "    FU: ";
    M.printFUBreakdown(OS);
    OS << "\n";
  }
}

void KernelPerfReport::print(raw_ostream &OS, StringRef FuncName) const {
  OS << "; ============================================================\n";
  if (!FuncName.empty())
    OS << "; " << FuncName << " - STATIC PERFORMANCE ESTIMATE (gfx1250)\n";
  else
    OS << "; STATIC PERFORMANCE ESTIMATE (gfx1250)\n";
  OS << "; ============================================================\n";
  OS << ";\n";

  OS << "; === Raw Metrics (each block executed once) ===\n";
  OS << formatv(";   Instructions: {0}\n", Raw.NumInstructions);
  OS << formatv(";   Cycles:       {0}\n", Raw.TotalCycles);
  printStallBreakdownReport(OS, Raw);
  OS << formatv(";   Waitcnts: {0} | False waits: {1}\n", Raw.NumWaitcnt,
                Raw.NumFalseWaits);
  OS << formatv(";   WMMA windows: {0} | Co-executed: {1}\n",
                Raw.WMMAWindowCycles, Raw.WMMACoExecUsed);
  if (Raw.ISlotTotal > 0) {
    OS << formatv(
        ";   I-slots: {0} used | {1} wasted on non-VALU ({2:F0}% VALU)\n",
        Raw.ISlotTotal, Raw.ISlotWastedOnNonVALU,
        Raw.ISlotTotal > 0 ? 100.0f * Raw.ISlotUsedByVALU / Raw.ISlotTotal
                           : 0.0f);
  }
  OS << ";\n";

  OS << "; === Scaled Metrics (loops x trip count) ===\n";
  OS << formatv(";   Instructions: {0}\n", Scaled.NumInstructions);
  OS << formatv(";   Cycles:       {0}\n", Scaled.TotalCycles);
  printStallBreakdownReport(OS, Scaled);
  OS << formatv(";   Waitcnts: {0} | False waits: {1}\n", Scaled.NumWaitcnt,
                Scaled.NumFalseWaits);
  OS << formatv(";   WMMA windows: {0} | Co-executed: {1} ({2:F0}%)\n",
                Scaled.WMMAWindowCycles, Scaled.WMMACoExecUsed,
                CoExecEfficiency * 100.0f);
  if (Scaled.ISlotTotal > 0) {
    OS << formatv(
        ";   I-slots: {0} used | {1} wasted on non-VALU ({2:F0}% VALU)\n",
        Scaled.ISlotTotal, Scaled.ISlotWastedOnNonVALU,
        Scaled.ISlotTotal > 0
            ? 100.0f * Scaled.ISlotUsedByVALU / Scaled.ISlotTotal
            : 0.0f);
  }
  OS << ";\n";

  OS << "; === Instruction Breakdown (Raw / Scaled) ===\n";
  unsigned RawVALUInst = Raw.NumVALU - Raw.NumVOPD - Raw.NumPacked;
  unsigned ScaledVALUInst = Scaled.NumVALU - Scaled.NumVOPD - Scaled.NumPacked;
  OS << formatv(";   VALU: {0}/{1}", RawVALUInst, ScaledVALUInst);
  if (Raw.NumVOPD || Scaled.NumVOPD || Raw.NumPacked || Scaled.NumPacked) {
    OS << " (";
    bool First = true;
    if (Raw.NumVOPD || Scaled.NumVOPD) {
      OS << formatv("VOPD:{0}/{1}", Raw.NumVOPD, Scaled.NumVOPD);
      First = false;
    }
    if (Raw.NumPacked || Scaled.NumPacked) {
      if (!First)
        OS << "+";
      OS << formatv("PK:{0}/{1}", Raw.NumPacked, Scaled.NumPacked);
    }
    OS << ")";
  }
  OS << formatv(" | SALU: {0}/{1} | TRANS: {2}/{3} | WMMA: {4}/{5}\n",
                Raw.NumSALU, Scaled.NumSALU, Raw.NumTRANS, Scaled.NumTRANS,
                Raw.NumWMMA, Scaled.NumWMMA);
  OS << formatv(
      ";   DS_RD: {0}/{1} | DS_WR: {2}/{3} | VMEM: {4}/{5} | TDM: {6}/{7}\n",
      Raw.NumDSRead, Scaled.NumDSRead, Raw.NumDSWrite, Scaled.NumDSWrite,
      Raw.NumVMEM, Scaled.NumVMEM, Raw.NumTDM, Scaled.NumTDM);
  if (Raw.NumSpill || Raw.NumReload || Scaled.NumSpill || Scaled.NumReload) {
    OS << formatv(";   Spill: {0}/{1} | Reload: {2}/{3}\n", Raw.NumSpill,
                  Scaled.NumSpill, Raw.NumReload, Scaled.NumReload);
  }
  if (Raw.NumSGPRToVGPR || Raw.NumVGPRToSGPR) {
    OS << formatv(";   SGPR->Lane: {0}/{1} | Lane->SGPR: {2}/{3}\n",
                  Raw.NumSGPRToVGPR, Scaled.NumSGPRToVGPR, Raw.NumVGPRToSGPR,
                  Scaled.NumVGPRToSGPR);
  }
  if (Raw.NumDelayAlu || Scaled.NumDelayAlu) {
    OS << formatv(
        ";   delay_alu: {0}/{1} | MSB_set: {2}/{3} (exposed: {4}/{5})\n",
        Raw.NumDelayAlu, Scaled.NumDelayAlu, Raw.NumMSBSet, Scaled.NumMSBSet,
        Raw.NumMSBSetExposed, Scaled.NumMSBSetExposed);
  }
  OS << ";\n";

  unsigned RawTotal = Raw.VGPRCacheHits + Raw.VGPRCacheMisses;
  unsigned ScaledTotal = Scaled.VGPRCacheHits + Scaled.VGPRCacheMisses;
  if (RawTotal > 0 || ScaledTotal > 0) {
    OS << "; === VGPR Operand Cache ===\n";
    OS << formatv(";   VGPR reads: {0}/{1} | From cache: {2}/{3}", RawTotal,
                  ScaledTotal, Raw.VGPRCacheHits, Scaled.VGPRCacheHits);
    if (ScaledTotal > 0)
      OS << formatv(" ({0:F0}%)", Scaled.VGPRCacheHitRate() * 100.0f);
    OS << "\n";
    if (Raw.VGPRCacheEvictions > 0 || Scaled.VGPRCacheEvictions > 0) {
      OS << formatv(";   Evictions: {0}/{1}\n", Raw.VGPRCacheEvictions,
                    Scaled.VGPRCacheEvictions);
    }
    OS << ";\n";
  }

  if (NumLoops > 0 || NumBranches > 0) {
    OS << "; === CFG Analysis ===\n";
    if (NumLoops > 0) {
      OS << formatv(";   Loops: {0} | Max depth: {1} | Trip count: {2}\n",
                    NumLoops, MaxLoopDepth, MaxTripCount);
      OS << formatv(";   Cold: {0} cycles | Warm: {1} cycles",
                    ColdTotal.TotalCycles, WarmTotal.TotalCycles);
      if (ColdTotal.TotalCycles > 0 && WarmTotal.TotalCycles > 0) {
        float Speedup = static_cast<float>(ColdTotal.TotalCycles) /
                        WarmTotal.TotalCycles;
        OS << formatv(" | Speedup: {0:F2}x", Speedup);
      }
      OS << "\n";
    }
    if (NumBranches > 0) {
      OS << formatv(
          ";   Branches: {0} (scaled metrics use uniform probability)\n",
          NumBranches);
    }
    OS << ";\n";
  }

  OS << "; === Derived Metrics ===\n";
  OS << formatv(";   IPC: {0:F2} | Stall ratio: {1:F1}%\n", IPC,
                StallRatio * 100.0f);
  if (Scaled.NumWaitcnt > 0) {
    float AvgFalsePerWait =
        static_cast<float>(Scaled.NumFalseWaits) / Scaled.NumWaitcnt;
    OS << formatv(";   False wait ratio: {0:F2} per waitcnt\n",
                  AvgFalsePerWait);
  }
  OS << ";\n";

  OS << "; ============================================================\n";
}

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

PreservedAnalyses
AMDGPUStaticSimulatorPass::run(MachineFunction &MF,
                               MachineFunctionAnalysisManager &MFAM) {
  MachineLoopInfo &MLI = MFAM.getResult<MachineLoopAnalysis>(MF);
  auto &MBFI = MFAM.getResult<MachineBlockFrequencyAnalysis>(MF);
  runStaticSimulator(MF, &MLI, &MBFI);
  return PreservedAnalyses::all();
}

namespace {

class AMDGPUStaticSimulatorLegacy : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUStaticSimulatorLegacy() : MachineFunctionPass(ID) {
    initializeAMDGPUStaticSimulatorLegacyPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    MachineLoopInfo &MLI =
        getAnalysis<MachineLoopInfoWrapperPass>().getLI();
    MachineBlockFrequencyInfo &MBFI =
        getAnalysis<MachineBlockFrequencyInfoWrapperPass>().getMBFI();
    runStaticSimulator(MF, &MLI, &MBFI);
    return false;
  }

  StringRef getPassName() const override {
    return "AMDGPU Static Performance Simulator";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineLoopInfoWrapperPass>();
    AU.addRequired<MachineBlockFrequencyInfoWrapperPass>();
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // anonymous namespace

char AMDGPUStaticSimulatorLegacy::ID = 0;
char &llvm::AMDGPUStaticSimulatorLegacyID = AMDGPUStaticSimulatorLegacy::ID;

INITIALIZE_PASS_BEGIN(AMDGPUStaticSimulatorLegacy, DEBUG_TYPE,
                      "AMDGPU Static Performance Simulator", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineBlockFrequencyInfoWrapperPass)
INITIALIZE_PASS_END(AMDGPUStaticSimulatorLegacy, DEBUG_TYPE,
                    "AMDGPU Static Performance Simulator", false, false)

FunctionPass *llvm::createAMDGPUStaticSimulatorPass() {
  return new AMDGPUStaticSimulatorLegacy();
}
