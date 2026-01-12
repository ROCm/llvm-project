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
/// the MachineFunction, simulating instruction execution to produce:
/// - Instruction counts by type (VALU, SALU, WMMA, DS_READ, etc.)
/// - Stall cycle estimates (RAW dependencies, memory waits)
/// - WMMA co-execution efficiency
/// - IPC and other derived metrics
///
/// Results are stored in SIMachineFunctionInfo and emitted as assembly comments.
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
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineBranchProbabilityInfo.h"
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

static cl::opt<bool> EnableScoreboard(
    "amdgpu-static-sim-scoreboard",
    cl::desc("Enable register scoreboard for RAW detection without s_delay_alu"),
    cl::init(false), cl::Hidden);

/// Check if enabled via cl::opt or AMDGPU_ENABLE_STATIC_SIM env var.
static bool isStaticSimulatorEnabled() {
  if (const char *EnvVal = std::getenv("AMDGPU_ENABLE_STATIC_SIM"))
    return StringRef(EnvVal) == "1";
  return EnableStaticSimulator;
}

void GPUSimState::retireCompletedMemOps() {
  auto RetireFrom = [this](std::deque<PendingMemOp> &Queue,
                           const char *Name) -> unsigned {
    unsigned Retired = 0;
    while (!Queue.empty() && Queue.front().CompletionCycle <= CurrentCycle) {
      if (VerboseSimulation && Retired == 0) {
        dbgs() << "  [Retire " << Name << " @ cycle " << CurrentCycle << "] ";
      }
      const auto &Op = Queue.front();
      if (VerboseSimulation) {
        if (Retired > 0) dbgs() << ", ";
        dbgs() << "v" << Op.DestVGPR;
        if (Op.NumRegs > 1)
          dbgs() << "-v" << (Op.DestVGPR + Op.NumRegs - 1);
        dbgs() << "@" << Op.CompletionCycle;
      }
      Queue.pop_front();
      Retired++;
    }
    if (VerboseSimulation && Retired > 0) {
      dbgs() << " (" << Retired << " ops)\n";
    }
    return Retired;
  };

  RetireFrom(PendingDS, "DS");
  RetireFrom(PendingVMEMLoad, "VMEM_LD");
  RetireFrom(PendingVMEMStore, "VMEM_ST");
  RetireFrom(PendingSMEM, "SMEM");
  RetireFrom(PendingTDM, "TDM");
}

namespace {

InstClass classifyInst(const MachineInstr &MI, const SIInstrInfo &TII) {
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

  if (TII.isWaitcnt(Opc) ||
      Opc == AMDGPU::S_WAIT_XCNT ||
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

#ifndef NDEBUG
static const char *getInstClassName(InstClass IC) {
  switch (IC) {
  case InstClass::VALU:       return "VALU";
  case InstClass::SALU:       return "SALU";
  case InstClass::TRANS:      return "TRANS";
  case InstClass::WMMA:       return "WMMA";
  case InstClass::DS_READ:    return "DS_READ";
  case InstClass::DS_WRITE:   return "DS_WRITE";
  case InstClass::VMEM_READ:  return "VMEM_READ";
  case InstClass::VMEM_WRITE: return "VMEM_WRITE";
  case InstClass::SMEM:       return "SMEM";
  case InstClass::TDM:        return "TDM";
  case InstClass::BARRIER:    return "BARRIER";
  case InstClass::WAITCNT:    return "WAITCNT";
  case InstClass::DELAY_ALU:  return "DELAY_ALU";
  case InstClass::MSB_SET:    return "MSB_SET";
  case InstClass::NOP:        return "NOP";
  case InstClass::BRANCH:     return "BRANCH";
  case InstClass::OTHER:      return "OTHER";
  }
  return "UNKNOWN";
}

static const char *getUnitName(FunctionalUnit Unit) {
  switch (Unit) {
  case FunctionalUnit::NONE:   return "NONE";
  case FunctionalUnit::XDL:    return "XDL";
  case FunctionalUnit::VALU:   return "VALU";
  case FunctionalUnit::SALU:   return "SALU";
  case FunctionalUnit::TRANS:  return "TRANS";
  case FunctionalUnit::LDS:    return "LDS";
  case FunctionalUnit::VMEM:   return "VMEM";
  case FunctionalUnit::SMEM:   return "SMEM";
  case FunctionalUnit::BRANCH: return "BRANCH";
  case FunctionalUnit::NUM_UNITS: return "NUM_UNITS";
  }
  return "UNKNOWN";
}
#endif // NDEBUG

//===----------------------------------------------------------------------===//
// s_delay_alu Parsing (gfx1250)
//===----------------------------------------------------------------------===//

// Decode a single delay dependency based on current state
unsigned decodeDelayDep(unsigned Dep, const GPUSimState &State) {
  if (Dep == 0)
    return 0;

  // VALU_DEP_1 to VALU_DEP_4 (values 1-4)
  if (Dep >= 1 && Dep <= 4) {
    unsigned Index = Dep - 1;
    if (Index < State.RecentVALU.size()) {
      auto &Recent = State.RecentVALU[State.RecentVALU.size() - 1 - Index];
      unsigned Elapsed = State.CurrentCycle - Recent.IssueCycle;
      if (Elapsed < Recent.Latency)
        return Recent.Latency - Elapsed;
    }
    return 0;
  }

  // TRANS32_DEP_1 to TRANS32_DEP_3 (values 5-7)
  if (Dep >= 5 && Dep <= 7) {
    unsigned Index = Dep - 5;
    if (Index < State.RecentTRANS.size()) {
      auto &Recent = State.RecentTRANS[State.RecentTRANS.size() - 1 - Index];
      unsigned Elapsed = State.CurrentCycle - Recent.IssueCycle;
      if (Elapsed < Recent.Latency)
        return Recent.Latency - Elapsed;
    }
    return 0;
  }

  // SALU_CYCLE_1 to SALU_CYCLE_4 (values 9-12)
  if (Dep >= 9 && Dep <= 12) {
    unsigned WaitCycles = Dep - 8;
    unsigned Elapsed = State.CurrentCycle - State.LastSALUCycle;
    if (Elapsed < WaitCycles)
      return WaitCycles - Elapsed;
    return 0;
  }

  return 0;
}

// Check pending instid1 from previous s_delay_alu
// SkipApply: decrement only, don't apply stall (for MSB_SET)
unsigned checkPendingDelayAlu(GPUSimState &State, bool SkipApply = false) {
  if (!State.PendingInstId1)
    return 0;

  auto &Pending = *State.PendingInstId1;
  if (Pending.InstructionsLeft > 0) {
    Pending.InstructionsLeft--;
    return 0;
  }

  if (SkipApply)
    return 0;

  unsigned Stall = decodeDelayDep(Pending.DepType, State);
  if (VerboseSimulation && Stall > 0) {
    dbgs() << "    PendingInstId1: Dep=" << Pending.DepType
           << " stall=" << Stall << "\n";
  }
  State.PendingInstId1.reset();
  return Stall;
}

unsigned parseDelayAlu(const MachineInstr &MI, GPUSimState &State) {
  if (MI.getOpcode() != AMDGPU::S_DELAY_ALU)
    return 0;

  unsigned Imm = MI.getOperand(0).getImm();
  unsigned Dep1 = Imm & 0xF;
  unsigned Skip = (Imm >> 4) & 0x7;
  unsigned Dep2 = (Imm >> 7) & 0xF;

  unsigned Stall1 = decodeDelayDep(Dep1, State);

  // If there's an instid1, set up pending delay for later
  // Skip = count of instructions between s_delay_alu and target
  // After Skip decrements to 0, instid1 applies
  if (Dep2 != 0) {
    State.PendingInstId1 = GPUSimState::PendingDelayAlu{
        Dep2, Skip, State.CurrentCycle};
    if (VerboseSimulation) {
      dbgs() << "    DelayALU: instid0=" << Dep1 << " (stall " << Stall1
             << "), skip=" << Skip << ", instid1=" << Dep2 << " (pending)\n";
    }
  } else if (VerboseSimulation) {
    dbgs() << "    DelayALU: instid0=" << Dep1 << " (stall " << Stall1 << ")\n";
  }

  return Stall1;
}

// Parse s_wait_alu depctr_va_vdst(N): wait until pending VALU writes <= N
unsigned parseWaitAluVaVdst(const MachineInstr &MI, GPUSimState &State) {
  if (MI.getOpcode() != AMDGPU::S_WAITCNT_DEPCTR)
    return 0;

  unsigned Imm = MI.getOperand(0).getImm();
  unsigned VaVdstTarget = AMDGPU::DepCtr::decodeFieldVaVdst(Imm);

  // VaVdst field is 4 bits (0-15). 15 means "don't wait"
  if (VaVdstTarget == 15)
    return 0;

  unsigned ReadyCycle = State.getVaVdstReadyCycle(VaVdstTarget);
  unsigned Stall = ReadyCycle > State.CurrentCycle ? ReadyCycle - State.CurrentCycle : 0;

  if (VerboseSimulation && Stall > 0) {
    dbgs() << "    s_wait_alu: va_vdst(" << VaVdstTarget << "), pending="
           << State.getVaVdst() << ", stall=" << Stall << "\n";
  }

  return Stall;
}

//===----------------------------------------------------------------------===//
// Latency and Throughput Queries
//===----------------------------------------------------------------------===//

unsigned getInstrLatency(const MachineInstr &MI, const SIInstrInfo &TII,
                         InstClass IC) {
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

unsigned getResourceCycles(const MachineInstr &MI, const SIInstrInfo &TII,
                           InstClass IC) {
  // Use getRepeatRate() for VALU/TRANS to get canonical long-lat VALU resource cycles
  // getRepeatRate returns 1 for regular VALU, >1 for long-lat VALU (PK8=4, PK16=8, F64=32, etc.)
  if (IC == InstClass::VALU || IC == InstClass::TRANS) {
    unsigned RepeatRate = TII.getRepeatRate(MI);
    if (RepeatRate > 1)
      return RepeatRate;
  }

  if (AMDGPU::isVOPD(MI.getOpcode()))
    return 1;

  if (IC == InstClass::DS_READ || IC == InstClass::DS_WRITE)
    return 1;

  const TargetSchedModel &SchedModel = TII.getSchedModel();
  if (SchedModel.hasInstrSchedModel()) {
    double RecipThroughput = SchedModel.computeReciprocalThroughput(&MI);
    if (RecipThroughput > 0.0) {
      unsigned Cycles = std::max(1u, static_cast<unsigned>(std::ceil(RecipThroughput)));
      if (IC == InstClass::TRANS && Cycles < 2)
        return 2;
      return Cycles;
    }
    LLVM_DEBUG(dbgs() << "StaticSim: No throughput for " << MI << "\n");
  }

  if (IC == InstClass::WMMA)
    return 8;
  if (IC == InstClass::TRANS)
    return 2;

  return 1;
}

//===----------------------------------------------------------------------===//

unsigned computeWaitStall(const MachineInstr &MI, GPUSimState &State) {
  unsigned Opc = MI.getOpcode();
  unsigned Stall = 0;
  unsigned WaitCount = 0;
  if (MI.getNumOperands() > 0 && MI.getOperand(0).isImm())
    WaitCount = MI.getOperand(0).getImm();

  const char *WaitName = "UNKNOWN";
  unsigned QueueSizeBefore = 0;
  unsigned QueueSizeAfter = 0;

  switch (Opc) {
  case AMDGPU::S_WAIT_DSCNT:
    WaitName = "DSCNT";
    QueueSizeBefore = State.PendingDS.size();
    Stall = State.waitDS(WaitCount);
    QueueSizeAfter = State.PendingDS.size();
    break;
  case AMDGPU::S_WAIT_LOADCNT:
    WaitName = "LOADCNT";
    QueueSizeBefore = State.PendingVMEMLoad.size();
    Stall = State.waitVMEMLoad(WaitCount);
    QueueSizeAfter = State.PendingVMEMLoad.size();
    break;
  case AMDGPU::S_WAIT_STORECNT:
    WaitName = "STORECNT";
    QueueSizeBefore = State.PendingVMEMStore.size();
    Stall = State.waitVMEMStore(WaitCount);
    QueueSizeAfter = State.PendingVMEMStore.size();
    break;
  case AMDGPU::S_WAIT_KMCNT:
    WaitName = "KMCNT";
    QueueSizeBefore = State.PendingSMEM.size();
    Stall = State.waitSMEM(WaitCount);
    QueueSizeAfter = State.PendingSMEM.size();
    break;
  case AMDGPU::S_WAIT_TENSORCNT:
    WaitName = "TENSORCNT";
    QueueSizeBefore = State.PendingTDM.size();
    Stall = State.waitTensor(WaitCount);
    QueueSizeAfter = State.PendingTDM.size();
    break;
  case AMDGPU::S_WAIT_XCNT:
    WaitName = "XCNT";
    Stall = 0;
    break;
  default:
    break;
  }

  if (VerboseSimulation) {
    unsigned Retired = QueueSizeBefore - QueueSizeAfter;
    dbgs() << "    Wait decode: " << WaitName << " " << WaitCount
           << " (queue " << QueueSizeBefore << " → " << QueueSizeAfter;
    if (Retired > 0)
      dbgs() << ", retired " << Retired;
    dbgs() << ") → stall " << Stall << "\n";
  }

  return Stall;
}

//===----------------------------------------------------------------------===//
// Register Info Helpers
//===----------------------------------------------------------------------===//

std::pair<unsigned, unsigned> getDestRegInfo(const MachineInstr &MI,
                                              const SIInstrInfo &TII,
                                              bool IsVGPR) {
  if (MI.getNumOperands() == 0 || !MI.getOperand(0).isReg())
    return {0, 0};

  Register Reg = MI.getOperand(0).getReg();
  if (!Reg.isPhysical())
    return {0, 0};

  const SIRegisterInfo &TRI = TII.getRegisterInfo();
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

static const MachineInstr *findNextConsumer(MachineBasicBlock::const_instr_iterator It,
                                             MachineBasicBlock::const_instr_iterator End,
                                             const SIInstrInfo &TII) {
  for (++It; It != End; ++It) {
    const MachineInstr &MI = *It;
    if (MI.isBundle() || MI.isMetaInstruction() || MI.isDebugInstr())
      continue;
    if (MI.isImplicitDef())
      continue;
    InstClass IC = classifyInst(MI, TII);
    if (IC == InstClass::WAITCNT || IC == InstClass::NOP ||
        IC == InstClass::DELAY_ALU || IC == InstClass::MSB_SET)
      continue;
    return &MI;
  }
  return nullptr;
}

struct FalseWaitResult {
  unsigned Count = 0;
  unsigned WastedCycles = 0;
};

static FalseWaitResult analyzeFalseWaitsInQueue(
    const MachineInstr &WaitMI,
    unsigned WaitCount,
    const std::deque<PendingMemOp> &Queue,
    const MachineInstr *Consumer,
    const SIInstrInfo &TII,
    unsigned CurrentCycle) {

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

    bool IsNeeded = Op.writesToAnyOf(ConsumerUses);
    if (IsNeeded) {
      MaxTrueWaitCompletion = std::max(MaxTrueWaitCompletion, Op.CompletionCycle);
    } else {
      Result.Count++;
      if (VerboseSimulation) {
        dbgs() << "    False wait: op writes v" << Op.DestVGPR;
        if (Op.NumRegs > 1)
          dbgs() << "-v" << (Op.DestVGPR + Op.NumRegs - 1);
        dbgs() << " (completes @ " << Op.CompletionCycle
               << ") not used by consumer\n";
      }
    }
  }

  if (MaxAllWaitCompletion > MaxTrueWaitCompletion) {
    unsigned ActualStall = (MaxAllWaitCompletion > CurrentCycle)
                         ? (MaxAllWaitCompletion - CurrentCycle) : 0;
    unsigned OptimalStall = (MaxTrueWaitCompletion > CurrentCycle)
                          ? (MaxTrueWaitCompletion - CurrentCycle) : 0;
    Result.WastedCycles = ActualStall - OptimalStall;

    if (VerboseSimulation && Result.WastedCycles > 0) {
      dbgs() << "    Wasted cycles: " << Result.WastedCycles
             << " (actual stall " << ActualStall
             << ", optimal " << OptimalStall << ")\n";
    }
  }

  return Result;
}
static FalseWaitResult analyzeFalseWaitsForWait(const MachineInstr &MI,
                                                 MachineBasicBlock::const_instr_iterator It,
                                                 MachineBasicBlock::const_instr_iterator End,
                                                 GPUSimState &State,
                                                 const SIInstrInfo &TII) {
  unsigned Opc = MI.getOpcode();
  if (Opc != AMDGPU::S_WAIT_DSCNT && Opc != AMDGPU::S_WAIT_LOADCNT)
    return {};

  unsigned WaitCount = 0;
  if (MI.getNumOperands() > 0 && MI.getOperand(0).isImm())
    WaitCount = MI.getOperand(0).getImm();

  const MachineInstr *Consumer = findNextConsumer(It, End, TII);
  if (VerboseSimulation && Consumer)
    dbgs() << "    Consumer: " << *Consumer;

  if (Opc == AMDGPU::S_WAIT_DSCNT) {
    return analyzeFalseWaitsInQueue(MI, WaitCount, State.PendingDS, Consumer,
                                     TII, State.CurrentCycle);
  }
  return analyzeFalseWaitsInQueue(MI, WaitCount, State.PendingVMEMLoad, Consumer,
                                   TII, State.CurrentCycle);
}

//===----------------------------------------------------------------------===//
// Core Simulation Helpers
//===----------------------------------------------------------------------===//

struct InstTiming {
  InstClass IC;
  unsigned Latency;
  FunctionalUnit Unit;
  unsigned ResourceCycles;
};

InstTiming getInstTiming(const MachineInstr &MI, const SIInstrInfo &TII) {
  InstClass IC = classifyInst(MI, TII);
  return {IC, getInstrLatency(MI, TII, IC), getUnitForClass(IC),
          getResourceCycles(MI, TII, IC)};
}

/// Check if instruction implicitly waits for all VALU to complete (VA_VDST==0)
/// Same logic as AMDGPUInsertDelayAlu::instructionWaitsForVALU
static bool instructionWaitsForVALU(const MachineInstr &MI) {
  const uint64_t VA_VDST_0 = SIInstrFlags::DS | SIInstrFlags::EXP |
                             SIInstrFlags::FLAT | SIInstrFlags::MIMG |
                             SIInstrFlags::MTBUF | SIInstrFlags::MUBUF;
  if (MI.getDesc().TSFlags & VA_VDST_0)
    return true;
  if (MI.getOpcode() == AMDGPU::S_SENDMSG_RTN_B32 ||
      MI.getOpcode() == AMDGPU::S_SENDMSG_RTN_B64)
    return true;
  if (MI.getOpcode() == AMDGPU::S_WAITCNT_DEPCTR &&
      AMDGPU::DepCtr::decodeFieldVaVdst(MI.getOperand(0).getImm()) == 0)
    return true;
  return false;
}

/// Check if instruction has explicit SGPR operands
static bool hasSGPROperands(const MachineInstr &MI, const SIRegisterInfo &TRI) {
  const MachineRegisterInfo &MRI = MI.getMF()->getRegInfo();
  for (const MachineOperand &MO : MI.explicit_operands()) {
    if (!MO.isReg() || !MO.getReg().isPhysical())
      continue;
    if (TRI.isSGPRReg(MRI, MO.getReg()))
      return true;
  }
  return false;
}

struct StallSources {
  unsigned Unit = 0;
  unsigned VALUSlot = 0;
  unsigned CoExec = 0;
  unsigned DelayAlu = 0;
  unsigned WaitCnt = 0;
  unsigned MemFIFO = 0;
  unsigned RegBank = 0;
  unsigned LongLatVALU = 0;
  unsigned LOLVALUTRANSHazard = 0; // 1-cycle mutual exclusion: LOLVALU <-> TRANS
  unsigned SSRC = 0;
  unsigned VaVdst = 0;
  unsigned RAW = 0;  // Scoreboard-detected RAW dependency
  std::string CachePattern;

  unsigned CacheHits = 0;
  unsigned CacheMisses = 0;
  unsigned CacheEvictions = 0;

  unsigned EffectiveCycle = 0;
  unsigned CoExecFromEffective = 0;
  bool HasFUCoExecInteraction = false;
  bool RegBankInWMMAWindow = false;

  // For WMMA_SCALE decomposition: track when WMMA phase starts
  unsigned WMMAStartCycle = 0;
  bool IsScaledWMMA = false;

  unsigned total() const {
    unsigned EffectiveRegBank = RegBankInWMMAWindow ? 0 : RegBank;
    return std::max({Unit, VALUSlot, CoExec, DelayAlu, WaitCnt, MemFIFO,
                     EffectiveRegBank, LongLatVALU, LOLVALUTRANSHazard, SSRC,
                     VaVdst, RAW});
  }
};

//===----------------------------------------------------------------------===//
// MSB_SET Handling
//===----------------------------------------------------------------------===//

enum class MSBSetOutcome { Fused, Exposed };

bool canMSBSetFuse(InstClass PrevIC) {
  switch (PrevIC) {
  case InstClass::DS_READ:
  case InstClass::DS_WRITE:
  case InstClass::BARRIER:
  case InstClass::WAITCNT:
    return false;
  case InstClass::VALU:
  case InstClass::TRANS:
  case InstClass::SALU:
  case InstClass::WMMA:
  case InstClass::VMEM_READ:
  case InstClass::VMEM_WRITE:
  case InstClass::SMEM:
  case InstClass::TDM:
    return true;
  default:
    return false;
  }
}

MSBSetOutcome classifyMSBSet(const GPUSimState &State) {
  return canMSBSetFuse(State.PreviousInstClass)
         ? MSBSetOutcome::Fused : MSBSetOutcome::Exposed;
}

void populateMSBSetInfo(MSBSetOutcome Outcome, bool IsMasked, InstrSimInfo &Info) {
  if (Outcome == MSBSetOutcome::Fused) {
    Info.WasFused = true;
  } else if (IsMasked) {
    Info.WasExposed = true;
    Info.WasMasked = true;
  } else {
    Info.WasExposed = true;
    Info.StallCycles = 1;
    Info.Reason = StallReason::MSB_SET_EXPOSED;
  }
}

bool handleMSBSet(InstClass IC, GPUSimState &State, BlockMetrics &Metrics,
                  KernelPerfReport *Report, const MachineInstr &MI,
                  const SIInstrInfo &TII, unsigned EntryCycle) {
  if (IC != InstClass::MSB_SET)
    return false;

  // MSB_SET counts toward s_delay_alu skip (LowerVGPREncoding runs before InsertDelayAlu)
  (void)checkPendingDelayAlu(State, /*SkipApply=*/true);

  MSBSetOutcome Outcome = classifyMSBSet(State);
  bool IsMasked = false;

  // Check if exposure is masked by next instruction's co-exec stall
  if (Outcome == MSBSetOutcome::Exposed && State.inWMMAWindow()) {
    if (MachineInstr *NextMI = SIInstrInfo::getNextRealInstr(const_cast<MachineInstr *>(&MI))) {
      InstClass NextIC = classifyInst(*NextMI, TII);
      unsigned NextCoExecStall = State.getCoExecStall(NextIC);
      IsMasked = (NextCoExecStall >= 1);
    }
  }

  // Apply outcome
  Metrics.NumInstructions++;
  Metrics.NumMSBSet++;
  if (Outcome == MSBSetOutcome::Exposed) {
    if (IsMasked) {
      Metrics.NumMSBSetMasked++;
    } else {
      Metrics.NumMSBSetExposed++;
      State.advanceCycle(1);
      if (State.inWMMAWindow()) {
        Metrics.StallCoExec++;
        Metrics.CoExecMissOther++;
      }
    }
  }
  State.PreviousInstClass = InstClass::SALU;

  if (VerboseSimulation) {
    unsigned DisplayCycle = (Outcome == MSBSetOutcome::Fused)
                            ? (EntryCycle > 0 ? EntryCycle - 1 : 0)
                            : EntryCycle;
    dbgs() << "\n[Cycle " << DisplayCycle << "] ";
    MI.print(dbgs(), /*IsStandalone=*/true, /*SkipOpers=*/false,
             /*SkipDebugLoc=*/true, /*AddNewLine=*/false);
    dbgs() << "\n  → MSB_SET ";
    if (Outcome == MSBSetOutcome::Fused) {
      dbgs() << "fused with prev (free)";
    } else if (IsMasked) {
      dbgs() << "exposed but MASKED (next instr stalls anyway)";
    } else {
      dbgs() << "EXPOSED (+1 cycle)";
      if (State.inWMMAWindow())
        dbgs() << " [in WMMA window]";
    }
    dbgs() << "\n";
  }

  if (Report) {
    InstrSimInfo Info;
    populateMSBSetInfo(Outcome, IsMasked, Info);
    Report->PerInstr[&MI] = Info;
  }

  return true;
}

// Helper: update IssueCycle if stall source is larger
static void applyStall(unsigned &IssueCycle, unsigned CurrentCycle,
                       unsigned StallUntil) {
  if (StallUntil > IssueCycle)
    IssueCycle = StallUntil;
}

// Compute VALU resource stalls (LOLVALU holds VALU, TRANS in WMMA window)
static unsigned computeVALUResourceStall(InstClass IC, const GPUSimState &State,
                                         unsigned IssueCycle) {
  unsigned Stall = 0;
  if (IC == InstClass::VALU && State.VALUResourceBusyUntil > IssueCycle)
    Stall = State.VALUResourceBusyUntil - IssueCycle;
  // TRANS checks VALUResourceBusyUntil only in WMMA window
  if (IC == InstClass::TRANS && State.inWMMAWindow() &&
      State.VALUResourceBusyUntil > IssueCycle)
    Stall = std::max(Stall, State.VALUResourceBusyUntil - IssueCycle);
  return Stall;
}

// Compute WMMA_SCALE timing using decomposition model:
// Scale Read (1-cycle VALU) followed by WMMA (XDL)
static void computeWMMAScaleStall(const GPUSimState &State,
                                  unsigned &IssueCycle, StallSources &S) {
  unsigned ScaleReadCycle = IssueCycle;

  // Scale read needs VALU resource free
  ScaleReadCycle = std::max(ScaleReadCycle, State.VALUResourceBusyUntil);

  // In WMMA window: scale read must be in I-slot
  if (State.inWMMAWindow()) {
    unsigned CoExecStall = State.getCoExecStall(InstClass::VALU);
    ScaleReadCycle = std::max(ScaleReadCycle, State.CurrentCycle + CoExecStall);
  }

  // WMMA phase needs XDL at (ScaleReadCycle + 1)
  unsigned WMMAStartCycle = ScaleReadCycle + 1;
  unsigned XDLFreeAt = State.getUnitBusyUntil(FunctionalUnit::XDL);

  if (WMMAStartCycle < XDLFreeAt) {
    // Delay scale read so WMMA starts when XDL is free
    unsigned DesiredScaleSlot = XDLFreeAt - 1;

    if (State.inWMMAWindow() && DesiredScaleSlot >= State.ActiveWMMA.StartCycle &&
        DesiredScaleSlot < State.ActiveWMMA.EndCycle) {
      // Check if VALU allowed at this slot
      auto StageOpt = State.ActiveWMMA.getCurrentStage(DesiredScaleSlot);
      if (StageOpt && State.ActiveWMMA.Info.canCoExec(InstClass::VALU, *StageOpt)) {
        ScaleReadCycle = DesiredScaleSlot;
        WMMAStartCycle = XDLFreeAt;
      } else {
        // V/E-slot: wait for window to end
        ScaleReadCycle = State.ActiveWMMA.EndCycle;
        WMMAStartCycle = ScaleReadCycle + 1;
      }
    } else {
      // Outside window or after window: just delay
      ScaleReadCycle = DesiredScaleSlot;
      WMMAStartCycle = XDLFreeAt;
    }
  }

  IssueCycle = ScaleReadCycle;
  S.WMMAStartCycle = WMMAStartCycle;
  S.VALUSlot = IssueCycle - State.CurrentCycle;
}

// Compute scoreboard-based RAW stalls
static unsigned computeRAWStall(const MachineInstr &MI, GPUSimState &State) {
  if (!EnableScoreboard || !State.RegFile.TRI)
    return 0;

  unsigned MaxRAW = 0;
  if (instructionWaitsForVALU(MI)) {
    // Memory ops implicitly wait for ALL pending VALU writes
    MaxRAW = State.getMaxPendingRAW();
    LLVM_DEBUG(if (MaxRAW > 0 && VerboseSimulation)
      dbgs() << "    RAW implicit wait (VA_VDST==0): stall=" << MaxRAW << "\n";);
  } else {
    for (const MachineOperand &MO : MI.explicit_uses()) {
      if (MO.isReg() && MO.getReg().isPhysical()) {
        unsigned RAWStall = State.getRAWStall(MO.getReg(), State.RegFile.TRI);
        MaxRAW = std::max(MaxRAW, RAWStall);
      }
    }
    LLVM_DEBUG(if (MaxRAW > 0 && VerboseSimulation)
      dbgs() << "    RAW scoreboard: stall=" << MaxRAW << "\n";);
  }
  return MaxRAW;
}

// Compute memory FIFO stalls
static unsigned computeMemFIFOStall(InstClass IC, const GPUSimState &State) {
  switch (IC) {
  case InstClass::DS_READ:
  case InstClass::DS_WRITE:
    return State.getDSFIFOStall();
  case InstClass::VMEM_READ:
  case InstClass::VMEM_WRITE:
    return State.getVMEMBufferStall();
  case InstClass::TDM:
    return State.getTDMFIFOStall();
  default:
    return 0;
  }
}

StallSources computeStallSources(const MachineInstr &MI, InstClass IC,
                                 FunctionalUnit Unit, const SIInstrInfo &TII,
                                 GPUSimState &State) {
  StallSources S;
  unsigned IssueCycle = State.CurrentCycle;
  bool IsLOLVALU = IC == InstClass::VALU && TII.getRepeatRate(MI) > 1;

  // 1. Pending delay_alu from previous instruction
  unsigned PendingDelay = checkPendingDelayAlu(State);
  if (PendingDelay > 0) {
    S.DelayAlu = PendingDelay;
    applyStall(IssueCycle, State.CurrentCycle, State.CurrentCycle + PendingDelay);
  }

  // 2. Functional unit availability
  unsigned BusyUntil = State.getUnitBusyUntil(Unit);
  if (BusyUntil > IssueCycle) {
    S.Unit = BusyUntil - State.CurrentCycle;
    IssueCycle = BusyUntil;
  }

  // 3. VALU resource stalls (LOLVALU, TRANS in WMMA window)
  unsigned VALUResStall = computeVALUResourceStall(IC, State, IssueCycle);
  if (VALUResStall > 0) {
    S.Unit = std::max(S.Unit, VALUResStall);
    IssueCycle = std::max(IssueCycle, State.VALUResourceBusyUntil);
  }

  // 4. LOLVALU <-> TRANS 1-cycle mutual exclusion hazard
  if ((IC == InstClass::TRANS || IsLOLVALU) &&
      State.LOLVALUTRANSHazardUntil > IssueCycle) {
    S.LOLVALUTRANSHazard = State.LOLVALUTRANSHazardUntil - IssueCycle;
    IssueCycle = State.LOLVALUTRANSHazardUntil;
  }

  // 5. WMMA-specific stalls
  if (IC == InstClass::WMMA) {
    StringRef Name = TII.getName(MI.getOpcode());
    bool HasScaling = Name.contains_insensitive("scale");
    S.IsScaledWMMA = HasScaling;

    // TRANS holds VALU for 2 cycles, blocks WMMA
    unsigned TRANSStall = State.getWMMATRANSStall();
    applyStall(IssueCycle, State.CurrentCycle, State.CurrentCycle + TRANSStall);

    if (HasScaling) {
      computeWMMAScaleStall(State, IssueCycle, S);
    } else {
      S.WMMAStartCycle = IssueCycle;
    }

    // Track WMMA cache hits/misses
    auto RB = State.RegFile.updateCacheForWMMA(MI, TII);
    S.CachePattern = RB.CachePattern;
    S.CacheHits = RB.CacheHits;
    S.CacheMisses = RB.CacheMisses;
    S.CacheEvictions = RB.CacheEvictions;
  }

  // 6. Register bank stalls (VALU/TRANS/SALU)
  if (IC == InstClass::VALU || IC == InstClass::TRANS || IC == InstClass::SALU) {
    auto RB = State.RegFile.getRegBankStalls(MI);
    S.RegBank = RB.Stalls;
    S.CachePattern = RB.CachePattern;
    S.CacheHits = RB.CacheHits;
    S.CacheMisses = RB.CacheMisses;
    S.CacheEvictions = RB.CacheEvictions;
    if (State.inWMMAWindow()) {
      S.RegBankInWMMAWindow = true; // Track but don't stall
    } else {
      IssueCycle += RB.Stalls;
    }
  }

  // 7. VA_SSRC: VALU with SGPR blocks SALU until complete
  if (IC == InstClass::SALU && State.VaSSRCBusyUntil > IssueCycle) {
    S.SSRC = State.VaSSRCBusyUntil - IssueCycle;
    IssueCycle = State.VaSSRCBusyUntil;
  }

  // 8. Scoreboard RAW hazards (optional mode)
  unsigned RAW = computeRAWStall(MI, State);
  if (RAW > 0) {
    S.RAW = RAW;
    applyStall(IssueCycle, State.CurrentCycle, State.CurrentCycle + RAW);
  }

  // 9. s_wait_alu va_vdst(N)
  if (MI.getOpcode() == AMDGPU::S_WAITCNT_DEPCTR) {
    unsigned VaVdstStall = parseWaitAluVaVdst(MI, State);
    if (VaVdstStall > 0) {
      S.VaVdst = VaVdstStall;
      applyStall(IssueCycle, State.CurrentCycle, State.CurrentCycle + VaVdstStall);
    }
  }

  // 10. WMMA co-execution window rules
  if (State.inWMMAWindow() && IC != InstClass::WMMA) {
    unsigned LongLatVALU = TII.isTRANS(MI) ? 0 : TII.getRepeatRate(MI);
    if (LongLatVALU > 1) {
      // LOLVALU can't co-execute - waits for entire window
      if (State.ActiveWMMA.EndCycle > IssueCycle) {
        S.LongLatVALU = State.ActiveWMMA.EndCycle - IssueCycle;
        IssueCycle = State.ActiveWMMA.EndCycle;
      }
    } else {
      // Regular co-execution slot check
      unsigned CoExecStall = State.getCoExecStallAt(IC, IssueCycle);
      if (CoExecStall > 0) {
        S.EffectiveCycle = IssueCycle;
        S.CoExecFromEffective = CoExecStall;
        S.HasFUCoExecInteraction = (IssueCycle > State.CurrentCycle);
        IssueCycle += CoExecStall;
      }
    }
    S.CoExec = IssueCycle - State.CurrentCycle;
  }

  // 11. s_delay_alu parsing
  if (IC == InstClass::DELAY_ALU) {
    unsigned DelayStall = parseDelayAlu(MI, State);
    S.DelayAlu = DelayStall;
    applyStall(IssueCycle, State.CurrentCycle, State.CurrentCycle + DelayStall);
  }

  // 12. Waitcnt stalls
  if (IC == InstClass::WAITCNT) {
    unsigned WaitStall = computeWaitStall(MI, State);
    S.WaitCnt = WaitStall;
    applyStall(IssueCycle, State.CurrentCycle, State.CurrentCycle + WaitStall);
  }

  // 13. Memory FIFO stalls
  S.MemFIFO = computeMemFIFOStall(IC, State);
  applyStall(IssueCycle, State.CurrentCycle, State.CurrentCycle + S.MemFIFO);

  return S;
}

void attributeStall(const StallSources &S, FunctionalUnit Unit, InstClass IC,
                    BlockMetrics &Metrics) {
  Metrics.VGPRCacheHits += S.CacheHits;
  Metrics.VGPRCacheMisses += S.CacheMisses;
  Metrics.VGPRCacheEvictions += S.CacheEvictions;

  // Track reg bank conflicts in WMMA window separately (not added to stalls)
  if (S.RegBankInWMMAWindow && S.RegBank > 0)
    Metrics.RegBankConflictsInWMMAWindow += S.RegBank;

  unsigned TotalStall = S.total();
  if (TotalStall == 0)
    return;

  if (S.WaitCnt == TotalStall) {
    Metrics.StallWaitCnt += TotalStall;
  } else if (S.MemFIFO == TotalStall) {
    Metrics.StallMemFIFO += TotalStall;
  } else if (S.Unit == TotalStall) {
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
  } else if (S.VALUSlot == TotalStall) {
    Metrics.StallFunctionalUnit += TotalStall;
    Metrics.StallVALU += TotalStall;
  } else if (S.CoExec == TotalStall) {
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
  } else if (S.DelayAlu == TotalStall) {
    Metrics.StallDelayAlu += TotalStall;
  } else if (S.LongLatVALU == TotalStall) {
    Metrics.StallCoExec += TotalStall;
    Metrics.StallLongLatVALU += TotalStall;
  } else if (S.LOLVALUTRANSHazard == TotalStall) {
    Metrics.StallLOLVALUTRANS += TotalStall;
  } else if (S.SSRC == TotalStall) {
    Metrics.StallVaSSRC += TotalStall;
  } else if (S.VaVdst == TotalStall) {
    Metrics.StallVaVdst += TotalStall;
  } else if (S.RAW == TotalStall) {
    Metrics.StallRAW += TotalStall;
  } else if (S.RegBank == TotalStall && !S.RegBankInWMMAWindow) {
    Metrics.StallRegBankConflict += TotalStall;
  }
}

void trackWMMACoExec(InstClass IC, const StallSources &S,
                     GPUSimState &State, BlockMetrics &Metrics) {
  bool InWMMAWindow = State.inWMMAWindow() && IC != InstClass::WMMA;
  if (InWMMAWindow) {
    if (S.CoExec > 0)
      Metrics.WMMACoExecBlocked++;
    else
      Metrics.WMMACoExecUsed++;

    // Track I-slot utilization
    auto StageOpt = State.getWMMAStage();
    if (StageOpt) {
      uint8_t StageMask = State.ActiveWMMA.Info.StageMask[*StageOpt];
      bool IsISlot = (StageMask & CoExecMask::VALU) != 0;

      if (IsISlot && S.CoExec == 0) {
        Metrics.ISlotTotal++;
        if (IC == InstClass::VALU || IC == InstClass::TRANS)
          Metrics.ISlotUsedByVALU++;
        else
          Metrics.ISlotWastedOnNonVALU++;
      }
    }
  }
}

void recordInstruction(const MachineInstr &MI, const InstTiming &T,
                       const SIInstrInfo &TII, const StallSources &Stalls,
                       GPUSimState &State, BlockMetrics &Metrics) {
  Metrics.NumInstructions++;

  switch (T.IC) {
  case InstClass::VALU: {
    Metrics.NumVALU++;
    if (AMDGPU::isVOPD(MI.getOpcode())) {
      Metrics.NumVOPD++;
      Metrics.NumVALU++;  // VOPD = 2 VALU ops
    } else if (TII.isPacked(MI)) {
      Metrics.NumPacked++;
      Metrics.NumVALU++;  // Packed = 2 VALU ops
    }
    State.trackVALU(T.Latency);
    State.trackVALUForWMMA(T.IC);
    unsigned LongLatVALU = TII.isTRANS(MI) ? 0 : TII.getRepeatRate(MI);
    if (LongLatVALU > 1) {
      State.VALUResourceBusyUntil = std::max(State.VALUResourceBusyUntil,
                                              State.CurrentCycle + LongLatVALU);
      // LOLVALU sets 1-cycle hazard for TRANS (next TRANS must wait 1 cycle)
      State.LOLVALUTRANSHazardUntil = std::max(State.LOLVALUTRANSHazardUntil,
                                                State.CurrentCycle + 2);
    }
    // VALU with SSRC blocks following SALU until VALU result is ready
    if (State.RegFile.TRI && hasSGPROperands(MI, *State.RegFile.TRI)) {
      State.VaSSRCBusyUntil = std::max(State.VaSSRCBusyUntil,
                                          State.CurrentCycle + T.Latency);
    }
    // Track pending VGPR write for va_vdst
    State.PendingVaVdst.push_back({State.CurrentCycle + T.Latency});
    break;
  }

  case InstClass::SALU:
    Metrics.NumSALU++;
    State.LastSALUCycle = State.CurrentCycle;
    break;

  case InstClass::TRANS:
    Metrics.NumTRANS++;
    State.trackTRANS(T.Latency);
    State.trackVALUForWMMA(T.IC);
    State.holdVALUResourceInWindow(T.ResourceCycles);
    // TRANS sets 1-cycle hazard for LOLVALU (next LOLVALU must wait 1 cycle)
    State.LOLVALUTRANSHazardUntil = std::max(State.LOLVALUTRANSHazardUntil,
                                              State.CurrentCycle + 2);
    // Track pending VGPR write for va_vdst
    State.PendingVaVdst.push_back({State.CurrentCycle + T.Latency});
    break;

  case InstClass::WMMA: {
    Metrics.NumWMMA++;
    State.trackTRANS(T.Latency);

    // use WMMAStartCycle from stall computation
    unsigned WMMAStartCycle = Stalls.IsScaledWMMA ? Stalls.WMMAStartCycle
                                                   : State.CurrentCycle;
    unsigned Occupancy = State.startWMMAWindow(MI, TII, WMMAStartCycle);
    Metrics.WMMAWindowCycles += Occupancy;

    // For scaled WMMA: scale read updates VALU state
    if (Stalls.IsScaledWMMA) {
      // Scale read at CurrentCycle occupies VALU for 1 cycle
      State.VALUResourceBusyUntil = std::max(State.VALUResourceBusyUntil,
                                              State.CurrentCycle + 1);
      State.LastVALUCycle = State.CurrentCycle;
    }

    // WMMA with SGPR operands (scale) blocks SALU (VA_SSRC)
    // Use co-execution window occupancy, not SchedModel latency
    if (State.RegFile.TRI && hasSGPROperands(MI, *State.RegFile.TRI)) {
      State.VaSSRCBusyUntil = std::max(State.VaSSRCBusyUntil,
                                          WMMAStartCycle + Occupancy);
    }

    // XDL (WMMA) is recorded as TRANS in the scoreboard for VA_VDST tracking
    // Per ISA: "XDL (WMMA, SWMMAC) instructions are recorded as TRANS"
    // Use co-execution window occupancy for timing consistency
    State.PendingVaVdst.push_back({WMMAStartCycle + Occupancy});

    if (VerboseSimulation) {
      dbgs() << "  Class: WMMA | Unit: XDL | Occupancy: " << Occupancy
             << " | Window: " << State.ActiveWMMA.Info.TotalWindow << "\n";
    }
    break;
  }

  case InstClass::DS_READ: {
    Metrics.NumDSRead++;
    auto [BaseVGPR, NumRegs] = getDestRegInfo(MI, TII, /*IsVGPR=*/true);
    State.issueDS(T.Latency, BaseVGPR, std::max(NumRegs, 1u), /*IsLoad=*/true);
    break;
  }
  case InstClass::DS_WRITE:
    Metrics.NumDSWrite++;
    State.issueDS(T.Latency, 0, 0, /*IsLoad=*/false);
    break;

  case InstClass::VMEM_READ: {
    Metrics.NumVMEM++;
    auto [BaseVGPR, NumRegs] = getDestRegInfo(MI, TII, /*IsVGPR=*/true);
    State.issueVMEM(T.Latency, BaseVGPR, std::max(NumRegs, 1u), /*IsLoad=*/true);
    break;
  }
  case InstClass::VMEM_WRITE:
    Metrics.NumVMEM++;
    State.issueVMEM(T.Latency, 0, 0, /*IsLoad=*/false);
    break;

  case InstClass::SMEM: {
    Metrics.NumSMEM++;
    auto [BaseSGPR, NumRegs] = getDestRegInfo(MI, TII, /*IsVGPR=*/false);
    State.issueSMEM(T.Latency, BaseSGPR, std::max(NumRegs, 1u));
    break;
  }

  case InstClass::BRANCH:
    Metrics.NumBranch++;
    break;

  case InstClass::TDM:
    Metrics.NumTDM++;
    State.issueTDM(T.Latency);
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
    llvm_unreachable("MSB_SET should return early");

  case InstClass::NOP:
    Metrics.NumNop++;
    break;

  default:
    break;
  }

  unsigned Opc = MI.getOpcode();
  if (Opc == AMDGPU::V_WRITELANE_B32)
    Metrics.NumSGPRToVGPR++;
  else if (Opc == AMDGPU::V_READLANE_B32)
    Metrics.NumVGPRToSGPR++;

  if (SIInstrInfo::isSpill(MI) || SIInstrInfo::isFLATScratch(MI)) {
    if (MI.mayStore()) Metrics.NumSpill++;
    if (MI.mayLoad()) Metrics.NumReload++;
  }

  if (T.IC != InstClass::WMMA)
    State.setUnitBusyUntil(T.Unit, State.CurrentCycle + T.ResourceCycles);

  // Scoreboard: record destination registers and handle implicit waits
  // Only track synchronous ALU ops (VALU, TRANS, SALU) - NOT async memory ops
  // Async ops (DS, VMEM) are handled via s_waitcnt, not scoreboard
  if (EnableScoreboard && State.RegFile.TRI) {
    // Memory ops implicitly wait for VA_VDST==0, clearing pending RAW hazards
    if (instructionWaitsForVALU(MI)) {
      State.clearRegScoreboard();
      if (VerboseSimulation)
        dbgs() << "  → Scoreboard cleared (implicit VA_VDST wait)\n";
    }
    // Only record writes from synchronous ALU instructions
    // WMMA is treated like TRANS for delay purposes (per AMDGPUInsertDelayAlu)
    if (T.IC == InstClass::VALU || T.IC == InstClass::TRANS ||
        T.IC == InstClass::SALU || T.IC == InstClass::WMMA) {
      for (const MachineOperand &MO : MI.defs()) {
        if (MO.isReg() && MO.getReg().isPhysical()) {
          State.recordRegWrite(MO.getReg(), T.Latency, State.RegFile.TRI);
        }
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Verbose Logging Helpers
//===----------------------------------------------------------------------===//

void logInstHeader(unsigned Cycle, const MachineInstr &MI, const InstTiming &T) {
  dbgs() << "\n[Cycle " << Cycle << "] ";
  MI.print(dbgs(), /*IsStandalone=*/true, /*SkipOpers=*/false,
           /*SkipDebugLoc=*/true, /*AddNewLine=*/false);
  dbgs() << "\n";
  if (T.IC != InstClass::WMMA) {
    dbgs() << "  Class: " << getInstClassName(T.IC)
           << " | Unit: " << getUnitName(T.Unit)
           << " | Latency: " << T.Latency
           << " | ResourceCycles: " << T.ResourceCycles << "\n";
  }
}

void logStalls(const StallSources &Stalls, const GPUSimState &State) {
  dbgs() << "  Stalls: ";
  if (Stalls.total() == 0) {
    dbgs() << "(none)";
  } else {
    bool First = true;
    auto printStall = [&](const char *Name, unsigned Val) {
      if (Val > 0) {
        if (!First) dbgs() << ", ";
        dbgs() << Name << "=" << Val;
        First = false;
      }
    };
    printStall("FU", Stalls.Unit);
    printStall("VALUSlot", Stalls.VALUSlot);
    printStall("WMMACoExecMiss", Stalls.CoExecFromEffective);
    printStall("LongLatVALU", Stalls.LongLatVALU);
    printStall("LOLVALUxTRANS", Stalls.LOLVALUTRANSHazard);
    printStall("SSRC", Stalls.SSRC);
    printStall("VaVdst", Stalls.VaVdst);
    printStall("RAW", Stalls.RAW);
    printStall("DelayALU", Stalls.DelayAlu);
    printStall("WaitCnt", Stalls.WaitCnt);
    printStall("MemFIFO", Stalls.MemFIFO);
    printStall("RegBank", Stalls.RegBank);
    if (Stalls.RegBankInWMMAWindow && Stalls.RegBank > 0)
      dbgs() << " [in WMMA window, not counted]";
  }
  dbgs() << " → Total: " << Stalls.total();
  if (!Stalls.CachePattern.empty())
    dbgs() << " Cache" << Stalls.CachePattern;
  dbgs() << "\n";

  if (Stalls.HasFUCoExecInteraction) {
    auto EffectiveStage = State.ActiveWMMA.getCurrentStage(Stalls.EffectiveCycle);
    dbgs() << "    (Base stall lands at cycle " << Stalls.EffectiveCycle;
    if (EffectiveStage) {
      uint8_t Mask = State.ActiveWMMA.Info.StageMask[*EffectiveStage];
      WMMAStageType StageType = CoExecMask::getStageType(Mask);
      const char *StageName =
          StageType == WMMAStageType::E0 ? "E0" :
          StageType == WMMAStageType::E  ? "E" :
          StageType == WMMAStageType::I  ? "I" :
          StageType == WMMAStageType::V  ? "V" : "?";
      dbgs() << " [stage " << *EffectiveStage << "/"
             << State.ActiveWMMA.Info.TotalWindow << " "
             << StageName << " - blocked]";
    } else {
      dbgs() << " [outside window]";
    }
    dbgs() << " → additional CoExec=" << Stalls.CoExecFromEffective << ")\n";
  }
}

void logWMMAWindow(const GPUSimState &State, InstClass IC) {
  if (!State.inWMMAWindow() || IC == InstClass::WMMA)
    return;

  auto Stage = State.ActiveWMMA.getCurrentStage(State.CurrentCycle);
  dbgs() << "  WMMA Window: [" << (Stage ? *Stage : ~0U)
         << "/" << State.ActiveWMMA.Info.TotalWindow << "]";
  if (Stage) {
    uint8_t Mask = State.ActiveWMMA.Info.StageMask[*Stage];
    WMMAStageType ST = CoExecMask::getStageType(Mask);
    const char *StageNames[] = {"?", "E0", "E", "I", "V"};
    dbgs() << " " << StageNames[(int)ST];
  }
  dbgs() << " (cycles " << State.ActiveWMMA.StartCycle
         << "-" << State.ActiveWMMA.EndCycle << ")\n";
}

void logUnitAndMemState(const GPUSimState &State, const InstTiming &T) {
  if (T.Unit != FunctionalUnit::NONE) {
    dbgs() << "  → UnitBusyUntil[" << getUnitName(T.Unit) << "] = "
           << State.getUnitBusyUntil(T.Unit) << "\n";
  }

  if (T.IC == InstClass::VALU)
    dbgs() << "  → LastVALUCycle = " << State.LastVALUCycle << "\n";
  else if (T.IC == InstClass::TRANS)
    dbgs() << "  → LastTRANSCycle = " << State.LastTRANSCycle << "\n";

  switch (T.IC) {
  case InstClass::DS_READ:
  case InstClass::DS_WRITE:
    dbgs() << "  → PendingDS: " << State.PendingDS.size()
           << ", Counter[LGKM]=" << State.MemCounters[(unsigned)MemCounter::LGKM] << "\n";
    break;
  case InstClass::VMEM_READ:
    dbgs() << "  → PendingVMEMLoad: " << State.PendingVMEMLoad.size()
           << ", Counter[VMEM]=" << State.MemCounters[(unsigned)MemCounter::VMEM] << "\n";
    break;
  case InstClass::VMEM_WRITE:
    dbgs() << "  → PendingVMEMStore: " << State.PendingVMEMStore.size()
           << ", Counter[VS]=" << State.MemCounters[(unsigned)MemCounter::VS] << "\n";
    break;
  case InstClass::SMEM:
    dbgs() << "  → PendingSMEM: " << State.PendingSMEM.size()
           << ", Counter[LGKM]=" << State.MemCounters[(unsigned)MemCounter::LGKM] << "\n";
    break;
  case InstClass::WMMA:
    dbgs() << "  → ActiveWMMA: cycles " << State.ActiveWMMA.StartCycle
           << "-" << State.ActiveWMMA.EndCycle;
    if (State.ActiveWMMA.IsBackToBack)
      dbgs() << " [back-to-back]";
    dbgs() << "\n";
    break;
  default:
    break;
  }
}

//===----------------------------------------------------------------------===//
// WMMA Window State Capture
//===----------------------------------------------------------------------===//

struct WMMAWindowCapture {
  bool WasInWindow = false;
  std::optional<unsigned> Stage;
  WMMAStageType StageType = WMMAStageType::NONE;
  unsigned TotalWindow = 0;
};

WMMAWindowCapture captureWMMAWindowState(const GPUSimState &State,
                                          unsigned EntryCycle, InstClass IC) {
  WMMAWindowCapture Capture;
  if (!State.inWMMAWindow() || IC == InstClass::WMMA)
    return Capture;

  Capture.WasInWindow = true;
  Capture.Stage = State.ActiveWMMA.getCurrentStage(EntryCycle);
  Capture.TotalWindow = State.ActiveWMMA.Info.TotalWindow;

  if (Capture.Stage) {
    uint8_t Mask = State.ActiveWMMA.Info.StageMask[*Capture.Stage];
    Capture.StageType = CoExecMask::getStageType(Mask);
  }
  return Capture;
}

//===----------------------------------------------------------------------===//
// InstrSimInfo Population
//===----------------------------------------------------------------------===//

static StallReason getDominantStallReason(const StallSources &Stalls) {
  unsigned Max = 0;
  StallReason Reason = StallReason::NONE;

  if (Stalls.WaitCnt > Max) { Max = Stalls.WaitCnt; Reason = StallReason::WAITCNT; }
  if (Stalls.DelayAlu > Max) { Max = Stalls.DelayAlu; Reason = StallReason::DELAY_ALU; }
  if (Stalls.CoExec > Max) { Max = Stalls.CoExec; Reason = StallReason::COEXEC_BLOCKED; }
  if (Stalls.LongLatVALU > Max) { Max = Stalls.LongLatVALU; Reason = StallReason::LONG_LAT_VALU; }
  if (Stalls.LOLVALUTRANSHazard > Max) { Max = Stalls.LOLVALUTRANSHazard; Reason = StallReason::LOLVALU_TRANS_HAZARD; }
  if (Stalls.SSRC > 0 && Stalls.SSRC >= Max) { Max = Stalls.SSRC; Reason = StallReason::VA_SSRC_STALL; }
  if (Stalls.VaVdst > 0 && Stalls.VaVdst >= Max) { Max = Stalls.VaVdst; Reason = StallReason::VA_VDST_WAIT; }
  if (Stalls.MemFIFO > Max) { Max = Stalls.MemFIFO; Reason = StallReason::MEM_FIFO; }
  if (Stalls.Unit > Max) { Max = Stalls.Unit; Reason = StallReason::FU_BUSY; }
  // Only count reg bank if not in WMMA window
  if (!Stalls.RegBankInWMMAWindow && Stalls.RegBank > Max) {
    Max = Stalls.RegBank;
    Reason = StallReason::REG_BANK;
  }

  return Reason;
}

void populateInstrSimInfo(InstrSimInfo &Info, const StallSources &Stalls,
                          const WMMAWindowCapture &WMMAState, InstClass IC) {
  Info.StallCycles = Stalls.total();
  Info.Reason = getDominantStallReason(Stalls);
  Info.CachePattern = Stalls.CachePattern;

  if (IC == InstClass::DELAY_ALU)
    Info.WasFused = true;

  if (WMMAState.WasInWindow) {
    Info.InWMMAWindow = true;
    Info.WMMATotalWindow = WMMAState.TotalWindow;

    if (WMMAState.Stage) {
      Info.WMMAStage = *WMMAState.Stage;
      Info.StageType = WMMAState.StageType;
    }

    Info.CoExecuted = (Stalls.CoExec == 0);
  }
}

void simulateInst(const MachineInstr &MI, const SIInstrInfo &TII,
                  GPUSimState &State, BlockMetrics &Metrics,
                  KernelPerfReport *Report = nullptr) {

  unsigned EntryCycle = State.CurrentCycle;
  InstTiming T = getInstTiming(MI, TII);

  if (handleMSBSet(T.IC, State, Metrics, Report, MI, TII, EntryCycle))
    return;

  if (VerboseSimulation)
    logInstHeader(EntryCycle, MI, T);

  if (T.IC == InstClass::WAITCNT) {
    const MachineBasicBlock *MBB = MI.getParent();
    MachineBasicBlock::const_instr_iterator It(&MI);
    FalseWaitResult FWR = analyzeFalseWaitsForWait(MI, It, MBB->instr_end(),
                                                    State, TII);
    Metrics.NumFalseWaits += FWR.Count;
    Metrics.StallFalseWait += FWR.WastedCycles;

    if (VerboseSimulation && (FWR.Count > 0 || FWR.WastedCycles > 0))
      dbgs() << "  → False waits: " << FWR.Count
             << ", wasted cycles: " << FWR.WastedCycles << "\n";
  }

  WMMAWindowCapture WMMAState = captureWMMAWindowState(State, EntryCycle, T.IC);
  StallSources Stalls = computeStallSources(MI, T.IC, T.Unit, TII, State);

  if (VerboseSimulation)
    logStalls(Stalls, State);

  attributeStall(Stalls, T.Unit, T.IC, Metrics);

  unsigned ReadyCycle = State.CurrentCycle + Stalls.total();
  if (ReadyCycle > State.CurrentCycle) {
    if (VerboseSimulation)
      dbgs() << "  → Advancing cycle: " << State.CurrentCycle
             << " → " << ReadyCycle << "\n";
    State.advanceToCycle(ReadyCycle);
  }

  trackWMMACoExec(T.IC, Stalls, State, Metrics);

  if (VerboseSimulation)
    logWMMAWindow(State, T.IC);

  recordInstruction(MI, T, TII, Stalls, State, Metrics);
  State.RegFile.invalidateWrites(MI);

  if (VerboseSimulation)
    logUnitAndMemState(State, T);

  State.advanceCycle(1);
  State.PreviousInstClass = T.IC;

  if (Report) {
    InstrSimInfo Info;
    populateInstrSimInfo(Info, Stalls, WMMAState, T.IC);
    if (T.IC == InstClass::WMMA) {
      Info.IsWMMA = true;
      Info.WMMAPattern = State.ActiveWMMA.Info.Pattern;
    }
    Report->PerInstr[&MI] = Info;
  }

  if (VerboseSimulation)
    dbgs() << "  → NextCycle: " << State.CurrentCycle << "\n";
}

BlockMetrics analyzeBlock(MachineBasicBlock &MBB, const SIInstrInfo &TII,
                          GPUSimState &State,
                          KernelPerfReport *Report = nullptr) {
  if (VerboseSimulation) {
    dbgs() << "\n=== BB#" << MBB.getNumber();
    if (const BasicBlock *BB = MBB.getBasicBlock())
      if (BB->hasName())
        dbgs() << " (" << BB->getName() << ")";
    dbgs() << " [Cycle " << State.CurrentCycle << "] ===\n";
  }

  BlockMetrics Metrics;
  unsigned StartCycle = State.CurrentCycle;

  for (MachineInstr &MI : MBB.instrs()) {
    if (MI.isBundle() || MI.isMetaInstruction())
      continue;
    if (MI.isDebugInstr() || MI.isImplicitDef())
      continue;
    simulateInst(MI, TII, State, Metrics, Report);
  }

  Metrics.TotalCycles = State.CurrentCycle - StartCycle;

  if (VerboseSimulation) {
    dbgs() << "=== End BB#" << MBB.getNumber()
           << ": " << Metrics.NumInstructions << " insts, "
           << Metrics.TotalCycles << " cycles, "
           << Metrics.StallCycles() << " stalls ===\n";
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

unsigned getLoopTripCount(MachineLoop *L,
                          const MachineBlockFrequencyInfo *MBFI = nullptr) {
  if (MBFI) {
    MachineBasicBlock *Header = L->getHeader();
    MachineBasicBlock *Preheader = L->getLoopPreheader();

    if (Header && Preheader) {
      float HeaderFreq = getBlockFrequency(MBFI, Header);
      float PreheaderFreq = getBlockFrequency(MBFI, Preheader);

      if (PreheaderFreq > 0.0f) {
        unsigned DerivedTC = static_cast<unsigned>(HeaderFreq / PreheaderFreq + 0.5f);
        if (DerivedTC >= 1) {
          if (VerboseSimulation) {
            dbgs() << "  Trip count from MBFI: " << DerivedTC
                   << " (header=" << format("%.1f", HeaderFreq)
                   << " / preheader=" << format("%.1f", PreheaderFreq) << ")\n";
          }
          return DerivedTC;
        }
      }
    }
  }
  return DefaultTripCount;
}
BlockMetrics analyzeLoop(MachineLoop *L, MachineLoopInfo &MLI,
                         const SIInstrInfo &TII, GPUSimState &EntryState,
                         DenseSet<MachineBasicBlock *> &Visited,
                         KernelPerfReport &Report,
                         const MachineBlockFrequencyInfo *MBFI) {
  unsigned TripCount = getLoopTripCount(L, MBFI);
  unsigned LoopDepth = L->getLoopDepth();

  Report.NumLoops++;
  Report.MaxLoopDepth = std::max(Report.MaxLoopDepth, LoopDepth);
  Report.MaxTripCount = std::max(Report.MaxTripCount, TripCount);

  MachineBasicBlock *Header = L->getHeader();
  float HeaderFreq = getBlockFrequency(MBFI, Header);

  if (VerboseSimulation) {
    dbgs() << "\n=== Analyzing Loop (depth " << LoopDepth
           << ", trip count " << TripCount << ") ===\n";
    dbgs() << "  Header: " << Header->getName()
           << " (freq=" << format("%.3f", HeaderFreq) << ")\n";
  }

  DenseMap<MachineBasicBlock *, BlockMetrics> ColdPerBlock;
  DenseMap<MachineBasicBlock *, BlockMetrics> WarmPerBlock;
  DenseMap<MachineLoop *, BlockMetrics> InnerLoopMetrics;
  BlockMetrics DirectBlocksRaw;

  auto simulateIteration = [&](GPUSimState &State, const char *Label,
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
          InnerMetrics = analyzeLoop(InnerLoop, MLI, TII, State,
                                     Visited, Report, MBFI);
          InnerLoopMetrics[InnerLoop] = InnerMetrics;
        } else {
          InnerMetrics = InnerLoopMetrics.lookup(InnerLoop);
        }

        float InnerEntryFreq;
        if (MachineBasicBlock *InnerPreheader = InnerLoop->getLoopPreheader()) {
          InnerEntryFreq = getBlockFrequency(MBFI, InnerPreheader);
        } else {
          float InnerHeaderFreq = getBlockFrequency(MBFI, MBB);
          unsigned InnerTripCount = getLoopTripCount(InnerLoop, MBFI);
          InnerEntryFreq = (InnerTripCount > 0) ? InnerHeaderFreq / InnerTripCount : InnerHeaderFreq;
        }
        float RelativeFreq = (HeaderFreq > 0) ? InnerEntryFreq / HeaderFreq : 1.0f;

        if (VerboseSimulation) {
          dbgs() << "  Inner loop " << MBB->getName()
                 << " entry freq: " << format("%.3f", InnerEntryFreq)
                 << " relative: " << format("%.3f", RelativeFreq)
                 << (isCold ? "" : " (cached)") << "\n";
        }

        IterMetrics = IterMetrics + InnerMetrics * RelativeFreq;
      } else if (MLI.getLoopFor(MBB) == L) {
        BlockMetrics BM = analyzeBlock(*MBB, TII, State, &Report);
        if (isCold)
          DirectBlocksRaw = DirectBlocksRaw + BM;

        float BlockFreq = getBlockFrequency(MBFI, MBB);
        float RelativeFreq = (HeaderFreq > 0) ? BlockFreq / HeaderFreq : 1.0f;
        IterMetrics = IterMetrics + BM * RelativeFreq;
        PerBlockOut[MBB] = BM;
      }
    }
    return IterMetrics;
  };

  GPUSimState ColdState = EntryState;
  BlockMetrics ColdMetrics = simulateIteration(ColdState, "Cold", ColdPerBlock, true);

  if (VerboseSimulation)
    dbgs() << "  Cold iteration: " << ColdMetrics.TotalCycles << " cycles, "
           << ColdMetrics.StallCycles() << " stall\n";

  BlockMetrics WarmMetrics = simulateIteration(ColdState, "Warm", WarmPerBlock, false);

  if (VerboseSimulation)
    dbgs() << "  Warm iteration: " << WarmMetrics.TotalCycles << " cycles, "
           << WarmMetrics.StallCycles() << " stall\n";

  for (MachineBasicBlock *MBB : L->blocks())
    Visited.insert(MBB);

  EntryState = ColdState;
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

  return ScaledMetrics;
}

KernelPerfReport analyzeFunction(MachineFunction &MF, const SIInstrInfo &TII,
                                 MachineLoopInfo *MLI,
                                 const MachineBlockFrequencyInfo *MBFI) {
  KernelPerfReport Report;
  GPUSimState State;

  const SIRegisterInfo *TRI = &TII.getRegisterInfo();
  State.RegFile = RegisterFile(TRI);

  DenseSet<MachineBasicBlock *> Visited;
  printBlockFrequencies(MF, MBFI);

  ReversePostOrderTraversal<MachineFunction *> RPOT(&MF);

  for (MachineBasicBlock *MBB : RPOT) {
    if (Visited.contains(MBB))
      continue;

    MachineLoop *L = MLI ? MLI->getLoopFor(MBB) : nullptr;

    if (L && L->getHeader() == MBB) {
      BlockMetrics LoopMetrics = analyzeLoop(L, *MLI, TII, State, Visited, Report, MBFI);

      float LoopEntryFreq = 1.0f;
      if (MachineBasicBlock *Preheader = L->getLoopPreheader()) {
        LoopEntryFreq = getBlockFrequency(MBFI, Preheader);
      } else {
        float HeaderFreq = getBlockFrequency(MBFI, MBB);
        unsigned TripCount = getLoopTripCount(L, MBFI);
        LoopEntryFreq = (TripCount > 0) ? HeaderFreq / TripCount : 1.0f;
      }

      if (VerboseSimulation)
        dbgs() << "  Loop entry frequency: " << format("%.3f", LoopEntryFreq) << "\n";

      Report.Scaled = Report.Scaled + LoopMetrics * LoopEntryFreq;
    } else {
      BlockMetrics BM = analyzeBlock(*MBB, TII, State, &Report);
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

bool runStaticSimulator(MachineFunction &MF, MachineLoopInfo *MLI,
                        const MachineBlockFrequencyInfo *MBFI) {
  if (!isStaticSimulatorEnabled())
    return false;

  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  if (!ST.hasGFX1250Insts())
    return false;

  const SIInstrInfo *TII = ST.getInstrInfo();
  if (!TII)
    return false;

  LLVM_DEBUG(dbgs() << "Running Static Simulator on: " << MF.getName() << "\n");

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
// KernelPerfReport Printing
//===----------------------------------------------------------------------===//

static void printStallBreakdown(raw_ostream &OS, const BlockMetrics &M,
                                const char *Indent = ";   ") {
  float StallPct = M.TotalCycles > 0
      ? 100.0f * M.StallCycles() / M.TotalCycles : 0.0f;
  OS << formatv("{0}Stall: {1} cycles ({2:F1}%)\n", Indent, M.StallCycles(), StallPct);
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

  // === Raw Metrics (each block executed once) ===
  OS << "; === Raw Metrics (each block executed once) ===\n";
  OS << formatv(";   Instructions: {0}\n", Raw.NumInstructions);
  OS << formatv(";   Cycles:       {0}\n", Raw.TotalCycles);
  printStallBreakdown(OS, Raw);
  OS << formatv(";   Waitcnts: {0} | False waits: {1}\n",
                Raw.NumWaitcnt, Raw.NumFalseWaits);
  OS << formatv(";   WMMA windows: {0} | Co-executed: {1}\n",
                Raw.WMMAWindowCycles, Raw.WMMACoExecUsed);
  if (Raw.ISlotTotal > 0) {
    OS << formatv(";   I-slots: {0} used | {1} wasted on non-VALU ({2:F0}% VALU)\n",
                  Raw.ISlotTotal, Raw.ISlotWastedOnNonVALU,
                  Raw.ISlotTotal > 0
                      ? 100.0f * Raw.ISlotUsedByVALU / Raw.ISlotTotal : 0.0f);
  }
  OS << ";\n";

  // === Scaled Metrics (loops × trip count) ===
  OS << "; === Scaled Metrics (loops x trip count) ===\n";
  OS << formatv(";   Instructions: {0}\n", Scaled.NumInstructions);
  OS << formatv(";   Cycles:       {0}\n", Scaled.TotalCycles);
  printStallBreakdown(OS, Scaled);
  OS << formatv(";   Waitcnts: {0} | False waits: {1}\n",
                Scaled.NumWaitcnt, Scaled.NumFalseWaits);
  OS << formatv(";   WMMA windows: {0} | Co-executed: {1} ({2:F0}%)\n",
                Scaled.WMMAWindowCycles, Scaled.WMMACoExecUsed,
                CoExecEfficiency * 100.0f);
  if (Scaled.ISlotTotal > 0) {
    OS << formatv(";   I-slots: {0} used | {1} wasted on non-VALU ({2:F0}% VALU)\n",
                  Scaled.ISlotTotal, Scaled.ISlotWastedOnNonVALU,
                  Scaled.ISlotTotal > 0
                      ? 100.0f * Scaled.ISlotUsedByVALU / Scaled.ISlotTotal : 0.0f);
  }
  OS << ";\n";

  // === Instruction Breakdown ===
  // NumVALU = ops, NumVOPD/NumPacked = instructions (each = 2 ops)
  // VALU instructions = NumVALU - NumVOPD - NumPacked
  OS << "; === Instruction Breakdown (Raw / Scaled) ===\n";
  unsigned RawVALUInst = Raw.NumVALU - Raw.NumVOPD - Raw.NumPacked;
  unsigned ScaledVALUInst = Scaled.NumVALU - Scaled.NumVOPD - Scaled.NumPacked;
  OS << formatv(";   VALU: {0}/{1}", RawVALUInst, ScaledVALUInst);
  // Show dual-issue breakdown (these are instructions, each = 2 ops)
  if (Raw.NumVOPD || Scaled.NumVOPD || Raw.NumPacked || Scaled.NumPacked) {
    OS << " (";
    bool First = true;
    if (Raw.NumVOPD || Scaled.NumVOPD) {
      OS << formatv("VOPD:{0}/{1}", Raw.NumVOPD, Scaled.NumVOPD);
      First = false;
    }
    if (Raw.NumPacked || Scaled.NumPacked) {
      if (!First) OS << "+";
      OS << formatv("PK:{0}/{1}", Raw.NumPacked, Scaled.NumPacked);
    }
    OS << ")";
  }
  OS << formatv(" | SALU: {0}/{1} | TRANS: {2}/{3} | WMMA: {4}/{5}\n",
                Raw.NumSALU, Scaled.NumSALU,
                Raw.NumTRANS, Scaled.NumTRANS, Raw.NumWMMA, Scaled.NumWMMA);
  OS << formatv(";   DS_RD: {0}/{1} | DS_WR: {2}/{3} | VMEM: {4}/{5} | TDM: {6}/{7}\n",
                Raw.NumDSRead, Scaled.NumDSRead, Raw.NumDSWrite, Scaled.NumDSWrite,
                Raw.NumVMEM, Scaled.NumVMEM, Raw.NumTDM, Scaled.NumTDM);
  if (Raw.NumSpill || Raw.NumReload || Scaled.NumSpill || Scaled.NumReload) {
    OS << formatv(";   Spill: {0}/{1} | Reload: {2}/{3}\n",
                  Raw.NumSpill, Scaled.NumSpill, Raw.NumReload, Scaled.NumReload);
  }
  if (Raw.NumSGPRToVGPR || Raw.NumVGPRToSGPR) {
    OS << formatv(";   SGPR->Lane: {0}/{1} | Lane->SGPR: {2}/{3}\n",
                  Raw.NumSGPRToVGPR, Scaled.NumSGPRToVGPR,
                  Raw.NumVGPRToSGPR, Scaled.NumVGPRToSGPR);
  }
  if (Raw.NumDelayAlu || Scaled.NumDelayAlu) {
    OS << formatv(";   delay_alu: {0}/{1} | MSB_set: {2}/{3} (exposed: {4}/{5})\n",
                  Raw.NumDelayAlu, Scaled.NumDelayAlu,
                  Raw.NumMSBSet, Scaled.NumMSBSet,
                  Raw.NumMSBSetExposed, Scaled.NumMSBSetExposed);
  }
  OS << ";\n";

  // === VGPR Operand Cache ===
  unsigned RawTotal = Raw.VGPRCacheHits + Raw.VGPRCacheMisses;
  unsigned ScaledTotal = Scaled.VGPRCacheHits + Scaled.VGPRCacheMisses;
  if (RawTotal > 0 || ScaledTotal > 0) {
    OS << "; === VGPR Operand Cache ===\n";
    OS << formatv(";   VGPR reads: {0}/{1} | From cache: {2}/{3}",
                  RawTotal, ScaledTotal,
                  Raw.VGPRCacheHits, Scaled.VGPRCacheHits);
    if (ScaledTotal > 0) {
      OS << formatv(" ({0:F0}%)", Scaled.VGPRCacheHitRate() * 100.0f);
    }
    OS << "\n";
    if (Raw.VGPRCacheEvictions > 0 || Scaled.VGPRCacheEvictions > 0) {
      OS << formatv(";   Evictions: {0}/{1}\n",
                    Raw.VGPRCacheEvictions, Scaled.VGPRCacheEvictions);
    }
    OS << ";\n";
  }

  // === CFG Analysis ===
  if (NumLoops > 0 || NumBranches > 0) {
    OS << "; === CFG Analysis ===\n";
    if (NumLoops > 0) {
      OS << formatv(";   Loops: {0} | Max depth: {1} | Trip count: {2}\n",
                    NumLoops, MaxLoopDepth, MaxTripCount);
      OS << formatv(";   Cold: {0} cycles | Warm: {1} cycles",
                    ColdTotal.TotalCycles, WarmTotal.TotalCycles);
      if (ColdTotal.TotalCycles > 0 && WarmTotal.TotalCycles > 0) {
        float Speedup = static_cast<float>(ColdTotal.TotalCycles) / WarmTotal.TotalCycles;
        OS << formatv(" | Speedup: {0:F2}x", Speedup);
      }
      OS << "\n";
    }
    if (NumBranches > 0) {
      OS << formatv(";   Branches: {0} (scaled metrics use uniform probability)\n",
                    NumBranches);
    }
    OS << ";\n";
  }

  // === Derived Metrics ===
  OS << "; === Derived Metrics ===\n";
  OS << formatv(";   IPC: {0:F2} | Stall ratio: {1:F1}%\n", IPC, StallRatio * 100.0f);
  if (Scaled.NumWaitcnt > 0) {
    float AvgFalsePerWait = static_cast<float>(Scaled.NumFalseWaits) / Scaled.NumWaitcnt;
    OS << formatv(";   False wait ratio: {0:F2} per waitcnt\n", AvgFalsePerWait);
  }
  OS << ";\n";

  OS << "; ============================================================\n";
}

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
    MachineLoopInfo &MLI = getAnalysis<MachineLoopInfoWrapperPass>().getLI();
    MachineBlockFrequencyInfo &MBFI =
        getAnalysis<MachineBlockFrequencyInfoWrapperPass>().getMBFI();
    runStaticSimulator(MF, &MLI, &MBFI);
    return false; // Does not modify the function
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

