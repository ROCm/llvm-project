//===-- TraceUtil.cpp - Trace processing utilities ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TraceUtil.h"
#include "AMDGPUSim/AMDGPUSim.h"
#include "AMDGPUSim/MCAdapter.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
namespace tracecp {

WaveView::WaveView(ArrayRef<InstEntry> Entries) {
  auto It = Entries.begin();
  while (It != Entries.end()) {
    unsigned WaveId = It->WaveId;
    auto IsInstructonFromSameWave = [WaveId](const InstEntry &Entry) {
      return Entry.WaveId != WaveId;
    };
    auto End = std::find_if(It + 1, Entries.end(), IsInstructonFromSameWave);
    WaveToEntries[WaveId] = ArrayRef<InstEntry>(It, End);
    It = End;
  }
}

using namespace AMDGPUSim;
using AMDGPU::BlockMetrics;

static bool parseHexBytes(StringRef HexStr, SmallVectorImpl<uint8_t> &Bytes) {
  SmallVector<StringRef, 8> HexParts;
  HexStr.split(HexParts, ' ', -1, false);

  for (StringRef Part : HexParts) {
    if (Part.size() % 2 != 0)
      return false;

    SmallVector<uint8_t, 4> WordBytes;
    for (size_t I = 0; I < Part.size(); I += 2) {
      unsigned ByteVal;
      if (Part.substr(I, 2).getAsInteger(16, ByteVal))
        return false;
      WordBytes.push_back(static_cast<uint8_t>(ByteVal));
    }
    // Reverse for little-endian
    append_range(Bytes, reverse(WordBytes));
  }
  return true;
}

Expected<std::vector<InstEntry>> parseAndDisassemble(StringRef FilePath,
                                                      const TraceFilter &Filter,
                                                      MCDisassembler &DisAsm) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> FileOrErr =
      MemoryBuffer::getFile(FilePath);
  if (std::error_code EC = FileOrErr.getError())
    return createStringError(EC, "Error opening file '%s': %s",
                             FilePath.str().c_str(), EC.message().c_str());

  std::vector<InstEntry> Results;
  StringRef Content = FileOrErr.get()->getBuffer();
  SmallVector<StringRef, 0> Lines;
  Content.split(Lines, '\n');

  for (size_t LineNum = 0; LineNum < Lines.size(); ++LineNum) {
    StringRef Line = Lines[LineNum].trim();
    if (Line.empty())
      continue;

    Expected<json::Value> ParseResult = json::parse(Line);
    if (!ParseResult)
      return createStringError(inconvertibleErrorCode(),
                               "Line %zu: JSON parse error: %s", LineNum + 1,
                               toString(ParseResult.takeError()).c_str());

    json::Object *Obj = ParseResult->getAsObject();
    if (!Obj)
      continue;

    std::optional<StringRef> InstructionText =
        Obj->getString("instruction_text");
    if (!InstructionText)
      continue;

    // Apply filters
    int64_t DispatchId;
    int64_t ClusterId;
    int64_t WorkgroupId;
    int64_t WaveId;
    const char *EntryLabels[] = {"dispatch_id", "cluster_id", "workgroup_id",
                                 "wave_id"};
    int64_t *EntryValues[] = {&DispatchId, &ClusterId, &WorkgroupId, &WaveId};

    for (auto [Label, Value] : zip(EntryLabels, EntryValues)) {
      std::optional<int64_t> EntryValue = Obj->getInteger(Label);
      if (!EntryValue)
        return createStringError(inconvertibleErrorCode(),
                                 "Line %zu: Missing \"%s\"", LineNum + 1,
                                 Label);

      *Value = *EntryValue;
    }

    if (!Filter.match(DispatchId, ClusterId, WorkgroupId, WaveId))
      continue;

    InstEntry Entry;
    Entry.DispatchId = DispatchId;
    Entry.ClusterId = ClusterId;
    Entry.WorkgroupId = WorkgroupId;
    Entry.WavegroupId = Obj->getInteger("wavegroup_id").value_or(0);
    Entry.WaveId = WaveId;
    Entry.InstructionId = Obj->getInteger("instruction_id").value_or(0);
    Entry.PC = static_cast<uint64_t>(Obj->getInteger("pc").value_or(0));
    Entry.Opcode = Obj->getInteger("opcode").value_or(0);
    Entry.InstructionText = InstructionText->trim().str();

    // Parse the hex encoding from the comment section
    // Format: "... // ADDRESS: HEXBYTES\n"
    size_t CommentPos = InstructionText->find("//");
    if (CommentPos == StringRef::npos)
      return createStringError(
          inconvertibleErrorCode(),
          "Line %zu: No comment section found in instruction: %s", LineNum + 1,
          Entry.InstructionText.c_str());

    StringRef Comment = InstructionText->substr(CommentPos + 2).trim();
    size_t ColonPos = Comment.find(':');
    if (ColonPos == StringRef::npos)
      return createStringError(inconvertibleErrorCode(),
                               "Line %zu: No colon found in comment: %s",
                               LineNum + 1, Entry.InstructionText.c_str());

    StringRef HexStr = Comment.substr(ColonPos + 1).trim();
    SmallVector<uint8_t, 16> Bytes;
    if (!parseHexBytes(HexStr, Bytes))
      return createStringError(inconvertibleErrorCode(),
                               "Line %zu: Failed to parse hex bytes: %s",
                               LineNum + 1, HexStr.str().c_str());

    MCDisassembler::DecodeStatus Status =
        DisAsm.getInstruction(Entry.Inst, Entry.InstSize, Bytes, 0, nulls());
    if (Status != MCDisassembler::Success)
      return createStringError(
          inconvertibleErrorCode(),
          "Line %zu: Disassembly failed for instruction: %s", LineNum + 1,
          Entry.InstructionText.c_str());

    Results.push_back(std::move(Entry));
  }

  return Results;
}

/// Count an instruction into BlockMetrics (adapted from AMDGPUStaticSimulator).
static void countInstruction(const SimInst &SI, MCInstInfo &InstInfo,
                             BlockMetrics &Metrics) {
  Metrics.NumInstructions++;

  switch (SI.Class) {
  case InstClass::VALU:
    Metrics.NumVALU++;
    break;
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
  case InstClass::NOP:
    Metrics.NumNop++;
    break;
  default:
    break;
  }
}

/// Attribute stall cycles to BlockMetrics (adapted from AMDGPUStaticSimulator).
static void attributeStall(const StallBreakdown &B, FunctionalUnit Unit,
                           InstClass IC, BlockMetrics &Metrics) {
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
  }
}

/// Log instruction header in pass-like format.
static void logInstHeader(unsigned Cycle, const SimInst &SI,
                          MCInstInfo &InstInfo, StringRef InstructionText) {
  errs() << "\n[Cycle " << Cycle << "] " << InstructionText << "\n";

  unsigned InstBytes = InstInfo.getInstBytes(SI);
  errs() << "  Class: " << getInstClassName(SI.Class)
         << " | Unit: " << getUnitName(SI.Unit) << " | Latency: " << SI.Latency
         << " | ResourceCycles: " << InstInfo.getResourceCycles(SI)
         << " | Size: " << InstBytes << " bytes\n";
}

/// Simulate a single instruction and update BlockMetrics.
static void simulateInst(const MCInst &Inst, size_t EntryIdx,
                         const std::vector<InstEntry> &Entries,
                         Simulator &Sim, MCInstInfo &InstInfo,
                         BlockMetrics &Metrics, bool Verbose,
                         StringRef InstructionText) {
  const GPUSimState &State = Sim.getState();
  unsigned EntryCycle = State.CurrentCycle;
  SimInst SI = InstInfo.createSimInst(Inst);

  // Build lookahead for MSB_SET masking
  SmallVector<SimInst, 1> Lookahead;
  if (SI.Class == InstClass::MSB_SET && EntryIdx + 1 < Entries.size()) {
    Lookahead.push_back(InstInfo.createSimInst(Entries[EntryIdx + 1].Inst));
  }

  // MSB_SET handling
  if (SI.Class == InstClass::MSB_SET) {
    InstrSimInfo Info = Sim.simulateInst(SI, Lookahead);

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

    if (Verbose) {
      unsigned DisplayCycle =
          Info.WasFused ? (EntryCycle > 0 ? EntryCycle - 1 : 0) : EntryCycle;
      errs() << "\n[Cycle " << DisplayCycle << "] " << InstructionText << "\n";
      unsigned InstBytes = InstInfo.getInstBytes(SI);
      errs() << "  Class: MSB_SET | Size: " << InstBytes << " bytes\n";
      errs() << "  \xe2\x86\x92 MSB_SET ";
      if (Info.WasFused) {
        errs() << "fused with prev (free)";
      } else if (Info.WasMasked) {
        errs() << "exposed but MASKED (next instr stalls anyway)";
      } else {
        errs() << "EXPOSED (+1 cycle)";
        if (Sim.getState().inWMMAWindow())
          errs() << " [in WMMA window]";
      }
      errs() << "\n";
    }
    return;
  }

  // Regular instruction: log header before simulation
  if (Verbose)
    logInstHeader(EntryCycle, SI, InstInfo, InstructionText);

  // Core simulation
  InstrSimInfo Info = Sim.simulateInst(SI);

  // Attribute stalls
  attributeStall(Info.Breakdown, SI.Unit, SI.Class, Metrics);

  // Count instruction
  countInstruction(SI, InstInfo, Metrics);

  // WMMA window tracking
  if (Info.IsWMMA) {
    Metrics.WMMAWindowCycles += Sim.getState().ActiveWMMA.Info.Occupancy;
  }

  // WMMA co-execution tracking
  bool InWMMAWindow =
      Sim.getState().inWMMAWindow() && SI.Class != InstClass::WMMA;
  if (InWMMAWindow) {
    if (Info.Breakdown.CoExec > 0)
      Metrics.WMMACoExecBlocked++;
    else
      Metrics.WMMACoExecUsed++;
  }

  // Log next cycle (stall breakdown already printed by Simulator)
  if (Verbose) {
    errs() << "  \xe2\x86\x92 NextCycle: " << Sim.getState().CurrentCycle << "\n";
  }
}

TraceMetrics simulateTrace(const std::vector<InstEntry> &Entries,
                              const TraceCFG &CFG, const MCInstrInfo &MCII,
                              const MCRegisterInfo &MRI, bool Verbose) {
  TraceMetrics Result;

  if (Entries.empty())
    return Result;

  MCInstInfo InstInfo(MCII, MRI);
  HWModel Model = createHWModel(GPUTarget::GFX1250);

  SimulatorConfig Cfg;
  Cfg.Log = Verbose ? &errs() : nullptr;
  Cfg.EnableScoreboard = true;  // Enable RAW hazard detection
  Cfg.EnableISCache = true;     // Enable IS cache modeling

  Simulator Sim(InstInfo, Model, Cfg);

  // Build PC -> block mapping
  std::map<uint64_t, uint64_t> PCToBlock;
  for (const auto &E : Entries) {
    auto It = CFG.Blocks.upper_bound(E.PC);
    if (It != CFG.Blocks.begin()) {
      --It;
      if (E.PC >= It->second.StartPC && E.PC <= It->second.EndPC)
        PCToBlock[E.PC] = It->first;
    }
  }

  // Track current block and its metrics
  uint64_t CurrentBlockPC = 0;
  BlockMetrics CurrentMetrics;
  unsigned BlockStartCycle = Sim.getState().CurrentCycle;
  bool InBlock = false;

  for (size_t EntryIdx = 0; EntryIdx < Entries.size(); ++EntryIdx) {
    const InstEntry &Entry = Entries[EntryIdx];

    // Find which block this instruction belongs to
    auto BlockIt = PCToBlock.find(Entry.PC);
    uint64_t BlockPC = (BlockIt != PCToBlock.end()) ? BlockIt->second : Entry.PC;

    // Check if we're starting a new block execution.
    // This happens when:
    // 1. We haven't started any block yet (!InBlock)
    // 2. We moved to a different block (BlockPC != CurrentBlockPC)
    // 3. We're at the start of the same block again (loop back-edge)
    bool IsBlockStart = (Entry.PC == BlockPC);
    bool StartNewBlock = !InBlock || BlockPC != CurrentBlockPC ||
                         (IsBlockStart && InBlock);

    if (StartNewBlock) {
      // Finish previous block if we had one
      if (InBlock) {
        CurrentMetrics.TotalCycles =
            Sim.getState().CurrentCycle - BlockStartCycle;
        Result.Blocks[CurrentBlockPC].push_back(CurrentMetrics);

        if (Verbose) {
          errs() << format("\n=== End Block 0x%04x: %u instrs, %u cycles, "
                           "%u stalls ===\n",
                           CurrentBlockPC, CurrentMetrics.NumInstructions,
                           CurrentMetrics.TotalCycles,
                           CurrentMetrics.StallCycles());
        }
      }

      // Start new block
      CurrentBlockPC = BlockPC;
      CurrentMetrics = BlockMetrics();
      BlockStartCycle = Sim.getState().CurrentCycle;
      InBlock = true;

      if (Verbose) {
        errs() << format("\n=== Block 0x%04x [Cycle %u] ===",
                         CurrentBlockPC, BlockStartCycle);
      }
    }

    // Simulate this instruction
    simulateInst(Entry.Inst, EntryIdx, Entries, Sim, InstInfo, CurrentMetrics,
                 Verbose, Entry.InstructionText);
  }

  // Finish last block
  if (InBlock) {
    CurrentMetrics.TotalCycles = Sim.getState().CurrentCycle - BlockStartCycle;
    Result.Blocks[CurrentBlockPC].push_back(CurrentMetrics);

    if (Verbose) {
      errs() << format("\n=== End Block 0x%04x: %u instrs, %u cycles, "
                       "%u stalls ===\n",
                       CurrentBlockPC, CurrentMetrics.NumInstructions,
                       CurrentMetrics.TotalCycles,
                       CurrentMetrics.StallCycles());
    }
  }

  return Result;
}

/// Helper to print a BlockMetrics in the same format as AMDGPUAsmPrinter.
static void printBlockMetricsLLVMStyle(raw_ostream &OS, const BlockMetrics &M,
                                       const char *BlockName) {
  // Header: ;=== Block: N cycles ===
  OS << ";=== " << BlockName << ": " << M.TotalCycles << " cycles ===\n";

  // Instruction breakdown
  OS << ";  ";
  M.printInstBreakdown(OS);
  OS << "\n";

  // Stall summary and breakdown
  if (M.StallCycles() > 0) {
    float StallPct = M.TotalCycles > 0
                         ? 100.0f * M.StallCycles() / M.TotalCycles
                         : 0.0f;
    OS << format(";  Stall: %u cycles (%.0f%%)\n", M.StallCycles(), StallPct);
    OS << ";    ";
    M.printStallBreakdown(OS);
    OS << "\n";

    if (M.StallFunctionalUnit > 0) {
      OS << ";      FU: ";
      M.printFUBreakdown(OS);
      OS << "\n";
    }
  }

  // WMMA efficiency if applicable
  if (M.WMMAWindowCycles > 0 && M.TotalCycles > 0) {
    float WMMAEff = 100.0f * M.WMMAWindowCycles / M.TotalCycles;
    OS << format(";  WMMA efficiency: %u / %u cycles (%.0f%%)\n",
                 M.WMMAWindowCycles, M.TotalCycles, WMMAEff);
  }
}

void TraceMetrics::print() const {
  outs() << "\n";
  outs() << "; ============================================================\n";
  outs() << "; TRACE SIMULATION METRICS\n";
  outs() << "; ============================================================\n";
  outs() << ";\n";

  // Print first execution of each block using LLVM-style format
  unsigned BlockIdx = 0;
  for (const auto &[BlockPC, Executions] : Blocks) {
    if (Executions.empty())
      continue;

    // Only print the first execution
    const BlockMetrics &BM = Executions[0];
    std::string BlockName = formatv("Block {0} (PC {1:X})", BlockIdx, BlockPC);
    printBlockMetricsLLVMStyle(outs(), BM, BlockName.c_str());

    // For blocks with multiple executions, show average cycles and stalls
    if (Executions.size() > 1) {
      unsigned TotalCycles = 0, TotalStalls = 0;
      for (const auto &M : Executions) {
        TotalCycles += M.TotalCycles;
        TotalStalls += M.StallCycles();
      }
      float AvgCycles = static_cast<float>(TotalCycles) / Executions.size();
      float AvgStalls = static_cast<float>(TotalStalls) / Executions.size();
      outs() << format(";  Avg over %zu executions: %.1f cycles, %.1f stalls\n",
                       Executions.size(), AvgCycles, AvgStalls);
    }

    outs() << ";\n";
    BlockIdx++;
  }

  // Aggregate all block executions into total
  BlockMetrics Total;
  unsigned TotalExecutions = 0;

  for (const auto &[BlockPC, Executions] : Blocks) {
    for (const auto &BM : Executions) {
      Total = Total + BM;
      TotalExecutions++;
    }
  }

  // Print aggregated total using same format
  outs() << "; ============================================================\n";
  printBlockMetricsLLVMStyle(outs(), Total, "TOTAL");

  // Additional summary info
  float IPC = Total.TotalCycles > 0
                  ? static_cast<float>(Total.NumInstructions) / Total.TotalCycles
                  : 0.0f;
  outs() << format(";  IPC: %.2f | Block Executions: %u\n", IPC, TotalExecutions);

  if (Total.WMMAWindowCycles > 0) {
    float CoExecEff = 100.0f * Total.WMMACoExecUsed / Total.WMMAWindowCycles;
    outs() << format(";  WMMA Co-exec: %u used / %u window cycles (%.1f%% utilized)\n",
                     Total.WMMACoExecUsed, Total.WMMAWindowCycles, CoExecEff);
  }

  outs() << "; ============================================================\n";
}

TraceCFG reconstructCFG(const WaveView &WaveView, const MCInstrInfo &MCII) {
  TraceCFG CFG;

  // Collect all block start PCs.
  // A PC is a block start if:
  // 1. It's the first instruction
  // 2. It's the target of a non-sequential transition (branch target)
  // 3. It follows a branch/terminator instruction (even if fall-through)
  DenseSet<uint64_t> BlockStartPCs;

  for (const auto &[_, Entries] : WaveView.entries_per_wave()) {
    bool LastInstMayAffectControlFlow = true;
    for (const InstEntry &E : Entries) {
      if (LastInstMayAffectControlFlow)
        BlockStartPCs.insert(E.PC);

      const MCInstrDesc &Desc = MCII.get(E.Inst.getOpcode());
      LastInstMayAffectControlFlow = Desc.isBranch() || Desc.isTerminator() ||
                                     Desc.isCall() || Desc.isReturn() ||
                                     Desc.isIndirectBranch();
    }
  }

  // Build blocks using known block starts.
  uint64_t CurrentBlockStart;
  uint64_t CurrentBlockEnd;
  unsigned CurrentBlockInstCount;

  // Track edge counts: (FromBlockPC, FromPC, ToPC) -> count
  std::map<std::tuple<uint64_t, uint64_t, uint64_t>, unsigned> EdgeCounts;

  auto updateBlocks = [&CFG](
      uint64_t BlockStart, uint64_t BlockEnd, uint64_t BlockInstCount) {
    auto It = CFG.Blocks.find(BlockStart);
    if (It != CFG.Blocks.end()) {
      assert(It->second.EndPC == BlockEnd && "Block end PC mismatch");
      It->second.ExecutionCount++;
    } else {
      TraceBlock BB;
      BB.StartPC = BlockStart;
      BB.EndPC = BlockEnd;
      BB.NumInstructions = BlockInstCount;
      BB.ExecutionCount = 1;
      CFG.Blocks[BlockStart] = BB;
    }
  };

  for (const auto &[_, Entries] : WaveView.entries_per_wave()) {
    const InstEntry &FirstEntry = Entries.front();
    CurrentBlockStart = FirstEntry.PC;
    CurrentBlockEnd = FirstEntry.PC + FirstEntry.InstSize;
    CurrentBlockInstCount = 1;

    for (const InstEntry &E : drop_begin(Entries)) {
      // Check if this instruction starts a new block
      if (BlockStartPCs.contains(E.PC)) {
        updateBlocks(CurrentBlockStart, CurrentBlockEnd, CurrentBlockInstCount);

        // Record edge from previous block to this one
        EdgeCounts[{CurrentBlockStart, CurrentBlockEnd, E.PC}]++;

        // Start new block
        CurrentBlockStart = E.PC;
        CurrentBlockInstCount = 0;
      }
      CurrentBlockEnd = E.PC + E.InstSize;
      CurrentBlockInstCount++;
    }

    // Handle last block
    if (CurrentBlockInstCount > 0) {
      updateBlocks(CurrentBlockStart, CurrentBlockEnd, CurrentBlockInstCount);
    }
  }

  // Convert edge counts to TraceEdge vector
  for (const auto &[Edge, Count] : EdgeCounts) {
    TraceEdge E;
    E.FromBlockPC = std::get<0>(Edge);
    E.FromPC = std::get<1>(Edge);
    E.ToPC = std::get<2>(Edge);
    E.Count = Count;
    CFG.Edges.push_back(E);
  }

  return CFG;
}

void TraceCFG::print() const {
  outs() << "\n";
  outs() << "============================================================\n";
  outs() << "CONTROL FLOW GRAPH\n";
  outs() << "============================================================\n";
  outs() << "\n";

  outs() << format("Basic Blocks: %zu\n", Blocks.size());
  outs() << format("Edges: %zu\n", Edges.size());
  outs() << "\n";

  outs() << "=== Basic Blocks ===\n";
  for (const auto &[StartPC, BB] : Blocks) {
    outs() << format("  [0x%04x - 0x%04x] %u instrs, executed %u time(s)\n",
                     BB.StartPC, BB.EndPC, BB.NumInstructions,
                     BB.ExecutionCount);
  }
  outs() << "\n";

  outs() << "=== Edges ===\n";
  for (const TraceEdge &E : Edges) {
    outs() << format("  [0x%04x] 0x%04x -> 0x%04x (%u time(s))\n", E.FromBlockPC,
                     E.FromPC, E.ToPC, E.Count);
  }
  outs() << "\n";

  outs() << "============================================================\n";
}

} // namespace tracecp
} // namespace llvm
