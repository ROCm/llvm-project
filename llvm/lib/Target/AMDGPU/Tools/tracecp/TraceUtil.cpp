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
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include <set>
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
namespace tracecp {

using namespace AMDGPUSim;

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
    for (auto It = WordBytes.rbegin(); It != WordBytes.rend(); ++It)
      Bytes.push_back(*It);
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
    int64_t DispatchId = Obj->getInteger("dispatch_id").value_or(0);
    int64_t ClusterId = Obj->getInteger("cluster_id").value_or(0);
    int64_t WorkgroupId = Obj->getInteger("workgroup_id").value_or(0);
    int64_t WaveId = Obj->getInteger("wave_id").value_or(0);

    if (DispatchId != Filter.DispatchId || ClusterId != Filter.ClusterId ||
        WorkgroupId != Filter.WorkgroupId || WaveId != Filter.WaveId)
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

    ArrayRef<uint8_t> BytesRef(Bytes);
    MCDisassembler::DecodeStatus Status =
        DisAsm.getInstruction(Entry.Inst, Entry.InstSize, BytesRef, 0, nulls());
    if (Status != MCDisassembler::Success)
      return createStringError(
          inconvertibleErrorCode(),
          "Line %zu: Disassembly failed for instruction: %s", LineNum + 1,
          Entry.InstructionText.c_str());

    Results.push_back(std::move(Entry));
  }

  return Results;
}

void simulateTrace(std::vector<InstEntry> &Entries, const MCInstrInfo &MCII,
                   const MCRegisterInfo &MRI, bool Verbose) {
  MCInstInfo InstInfo(MCII, MRI);
  HWModel Model = createHWModel(GPUTarget::GFX1250);

  SimulatorConfig Cfg;
  Cfg.Verbose = Verbose;
  Cfg.Log = Verbose ? &errs() : nullptr;
  Cfg.EnableScoreboard = false;
  Cfg.EnableISCache = false;

  Simulator Sim(InstInfo, Model, Cfg);
  unsigned PrevCycle = Sim.getState().CurrentCycle;

  for (InstEntry &Entry : Entries) {
    SimInst SI = InstInfo.createSimInst(Entry.Inst);
    InstrSimInfo Info = Sim.simulateInst(SI);
    unsigned CurrentCycle = Sim.getState().CurrentCycle;

    Entry.InstClass = static_cast<unsigned>(SI.Class);
    Entry.Cycles = CurrentCycle - PrevCycle;
    Entry.StallCycles = Info.StallCycles;
    Entry.StallReason = static_cast<unsigned>(Info.Reason);
    PrevCycle = CurrentCycle;

    if (Verbose) {
      errs() << "[" << Sim.getState().CurrentCycle << "] "
             << Entry.InstructionText << "\n";
      if (Info.StallCycles > 0) {
        errs() << "  Stall: " << Info.StallCycles << " ("
               << Info.getReasonString() << ")\n";
      }
      errs() << "\n";
    }
  }
}

TraceMetrics::TraceMetrics(const std::vector<InstEntry> &Entries) {
  for (const auto &Entry : Entries)
    addInstruction(Entry);
}

void TraceMetrics::print() const {
  outs() << "\n";
  outs() << "============================================================\n";
  outs() << "TRACE SIMULATION METRICS\n";
  outs() << "============================================================\n";
  outs() << "\n";

  printBody();

  outs() << "============================================================\n";
}

void TraceMetrics::printBody(unsigned Iterations) const {
  outs() << "=== Summary ===\n";
  outs() << format("  Instructions: %u\n", NumInstructions);
  outs() << format("  Total Cycles: %u\n", TotalCycles);
  outs() << format("  Stall Cycles: %u (%.1f%%)\n", StallCycles,
                   getStallRatio() * 100.0f);
  outs() << format("  IPC: %.2f\n", getIPC());
  outs() << "\n";

  outs() << "=== Instruction Breakdown ===\n";
  outs() << format("  VALU: %u | SALU: %u | TRANS: %u | WMMA: %u\n", NumVALU,
                   NumSALU, NumTRANS, NumWMMA);
  outs() << format("  DS_RD: %u | DS_WR: %u | VMEM: %u | SMEM: %u\n", NumDSRead,
                   NumDSWrite, NumVMEM, NumSMEM);
  outs() << format("  Branch: %u | Waitcnt: %u | Other: %u\n", NumBranch,
                   NumWaitcnt, NumOther);
  outs() << "\n";

  if (StallCycles > 0) {
    outs() << "=== Stall Breakdown ===\n";
    if (StallFU > 0)
      outs() << format("  FU Busy: %u\n", StallFU);
    if (StallCoExec > 0)
      outs() << format("  WMMA CoExec: %u\n", StallCoExec);
    if (StallDelayAlu > 0)
      outs() << format("  DelayAlu: %u\n", StallDelayAlu);
    if (StallWaitCnt > 0)
      outs() << format("  WaitCnt: %u\n", StallWaitCnt);
    if (StallMemFIFO > 0)
      outs() << format("  MemFIFO: %u\n", StallMemFIFO);
    if (StallRegBank > 0)
      outs() << format("  RegBank: %u\n", StallRegBank);
    if (StallRAW > 0)
      outs() << format("  RAW: %u\n", StallRAW);
    if (StallTrans > 0)
      outs() << format("  Trans Hazard: %u\n", StallTrans);
    if (StallVASSrc > 0)
      outs() << format("  VA SSrc: %u\n", StallVASSrc);
    if (StallVAVDst > 0)
      outs() << format("  VA VDst: %u\n", StallVAVDst);
    if (StallISFetch > 0)
      outs() << format("  IS Fetch: %u\n", StallISFetch);
    if (StallMSBExposed > 0)
      outs() << format("  MSB Exposed: %u\n", StallMSBExposed);
    if (StallOther > 0)
      outs() << format("  Other: %u\n", StallOther);
    outs() << "\n";
  }

  if (Iterations > 0) {
    outs() << "=== Per Iteration (average) ===\n";
    float N = static_cast<float>(Iterations);
    outs() << format("  Instructions: %.1f\n", NumInstructions / N);
    outs() << format("  Total Cycles: %.1f\n", TotalCycles / N);
    outs() << format("  Stall Cycles: %.1f\n", StallCycles / N);
    outs() << format("  VALU: %.1f | SALU: %.1f | TRANS: %.1f | WMMA: %.1f\n",
                     NumVALU / N, NumSALU / N, NumTRANS / N, NumWMMA / N);
    outs() << format("  DS_RD: %.1f | DS_WR: %.1f | VMEM: %.1f | SMEM: %.1f\n",
                     NumDSRead / N, NumDSWrite / N, NumVMEM / N, NumSMEM / N);
    outs() << "\n";
  }
}

void TraceMetrics::addInstruction(const InstEntry &E) {
  NumInstructions++;
  TotalCycles += E.Cycles;

  // Count instruction class
  switch (static_cast<InstClass>(E.InstClass)) {
  case InstClass::VALU:
    NumVALU++;
    break;
  case InstClass::SALU:
    NumSALU++;
    break;
  case InstClass::TRANS:
    NumTRANS++;
    break;
  case InstClass::WMMA:
    NumWMMA++;
    break;
  case InstClass::DS_READ:
    NumDSRead++;
    break;
  case InstClass::DS_WRITE:
    NumDSWrite++;
    break;
  case InstClass::VMEM_READ:
  case InstClass::VMEM_WRITE:
    NumVMEM++;
    break;
  case InstClass::SMEM:
    NumSMEM++;
    break;
  case InstClass::BRANCH:
    NumBranch++;
    break;
  case InstClass::WAITCNT:
    NumWaitcnt++;
    break;
  default:
    NumOther++;
    break;
  }

  // Attribute stall cycles
  if (E.StallCycles == 0)
    return;

  StallCycles += E.StallCycles;

  switch (static_cast<StallReason>(E.StallReason)) {
  case StallReason::FU_BUSY:
    StallFU += E.StallCycles;
    break;
  case StallReason::COEXEC_BLOCKED:
  case StallReason::LONG_LAT_VALU:
    StallCoExec += E.StallCycles;
    break;
  case StallReason::DELAY_ALU:
    StallDelayAlu += E.StallCycles;
    break;
  case StallReason::WAITCNT:
    StallWaitCnt += E.StallCycles;
    break;
  case StallReason::MEM_FIFO:
    StallMemFIFO += E.StallCycles;
    break;
  case StallReason::REG_BANK:
    StallRegBank += E.StallCycles;
    break;
  case StallReason::RAW_HAZARD:
    StallRAW += E.StallCycles;
    break;
  case StallReason::LOLVALU_TRANS_HAZARD:
    StallTrans += E.StallCycles;
    break;
  case StallReason::VA_SSRC_STALL:
    StallVASSrc += E.StallCycles;
    break;
  case StallReason::VA_VDST_WAIT:
    StallVAVDst += E.StallCycles;
    break;
  case StallReason::IS_FETCH:
    StallISFetch += E.StallCycles;
    break;
  case StallReason::MSB_SET_EXPOSED:
    StallMSBExposed += E.StallCycles;
    break;
  case StallReason::NONE:
    StallOther += E.StallCycles;
    break;
  }
}

TraceCFG reconstructCFG(const std::vector<InstEntry> &Entries,
                        const MCInstrInfo &MCII) {
  TraceCFG CFG;

  if (Entries.empty())
    return CFG;

  // Collect all block start PCs.
  // A PC is a block start if:
  // 1. It's the first instruction
  // 2. It's the target of a non-sequential transition (branch target)
  // 3. It follows a branch/terminator instruction (even if fall-through)
  std::set<uint64_t> BlockStartPCs;
  BlockStartPCs.insert(Entries[0].PC);

  for (size_t I = 0; I + 1 < Entries.size(); ++I) {
    const InstEntry &E = Entries[I];
    uint64_t NextPC = Entries[I + 1].PC;

    if (NextPC != E.PC + E.InstSize) {
      BlockStartPCs.insert(NextPC);
    }

    const MCInstrDesc &Desc = MCII.get(E.Inst.getOpcode());
    if (Desc.isBranch() || Desc.isTerminator()) {
      BlockStartPCs.insert(NextPC);
    }
  }

  // Build blocks using known block starts.
  uint64_t CurrentBlockStart = Entries[0].PC;
  uint64_t CurrentBlockEnd = Entries[0].PC;
  unsigned CurrentBlockInstCount = 0;

  // Track edge counts: (FromBlockPC, FromPC, ToPC) -> count
  std::map<std::tuple<uint64_t, uint64_t, uint64_t>, unsigned> EdgeCounts;

  auto updateBlocks = [&CFG](
      uint64_t BlockStart, uint64_t BlockEnd, uint64_t BlockInstCount) {
    auto It = CFG.Blocks.find(BlockStart);
    if (It != CFG.Blocks.end()) {
      assert(It->second.EndPC == BlockEnd && "Block end PC mismatch");
      It->second.ExecutionCount++;
    } else {
      BasicBlock BB;
      BB.StartPC = BlockStart;
      BB.EndPC = BlockEnd;
      BB.NumInstructions = BlockInstCount;
      BB.ExecutionCount = 1;
      CFG.Blocks[BlockStart] = BB;
    }
  };

  for (size_t I = 0; I < Entries.size(); ++I) {
    const InstEntry &E = Entries[I];

    // Check if this instruction starts a new block
    if (I > 0 && BlockStartPCs.count(E.PC)) {
      updateBlocks(CurrentBlockStart, CurrentBlockEnd, CurrentBlockInstCount);

      // Record edge from previous block to this one
      EdgeCounts[{CurrentBlockStart, CurrentBlockEnd, E.PC}]++;

      // Start new block
      CurrentBlockStart = E.PC;
      CurrentBlockInstCount = 0;
    }

    CurrentBlockEnd = E.PC;
    CurrentBlockInstCount++;
  }

  // Handle last block
  if (CurrentBlockInstCount > 0) {
    updateBlocks(CurrentBlockStart, CurrentBlockEnd, CurrentBlockInstCount);
  }

  // Convert edge counts to CFGEdge vector
  for (const auto &[Edge, Count] : EdgeCounts) {
    CFGEdge E;
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
  for (const CFGEdge &E : Edges) {
    outs() << format("  [0x%04x] 0x%04x -> 0x%04x (%u time(s))\n", E.FromBlockPC,
                     E.FromPC, E.ToPC, E.Count);
  }
  outs() << "\n";

  outs() << "============================================================\n";
}

/// Check if A dominates B using the dominator map.
static bool dominates(uint64_t A, uint64_t B,
                      const std::map<uint64_t, uint64_t> &Doms) {
  // Walk up the dominator tree from B
  uint64_t Current = B;
  while (true) {
    if (Current == A)
      return true;
    auto It = Doms.find(Current);
    if (It == Doms.end() || It->second == Current)
      return false;
    Current = It->second;
  }
}

LoopInfo detectLoops(const TraceCFG &CFG,
                     const std::vector<InstEntry> &Entries) {
  LoopInfo LI;

  if (CFG.Blocks.empty())
    return LI;

  // Build PC -> block mapping for fast lookup
  std::map<uint64_t, uint64_t> PCToBlock;  // PC -> BlockStartPC
  for (const auto &E : Entries) {
    auto It = CFG.Blocks.upper_bound(E.PC);
    assert(It != CFG.Blocks.begin() && "PC before first block");
    --It;
    assert(E.PC >= It->second.StartPC && E.PC <= It->second.EndPC &&
           "PC not within block bounds");
    PCToBlock[E.PC] = It->first;
  }

  // Build predecessors and successors lists
  std::map<uint64_t, std::vector<uint64_t>> Preds, Succs;
  for (const auto &[StartPC, BB] : CFG.Blocks) {
    Preds[StartPC] = {};
    Succs[StartPC] = {};
  }
  for (const CFGEdge &E : CFG.Edges) {
    Succs[E.FromBlockPC].push_back(E.ToPC);
    Preds[E.ToPC].push_back(E.FromBlockPC);
  }

  // Find entry block (first block in trace, or block with no predecessors)
  uint64_t EntryPC = CFG.Blocks.begin()->first;

  // Compute dominators using iterative dataflow.
  // Dom[n] = {n} union (intersect Dom[p] for all predecessors p)
  std::map<uint64_t, uint64_t> &Doms = LI.Dominators;

  for (const auto &[StartPC, BB] : CFG.Blocks) {
    Doms[StartPC] = StartPC;
  }
  Doms[EntryPC] = EntryPC; // Entry's dominator is itself

  bool Changed = true;
  while (Changed) {
    Changed = false;
    for (const auto &[StartPC, BB] : CFG.Blocks) {
      if (StartPC == EntryPC)
        continue;

      const auto &PredList = Preds[StartPC];
      if (PredList.empty())
        continue;

      // Find common dominator of all predecessors
      // Skip predecessors that haven't been processed yet (dom == self, except entry)
      uint64_t NewDom = 0;
      bool FoundFirst = false;
      for (uint64_t Pred : PredList) {
        // Skip unprocessed predecessors (except entry)
        if (Pred != EntryPC && Doms[Pred] == Pred)
          continue;

        if (!FoundFirst) {
          NewDom = Pred;
          FoundFirst = true;
          continue;
        }

        // Intersect dominator paths
        uint64_t P = Pred;
        // Walk up both paths to find common ancestor
        std::set<uint64_t> Path1;
        uint64_t N = NewDom;
        while (true) {
          Path1.insert(N);
          if (N == EntryPC || N == Doms[N])
            break;
          N = Doms[N];
        }
        N = P;
        while (Path1.find(N) == Path1.end()) {
          if (N == EntryPC)
            break;
          auto It = Doms.find(N);
          if (It == Doms.end() || It->second == N)
            break;
          N = It->second;
        }
        NewDom = N;
      }

      if (!FoundFirst)
        continue;  // All predecessors unprocessed, skip for now

      if (Doms[StartPC] != NewDom) {
        Doms[StartPC] = NewDom;
        Changed = true;
      }
    }
  }

  // Find back-edges: edge N -> H where H dominates N.
  // Group by header and track edge counts
  std::map<uint64_t, std::vector<std::pair<uint64_t, unsigned>>> BackEdgesByHeader;

  for (const CFGEdge &E : CFG.Edges) {
    // Check if this is a back-edge: target dominates source
    if (dominates(E.ToPC, E.FromBlockPC, Doms)) {
      BackEdgesByHeader[E.ToPC].push_back({E.FromBlockPC, E.Count});
    }
  }

  // Build loops from back-edges
  for (const auto &[HeaderPC, Latches] : BackEdgesByHeader) {
    Loop L;
    L.HeaderPC = HeaderPC;
    L.ParentIdx = -1;
    L.TotalBackEdgeCount = 0;

    // Collect all latches and sum edge counts
    for (const auto &[LatchPC, Count] : Latches) {
      L.LatchPCs.push_back(LatchPC);
      L.TotalBackEdgeCount += Count;
    }

    // Compute loop body: header + all nodes that can reach any latch without
    // going through header
    std::set<uint64_t> Body;
    Body.insert(HeaderPC);

    std::vector<uint64_t> Worklist;
    for (uint64_t LatchPC : L.LatchPCs) {
      if (Body.find(LatchPC) == Body.end())
        Worklist.push_back(LatchPC);
    }

    while (!Worklist.empty()) {
      uint64_t Node = Worklist.back();
      Worklist.pop_back();

      if (Body.find(Node) != Body.end())
        continue;
      Body.insert(Node);

      // Add predecessors (except header, which stops the traversal)
      for (uint64_t Pred : Preds[Node]) {
        if (Pred != HeaderPC && Body.find(Pred) == Body.end())
          Worklist.push_back(Pred);
      }
    }

    L.BodyBlockPCs.assign(Body.begin(), Body.end());

    // Compute metrics for this loop by iterating through trace instructions
    for (const InstEntry &Entry : Entries) {
      auto It = PCToBlock.find(Entry.PC);
      if (It != PCToBlock.end() && Body.count(It->second)) {
        L.Metrics.addInstruction(Entry);
      }
    }

    LI.Loops.push_back(L);
  }

  // Detect nesting: inner loop's header is in outer loop's body
  for (size_t I = 0; I < LI.Loops.size(); ++I) {
    Loop &Inner = LI.Loops[I];
    int BestParent = -1;
    size_t SmallestParentSize = SIZE_MAX;

    for (size_t J = 0; J < LI.Loops.size(); ++J) {
      if (I == J)
        continue;
      const Loop &Outer = LI.Loops[J];

      bool Found = false;
      for (uint64_t PC : Outer.BodyBlockPCs) {
        if (PC == Inner.HeaderPC) {
          Found = true;
          break;
        }
      }

      if (Found && Outer.BodyBlockPCs.size() < SmallestParentSize) {
        SmallestParentSize = Outer.BodyBlockPCs.size();
        BestParent = static_cast<int>(J);
      }
    }
    Inner.ParentIdx = BestParent;
  }

  return LI;
}

void LoopInfo::print(const TraceCFG *CFG,
                     const std::vector<InstEntry> *Entries) const {
  outs() << "\n";
  outs() << "============================================================\n";
  outs() << "LOOP ANALYSIS\n";
  outs() << "============================================================\n";
  outs() << "\n";

  if (Loops.empty()) {
    outs() << "No loops detected.\n";
    outs() << "============================================================\n";
    return;
  }

  // Build PC -> block mapping if we have entries to print
  std::map<uint64_t, uint64_t> PCToBlock;
  if (CFG && Entries) {
    for (const auto &E : *Entries) {
      auto It = CFG->Blocks.upper_bound(E.PC);
      if (It != CFG->Blocks.begin()) {
        --It;
        if (E.PC >= It->second.StartPC && E.PC <= It->second.EndPC)
          PCToBlock[E.PC] = It->first;
      }
    }
  }

  // Print dominator info
  outs() << "=== Dominators ===\n";
  for (const auto &[Block, Dom] : Dominators) {
    if (Block != Dom)
      outs() << format("  0x%04x dominated by 0x%04x\n", Block, Dom);
  }
  outs() << "\n";

  outs() << format("Loops detected: %zu\n\n", Loops.size());

  for (size_t I = 0; I < Loops.size(); ++I) {
    const Loop &L = Loops[I];

    outs() << format("Loop %zu:\n", I);
    outs() << format("  Header: 0x%04x\n", L.HeaderPC);
    outs() << "  Latches:";
    for (uint64_t LatchPC : L.LatchPCs)
      outs() << format(" 0x%04x", LatchPC);
    outs() << "\n";
    outs() << format("  Back-edge count: %u\n", L.TotalBackEdgeCount);
    outs() << format("  Body blocks: %zu\n", L.BodyBlockPCs.size());

    if (L.ParentIdx >= 0) {
      const Loop &Parent = Loops[L.ParentIdx];
      unsigned IterPerParent = L.getIterations() / Parent.getIterations();
      outs() << format("  Nested in loop %d (~%u iters per parent)\n",
                       L.ParentIdx, IterPerParent);
    }
    outs() << "\n";

    // Print instructions in this loop (first iteration only)
    if (Entries) {
      std::set<uint64_t> BodySet(L.BodyBlockPCs.begin(), L.BodyBlockPCs.end());
      outs() << "  Instructions:\n";
      bool SeenHeader = false;
      for (const InstEntry &E : *Entries) {
        auto It = PCToBlock.find(E.PC);
        if (It != PCToBlock.end() && BodySet.count(It->second)) {
          if (E.PC == L.HeaderPC) {
            if (SeenHeader)
              break;  // Second time at header = end of first iteration
            SeenHeader = true;
          }
          const char *ClassName = getInstClassName(static_cast<InstClass>(E.InstClass));
          const char *StallName = getStallReasonName(static_cast<StallReason>(E.StallReason));
          std::string Prefix = formatv("    0x{0:X4}: [{1} cy={2} stall={3}{4}]",
                                        E.PC, ClassName, E.Cycles, E.StallCycles,
                                        StallName ? formatv(" reason={0}", StallName).str() : "").str();
          outs() << format("%-60s %s\n", Prefix.c_str(), E.InstructionText.c_str());
        }
      }
      outs() << "\n";
    }

    if (L.Metrics.NumInstructions > 0) {
      outs() << format("--- Loop %zu Metrics (Iterations: %u) ---\n", I,
                       L.getIterations());
      L.Metrics.printBody(L.getIterations());
    }
  }

  outs() << "============================================================\n";
}

} // namespace tracecp
} // namespace llvm
