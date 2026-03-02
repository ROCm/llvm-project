//===-- TraceUtil.h - Trace processing utilities --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_TOOLS_TRACECP_TRACEUTIL_H
#define LLVM_LIB_TARGET_AMDGPU_TOOLS_TRACECP_TRACEUTIL_H

#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace llvm {
namespace tracecp {

struct InstEntry {
  // Parsed from trace file
  int64_t DispatchId;
  int64_t ClusterId;
  int64_t WorkgroupId;
  int64_t WavegroupId;
  int64_t WaveId;
  int64_t InstructionId;
  uint64_t PC;
  int64_t Opcode;
  uint64_t InstSize;
  std::string InstructionText;
  MCInst Inst;

  // Simulation metrics (populated by simulateTrace)
  unsigned InstClass = 0;
  unsigned Cycles = 0;
  unsigned StallCycles = 0;
  unsigned StallReason = 0;
};

/// Metrics collected from simulating a trace (or a subset like a loop).
struct TraceMetrics {
  unsigned NumInstructions = 0;
  unsigned TotalCycles = 0;
  unsigned StallCycles = 0;

  // Instruction counts by class
  unsigned NumVALU = 0;
  unsigned NumSALU = 0;
  unsigned NumTRANS = 0;
  unsigned NumWMMA = 0;
  unsigned NumDSRead = 0;
  unsigned NumDSWrite = 0;
  unsigned NumVMEM = 0;
  unsigned NumSMEM = 0;
  unsigned NumBranch = 0;
  unsigned NumWaitcnt = 0;
  unsigned NumOther = 0;

  // Stall breakdown
  unsigned StallFU = 0;
  unsigned StallCoExec = 0;
  unsigned StallDelayAlu = 0;
  unsigned StallWaitCnt = 0;
  unsigned StallMemFIFO = 0;
  unsigned StallRegBank = 0;
  unsigned StallRAW = 0;
  unsigned StallTrans = 0;
  unsigned StallVASSrc = 0;
  unsigned StallVAVDst = 0;
  unsigned StallISFetch = 0;
  unsigned StallMSBExposed = 0;
  unsigned StallOther = 0;

  TraceMetrics() = default;

  /// Construct metrics from trace entries (after simulation).
  explicit TraceMetrics(const std::vector<InstEntry> &Entries);

  float getIPC() const {
    return TotalCycles > 0 ? static_cast<float>(NumInstructions) / TotalCycles
                           : 0.0f;
  }

  float getStallRatio() const {
    return TotalCycles > 0 ? static_cast<float>(StallCycles) / TotalCycles
                           : 0.0f;
  }

  void print() const;

  /// Print metrics body (without banner). If Iterations > 0, also prints
  /// per-iteration averages.
  void printBody(unsigned Iterations = 0) const;

  /// Add metrics from a single trace entry.
  void addInstruction(const InstEntry &Entry);
};

/// Filter criteria for trace entries.
struct TraceFilter {
  int64_t DispatchId;
  int64_t ClusterId;
  int64_t WorkgroupId;
  int64_t WaveId;
};

/// Parse a trace file and disassemble the instructions.
/// \param FilePath Path to the JSON trace file.
/// \param Filter Only include entries matching these criteria.
/// \param DisAsm The disassembler to use.
/// \returns A vector of InstEntry on success, or an error.
Expected<std::vector<InstEntry>> parseAndDisassemble(StringRef FilePath,
                                                      const TraceFilter &Filter,
                                                      MCDisassembler &DisAsm);

/// Simulate a trace and populate per-instruction metrics in each entry.
/// \param Entries The trace entries to simulate (metrics fields will be filled).
/// \param MCII The MCInstrInfo for the target.
/// \param MRI The MCRegisterInfo for the target.
/// \param Verbose Enable verbose per-instruction logging.
void simulateTrace(std::vector<InstEntry> &Entries, const MCInstrInfo &MCII,
                   const MCRegisterInfo &MRI, bool Verbose = false);

/// CFG Analysis.
struct BasicBlock {
  uint64_t StartPC;
  uint64_t EndPC;
  unsigned NumInstructions;
  unsigned ExecutionCount;
};

struct CFGEdge {
  uint64_t FromBlockPC; // Start of source block
  uint64_t FromPC;      // End of source block (branch instruction)
  uint64_t ToPC;        // Start of target block
  unsigned Count;       // How many times this edge was taken
};

/// CFG reconstructed from the trace.
struct TraceCFG {
  std::map<uint64_t, BasicBlock> Blocks; // Keyed by StartPC
  std::vector<CFGEdge> Edges;

  void print() const;
};

/// Reconstruct CFG from the trace.
/// \param Entries The trace entries to analyze.
/// \param MCII The MCInstrInfo to identify branch instructions.
/// \returns The reconstructed CFG.
TraceCFG reconstructCFG(const std::vector<InstEntry> &Entries,
                        const MCInstrInfo &MCII);

/// A natural loop detected via dominator analysis.
struct Loop {
  uint64_t HeaderPC;                   // Loop header (dominates all body nodes)
  std::vector<uint64_t> LatchPCs;      // Blocks with back-edges to header
  std::vector<uint64_t> BodyBlockPCs;  // All blocks in the loop body
  unsigned TotalBackEdgeCount;         // Sum of all back-edge counts (iterations)
  int ParentIdx;                       // Index of parent loop (-1 if top-level)
  TraceMetrics Metrics;                // Metrics for this loop

  // Per-iteration averages
  float getAvgInstructions() const {
    return TotalBackEdgeCount > 0
               ? static_cast<float>(Metrics.NumInstructions) / TotalBackEdgeCount
               : 0.0f;
  }
  float getAvgStallCycles() const {
    return TotalBackEdgeCount > 0
               ? static_cast<float>(Metrics.StallCycles) / TotalBackEdgeCount
               : 0.0f;
  }
};

/// Loops detected from the CFG.
struct LoopInfo {
  std::vector<Loop> Loops;
  std::map<uint64_t, uint64_t> Dominators; // Block -> immediate dominator

  void print() const;
};

/// Detect loops from the reconstructed CFG and compute per-loop metrics.
/// \param CFG The control flow graph.
/// \param Entries The trace entries (with simulation metrics populated).
/// \returns Detected loops with nesting information and metrics.
LoopInfo detectLoops(const TraceCFG &CFG,
                     const std::vector<InstEntry> &Entries);

} // namespace tracecp
} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_TOOLS_TRACECP_TRACEUTIL_H
