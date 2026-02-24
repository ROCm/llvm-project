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

struct TraceEntry {
  int64_t DispatchId;
  int64_t ClusterId;
  int64_t WorkgroupId;
  int64_t WavegroupId;
  int64_t WaveId;
  int64_t InstructionId;
  uint64_t PC;
  uint64_t Opcode;
  uint64_t InstSize;
  std::string InstructionText;
  MCInst Inst;
};

/// Metrics collected from simulating a trace.
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

  float getIPC() const {
    return TotalCycles > 0 ? static_cast<float>(NumInstructions) / TotalCycles
                           : 0.0f;
  }

  float getStallRatio() const {
    return TotalCycles > 0 ? static_cast<float>(StallCycles) / TotalCycles
                           : 0.0f;
  }

  void print() const;
};

/// Parse a trace file and disassemble the instructions.
/// \param FilePath Path to the JSON trace file.
/// \param SelectWaveId Only include entries with this wave_id.
/// \param DisAsm The disassembler to use.
/// \returns A vector of TraceEntry on success, or an error.
Expected<std::vector<TraceEntry>> parseAndDisassemble(StringRef FilePath,
                                                      int64_t SelectWaveId,
                                                      MCDisassembler &DisAsm);

/// Simulate a trace and collect metrics.
/// \param Entries The trace entries to simulate.
/// \param MCII The MCInstrInfo for the target.
/// \param MRI The MCRegisterInfo for the target.
/// \param Verbose Enable verbose per-instruction logging.
/// \returns Metrics collected from the simulation.
TraceMetrics simulateTrace(const std::vector<TraceEntry> &Entries,
                           const MCInstrInfo &MCII, const MCRegisterInfo &MRI,
                           bool Verbose = false);

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
TraceCFG reconstructCFG(const std::vector<TraceEntry> &Entries,
                        const MCInstrInfo &MCII);

/// A natural loop detected via dominator analysis.
struct Loop {
  uint64_t HeaderPC;                   // Loop header (dominates all body nodes)
  std::vector<uint64_t> LatchPCs;      // Blocks with back-edges to header
  std::vector<uint64_t> BodyBlockPCs;  // All blocks in the loop body
  unsigned TotalBackEdgeCount;         // Sum of all back-edge counts
  int ParentIdx;                       // Index of parent loop (-1 if top-level)
};

/// Loops detected from the CFG.
struct LoopInfo {
  std::vector<Loop> Loops;
  std::map<uint64_t, uint64_t> Dominators; // Block -> immediate dominator

  void print() const;
};

/// Detect loops from the reconstructed CFG.
/// \param CFG The control flow graph.
/// \returns Detected loops with nesting information.
LoopInfo detectLoops(const TraceCFG &CFG);

} // namespace tracecp
} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_TOOLS_TRACECP_TRACEUTIL_H
