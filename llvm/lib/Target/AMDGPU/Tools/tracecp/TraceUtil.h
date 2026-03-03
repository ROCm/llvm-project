//===-- TraceUtil.h - Trace processing utilities --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_TOOLS_TRACECP_TRACEUTIL_H
#define LLVM_LIB_TARGET_AMDGPU_TOOLS_TRACECP_TRACEUTIL_H

#include "AMDGPUStaticSimulator.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
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

/// CFG Analysis.
struct TraceBlock {
  uint64_t StartPC;
  uint64_t EndPC;
  unsigned NumInstructions;
  unsigned ExecutionCount;
};

struct TraceEdge {
  uint64_t FromBlockPC; // Start of source block
  uint64_t FromPC;      // End of source block (branch instruction)
  uint64_t ToPC;        // Start of target block
  unsigned Count;       // How many times this edge was taken
};

/// CFG reconstructed from the trace.
struct TraceCFG {
  std::map<uint64_t, TraceBlock> Blocks; // Keyed by StartPC
  std::vector<TraceEdge> Edges;

  void print() const;
};

/// Reconstruct CFG from the trace.
/// \param Entries The trace entries to analyze.
/// \param MCII The MCInstrInfo to identify branch instructions.
/// \returns The reconstructed CFG.
TraceCFG reconstructCFG(const std::vector<InstEntry> &Entries,
                        const MCInstrInfo &MCII);

/// Result of trace simulation: maps block start PC to a vector of BlockMetrics,
/// one per execution of that block.
struct TraceMetrics {
  std::map<uint64_t, std::vector<AMDGPU::BlockMetrics>> Blocks;

  /// Print aggregated metrics for all block executions.
  void print() const;
};

/// Simulate a trace and collect per-block metrics.
/// Each time a block is executed, a new BlockMetrics is collected.
/// \param Entries The trace entries to simulate.
/// \param CFG The reconstructed CFG (needed to know block boundaries).
/// \param MCII The MCInstrInfo for the target.
/// \param MRI The MCRegisterInfo for the target.
/// \param STI The MCSubtargetInfo for the target (for scheduling model access).
/// \param Verbose Enable verbose per-instruction logging.
/// \returns TraceMetrics with per-block metrics.
TraceMetrics simulateTrace(const std::vector<InstEntry> &Entries,
                           const TraceCFG &CFG, const MCInstrInfo &MCII,
                           const MCRegisterInfo &MRI, const MCSubtargetInfo &STI,
                           bool Verbose = false);

} // namespace tracecp
} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_TOOLS_TRACECP_TRACEUTIL_H
