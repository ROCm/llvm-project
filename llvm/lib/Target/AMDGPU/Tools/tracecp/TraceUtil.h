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
  std::vector<int64_t> WaveIds;

  TraceFilter(int64_t DispatchId, int64_t ClusterId, int64_t WorkgroupId,
              ArrayRef<int64_t> WaveIds)
      : DispatchId(DispatchId), ClusterId(ClusterId), WorkgroupId(WorkgroupId),
        WaveIds(WaveIds.begin(), WaveIds.end()) {
    sort(this->WaveIds);
  }

  bool match(int64_t OtherDispatchId, int64_t OtherClusterId,
             int64_t OtherWorkgroupId, int64_t OtherWaveId) const {
    if (OtherDispatchId != DispatchId || OtherClusterId != ClusterId ||
        OtherWorkgroupId != WorkgroupId)
      return false;
    return binary_search(WaveIds, OtherWaveId);
  }
};

class WaveView {
  llvm::DenseMap<int64_t, ArrayRef<InstEntry>> WaveToEntries;

public:
  WaveView(ArrayRef<InstEntry> Entries);

  auto entries_per_wave() const {
    return llvm::make_range(WaveToEntries.begin(), WaveToEntries.end());
  }

  bool empty() const { return WaveToEntries.empty(); }
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

  bool contains(uint64_t PC) const { return StartPC <= PC && PC <= EndPC; }
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

  const TraceBlock &getBlockForPC(uint64_t PC) const {
    auto It = Blocks.upper_bound(PC);
    if (It == Blocks.end()) {
      const TraceBlock &Last = Blocks.rbegin()->second;
      assert(Last.contains(PC) && "PC out of range, too large");
      return Last;
    }
    assert(It != Blocks.begin() && "PC out of range, too small");
    // The previous node contains the block
    --It;
    const TraceBlock &Block = It->second;
    assert(Block.contains(PC) && "PC out of range, not in block");
    return Block;
  }

  void print() const;
};

/// Reconstruct CFG from the trace.
/// \param Entries The trace entries to analyze.
/// \param MCII The MCInstrInfo to identify branch instructions.
/// \returns The reconstructed CFG.
TraceCFG reconstructCFG(const WaveView &WaveView, const MCInstrInfo &MCII);

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
/// \param Verbose Enable verbose per-instruction logging.
/// \returns TraceMetrics with per-block metrics.
TraceMetrics simulateTrace(const WaveView &WaveView, const TraceCFG &CFG,
                           const MCInstrInfo &MCII, const MCRegisterInfo &MRI,
                           bool Verbose = false);

} // namespace tracecp
} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_TOOLS_TRACECP_TRACEUTIL_H
