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
#include "llvm/Support/Format.h"
#include "llvm/Support/JSON.h"
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

Expected<std::vector<TraceEntry>> parseAndDisassemble(StringRef FilePath,
                                                      int64_t SelectWaveId,
                                                      MCDisassembler &DisAsm) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> FileOrErr =
      MemoryBuffer::getFile(FilePath);
  if (std::error_code EC = FileOrErr.getError())
    return createStringError(EC, "Error opening file '%s': %s",
                             FilePath.str().c_str(), EC.message().c_str());

  std::vector<TraceEntry> Results;
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

    std::optional<int64_t> WaveId = Obj->getInteger("wave_id");
    std::optional<StringRef> InstructionText =
        Obj->getString("instruction_text");

    if (!WaveId || !InstructionText)
      continue;

    if (*WaveId != SelectWaveId)
      continue;

    TraceEntry Entry;
    Entry.DispatchId = Obj->getInteger("dispatch_id").value_or(0);
    Entry.ClusterId = Obj->getInteger("cluster_id").value_or(0);
    Entry.WorkgroupId = Obj->getInteger("workgroup_id").value_or(0);
    Entry.WavegroupId = Obj->getInteger("wavegroup_id").value_or(0);
    Entry.WaveId = *WaveId;
    Entry.InstructionId = Obj->getInteger("instruction_id").value_or(0);
    Entry.PC = static_cast<uint64_t>(Obj->getInteger("pc").value_or(0));
    Entry.Opcode = static_cast<uint64_t>(Obj->getInteger("opcode").value_or(0));
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

    uint64_t Size;
    ArrayRef<uint8_t> BytesRef(Bytes);
    MCDisassembler::DecodeStatus Status =
        DisAsm.getInstruction(Entry.Inst, Size, BytesRef, 0, nulls());
    if (Status != MCDisassembler::Success)
      return createStringError(
          inconvertibleErrorCode(),
          "Line %zu: Disassembly failed for instruction: %s", LineNum + 1,
          Entry.InstructionText.c_str());

    Results.push_back(std::move(Entry));
  }

  return Results;
}

static void countInstruction(InstClass IC, TraceMetrics &Metrics) {
  Metrics.NumInstructions++;

  switch (IC) {
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
  case InstClass::BRANCH:
    Metrics.NumBranch++;
    break;
  case InstClass::WAITCNT:
    Metrics.NumWaitcnt++;
    break;
  default:
    Metrics.NumOther++;
    break;
  }
}

static void attributeStall(const InstrSimInfo &Info, TraceMetrics &Metrics) {
  unsigned TotalStall = Info.StallCycles;
  if (TotalStall == 0)
    return;

  Metrics.StallCycles += TotalStall;

  // Attribute based on the dominant stall reason computed by the simulator
  switch (Info.Reason) {
  case StallReason::FU_BUSY:
    Metrics.StallFU += TotalStall;
    break;
  case StallReason::COEXEC_BLOCKED:
  case StallReason::LONG_LAT_VALU:
    Metrics.StallCoExec += TotalStall;
    break;
  case StallReason::DELAY_ALU:
    Metrics.StallDelayAlu += TotalStall;
    break;
  case StallReason::WAITCNT:
    Metrics.StallWaitCnt += TotalStall;
    break;
  case StallReason::MEM_FIFO:
    Metrics.StallMemFIFO += TotalStall;
    break;
  case StallReason::REG_BANK:
    Metrics.StallRegBank += TotalStall;
    break;
  case StallReason::RAW_HAZARD:
    Metrics.StallRAW += TotalStall;
    break;
  case StallReason::LOLVALU_TRANS_HAZARD:
    Metrics.StallTrans += TotalStall;
    break;
  case StallReason::VA_SSRC_STALL:
    Metrics.StallVASSrc += TotalStall;
    break;
  case StallReason::VA_VDST_WAIT:
    Metrics.StallVAVDst += TotalStall;
    break;
  case StallReason::IS_FETCH:
    Metrics.StallISFetch += TotalStall;
    break;
  case StallReason::MSB_SET_EXPOSED:
    Metrics.StallMSBExposed += TotalStall;
    break;
  case StallReason::NONE:
    Metrics.StallOther += TotalStall;
    break;
  }
}

TraceMetrics simulateTrace(const std::vector<TraceEntry> &Entries,
                           const MCInstrInfo &MCII, const MCRegisterInfo &MRI,
                           bool Verbose) {
  MCInstInfo InstInfo(MCII, MRI);
  HWModel Model = createHWModel(GPUTarget::GFX1250);

  SimulatorConfig Cfg;
  Cfg.Verbose = Verbose;
  Cfg.Log = Verbose ? &errs() : nullptr;
  Cfg.EnableScoreboard = false;
  Cfg.EnableISCache = false;

  Simulator Sim(InstInfo, Model, Cfg);
  unsigned StartCycle = Sim.getState().CurrentCycle;

  TraceMetrics Metrics;
  for (const TraceEntry &Entry : Entries) {
    SimInst SI = InstInfo.createSimInst(Entry.Inst);
    countInstruction(SI.Class, Metrics);

    InstrSimInfo Info = Sim.simulateInst(SI);
    attributeStall(Info, Metrics);

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

  Metrics.TotalCycles = Sim.getState().CurrentCycle - StartCycle;
  return Metrics;
}

void TraceMetrics::print() const {
  outs() << "\n";
  outs() << "============================================================\n";
  outs() << "TRACE SIMULATION METRICS\n";
  outs() << "============================================================\n";
  outs() << "\n";

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

  outs() << "============================================================\n";
}

} // namespace tracecp
} // namespace llvm
