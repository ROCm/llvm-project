//===-- tracecp.cpp - Trace simulation utility ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TraceUtil.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static cl::OptionCategory TraceCPCategory("tracecp options");

static cl::opt<std::string> InputFilePath(cl::Positional,
                                          cl::desc("<input file>"),
                                          cl::Required,
                                          cl::cat(TraceCPCategory));

static cl::opt<std::string> MTriple("mtriple", cl::desc("Target triple"),
                                    cl::init("amdgcn"),
                                    cl::cat(TraceCPCategory));

static cl::opt<std::string> MCPU("mcpu", cl::desc("Target CPU"),
                                 cl::init("gfx1250"), cl::cat(TraceCPCategory));

static cl::opt<int64_t> DispatchId("dispatch-id",
                                    cl::desc("Filter by dispatch_id"),
                                    cl::Required, cl::cat(TraceCPCategory));

static cl::opt<int64_t> ClusterId("cluster-id",
                                   cl::desc("Filter by cluster_id"),
                                   cl::Required, cl::cat(TraceCPCategory));

static cl::opt<int64_t> WorkgroupId("workgroup-id",
                                     cl::desc("Filter by workgroup_id"),
                                     cl::init(0), cl::cat(TraceCPCategory));

static cl::opt<int64_t> WaveId("wave-id", cl::desc("Filter by wave_id"),
                               cl::Required, cl::cat(TraceCPCategory));

static cl::opt<bool> Verbose("verbose", cl::desc("Enable verbose output"),
                             cl::init(false), cl::cat(TraceCPCategory));

int main(int argc, char **argv) {
  cl::HideUnrelatedOptions(TraceCPCategory);
  cl::ParseCommandLineOptions(argc, argv, "tracecp - trace simulation tool\n");

  // Currently supporting only gfx1250
  if (MCPU != "gfx1250") {
    errs() << "Error: unsupported --mcpu value '" << MCPU << "'. "
           << "Supported: gfx1250\n";
    return 1;
  }

  // Initialize AMDGPU target for disassembly
  LLVMInitializeAMDGPUTargetInfo();
  LLVMInitializeAMDGPUTargetMC();
  LLVMInitializeAMDGPUDisassembler();

  // Set up the target
  std::string Error;
  Triple TheTriple(MTriple);
  const Target *TheTarget =
      TargetRegistry::lookupTarget(MTriple, TheTriple, Error);
  if (!TheTarget) {
    errs() << "Error: " << Error << "\n";
    return 1;
  }

  // Create MC components
  std::unique_ptr<MCRegisterInfo> MRI(TheTarget->createMCRegInfo(TheTriple));
  if (!MRI) {
    errs() << "Error: no register info\n";
    return 1;
  }

  MCTargetOptions MCOptions;
  std::unique_ptr<MCAsmInfo> MAI(
      TheTarget->createMCAsmInfo(*MRI, TheTriple, MCOptions));
  if (!MAI) {
    errs() << "Error: no asm info\n";
    return 1;
  }

  std::unique_ptr<MCSubtargetInfo> STI(
      TheTarget->createMCSubtargetInfo(TheTriple, MCPU, ""));
  if (!STI) {
    errs() << "Error: no subtarget info\n";
    return 1;
  }

  std::unique_ptr<MCInstrInfo> MII(TheTarget->createMCInstrInfo());
  if (!MII) {
    errs() << "Error: no instr info\n";
    return 1;
  }

  MCContext Ctx(TheTriple, MAI.get(), MRI.get(), STI.get());

  std::unique_ptr<MCDisassembler> DisAsm(
      TheTarget->createMCDisassembler(*STI, Ctx));
  if (!DisAsm) {
    errs() << "Error: no disassembler\n";
    return 1;
  }

  // Parse and disassemble
  tracecp::TraceFilter Filter{DispatchId, ClusterId, WorkgroupId, WaveId};
  Expected<std::vector<tracecp::InstEntry>> EntriesOrErr =
      tracecp::parseAndDisassemble(InputFilePath, Filter, *DisAsm);
  if (!EntriesOrErr) {
    errs() << toString(EntriesOrErr.takeError()) << "\n";
    return 1;
  }
  std::vector<tracecp::InstEntry> &Entries = *EntriesOrErr;

  // Run simulation (populates metrics in each entry)
  tracecp::simulateTrace(Entries, *MII, *MRI, Verbose);

  // Compute and print aggregate metrics
  tracecp::TraceMetrics Metrics(Entries);
  Metrics.print();

  // Reconstruct and print CFG
  tracecp::TraceCFG CFG = tracecp::reconstructCFG(Entries, *MII);
  CFG.print();

  // Detect loops and compute per-loop metrics
  tracecp::LoopInfo LI = tracecp::detectLoops(CFG, Entries);
  LI.print();

  return 0;
}
