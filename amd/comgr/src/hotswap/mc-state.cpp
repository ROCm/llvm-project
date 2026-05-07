//===- mc-state.cpp - Hotswap transpiler ----------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mc-state.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetSelect.h"

using namespace llvm;

namespace COMGR::hotswap {

const char kAMDGPUTriple[] = "amdgcn-amd-amdhsa";

std::unique_ptr<MCSubtargetInfo>
buildSubtargetInfo(const Target &Target, StringRef Isa) {
  Triple Triple(kAMDGPUTriple);
  std::unique_ptr<MCSubtargetInfo> Sti(
      Target.createMCSubtargetInfo(Triple, Isa, ""));
  if (!Sti)
    report_fatal_error(Twine("transpiler: failed to create MCSubtargetInfo "
                             "for ISA '") +
                       Isa + "'");
  return Sti;
}

bool initMCState(MCState &State, StringRef TargetIsa) {
  LLVMInitializeAMDGPUTargetInfo();
  LLVMInitializeAMDGPUTarget();
  LLVMInitializeAMDGPUTargetMC();
  LLVMInitializeAMDGPUDisassembler();
  LLVMInitializeAMDGPUAsmParser();
  LLVMInitializeAMDGPUAsmPrinter();

  Triple Triple(kAMDGPUTriple);
  std::string Error;
  State.Target = TargetRegistry::lookupTarget(Triple, Error);
  if (!State.Target)
    report_fatal_error(Twine("transpiler: Target lookup for '") +
                       kAMDGPUTriple + "' failed: " + Error);

  State.InstrInfo.reset(State.Target->createMCInstrInfo());
  State.RegInfo.reset(State.Target->createMCRegInfo(Triple));
  State.SubtargetInfo = buildSubtargetInfo(*State.Target, TargetIsa);
  State.AsmInfo.reset(State.Target->createMCAsmInfo(
      *State.RegInfo, Triple, MCTargetOptions()));
  State.Ctx = std::make_unique<MCContext>(Triple, *State.AsmInfo,
                                         *State.RegInfo,
                                         *State.SubtargetInfo);
  // Defensive consistency with the legacy hotswap path
  // (see `hotswap.cpp` / `src/hotswap.cpp`'s companion
  // `initInlineSourceManager` calls): the MCContext ctor defaults
  // `SourceMgr *Mgr = nullptr`, so any MC-layer diagnostic that
  // reaches `MCContext::reportCommon` or `MCContext::diagnose`
  // with a valid SMLoc and no SrcMgr trips the
  // `llvm_unreachable("Either SourceMgr should be available")`
  // abort at `llvm/lib/MC/MCContext.cpp:1093` / `:1120`.
  //
  // Hotswap's IR-raise pipeline doesn't currently exercise the MC
  // assembler (codegen runs through `llc`/`lld` on lifted IR, not
  // through this MCContext), so the abort doesn't fire on hotswap
  // today.  But the disassembler here can emit diagnostics on
  // malformed instruction bytes, and any future reuse of this
  // MCContext for an MC emission path (e.g. an assembly-based
  // post-rewrite pass or a new cross-widening lowering that
  // goes through MC) would hit the same abort.  Attaching an
  // inline SourceMgr here keeps the failure mode graceful for
  // both current and future callers — the cost is one pointer
  // and one default-constructed SourceMgr per MCState.
  State.Ctx->initInlineSourceManager();
  State.Disasm.reset(
      State.Target->createMCDisassembler(*State.SubtargetInfo, *State.Ctx));
  State.Printer.reset(State.Target->createMCInstPrinter(
      Triple, 0, *State.AsmInfo, *State.InstrInfo, *State.RegInfo));
  State.Printer->setPrintImmHex(true);
  return true;
}

std::string getMnemonic(const MCState &Mc, const MCInst &Inst) {
  std::string S;
  raw_string_ostream Os(S);
  Mc.Printer->printInst(&Inst, 0, "", *Mc.SubtargetInfo, Os);
  StringRef Sr(S);
  Sr = Sr.ltrim();
  return Sr.split('\t').first.split(' ').first.str();
}

StringRef stripEncoding(StringRef Mn) {
  for (StringRef Suffix : {"_e32", "_e64", "_vi"})
    if (Mn.ends_with(Suffix))
      return Mn.drop_back(Suffix.size());
  return Mn;
}

} // namespace COMGR::hotswap
