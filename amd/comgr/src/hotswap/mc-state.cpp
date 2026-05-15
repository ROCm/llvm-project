//===- mc-state.cpp - Hotswap transpiler ----------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mc-state.h"
#include "hotswap-error.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/Support/TargetSelect.h"

using namespace llvm;

namespace COMGR::hotswap {

const char kAMDGPUTriple[] = "amdgcn-amd-amdhsa";

Expected<std::unique_ptr<MCSubtargetInfo>>
buildSubtargetInfo(const Target &Target, StringRef Isa) {
  Triple Triple(kAMDGPUTriple);
  std::unique_ptr<MCSubtargetInfo> Sti(
      Target.createMCSubtargetInfo(Triple, Isa, ""));
  if (!Sti)
    return makeHotswapError("buildSubtargetInfo: failed to create "
                            "MCSubtargetInfo for ISA '" +
                            Isa + "'");
  return Sti;
}

Error initMCState(MCState &State, StringRef TargetIsa) {
  LLVMInitializeAMDGPUTargetInfo();
  LLVMInitializeAMDGPUTarget();
  LLVMInitializeAMDGPUTargetMC();
  LLVMInitializeAMDGPUDisassembler();
  LLVMInitializeAMDGPUAsmParser();
  LLVMInitializeAMDGPUAsmPrinter();

  Triple Triple(kAMDGPUTriple);
  std::string LookupError;
  State.Target = TargetRegistry::lookupTarget(Triple, LookupError);
  if (!State.Target)
    return makeHotswapError(Twine("initMCState: Target lookup for '") +
                            kAMDGPUTriple + "' failed: " + LookupError);

  State.InstrInfo.reset(State.Target->createMCInstrInfo());
  State.RegInfo.reset(State.Target->createMCRegInfo(Triple));
  Expected<std::unique_ptr<MCSubtargetInfo>> StiOrErr =
      buildSubtargetInfo(*State.Target, TargetIsa);
  if (!StiOrErr)
    return StiOrErr.takeError();
  State.SubtargetInfo = std::move(*StiOrErr);
  State.AsmInfo.reset(State.Target->createMCAsmInfo(
      *State.RegInfo, Triple, MCTargetOptions()));
  if (!State.AsmInfo)
    return makeHotswapError("initMCState: createMCAsmInfo returned null");
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
  if (!State.Disasm)
    return makeHotswapError("initMCState: createMCDisassembler returned null");
  State.Printer.reset(State.Target->createMCInstPrinter(
      Triple, 0, *State.AsmInfo, *State.InstrInfo, *State.RegInfo));
  if (!State.Printer)
    return makeHotswapError("initMCState: createMCInstPrinter returned null");
  State.Printer->setPrintImmHex(true);
  return Error::success();
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
