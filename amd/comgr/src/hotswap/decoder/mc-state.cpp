//===- mc-state.cpp - Hotswap transpiler ----------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mc-state.h"
#include "comgr.h"
#include "hotswap/common/hotswap-error.h"

// AMDGPU target-private headers.
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"

#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/Support/Error.h"
#include "llvm/TargetParser/AMDGPUTargetParser.h"

#include <cassert>

using namespace llvm;

namespace COMGR::hotswap {

Expected<std::unique_ptr<MCSubtargetInfo>>
buildSubtargetInfo(const Target &Target, StringRef Isa) {
  // createMCSubtargetInfo does not reject an unknown CPU: it diagnoses to
  // stderr and returns a featureless default, and building the AMDGPU
  // disassembler from that trips reportFatalUsageError and aborts the host.
  // Reject the bare processor name up front so malformed ISA input returns an
  // error instead.
  if (AMDGPU::parseArchAMDGCN(Isa) == AMDGPU::GK_NONE) {
    return makeHotswapError("buildSubtargetInfo: unknown AMDGPU processor '" +
                            Isa + "'");
  }
  Triple Triple(kAMDGPUTriple);
  std::unique_ptr<MCSubtargetInfo> STI(
      Target.createMCSubtargetInfo(Triple, Isa, ""));
  // The ISA is a validated AMDGPU processor by here, so the registered factory
  // always returns a subtarget.
  assert(STI && "AMDGPU target must provide an MCSubtargetInfo");
  return STI;
}

llvm::Expected<MCState> initMCState(StringRef TargetIsa) {
  // Registering the AMDGPU target mutates the process-global TargetRegistry,
  // which is not thread-safe. Reuse COMGR's shared one-time initializer (mutex
  // plus run-once guard) rather than re-registering on every call.
  COMGR::ensureLLVMInitialized();

  Triple Triple(kAMDGPUTriple);
  std::string LookupError;
  MCState State;
  State.Target = TargetRegistry::lookupTarget(Triple, LookupError);
  if (!State.Target) {
    return makeHotswapError("initMCState: Target lookup for '" + kAMDGPUTriple +
                            "' failed: " + LookupError);
  }

  // Once the AMDGPU target is registered, its instr/reg tables are built from
  // static data and never fail; a null here is a broken build, not bad input.
  State.InstrInfo.reset(State.Target->createMCInstrInfo());
  assert(State.InstrInfo && "AMDGPU target must provide an MCInstrInfo");
  State.RegInfo.reset(State.Target->createMCRegInfo(Triple));
  assert(State.RegInfo && "AMDGPU target must provide an MCRegInfo");
  Expected<std::unique_ptr<MCSubtargetInfo>> STIOrErr =
      buildSubtargetInfo(*State.Target, TargetIsa);
  if (!STIOrErr) {
    return STIOrErr.takeError();
  }

  State.SubtargetInfo = std::move(*STIOrErr);
  State.AsmInfo.reset(
      State.Target->createMCAsmInfo(*State.RegInfo, Triple, MCTargetOptions()));
  assert(State.AsmInfo && "AMDGPU target must provide an MCAsmInfo");

  State.Ctx = std::make_unique<MCContext>(Triple, *State.AsmInfo,
                                          *State.RegInfo, *State.SubtargetInfo);
  // The MCContext ctor defaults `SourceMgr *Mgr = nullptr`, so any
  // MC-layer diagnostic that reaches `MCContext::reportCommon` or
  // `MCContext::diagnose` with a valid SMLoc and no SrcMgr trips an
  // `llvm_unreachable("Either SourceMgr should be available")` abort
  // inside `llvm/lib/MC/MCContext.cpp`.
  //
  // The hotswap IR-raise pipeline does not currently exercise the MC
  // assembler (codegen runs through `llc`/`lld` on lifted IR, not
  // through this MCContext), so the abort does not fire on hotswap
  // today. But the disassembler here can emit diagnostics on malformed
  // instruction bytes, and any future reuse of this MCContext for an
  // MC emission path (e.g. an assembly-based post-rewrite pass or a
  // new widening lowering that goes through MC) would hit the
  // same abort. Attaching an inline SourceMgr here keeps the failure
  // mode graceful for both current and future callers -- the cost is
  // one pointer and one default-constructed SourceMgr per MCState.
  State.Ctx->initInlineSourceManager();
  State.Disasm.reset(
      State.Target->createMCDisassembler(*State.SubtargetInfo, *State.Ctx));
  assert(State.Disasm && "AMDGPU target must provide an MCDisassembler");

  State.Printer.reset(State.Target->createMCInstPrinter(
      Triple, 0, *State.AsmInfo, *State.InstrInfo, *State.RegInfo));
  assert(State.Printer && "AMDGPU target must provide an MCInstPrinter");

  State.Printer->setPrintImmHex(true);

  return State;
}

std::string getMnemonic(const MCState &State, const MCInst &Inst) {
  std::string S;
  raw_string_ostream Os(S);
  State.Printer->printInst(&Inst, 0, "", *State.SubtargetInfo, Os);
  StringRef Sr(S);
  Sr = Sr.ltrim();
  return Sr.split('\t').first.split(' ').first.str();
}

std::string printInst(const MCState &State, const MCInst &Inst) {
  std::string S;
  raw_string_ostream Os(S);
  State.Printer->printInst(&Inst, 0, "", *State.SubtargetInfo, Os);
  return StringRef(S).ltrim().str();
}

StringRef stripEncoding(StringRef Mnemonic) {
  for (StringRef Suffix : {"_e32", "_e64", "_vi"}) {
    if (Mnemonic.ends_with(Suffix)) {
      return Mnemonic.drop_back(Suffix.size());
    }
  }
  return Mnemonic;
}

std::string strippedMnemonic(const MCState &State, const MCInst &Inst) {
  return stripEncoding(getMnemonic(State, Inst)).str();
}

// Spelling the enumerators through concatenation keeps a renamed or dropped
// register a compile error rather than a silent mismatch.
#define CASE_CI_VI(Node)                                                       \
  case AMDGPU::Node##_ci:                                                      \
  case AMDGPU::Node##_vi:                                                      \
    return AMDGPU::Node;
#define CASE_VI_GFX9PLUS(Node)                                                 \
  case AMDGPU::Node##_vi:                                                      \
  case AMDGPU::Node##_gfx9plus:                                                \
    return AMDGPU::Node;
#define CASE_GFXPRE11_GFX11PLUS(Node)                                          \
  case AMDGPU::Node##_gfxpre11:                                                \
  case AMDGPU::Node##_gfx11plus:                                               \
    return AMDGPU::Node;

MCRegister stripRegEncoding(MCRegister Reg) {
  switch (Reg.id()) {
  default:
    return Reg;
    CASE_CI_VI(FLAT_SCR)
    CASE_CI_VI(FLAT_SCR_LO)
    CASE_CI_VI(FLAT_SCR_HI)
    CASE_VI_GFX9PLUS(TTMP0)
    CASE_VI_GFX9PLUS(TTMP1)
    CASE_VI_GFX9PLUS(TTMP2)
    CASE_VI_GFX9PLUS(TTMP3)
    CASE_VI_GFX9PLUS(TTMP4)
    CASE_VI_GFX9PLUS(TTMP5)
    CASE_VI_GFX9PLUS(TTMP6)
    CASE_VI_GFX9PLUS(TTMP7)
    CASE_VI_GFX9PLUS(TTMP8)
    CASE_VI_GFX9PLUS(TTMP9)
    CASE_VI_GFX9PLUS(TTMP10)
    CASE_VI_GFX9PLUS(TTMP11)
    CASE_VI_GFX9PLUS(TTMP12)
    CASE_VI_GFX9PLUS(TTMP13)
    CASE_VI_GFX9PLUS(TTMP14)
    CASE_VI_GFX9PLUS(TTMP15)
    CASE_VI_GFX9PLUS(TTMP0_TTMP1)
    CASE_VI_GFX9PLUS(TTMP2_TTMP3)
    CASE_VI_GFX9PLUS(TTMP4_TTMP5)
    CASE_VI_GFX9PLUS(TTMP6_TTMP7)
    CASE_VI_GFX9PLUS(TTMP8_TTMP9)
    CASE_VI_GFX9PLUS(TTMP10_TTMP11)
    CASE_VI_GFX9PLUS(TTMP12_TTMP13)
    CASE_VI_GFX9PLUS(TTMP14_TTMP15)
    CASE_VI_GFX9PLUS(TTMP0_TTMP1_TTMP2_TTMP3)
    CASE_VI_GFX9PLUS(TTMP4_TTMP5_TTMP6_TTMP7)
    CASE_VI_GFX9PLUS(TTMP8_TTMP9_TTMP10_TTMP11)
    CASE_VI_GFX9PLUS(TTMP12_TTMP13_TTMP14_TTMP15)
    CASE_VI_GFX9PLUS(TTMP0_TTMP1_TTMP2_TTMP3_TTMP4_TTMP5_TTMP6_TTMP7)
    CASE_VI_GFX9PLUS(TTMP4_TTMP5_TTMP6_TTMP7_TTMP8_TTMP9_TTMP10_TTMP11)
    CASE_VI_GFX9PLUS(TTMP8_TTMP9_TTMP10_TTMP11_TTMP12_TTMP13_TTMP14_TTMP15)
    CASE_VI_GFX9PLUS(
        TTMP0_TTMP1_TTMP2_TTMP3_TTMP4_TTMP5_TTMP6_TTMP7_TTMP8_TTMP9_TTMP10_TTMP11_TTMP12_TTMP13_TTMP14_TTMP15)
    CASE_GFXPRE11_GFX11PLUS(M0)
    CASE_GFXPRE11_GFX11PLUS(SGPR_NULL)
  }
}

#undef CASE_CI_VI
#undef CASE_VI_GFX9PLUS
#undef CASE_GFXPRE11_GFX11PLUS

} // namespace COMGR::hotswap
