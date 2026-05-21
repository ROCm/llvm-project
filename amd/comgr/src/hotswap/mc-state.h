//===- mc-state.h - Hotswap transpiler ------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Per-ISA AMDGPU MC stack used by the hotswap raiser: target lookup, MC
// instr/reg/subtarget info, MCContext (with an inline SourceMgr attached
// so MC-layer diagnostics do not abort), MC disassembler, and MC inst
// printer. `initMCState` constructs the bundle for a given AMDGPU CPU
// name (e.g. `gfx942`); `getMnemonic` and `stripEncoding` are
// thin helpers for canonicalizing MCInst mnemonic strings.
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_MC_STATE_H
#define HOTSWAP_TRANSPILER_MC_STATE_H

#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Error.h"

#include <memory>
#include <string>

namespace COMGR::hotswap {

/// Owns the LLVM MC objects used by the hotswap raiser for one AMDGPU
/// subtarget. Populated by `initMCState`; every member is non-null after
/// a successful init.
struct MCState {
  const llvm::Target *Target = nullptr;
  std::unique_ptr<llvm::MCInstrInfo> InstrInfo;
  std::unique_ptr<llvm::MCRegisterInfo> RegInfo;
  std::unique_ptr<llvm::MCSubtargetInfo> SubtargetInfo;
  std::unique_ptr<llvm::MCAsmInfo> AsmInfo;
  std::unique_ptr<llvm::MCContext> Ctx;
  std::unique_ptr<llvm::MCDisassembler> Disasm;
  std::unique_ptr<llvm::MCInstPrinter> Printer;
};

/// The AMDGPU triple shared by every MC object we construct here.
inline constexpr llvm::StringLiteral kAMDGPUTriple = "amdgcn-amd-amdhsa";

/// Populate `State` with the AMDGPU MC stack for `TargetIsa`. Returns
/// `Error::success()` on success; on failure returns either a
/// hotswap-originated `HotswapError` (Target lookup, MCSubtargetInfo /
/// MCAsmInfo / MCDisassembler / MCInstPrinter creation failures) or
/// forwards an upstream LLVM ErrorInfo unchanged.
llvm::Error initMCState(MCState &State, llvm::StringRef TargetIsa);

/// Thin wrapper around `Target::createMCSubtargetInfo` for the AMDGPU
/// triple. Returns a fully populated MCSubtargetInfo (feature bits honor
/// the CPU name) or a `HotswapError` when the Target rejects the ISA
/// string.
llvm::Expected<std::unique_ptr<llvm::MCSubtargetInfo>>
buildSubtargetInfo(const llvm::Target &Target, llvm::StringRef Isa);

/// Return the mnemonic token at the start of the printed disassembly of
/// `Inst`. Strips leading whitespace and discards everything from the
/// first space or tab onwards.
std::string getMnemonic(const MCState &State, const llvm::MCInst &Inst);

/// Drop a recognised LLVM mnemonic-suffix token (e.g. `_e32`, `_e64`,
/// `_vi`) from `Mn` if present; otherwise return `Mn` unchanged.
llvm::StringRef stripEncoding(llvm::StringRef Mn);

} // namespace COMGR::hotswap

#endif
