//===- mc-state.h - Hotswap transpiler ------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_MC_STATE_H
#define HOTSWAP_TRANSPILER_MC_STATE_H
// Include LLVM MC headers needed for the unique_ptrs
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

struct MCState {
  const llvm::Target *Target = nullptr;
  std::unique_ptr<llvm::MCInstrInfo> InstrInfo;
  std::unique_ptr<llvm::MCRegisterInfo> RegInfo;
  std::unique_ptr<llvm::MCSubtargetInfo> SubtargetInfo;
  std::unique_ptr<const llvm::MCAsmInfo> AsmInfo;
  std::unique_ptr<llvm::MCContext> Ctx;
  std::unique_ptr<llvm::MCDisassembler> Disasm;
  std::unique_ptr<llvm::MCInstPrinter> Printer;
};

// The AMDGPU triple shared by every MC object we construct here.
extern const char kAMDGPUTriple[];

// Populate `State` with the AMDGPU MC stack for `TargetIsa`. Returns
// `Error::success()` on success; on failure returns either a
// hotswap-originated `HotswapError` (Target lookup, MCSubtargetInfo /
// MCAsmInfo / MCDisassembler / MCInstPrinter creation failures) or
// forwards an upstream LLVM ErrorInfo unchanged.
llvm::Error initMCState(MCState &State, llvm::StringRef TargetIsa);

// Thin wrapper around Target::createMCSubtargetInfo for the AMDGPU triple.
// Returns a fully populated MCSubtargetInfo (feature bits honour the CPU
// name) or a `HotswapError` when the Target rejects the ISA string.
llvm::Expected<std::unique_ptr<llvm::MCSubtargetInfo>>
buildSubtargetInfo(const llvm::Target &Target, llvm::StringRef Isa);

std::string getMnemonic(const MCState &Mc, const llvm::MCInst &Inst);
llvm::StringRef stripEncoding(llvm::StringRef Mn);

} // namespace COMGR::hotswap

#endif
