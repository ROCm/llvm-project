//===- raiser.h - Hotswap MC -> LLVM IR raiser entry point --------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_RAISER_H
#define HOTSWAP_TRANSPILER_RAISER_H

#include "code-object-utils.h"
#include "raise-failure.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <memory>
#include <string>

namespace llvm {
class LLVMContext;
class Module;
} // namespace llvm

namespace COMGR::hotswap {

// Instruction-lifting statistics collected during a raise. Passed to
// raiseToIR by pointer; a null pointer means the caller does not want stats.
struct RaiseStats {
  int LiftedCount = 0;
  int TotalCount = 0;
};

struct RaiseResult {
  std::unique_ptr<llvm::LLVMContext> Ctx;
  std::unique_ptr<llvm::Module> Module;
  // Source disassembly, populated only when HSA_HOTSWAP_DUMP_INPUT=1 for the
  // `.dis` debug dump; empty on the production path.
  std::string DisasmText;
  // Predicate-chain classifier observations that the cross-widening
  // analysis accepted (rather than refused) for this kernel. Surfaced
  // for diagnostic attribution; counters are zero on a clean lift.
  // TODO(naming): the `c5*` identifier is prototype-era jargon and
  // should be replaced with a domain-meaningful name before this lands.
  int C5SuppressedCount = 0;
  std::string C5SuppressionReason;
  bool UsesScratchPrivateSegment = false;
  uint32_t SourcePrivateSegmentFixedSize = 0;
  bool HasDivergentExec = false;
  bool HasEnumeratedSetpcDispatch = false;
  // Factor by which the runtime must scale the block's x extent under
  // ScaledModuloReplicationProjection; 1 means no scaling. See sec. 10 of
  // hotswap/docs/modrep-predicate-chain.md.
  unsigned ScaledDispatchFactor = 1;
};

// Raise one kernel from extracted source code-object sections. `TextBytes`
// remains the disassembly image; `TextBaseAddress` and `SourceImageSections`
// let PC-relative SMEM literal loads resolve source code-object addresses at
// raise time.
llvm::Expected<RaiseResult>
raiseToIR(llvm::ArrayRef<uint8_t> TextBytes, llvm::StringRef SourceIsa,
          llvm::StringRef KernelName, const KernelMeta &Meta,
          llvm::StringRef CompilationTargetIsa = "",
          bool EnableWritelaneRewrite = true, bool EnableWaveNative = true,
          uint64_t TextBaseAddress = 0,
          llvm::ArrayRef<TextSection::ImageSection> SourceImageSections = {},
          RaiseStats *Stats = nullptr);

llvm::Expected<RaiseResult>
raiseToIR(llvm::ArrayRef<uint8_t> TextBytes, llvm::StringRef SourceIsa,
          llvm::StringRef KernelName, const KernelMeta &Meta,
          uint64_t KernelOffset, uint64_t KernelSize,
          llvm::StringRef CompilationTargetIsa = "",
          bool EnableWritelaneRewrite = true, bool EnableWaveNative = true,
          bool AssumeHipGlobalOffsetZero = false,
          // Unconditionally select ScaledModuloReplicationProjection for
          // wave32->wave64 cross-widening (offline testing of the scaled
          // in-kernel virtualization independent of a C5 refusal trigger). The
          // normal path needs no flag: the raiser auto-upgrades the WaveNative
          // y/z-derived C5 refusal to a scaled dispatch by default.
          bool ForceScaledModrep = false, uint64_t TextBaseAddress = 0,
          llvm::ArrayRef<TextSection::ImageSection> SourceImageSections = {},
          // Text-relative extents of all function symbols in the code object
          // (from listTextFunctionExtents). Lets the raiser follow a
          // call/branch into an outlined helper outside the selected kernel's
          // own extent and lift it alongside the caller. Empty (the default)
          // keeps the legacy behavior: any out-of-extent target is a
          // kernel-boundary violation.
          llvm::ArrayRef<KernelSymbolExtent> FunctionExtents = {},
          RaiseStats *Stats = nullptr);

} // namespace COMGR::hotswap

#endif
