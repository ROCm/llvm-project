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

/// Inputs to `raiseToIR` beyond the code object and the kernel to lift.
struct RaiseOptions {
  // ISA the disassembly image is encoded in.
  llvm::StringRef SourceIsa;

  // ISA to compile for. Empty means the source ISA.
  llvm::StringRef CompilationTargetIsa;

  // Text-relative extent of the kernel to lift; both zero selects the whole
  // text section.
  uint64_t KernelOffset = 0;
  uint64_t KernelSize = 0;

  // Load address of `TextBytes` in the source code object, so a PC-relative
  // SMEM literal load can turn its PC into a source address.
  uint64_t TextBaseAddress = 0;

  // Allocated source sections, keyed by source code-object address, so a
  // resolved PC-relative literal can be read at raise time. Literal tables live
  // in `.rodata` as often as in `.text`, which is why this is not just
  // `TextBytes`.
  llvm::ArrayRef<TextSection::ImageSection> SourceImageSections;

  // Text-relative extents of every function symbol, from
  // `listTextFunctionExtents`. Lets the raiser follow a call or branch into an
  // outlined helper outside the kernel's own extent and lift it alongside the
  // caller. Empty makes any out-of-extent target a kernel-boundary violation.
  llvm::ArrayRef<KernelSymbolExtent> FunctionExtents;

  // Rewrite cross-lane writelane/readlane forms whose source-wave semantics do
  // not survive a wave-size change. Disabled on the ThreadLoop retry, which
  // gives those forms their source-wave scope by other means.
  bool EnableWritelaneRewrite = true;

  // Allow WaveNativeProjection for a wave-size change. Disabled on a retry that
  // has already chosen ThreadLoop or scaled modulo replication.
  bool EnableWaveNative = true;

  // Select ScaledModuloReplicationProjection for wave32->wave64 cross-widening
  // regardless of whether a C5 refusal triggered it. Offline testing only;
  // production relies on the raiser's own auto-upgrade.
  bool ForceScaledModrep = false;

  // Per-kernel lift counters, or null.
  RaiseStats *Stats = nullptr;
};

/// Raise one kernel from extracted source code-object sections to LLVM IR.
///
/// `TextBytes` is the disassembly image and `KernelName`/`Meta` select the
/// kernel within it; everything else, both ISAs included, is in `RaiseOptions`.
llvm::Expected<RaiseResult> raiseToIR(llvm::ArrayRef<uint8_t> TextBytes,
                                      llvm::StringRef KernelName,
                                      const KernelMeta &Meta,
                                      const RaiseOptions &Options = {});

} // namespace COMGR::hotswap

#endif
