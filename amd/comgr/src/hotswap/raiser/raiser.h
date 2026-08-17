//===- raiser.h - Hotswap MC -> LLVM IR raiser entry point --------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_RAISER_H
#define HOTSWAP_TRANSPILER_RAISER_H

#include "hotswap/common/kernel-meta.h"
#include "hotswap/loader/code-object-utils.h"
#include "raise_failure.h"

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
  // Predicate-chain observations the widening analysis accepted rather
  // than refused for this kernel. Zero on a clean lift.
  int C5SuppressedCount = 0;
  std::string C5SuppressionReason;
  bool UsesScratchPrivateSegment = false;
  uint32_t SourcePrivateSegmentFixedSize = 0;
  bool HasDivergentExec = false;
  // Set when the kernel was raised under ModRepDoubledDispatchProjection: the
  // block dimension (0=x,1=y,2=z) the runtime must scale and the factor
  // (W_t/W_s). Dim is -1 (factor 1) when no doubled dispatch is required. The
  // launch runtime reads these (threaded through the comgr transpile result and
  // the loader) to scale exactly the doubled kernels' dispatch.
  int DoubledDispatchDim = -1;
  unsigned DoubledDispatchFactor = 1;
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
          // Unconditionally select ModRepDoubledDispatchProjection for
          // wave32->wave64 widening (offline testing of the doubled
          // in-kernel virtualization independent of a C5 refusal trigger). The
          // normal path needs no flag: the raiser auto-upgrades the WaveNative
          // y/z-derived C5 refusal to a doubled dispatch by default.
          bool ForceReplicationDoubled = false, uint64_t TextBaseAddress = 0,
          llvm::ArrayRef<TextSection::ImageSection> SourceImageSections = {},
          RaiseStats *Stats = nullptr);

} // namespace COMGR::hotswap

#endif
