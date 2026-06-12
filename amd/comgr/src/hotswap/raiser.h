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

#include <memory>
#include <string>

namespace llvm {
class LLVMContext;
class Module;
} // namespace llvm

namespace COMGR::hotswap {

struct RaiseResult {
  std::unique_ptr<llvm::LLVMContext> Ctx;
  std::unique_ptr<llvm::Module> Module;
  int LiftedCount = 0;
  int TotalCount = 0;
  std::string IrText;
  std::string DisasmText;
  // Predicate-chain classifier observations that the cross-widening
  // analysis accepted (rather than refused) for this kernel. Surfaced
  // for diagnostic attribution; counters are zero on a clean lift.
  // TODO(naming): the `c5*` identifier is prototype-era jargon and
  // should be replaced with a domain-meaningful name before this lands.
  int C5SuppressedCount = 0;
  std::string C5SuppressionReason;
  // Structured failure description. `Failure.Reason == None` iff `Success`.
  // Holds the first failure encountered.
  RaiseFailure Failure;
  // All failures collected during the per-instruction raising phase.
  llvm::SmallVector<RaiseFailure> AllFailures;
  bool Success = false;

  bool UsesScratchPrivateSegment = false;
  uint32_t SourcePrivateSegmentFixedSize = 0;
  bool HasDivergentExec = false;
};

RaiseResult raiseToIR(llvm::ArrayRef<uint8_t> TextBytes,
                      llvm::StringRef SourceIsa,
                      llvm::StringRef KernelName,
                      const KernelMeta &Meta,
                      llvm::StringRef CompilationTargetIsa = "",
                      bool EnableWritelaneRewrite = true,
                      bool EnableWaveNative = true);

RaiseResult raiseToIR(llvm::ArrayRef<uint8_t> TextBytes,
                      llvm::StringRef SourceIsa,
                      llvm::StringRef KernelName,
                      const KernelMeta &Meta,
                      uint64_t KernelOffset,
                      uint64_t KernelSize,
                      llvm::StringRef CompilationTargetIsa = "",
                      bool EnableWritelaneRewrite = true,
                      bool EnableWaveNative = true);

} // namespace COMGR::hotswap

#endif
