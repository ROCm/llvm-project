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

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <memory>

namespace llvm {
class LLVMContext;
class Module;
} // namespace llvm

namespace COMGR::hotswap {

struct RaiseResult {
  std::unique_ptr<llvm::LLVMContext> Ctx;
  std::unique_ptr<llvm::Module> Module;
};

// Raise a kernel named `KernelName` whose source ISA is `SourceISA`. `Meta`
// carries the MsgPack-derived per-kernel metadata. The scaffolding
// implementation emits a `ret void` placeholder and refuses inputs the full
// pipeline would also refuse:
//
//   * Scaffolding / empty-input mode: when both `SourceISA` and
//     `KernelName` are empty, validation is skipped and a placeholder
//     module is returned. Useful for stubbing the raiser in tests
//     without setting up a real ISA.
//
//   * Non-empty mode (anything else): both strings must be non-empty,
//     `SourceISA` must parse via `llvm::AMDGPU::parseArchAMDGCN`, and
//     `Meta.HasKernelDescriptor` must be true.
//
// Returns a `HotswapError` on rejected input; once wired in, decoder
// failures will likewise propagate as `llvm::Error` (forwarded from
// the MC layer or freshly created `HotswapError`s for raiser-internal
// mismatches). The kernel-text bytes, kernel offset, and compilation-
// target ISA become real parameters once the decoder is wired in
// (subsequent commit).
llvm::Expected<RaiseResult> raiseToIR(llvm::StringRef SourceISA,
                                      llvm::StringRef KernelName,
                                      const KernelMeta &Meta);

} // namespace COMGR::hotswap

#endif
