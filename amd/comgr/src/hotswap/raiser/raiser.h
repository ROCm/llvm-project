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

#include "llvm/ADT/ArrayRef.h"
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

// One kernel to raise: the name the lifted function takes, the metadata
// `CodeObjectInfo` loaded and validated for it, and the extent
// [`StartOffset`, `EndOffset`) its code occupies in the shared text section. A
// zero `EndOffset` runs the kernel to the end of that section.
struct KernelRequest {
  llvm::StringRef Name;
  const KernelMeta &Meta;
  uint64_t StartOffset;
  uint64_t EndOffset;
};

// Raise every kernel in `Kernels` onto `TargetIsa`, into one module of
// amdgpu_kernel functions. `SourceIsa` names the ISA the code object was
// compiled for and `TargetIsa` the one the raised IR will be lowered for; each
// is either a bare processor (`gfx942`) or a canonical target identifier. The
// two differ whenever the raise moves a kernel to another GPU, and the wave
// projection reads both to translate a source lane into the target lane that
// runs it. The kernels share a text section and the source ISA, so they also
// share the MC layer built over it.
//
// The raise refuses rather than mislowers: either ISA naming something that is
// not an AMDGPU processor, a descriptor that does not describe a consistent
// user-SGPR layout, and any instruction outside the dispatched families all
// come back as a `RaiseFailure`. One refused kernel refuses the whole batch,
// since a module missing a kernel the caller asked for is not a usable partial
// result.
llvm::Expected<RaiseResult> raiseToIR(const TextSection &Text,
                                      llvm::StringRef SourceIsa,
                                      llvm::StringRef TargetIsa,
                                      llvm::ArrayRef<KernelRequest> Kernels);

} // namespace COMGR::hotswap

#endif
