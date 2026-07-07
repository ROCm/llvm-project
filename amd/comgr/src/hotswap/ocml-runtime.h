//===- ocml-runtime.h - Hotswap OCML device-library linking ---------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_OCML_RUNTIME_H
#define HOTSWAP_TRANSPILER_OCML_RUNTIME_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace llvm {
class FunctionCallee;
class Module;
} // namespace llvm

namespace COMGR::hotswap {

// OCML entry points used by HotSwap tanh lowering.
inline constexpr llvm::StringRef kOCMLTanhF32Symbol = "__ocml_tanh_f32";
inline constexpr llvm::StringRef kOCMLTanhF16Symbol = "__ocml_tanh_f16";

// Declare the OCML tanh helpers in the raised module. The raiser resolves
// these declarations by link-merging COMGR's embedded OCML bitcode before IR
// verification and codegen.
llvm::FunctionCallee declareOCMLTanhF32(llvm::Module &M);
llvm::FunctionCallee declareOCMLTanhF16(llvm::Module &M);

// True iff the raised module references OCML helper calls that must be resolved
// before final lowering.
bool moduleUsesOCMLRuntime(const llvm::Module &M);

// Link the embedded OCML/device-library bitcode needed by the module, inline
// the helper call chain, and DCE linked library bodies that are no longer
// reachable. Returns false after printing a precise diagnostic on failure.
llvm::Error linkOCMLRuntime(llvm::Module &M, llvm::StringRef TargetProcessor,
                            unsigned TargetWaveSize);

} // namespace COMGR::hotswap

#endif // HOTSWAP_TRANSPILER_OCML_RUNTIME_H
