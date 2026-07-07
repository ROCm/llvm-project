//===- tdm-runtime.h - Hotswap transpiler ---------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_TDM_RUNTIME_H
#define HOTSWAP_TRANSPILER_TDM_RUNTIME_H

// Bitcode-embedded HIP runtime helper for cross-target lifts of gfx1250
// Tensor Data Mover instructions onto gfx942.
//
// At CMake configure time, when `hipcc` is available, the project compiles
// `runtime/tdm.hip` to `tdm_runtime.gfx942.bc` and embeds the bytes via
// `cmake/EmbedBinary.cmake`. At raise time, `linkTDMRuntime` parses the
// embedded bitcode into an `llvm::Module` and link-merges it into the
// per-kernel module before `verifyModule` runs (raiser.cpp), so `llc`
// sees a self-contained module.
//
// Symbol naming is stable across builds -- `handle-vimage.cpp` declares
// the helpers via `declareTDMLoad/Store` and emits `CallInst`s by name;
// `linkTDMRuntime` resolves the calls by linking the bodies in.
//
// When the transpiler was built without hipcc, `tdmRuntimeAvailable()`
// returns false and the cross-target VIMAGE handler falls back to the
// pre-existing loud refusal path. This preserves today's no-hipcc build
// behaviour exactly.

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace llvm {
class FunctionCallee;
class Module;
} // namespace llvm

namespace COMGR::hotswap {

// Stable C-linkage names emitted by `runtime/tdm.hip`.
inline constexpr llvm::StringRef kTDMLoadSymbol  = "hotswap_tdm_load_to_lds";
inline constexpr llvm::StringRef kTDMStoreSymbol = "hotswap_tdm_store_from_lds";

// Declare (without body) the helper functions in `M`. Signature carries
// the four D# groups the emulation walk actually reads plus the source
// wave size used to scope descriptor uniformization and X striping:
//   void(<4 x i32>, <8 x i32>, <4 x i32>, <4 x i32>, i32 sourceWaveSize)
// The gfx1250 LLVM intrinsic's trailing `<8 x i32> grp4` (reserved) and
// `i32 cpol` arguments are deliberately NOT forwarded. Group 4 has no
// architectural meaning for gfx1250, and cpol is a cache-policy immediate
// with no equivalent target encoding in the helper's MUBUF-based lowering.
// The descriptor-visible semantics, including atomic-barrier updates, live
// in the D# groups passed here. `sourceWaveSize` is explicit because a
// target wave64 can contain two wave32 source waves under WaveNativeProjection;
// target-wave-local `readfirstlane` / lane-id striping would collapse those
// source waves together.
llvm::FunctionCallee declareTDMLoad(llvm::Module &M);
llvm::FunctionCallee declareTDMStore(llvm::Module &M);

// True iff the embedded bitcode blob is non-empty (i.e. hipcc was found
// at the transpiler's CMake configure time).
bool tdmRuntimeAvailable();

// True iff `M` already declares either of the helper symbols. Used by
// the raiser to decide whether a link-merge is required.
bool moduleUsesTDMRuntime(const llvm::Module &M);

// Parse the embedded `.bc` and link it into `M`. `TargetIsa` is
// informational today (we ship gfx942-only bitcode for v1); future
// per-target dispatch will key on it. On failure returns an Error; the
// caller MUST treat any error as a hard raise failure -- the unresolved
// `CallInst`s emitted by the handler would otherwise reach `llc` as
// undefined externals.
llvm::Error linkTDMRuntime(llvm::Module &M, llvm::StringRef TargetIsa);

} // namespace COMGR::hotswap

#endif // HOTSWAP_TRANSPILER_TDM_RUNTIME_H
