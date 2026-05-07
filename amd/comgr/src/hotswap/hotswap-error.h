//===- hotswap-error.h - Hotswap-originated llvm::Error payload ----------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// `HotswapError` is the dedicated `llvm::ErrorInfo` subclass for failures
// that the hotswap transpiler detects itself: missing ELF sections,
// kernels absent from the AMDGPU MsgPack metadata, MC target-stack
// construction failures, etc. Errors that the transpiler *forwards*
// from `llvm::object`, `MC`, or `COMGR::lookupSymbolByName` are passed
// through as their original ErrorInfo type, so callers can still
// `handleErrors` on them and tell hotswap-originated failures apart
// from upstream LLVM ones.
//
// The payload is intentionally just a message string -- there's no
// HotswapErrorCode enum to maintain. Discrimination across the two
// error origins (hotswap-internal vs. forwarded) happens at the
// ErrorInfo *type* level via `Err.isA<HotswapError>()` /
// `handleErrors(... [](const HotswapError &) { ... })`.
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_HOTSWAP_ERROR_H
#define HOTSWAP_TRANSPILER_HOTSWAP_ERROR_H

#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <system_error>

namespace COMGR::hotswap {

class HotswapError : public llvm::ErrorInfo<HotswapError> {
public:
  static char ID;
  std::string Msg;

  explicit HotswapError(const llvm::Twine &Detail) : Msg(Detail.str()) {}

  void log(llvm::raw_ostream &OS) const override { OS << "hotswap: " << Msg; }

  std::error_code convertToErrorCode() const override {
    return llvm::inconvertibleErrorCode();
  }
};

inline llvm::Error makeHotswapError(const llvm::Twine &Detail) {
  return llvm::make_error<HotswapError>(Detail);
}

} // namespace COMGR::hotswap

#endif
