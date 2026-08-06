//===- hotswap-error.h - Hotswap-originated llvm::Error payload ----------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

/// Error for malformed input the hotswap transpiler detects itself (a missing
/// ELF section, a kernel absent from the AMDGPU metadata, an invalid kernel
/// descriptor, ...), as opposed to errors forwarded unchanged from lower LLVM
/// layers.
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

/// Build a `HotswapError` wrapped in an `llvm::Error`.
inline llvm::Error makeHotswapError(const llvm::Twine &Detail) {
  return llvm::make_error<HotswapError>(Detail);
}

} // namespace COMGR::hotswap

#endif
