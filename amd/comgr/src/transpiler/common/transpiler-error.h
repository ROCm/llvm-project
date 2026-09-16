//===- transpiler-error.h - Transpiler-originated llvm::Error payload ----===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRANSPILER_ERROR_H
#define TRANSPILER_ERROR_H

#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <system_error>

namespace COMGR::transpiler {

/// Error for malformed input the transpiler detects itself (a missing
/// ELF section, a kernel absent from the AMDGPU metadata, an invalid kernel
/// descriptor, ...), as opposed to errors forwarded unchanged from lower LLVM
/// layers.
class TranspilerError : public llvm::ErrorInfo<TranspilerError> {
public:
  static char ID;
  std::string Msg;

  explicit TranspilerError(const llvm::Twine &Detail) : Msg(Detail.str()) {}

  void log(llvm::raw_ostream &OS) const override { OS << "transpiler: " << Msg; }

  std::error_code convertToErrorCode() const override {
    return llvm::inconvertibleErrorCode();
  }
};

/// Build a `TranspilerError` wrapped in an `llvm::Error`.
inline llvm::Error makeTranspilerError(const llvm::Twine &Detail) {
  return llvm::make_error<TranspilerError>(Detail);
}

} // namespace COMGR::transpiler

#endif
