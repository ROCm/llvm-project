//===- transpiler-error.cpp - Out-of-line TranspilerError ID symbol ------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "transpiler/common/transpiler-error.h"

namespace COMGR::transpiler {

char TranspilerError::ID = 0;

} // namespace COMGR::transpiler
