//===- hotswap-error.cpp - Out-of-line HotswapError ID symbol ------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap-error.h"

namespace COMGR::hotswap {

char HotswapError::ID = 0;

} // namespace COMGR::hotswap
