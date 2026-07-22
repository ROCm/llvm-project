//===- amdgpu-formats.h - Hotswap transpiler ------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_AMDGPU_FORMATS_H
#define HOTSWAP_TRANSPILER_AMDGPU_FORMATS_H

#include "llvm/ADT/StringRef.h"

#include <cstdint>

namespace COMGR::hotswap {

// The AMDGPU instruction-format label (e.g. "SOP1", "VOP3", "FLAT") for an
// instruction with the given MC `TSFlags` and `Opcode`, or "Unknown". Consumed
// only by diagnostics; there is no dispatch on the returned string.
llvm::StringRef formatName(uint64_t TSFlags, unsigned Opcode);

} // namespace COMGR::hotswap

#endif
