//===- source-image-address.h - Source image address arithmetic -*- C++ -*-===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_SOURCE_IMAGE_ADDRESS_H
#define HOTSWAP_TRANSPILER_SOURCE_IMAGE_ADDRESS_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>

namespace COMGR::hotswap {

struct DecodedInst;

/// Apply a signed immediate addend to a proven source code-object address. If
/// the arithmetic would wrap, refuse the translation rather than preserving an
/// invalid PC-relative source-address fact.
llvm::Expected<uint64_t> applySourceImageByteOffset(const DecodedInst &Di,
                                                    llvm::StringRef Format,
                                                    uint64_t SourceAddr,
                                                    int64_t ByteOffset);

/// Apply a signed immediate subtrahend to a proven source code-object address.
/// This models `source_addr - imm`; a negative subtrahend becomes addition, and
/// any wrap means the PC-relative address chain cannot be materialised safely.
llvm::Expected<uint64_t> subtractSourceImageByteOffset(const DecodedInst &Di,
                                                       llvm::StringRef Format,
                                                       uint64_t SourceAddr,
                                                       int64_t ByteOffset);

} // namespace COMGR::hotswap

#endif
