//===- source-image.h - Transpiler ----------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRANSPILER_SOURCE_IMAGE_H
#define TRANSPILER_SOURCE_IMAGE_H

#include "transpiler/decoder/decoded-inst.h"
#include "transpiler/raiser/raise-context.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>

namespace COMGR::transpiler {

// Move a source code-object address forward by a signed byte offset, or
// backward by one. An address that leaves the address space names nothing in
// the source image, so the arithmetic is refused rather than wrapped.
llvm::Expected<uint64_t> addSourceImageByteOffset(RaiseContext &Ctx,
                                                  const DecodedInst &Di,
                                                  uint64_t SourceAddress,
                                                  int64_t ByteOffset);
llvm::Expected<uint64_t> subtractSourceImageByteOffset(RaiseContext &Ctx,
                                                       const DecodedInst &Di,
                                                       uint64_t SourceAddress,
                                                       int64_t ByteOffset);

// Read the dword the source code object holds at a source address, or no value
// when none of its captured sections covers the four bytes there.
std::optional<uint32_t> readSourceImageDword(const RaiseContext &Ctx,
                                             uint64_t Address);

} // namespace COMGR::transpiler

#endif
