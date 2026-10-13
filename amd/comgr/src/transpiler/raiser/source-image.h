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

// Move a source code-object address by a byte displacement, which is negative
// when its top bit is set. Refuse a displacement that takes the address out of
// the address space, because the result names nothing in the source image.
llvm::Expected<uint64_t> moveSourceImageAddress(RaiseContext &Ctx,
                                                const DecodedInst &Di,
                                                uint64_t SourceAddress,
                                                uint64_t ByteDisplacement);

// Read the dword the source code object holds at a source address. Return no
// value if none of its captured sections covers the four bytes there.
std::optional<uint32_t> readSourceImageDword(const RaiseContext &Ctx,
                                             uint64_t Address);

// Return the source code-object address SGPR pair BaseIdx holds. Return no
// value if it holds none, and refuse the instruction if a block boundary
// dropped an address the pair was given.
llvm::Expected<std::optional<uint64_t>>
sourceImageSgprPairAddr(RaiseContext &Ctx, const DecodedInst &Di,
                        unsigned BaseIdx);

// Return the source code-object address the operand at Index names. Return no
// value if the operand is an immediate, or a register that holds no such
// address.
llvm::Expected<std::optional<uint64_t>>
sourceImageOperandAddr(RaiseContext &Ctx, const DecodedInst &Di,
                       unsigned Index);

} // namespace COMGR::transpiler

#endif
