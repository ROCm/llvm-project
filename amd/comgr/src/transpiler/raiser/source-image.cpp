//===- source-image.cpp - Transpiler --------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "transpiler/raiser/source-image.h"

#include "transpiler/loader/code-object-utils.h"
#include "transpiler/raiser/handlers.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Endian.h"

#include <limits>

using namespace llvm;

namespace COMGR::transpiler {

namespace {

constexpr uint64_t DwordBytes = 4;

} // namespace

Expected<uint64_t> addSourceImageByteOffset(RaiseContext &Ctx,
                                            const DecodedInst &Di,
                                            uint64_t SourceAddress,
                                            int64_t ByteOffset) {
  if (ByteOffset < 0) {
    // Negate in unsigned so that the most negative offset keeps its magnitude.
    uint64_t Magnitude = -static_cast<uint64_t>(ByteOffset);
    if (SourceAddress < Magnitude)
      return unsupported(Ctx, Di,
                         "moves a source address before the start of the "
                         "address space");
    return SourceAddress - Magnitude;
  }
  uint64_t Magnitude = static_cast<uint64_t>(ByteOffset);
  if (Magnitude > std::numeric_limits<uint64_t>::max() - SourceAddress)
    return unsupported(Ctx, Di,
                       "moves a source address past the end of the address "
                       "space");
  return SourceAddress + Magnitude;
}

Expected<uint64_t> subtractSourceImageByteOffset(RaiseContext &Ctx,
                                                 const DecodedInst &Di,
                                                 uint64_t SourceAddress,
                                                 int64_t ByteOffset) {
  // The most negative offset has no negation that fits, so subtracting it is
  // refused rather than wrapped.
  if (ByteOffset == std::numeric_limits<int64_t>::min())
    return unsupported(Ctx, Di,
                       "subtracts an offset whose negation does not fit in a "
                       "source address");
  return addSourceImageByteOffset(Ctx, Di, SourceAddress, -ByteOffset);
}

std::optional<uint32_t> readSourceImageDword(const RaiseContext &Ctx,
                                             uint64_t Address) {
  for (const TextSection::ImageSection &Section : Ctx.sourceImageSections()) {
    if (Address < Section.Address)
      continue;
    uint64_t Offset = Address - Section.Address;
    uint64_t Size = Section.Bytes.size();
    if (Offset > Size || DwordBytes > Size - Offset)
      continue;
    return support::endian::read32le(Section.Bytes.data() + Offset);
  }
  return std::nullopt;
}

} // namespace COMGR::transpiler
