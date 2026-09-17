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

// Move SourceAddress by Magnitude bytes in the direction Backwards names.
Expected<uint64_t> moveSourceImageAddress(RaiseContext &Ctx,
                                          const DecodedInst &Di,
                                          uint64_t SourceAddress,
                                          uint64_t Magnitude, bool Backwards) {
  if (Backwards) {
    if (SourceAddress < Magnitude)
      return unsupported(Ctx, Di,
                         "moves a source address before the start of the "
                         "address space");
    return SourceAddress - Magnitude;
  }
  if (Magnitude > std::numeric_limits<uint64_t>::max() - SourceAddress)
    return unsupported(Ctx, Di,
                       "moves a source address past the end of the address "
                       "space");
  return SourceAddress + Magnitude;
}

// Distance an offset names, as an unsigned count of bytes. The negation is
// unsigned so that the most negative offset keeps its magnitude.
uint64_t byteOffsetMagnitude(int64_t ByteOffset) {
  uint64_t Magnitude = static_cast<uint64_t>(ByteOffset);
  return ByteOffset < 0 ? -Magnitude : Magnitude;
}

} // namespace

Expected<uint64_t> addSourceImageByteOffset(RaiseContext &Ctx,
                                            const DecodedInst &Di,
                                            uint64_t SourceAddress,
                                            int64_t ByteOffset) {
  return moveSourceImageAddress(
      Ctx, Di, SourceAddress, byteOffsetMagnitude(ByteOffset), ByteOffset < 0);
}

Expected<uint64_t> subtractSourceImageByteOffset(RaiseContext &Ctx,
                                                 const DecodedInst &Di,
                                                 uint64_t SourceAddress,
                                                 int64_t ByteOffset) {
  return moveSourceImageAddress(
      Ctx, Di, SourceAddress, byteOffsetMagnitude(ByteOffset), ByteOffset >= 0);
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
