//===- source-image-address.cpp -------------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "source-image-address.h"

#include "decoded-inst.h"
#include "raise-failure.h"

#include "llvm/Support/CheckedArithmetic.h"

#include <cassert>
#include <optional>

using namespace llvm;

namespace COMGR::hotswap {

namespace {

/// Return `abs(Offset)` for a negative offset without evaluating
/// `-INT64_MIN`, which would overflow before conversion to uint64_t.
uint64_t signedOffsetMagnitude(int64_t Offset) {
  assert(Offset < 0 && "expected a negative offset");
  return static_cast<uint64_t>(-(Offset + 1)) + 1;
}

Error sourceImageAddressFailure(const DecodedInst &Di, StringRef Format,
                                const Twine &Detail) {
  return RaiseFailure::unsupportedInstructionForm(
      Di, Format, Twine("source-image ") + Format + " address " + Detail);
}

} // namespace

Expected<uint64_t> applySourceImageByteOffset(const DecodedInst &Di,
                                              StringRef Format,
                                              uint64_t SourceAddr,
                                              int64_t ByteOffset) {
  if (ByteOffset < 0) {
    uint64_t Magnitude = signedOffsetMagnitude(ByteOffset);
    assert(SourceAddr >= Magnitude &&
           "source-image address must not underflow signed offset");
    if (SourceAddr < Magnitude)
      return sourceImageAddressFailure(Di, Format,
                                       "underflows its signed constant offset");
    return SourceAddr - Magnitude;
  }

  if (std::optional<uint64_t> Sum =
          checkedAddUnsigned(SourceAddr, static_cast<uint64_t>(ByteOffset)))
    return *Sum;

  return sourceImageAddressFailure(Di, Format,
                                   "overflows its signed constant offset");
}

Expected<uint64_t> subtractSourceImageByteOffset(const DecodedInst &Di,
                                                 StringRef Format,
                                                 uint64_t SourceAddr,
                                                 int64_t ByteOffset) {
  if (ByteOffset < 0) {
    uint64_t Magnitude = signedOffsetMagnitude(ByteOffset);
    if (std::optional<uint64_t> Sum = checkedAddUnsigned(SourceAddr, Magnitude))
      return *Sum;
    return sourceImageAddressFailure(
        Di, Format, "overflows its negative constant subtrahend");
  }

  uint64_t Magnitude = static_cast<uint64_t>(ByteOffset);
  assert(SourceAddr >= Magnitude &&
         "source-image address must not underflow constant subtrahend");
  if (SourceAddr < Magnitude)
    return sourceImageAddressFailure(Di, Format,
                                     "underflows its constant subtrahend");
  return SourceAddr - Magnitude;
}

} // namespace COMGR::hotswap
