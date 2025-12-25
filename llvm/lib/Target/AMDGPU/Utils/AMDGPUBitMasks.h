//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_UTILS_AMDGPUBITMASKS_H
#define LLVM_LIB_TARGET_AMDGPU_UTILS_AMDGPUBITMASKS_H

namespace {

/// \returns Bit mask for given bit \p Shift and bit \p Width.
unsigned getBitMask(unsigned Shift, unsigned Width) {
  return ((1 << Width) - 1) << Shift;
}

/// Packs \p Src into \p Dst for given bit \p Shift and bit \p Width.
///
/// \returns Packed \p Dst.
unsigned packBits(unsigned Src, unsigned Dst, unsigned Shift, unsigned Width) {
  unsigned Mask = getBitMask(Shift, Width);
  return ((Src << Shift) & Mask) | (Dst & ~Mask);
}

/// Unpacks bits from \p Src for given bit \p Shift and bit \p Width.
///
/// \returns Unpacked bits.
unsigned unpackBits(unsigned Src, unsigned Shift, unsigned Width) {
  return (Src & getBitMask(Shift, Width)) >> Shift;
}

} // end anonymous namespace

#endif // LLVM_LIB_TARGET_AMDGPU_UTILS_AMDGPUBITMASKS_H
