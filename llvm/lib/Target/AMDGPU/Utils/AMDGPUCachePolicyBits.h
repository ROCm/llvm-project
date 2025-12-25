//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
/// \file
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_UTILS_AMDGPUCACHEPOLICYBITS_H
#define LLVM_LIB_TARGET_AMDGPU_UTILS_AMDGPUCACHEPOLICYBITS_H

namespace llvm {
namespace AMDGPU {
namespace CPol {

enum CPol {
  GLC = 1,
  SLC = 2,
  DLC = 4,
  SCC = 16,
  SC0 = GLC,
  SC1 = SCC,
  NT = SLC,
  ALL_pregfx12 = GLC | SLC | DLC | SCC,
  SWZ_pregfx12 = 8,

  // Below are GFX12+ cache policy bits

  // Temporal hint
  TH = 0x7,      // All TH bits
  TH_RT = 0,     // regular
  TH_NT = 1,     // non-temporal
  TH_HT = 2,     // high-temporal
  TH_LU = 3,     // last use
  TH_WB = 3,     // regular (CU, SE), high-temporal with write-back (MALL)
  TH_NT_RT = 4,  // non-temporal (CU, SE), regular (MALL)
  TH_RT_NT = 5,  // regular (CU, SE), non-temporal (MALL)
  TH_NT_HT = 6,  // non-temporal (CU, SE), high-temporal (MALL)
  TH_NT_WB = 7,  // non-temporal (CU, SE), high-temporal with write-back (MALL)
  TH_BYPASS = 3, // only to be used with scope = 3

  TH_RESERVED = 7, // unused value for load insts

  // Bits of TH for atomics
  TH_ATOMIC_RETURN = GLC, // Returning vs non-returning
  TH_ATOMIC_NT = SLC,     // Non-temporal vs regular
  TH_ATOMIC_CASCADE = 4,  // Cascading vs regular

  // Scope
  SCOPE_SHIFT = 3,
  SCOPE_MASK = 0x3,
  SCOPE = SCOPE_MASK << SCOPE_SHIFT, // All Scope bits
  SCOPE_CU = 0 << SCOPE_SHIFT,
  SCOPE_SE = 1 << SCOPE_SHIFT,
  SCOPE_DEV = 2 << SCOPE_SHIFT,
  SCOPE_SYS = 3 << SCOPE_SHIFT,

  NV = 1 << 5, // Non-volatile bit

  SWZ = 1 << 6, // Swizzle bit

  SCAL = 1 << 11, // Scale offset bit

  ALL = TH | SCOPE | NV,

  // Helper bits
  TH_TYPE_LOAD = 1 << 7,    // TH_LOAD policy
  TH_TYPE_STORE = 1 << 8,   // TH_STORE policy
  TH_TYPE_ATOMIC = 1 << 9,  // TH_ATOMIC policy
  TH_REAL_BYPASS = 1 << 10, // is TH=3 bypass policy or not

  // Volatile (used to preserve/signal operation volatility for buffer
  // operations not a real instruction bit)
  VOLATILE = 1 << 31,
  // The set of "cache policy" bits used for compiler features that
  // do not correspond to handware features.
  VIRTUAL_BITS = VOLATILE,
};

} // namespace CPol
} // namespace AMDGPU
} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_UTILS_AMDGPUCACHEPOLICYBITS_H
