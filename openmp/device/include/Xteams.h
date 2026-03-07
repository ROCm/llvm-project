//===-------- Xteams.h - Cross team scan --------------------------- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// DeviceRTL Header file: Xteams.h
//     External __kmpc headers for single-pass cross-team scan functions using
//     the decoupled look-back algorithm.
//
// Memory requirements per kernel invocation:
//   - block_status[NumTeams + 1]: uint32_t array, initialized to 0 (INVALID)
//       The extra entry at index NumTeams is an atomic done-counter used by
//       the self-reset logic (Step 4): the last block to finish resets all
//       status entries to 0, so callers only need to zero-initialize once.
//   - block_aggregates[NumTeams]: T array (uninitialized), written once at
//   PARTIAL
//   - block_prefixes[NumTeams]: T array (uninitialized), written once at
//   COMPLETE
//   - result[Grid]: T array -- output for per-thread scan results
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_DEVICERTL_XTEAMS_LOOKBACK_H
#define OMPTARGET_DEVICERTL_XTEAMS_LOOKBACK_H

#include "DeviceTypes.h"
#include "XteamCommon.h"

#define _CD double _Complex
#define _CF float _Complex
#define _UI unsigned int
#define _UL unsigned long

extern "C" {

/// Single-pass cross-team scan using decoupled look-back algorithm
///
/// This is a single-kernel scan that completes the entire operation without
/// needing a separate Phase 2 call. Each block:
///   1. Computes its local inclusive scan
///   2. Publishes its aggregate with PARTIAL status
///   3. Looks back at predecessor blocks to compute its prefix
///   4. Marks itself COMPLETE and writes final results
///
/// Out-of-bounds threads should pass rnv as v. They participate in block
/// status publishing.
///
/// \param v Input thread local value (use rnv for out-of-bounds threads)
/// \param result Output array for per-thread scan results (size: Grid)
/// \param status Block status array (size: NumTeams + 1, init to 0)
/// \param aggregates Block aggregates array (size: NumTeams)
/// \param prefixes Block prefixes array (size: NumTeams)
/// \param rf Function pointer to reduction function
/// \param rnv Reduction null value (identity element)
/// \param k Global thread index (0 to NumTeams * BlockSize - 1)

#define _XTEAMS_DECL(T, TS)                                                     \
  void _XTEAM_EXTERN_ATTR __kmpc_xteams_##TS(                                  \
      T v, T *result, uint32_t *status, T *aggregates, T *prefixes,            \
      void (*rf)(T *, T), const T rnv, const uint64_t k);

_XTEAMS_DECL(_CD, cd)
_XTEAMS_DECL(_CF, cf)
_XTEAMS_DECL(double, d)
_XTEAMS_DECL(float, f)
_XTEAMS_DECL(int, i)
_XTEAMS_DECL(_UI, ui)
_XTEAMS_DECL(long, l)
_XTEAMS_DECL(_UL, ul)

#undef _XTEAMS_DECL

} // extern "C"

#undef _CD
#undef _CF
#undef _UI
#undef _UL

#endif // OMPTARGET_DEVICERTL_XTEAMS_LOOKBACK_H
