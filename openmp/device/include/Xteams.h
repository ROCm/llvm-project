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
//   - block_status[NumTeams]: uint32_t array, initialized to 0 (INVALID)
//   - block_aggregates[NumTeams]: T array (uninitialized)
//   - block_prefixes[NumTeams]: T array (uninitialized)
//   - result[NumTeams * BlockSize]: T array for final scan results
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
/// \param result Output array for final scan results (grid-sized)
/// \param status Block status array (size: NumTeams, init to 0)
/// \param agg Block aggregates array (size: NumTeams)
/// \param prefix Block inclusive prefix array (size: NumTeams)
/// \param rf Function pointer to reduction function
/// \param rnv Reduction null value (identity element)
/// \param k Global thread index (0 to NumTeams * BlockSize - 1)
/// \param n Number of elements in the scan (loop trip count)
/// \param is_inclusive True for inclusive scan, false for exclusive

void _XTEAM_EXTERN_ATTR __kmpc_xteams_d(double v, double *result,
                                        uint32_t *status, double *agg,
                                        double *prefix,
                                        void (*rf)(double *, double),
                                        const double rnv, const uint64_t k,
                                        const uint64_t n, bool is_inclusive);

void _XTEAM_EXTERN_ATTR __kmpc_xteams_f(float v, float *result,
                                        uint32_t *status, float *agg,
                                        float *prefix,
                                        void (*rf)(float *, float),
                                        const float rnv, const uint64_t k,
                                        const uint64_t n, bool is_inclusive);

void _XTEAM_EXTERN_ATTR __kmpc_xteams_i(int v, int *result, uint32_t *status,
                                        int *agg, int *prefix,
                                        void (*rf)(int *, int), const int rnv,
                                        const uint64_t k, const uint64_t n,
                                        bool is_inclusive);

void _XTEAM_EXTERN_ATTR __kmpc_xteams_ui(_UI v, _UI *result, uint32_t *status,
                                         _UI *agg, _UI *prefix,
                                         void (*rf)(_UI *, _UI), const _UI rnv,
                                         const uint64_t k, const uint64_t n,
                                         bool is_inclusive);

void _XTEAM_EXTERN_ATTR __kmpc_xteams_l(long v, long *result, uint32_t *status,
                                        long *agg, long *prefix,
                                        void (*rf)(long *, long),
                                        const long rnv, const uint64_t k,
                                        const uint64_t n, bool is_inclusive);

void _XTEAM_EXTERN_ATTR __kmpc_xteams_ul(_UL v, _UL *result, uint32_t *status,
                                         _UL *agg, _UL *prefix,
                                         void (*rf)(_UL *, _UL), const _UL rnv,
                                         const uint64_t k, const uint64_t n,
                                         bool is_inclusive);

void _XTEAM_EXTERN_ATTR __kmpc_xteams_cd(_CD v, _CD *result, uint32_t *status,
                                         _CD *agg, _CD *prefix,
                                         void (*rf)(_CD *, _CD), const _CD rnv,
                                         const uint64_t k, const uint64_t n,
                                         bool is_inclusive);

void _XTEAM_EXTERN_ATTR __kmpc_xteams_cf(_CF v, _CF *result, uint32_t *status,
                                         _CF *agg, _CF *prefix,
                                         void (*rf)(_CF *, _CF), const _CF rnv,
                                         const uint64_t k, const uint64_t n,
                                         bool is_inclusive);

} // extern "C"

#undef _CD
#undef _CF
#undef _UI
#undef _UL

#endif // OMPTARGET_DEVICERTL_XTEAMS_LOOKBACK_H
