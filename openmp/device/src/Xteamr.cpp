//===---- Xteamr.cpp - OpenMP cross team helper functions ---- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains helper functions for cross team reductions
//
//===----------------------------------------------------------------------===//

#include "Xteamr.h"
#include "Mapping.h"
#ifdef __NVPTX__
#include "Interface.h"
#endif

using namespace ompx;

/// Templated internal function used by all extern typed reductions
///
/// Uses shared primitives from XteamCommon.h for wave and block operations.
///
/// \param T Template typename parameter T
/// \param _IS_FAST Template parameter for fast atomic path
/// \param val Input thread local value
/// \param r_ptr Pointer to result value
/// \param team_vals Global array of team values
/// \param teams_done_ptr Pointer to atomic teams done counter
/// \param _rf Function pointer to TLS pair reduction function
/// \param _rf_lds Function pointer to LDS pair reduction function
/// \param rnv Reduction null value
/// \param k The iteration value from 0 to (NumTeams*NumThreads)-1
/// \param NumTeams The number of teams
/// \param Scope The scope of the atomic operation
///
/// Note that block=team and warp=wave.
///
template <typename T, const bool _IS_FAST = false>
_XTEAM_INLINE_ATTR void
_xteam_reduction(T val, T *r_ptr, T *team_vals, uint32_t *teams_done_ptr,
                 void (*_rf)(T *, T), void (*_rf_lds)(_RF_LDS T *, _RF_LDS T *),
                 const T rnv, const uint64_t k, const uint32_t NumTeams,
                 ompx::atomic::MemScopeTy Scope) {

  const uint32_t block_size = mapping::getNumberOfThreadsInBlock();
  const uint32_t omp_thread_num = k % block_size;
  const uint32_t omp_team_num = k / block_size;

  // LDS array for wave results
  static _RF_LDS T xwave_lds[_XTEAM_MAX_NUM_WAVES];

// Cuda may restrict max threads, so clear unused wave values
#ifdef __NVPTX__
  const uint32_t number_of_waves = (block_size - 1) / _XTEAM_WARP_SIZE + 1;
  if (number_of_waves == 32) {
    if (omp_thread_num == 0) {
      for (uint32_t i = (omp_get_num_threads() / _XTEAM_WARP_SIZE);
           i < number_of_waves; i++)
        xwave_lds[i] = rnv;
    }
  }
#endif

  // Use shared block_reduce primitive for intra-team reduction
  // Note: this returns the reduced value *only* in thread 0
  T team_result = xteam::block_reduce(val, _rf, _rf_lds, rnv, xwave_lds);

  if constexpr (_IS_FAST) {
    // Fast path: use atomic add directly
    if (omp_thread_num == 0)
      ompx::atomic::add(r_ptr, team_result, ompx::atomic::relaxed, Scope);
  } else if (NumTeams == 1) {
    // Single team: just write result
    if (omp_thread_num == 0)
      *r_ptr = team_result;
    synchronize::threadsAligned(atomic::relaxed);
  } else {
    // No sync needed here from last reduction in LDS loop
    // because we only need team_result correct on thread 0.

    // Save the teams reduced value in team_vals global array
    // and atomically increment teams_done counter.
    static _RF_LDS uint32_t td;
    if (omp_thread_num == 0) {
      team_vals[omp_team_num] = team_result;
      td = atomic::inc(teams_done_ptr, NumTeams - 1u, atomic::acq_rel,
                       atomic::MemScopeTy::device);
    }

    // This sync needed so all threads from last team see the shared volatile
    // value td (teams done counter) so they know they are in the last team.
    synchronize::threadsAligned(atomic::acq_rel);

    // If td counter reaches NumTeams-1, this is the last team.
    // The team number of this last team is nondeterministic.
    if (td == (NumTeams - 1u)) {
      // Last team performs final reduction across all team values

      // Acquire all teams' team_vals before TLS shfl reduce
      val = (omp_thread_num < NumTeams) ? team_vals[omp_thread_num] : rnv;

      // Need sync here to prepare for TLS shfl reduce.
      synchronize::threadsAligned(atomic::relaxed);

      // Use block_reduce again for final reduction
      // Note: this returns the reduced value *only* in thread 0
      T final_result = xteam::block_reduce(val, _rf, _rf_lds, rnv, xwave_lds);

      if (omp_thread_num == 0) {
        // Reduce with the original result value.
        (*_rf)(&final_result, *r_ptr);

        // If more teams than threads, do non-parallel reduction of extra
        // team_vals. This loop iterates only if NumTeams > block_size.
        for (unsigned offset = block_size; offset < NumTeams; offset++)
          (*_rf)(&final_result, team_vals[offset]);

        *r_ptr = final_result;
      }

      // Prevent warps from starting next reduction early
      synchronize::threadsAligned(atomic::relaxed);
    }
  }
}

/// Internal macro used by extern intra-team reductions
///
/// \param T Template typename parameter T
/// \param val Input thread local (TLS) value for warp shfl reduce
/// \param r_ptr Pointer to result value, also used in final reduction
/// \param _rf Function pointer to TLS pair reduction function
/// \param _rf_lds Function pointer to LDS pair reduction function
/// \param rnv Reduction null value, used for partial waves
/// \param k The iteration value from 0 to (NumTeams*_NUM_THREADS)-1
///
#define _iteam_reduction(T, val, r_ptr, _rf, _rf_lds, rnv, k)                  \
  _xteam_reduction<T>((val), (r_ptr), nullptr, nullptr, (_rf), (_rf_lds),      \
                      (rnv), (k), 1, ompx::atomic::MemScopeTy::single)

//===----------------------------------------------------------------------===//
// Extern C wrapper functions
//
// Calls to these __kmpc extern C functions are created in clang codegen
// for FORTRAN, c, and C++. They may also be used for simulation and testing.
// The headers for these extern C functions are in ../include/Interface.h
// The compiler builds the name based on the data type.
//===----------------------------------------------------------------------===//

#define _EXT_ATTR extern "C" _XTEAM_EXTERN_ATTR void
#define _CD double _Complex
#define _CF float _Complex
#define _US unsigned short
#define _UI unsigned int
#define _UL unsigned long

#define _XTEAMR_DEF(T, TS)                                                     \
  _EXT_ATTR __kmpc_xteamr_##TS(                                                \
      T v, T *r_p, T *tvs, uint32_t *td, void (*rf)(T *, T),                   \
      void (*rflds)(_RF_LDS T *, _RF_LDS T *), const T rnv, const uint64_t k,  \
      const uint32_t nt, ompx::atomic::MemScopeTy Scope) {                     \
    _xteam_reduction<T>(v, r_p, tvs, td, rf, rflds, rnv, k, nt, Scope);        \
  }

_XTEAMR_DEF(__bf16, bf)
_XTEAMR_DEF(_Float16, h)
_XTEAMR_DEF(double, d)
_XTEAMR_DEF(float, f)
_XTEAMR_DEF(int, i)
_XTEAMR_DEF(_UI, ui)
_XTEAMR_DEF(long, l)
_XTEAMR_DEF(_UL, ul)
_XTEAMR_DEF(short, s)
_XTEAMR_DEF(_US, us)

#undef _XTEAMR_DEF

#define _XTEAMR_DEF_FAST_SUM(T, TS)                                            \
  _EXT_ATTR __kmpc_xteamr_##TS##_fast_sum(                                     \
      T v, T *r_p, T *tvs, uint32_t *td, void (*rf)(T *, T),                   \
      void (*rflds)(_RF_LDS T *, _RF_LDS T *), const T rnv, const uint64_t k,  \
      const uint32_t nt, ompx::atomic::MemScopeTy Scope) {                     \
    _xteam_reduction<T, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt, Scope);  \
  }

_XTEAMR_DEF_FAST_SUM(__bf16, bf)
_XTEAMR_DEF_FAST_SUM(_Float16, h)
_XTEAMR_DEF_FAST_SUM(double, d)
_XTEAMR_DEF_FAST_SUM(float, f)
_XTEAMR_DEF_FAST_SUM(int, i)
_XTEAMR_DEF_FAST_SUM(_UI, ui)
_XTEAMR_DEF_FAST_SUM(long, l)
_XTEAMR_DEF_FAST_SUM(_UL, ul)
_XTEAMR_DEF_FAST_SUM(short, s)
_XTEAMR_DEF_FAST_SUM(_US, us)

#undef _XTEAMR_DEF_FAST_SUM

#define _ITEAMR_DEF(T, TS)                                                     \
  _EXT_ATTR __kmpc_iteamr_##TS(T v, T *r_p, void (*rf)(T *, T),                \
                               void (*rflds)(_RF_LDS T *, _RF_LDS T *),        \
                               const T rnv, const uint64_t k) {                \
    _iteam_reduction(T, v, r_p, rf, rflds, rnv, k);                            \
  }

_ITEAMR_DEF(__bf16, bf)
_ITEAMR_DEF(_Float16, h)
_ITEAMR_DEF(double, d)
_ITEAMR_DEF(float, f)
_ITEAMR_DEF(int, i)
_ITEAMR_DEF(_UI, ui)
_ITEAMR_DEF(long, l)
_ITEAMR_DEF(_UL, ul)
_ITEAMR_DEF(short, s)
_ITEAMR_DEF(_US, us)

#undef _ITEAMR_DEF

//===----------------------------------------------------------------------===//
// Built-in pair reduction functions used as function pointers for
// cross team reduction functions.
//===----------------------------------------------------------------------===//

#define _REDUCTION_FUNCTION_SUM_IMPL(T, TS)                                    \
  _EXT_ATTR __kmpc_rfun_sum_##TS(T *val, T otherval) { *val += otherval; }
#define _REDUCTION_FUNCTION_LDS_SUM_IMPL(T, TS)                                \
  _EXT_ATTR __kmpc_rfun_sum_lds_##TS(_RF_LDS T *val, _RF_LDS T *otherval) {    \
    *val += *otherval;                                                         \
  }
#define _REDUCTION_FUNCTION_MAX_IMPL(T, TS)                                    \
  _EXT_ATTR __kmpc_rfun_max_##TS(T *val, T otherval) {                         \
    *val = (otherval > *val) ? otherval : *val;                                \
  }
#define _REDUCTION_FUNCTION_LDS_MAX_IMPL(T, TS)                                \
  _EXT_ATTR __kmpc_rfun_max_lds_##TS(_RF_LDS T *val, _RF_LDS T *otherval) {    \
    *val = (*otherval > *val) ? *otherval : *val;                              \
  }
#define _REDUCTION_FUNCTION_MIN_IMPL(T, TS)                                    \
  _EXT_ATTR __kmpc_rfun_min_##TS(T *val, T otherval) {                         \
    *val = (otherval < *val) ? otherval : *val;                                \
  }
#define _REDUCTION_FUNCTION_LDS_MIN_IMPL(T, TS)                                \
  _EXT_ATTR __kmpc_rfun_min_lds_##TS(_RF_LDS T *val, _RF_LDS T *otherval) {    \
    *val = (*otherval < *val) ? *otherval : *val;                              \
  }

#define _REDUCTION_FUNCTION_ALL_IMPL(T, TS)                                    \
  _REDUCTION_FUNCTION_SUM_IMPL(T, TS)                                          \
  _REDUCTION_FUNCTION_LDS_SUM_IMPL(T, TS)                                      \
  _REDUCTION_FUNCTION_MAX_IMPL(T, TS)                                          \
  _REDUCTION_FUNCTION_LDS_MAX_IMPL(T, TS)                                      \
  _REDUCTION_FUNCTION_MIN_IMPL(T, TS)                                          \
  _REDUCTION_FUNCTION_LDS_MIN_IMPL(T, TS)

_REDUCTION_FUNCTION_ALL_IMPL(__bf16, bf)
_REDUCTION_FUNCTION_ALL_IMPL(_Float16, h)
_REDUCTION_FUNCTION_ALL_IMPL(double, d)
_REDUCTION_FUNCTION_ALL_IMPL(float, f)
_REDUCTION_FUNCTION_ALL_IMPL(int, i)
_REDUCTION_FUNCTION_ALL_IMPL(_UI, ui)
_REDUCTION_FUNCTION_ALL_IMPL(long, l)
_REDUCTION_FUNCTION_ALL_IMPL(_UL, ul)
_REDUCTION_FUNCTION_ALL_IMPL(short, s)
_REDUCTION_FUNCTION_ALL_IMPL(_US, us)

#undef _REDUCTION_FUNCTION_ALL_IMPL
#undef _REDUCTION_FUNCTION_MAX_IMPL
#undef _REDUCTION_FUNCTION_LDS_MAX_IMPL
#undef _REDUCTION_FUNCTION_MIN_IMPL
#undef _REDUCTION_FUNCTION_LDS_MIN_IMPL
#undef _REDUCTION_FUNCTION_SUM_IMPL
#undef _REDUCTION_FUNCTION_LDS_SUM_IMPL

#undef _CD
#undef _CF
#undef _US
#undef _UI
#undef _UL
#undef _EXT_ATTR
