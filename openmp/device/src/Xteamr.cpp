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

//===----------------------------------------------------------------------===//
// Cross-team reduction implementation using shared primitives
//===----------------------------------------------------------------------===//

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
      for (uint32_t i = (omp_get_num_threads() / 32); i < number_of_waves; i++)
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

_EXT_ATTR
__kmpc_xteamr_d(double v, double *r_p, double *tvs, uint32_t *td,
                void (*rf)(double *, double),
                void (*rflds)(_RF_LDS double *, _RF_LDS double *),
                const double rnv, const uint64_t k, const uint32_t nt,
                ompx::atomic::MemScopeTy Scope) {
  _xteam_reduction<double>(v, r_p, tvs, td, rf, rflds, rnv, k, nt, Scope);
}
_EXT_ATTR
__kmpc_xteamr_d_fast_sum(double v, double *r_p, double *tvs, uint32_t *td,
                         void (*rf)(double *, double),
                         void (*rflds)(_RF_LDS double *, _RF_LDS double *),
                         const double rnv, const uint64_t k, const uint32_t nt,
                         ompx::atomic::MemScopeTy Scope) {
  _xteam_reduction<double, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt, Scope);
}
_EXT_ATTR
__kmpc_iteamr_d(double v, double *r_p, void (*rf)(double *, double),
                void (*rflds)(_RF_LDS double *, _RF_LDS double *),
                const double rnv, const uint64_t k) {
  _iteam_reduction(double, v, r_p, rf, rflds, rnv, k);
}

_EXT_ATTR
__kmpc_xteamr_f(float v, float *r_p, float *tvs, uint32_t *td,
                void (*rf)(float *, float),
                void (*rflds)(_RF_LDS float *, _RF_LDS float *),
                const float rnv, const uint64_t k, const uint32_t nt,
                ompx::atomic::MemScopeTy Scope) {
  _xteam_reduction<float>(v, r_p, tvs, td, rf, rflds, rnv, k, nt, Scope);
}
_EXT_ATTR
__kmpc_xteamr_f_fast_sum(float v, float *r_p, float *tvs, uint32_t *td,
                         void (*rf)(float *, float),
                         void (*rflds)(_RF_LDS float *, _RF_LDS float *),
                         const float rnv, const uint64_t k, const uint32_t nt,
                         ompx::atomic::MemScopeTy Scope) {
  _xteam_reduction<float, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt, Scope);
}
_EXT_ATTR
__kmpc_iteamr_f(float v, float *r_p, void (*rf)(float *, float),
                void (*rflds)(_RF_LDS float *, _RF_LDS float *),
                const float rnv, const uint64_t k) {
  _iteam_reduction(float, v, r_p, rf, rflds, rnv, k);
}

_EXT_ATTR
__kmpc_xteamr_h(_Float16 v, _Float16 *r_p, _Float16 *tvs, uint32_t *td,
                void (*rf)(_Float16 *, _Float16),
                void (*rflds)(_RF_LDS _Float16 *, _RF_LDS _Float16 *),
                const _Float16 rnv, const uint64_t k, const uint32_t nt,
                ompx::atomic::MemScopeTy Scope) {
  _xteam_reduction<_Float16>(v, r_p, tvs, td, rf, rflds, rnv, k, nt, Scope);
}
_EXT_ATTR
__kmpc_xteamr_h_fast_sum(_Float16 v, _Float16 *r_p, _Float16 *tvs, uint32_t *td,
                         void (*rf)(_Float16 *, _Float16),
                         void (*rflds)(_RF_LDS _Float16 *, _RF_LDS _Float16 *),
                         const _Float16 rnv, const uint64_t k,
                         const uint32_t nt, ompx::atomic::MemScopeTy Scope) {
  _xteam_reduction<_Float16, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt,
                                   Scope);
}
_EXT_ATTR
__kmpc_iteamr_h(_Float16 v, _Float16 *r_p, void (*rf)(_Float16 *, _Float16),
                void (*rflds)(_RF_LDS _Float16 *, _RF_LDS _Float16 *),
                const _Float16 rnv, const uint64_t k) {
  _iteam_reduction(_Float16, v, r_p, rf, rflds, rnv, k);
}

_EXT_ATTR
__kmpc_xteamr_bf(__bf16 v, __bf16 *r_p, __bf16 *tvs, uint32_t *td,
                 void (*rf)(__bf16 *, __bf16),
                 void (*rflds)(_RF_LDS __bf16 *, _RF_LDS __bf16 *),
                 const __bf16 rnv, const uint64_t k, const uint32_t nt,
                 ompx::atomic::MemScopeTy Scope) {
  _xteam_reduction<__bf16>(v, r_p, tvs, td, rf, rflds, rnv, k, nt, Scope);
}
_EXT_ATTR
__kmpc_xteamr_bf_fast_sum(__bf16 v, __bf16 *r_p, __bf16 *tvs, uint32_t *td,
                          void (*rf)(__bf16 *, __bf16),
                          void (*rflds)(_RF_LDS __bf16 *, _RF_LDS __bf16 *),
                          const __bf16 rnv, const uint64_t k, const uint32_t nt,
                          ompx::atomic::MemScopeTy Scope) {
  _xteam_reduction<__bf16, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt, Scope);
}
_EXT_ATTR
__kmpc_iteamr_bf(__bf16 v, __bf16 *r_p, void (*rf)(__bf16 *, __bf16),
                 void (*rflds)(_RF_LDS __bf16 *, _RF_LDS __bf16 *),
                 const __bf16 rnv, const uint64_t k) {
  _iteam_reduction(__bf16, v, r_p, rf, rflds, rnv, k);
}

_EXT_ATTR
__kmpc_xteamr_s(short v, short *r_p, short *tvs, uint32_t *td,
                void (*rf)(short *, short),
                void (*rflds)(_RF_LDS short *, _RF_LDS short *),
                const short rnv, const uint64_t k, const uint32_t nt,
                ompx::atomic::MemScopeTy Scope) {
  _xteam_reduction<short>(v, r_p, tvs, td, rf, rflds, rnv, k, nt, Scope);
}
_EXT_ATTR
__kmpc_xteamr_s_fast_sum(short v, short *r_p, short *tvs, uint32_t *td,
                         void (*rf)(short *, short),
                         void (*rflds)(_RF_LDS short *, _RF_LDS short *),
                         const short rnv, const uint64_t k, const uint32_t nt,
                         ompx::atomic::MemScopeTy Scope) {
  _xteam_reduction<short, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt, Scope);
}
_EXT_ATTR
__kmpc_iteamr_s(short v, short *r_p, void (*rf)(short *, short),
                void (*rflds)(_RF_LDS short *, _RF_LDS short *),
                const short rnv, const uint64_t k) {
  _iteam_reduction(short, v, r_p, rf, rflds, rnv, k);
}

_EXT_ATTR
__kmpc_xteamr_us(_US v, _US *r_p, _US *tvs, uint32_t *td,
                 void (*rf)(_US *, _US),
                 void (*rflds)(_RF_LDS _US *, _RF_LDS _US *), const _US rnv,
                 const uint64_t k, const uint32_t nt,
                 ompx::atomic::MemScopeTy Scope) {
  _xteam_reduction<_US>(v, r_p, tvs, td, rf, rflds, rnv, k, nt, Scope);
}
_EXT_ATTR
__kmpc_xteamr_us_fast_sum(_US v, _US *r_p, _US *tvs, uint32_t *td,
                          void (*rf)(_US *, _US),
                          void (*rflds)(_RF_LDS _US *, _RF_LDS _US *),
                          const _US rnv, const uint64_t k, const uint32_t nt,
                          ompx::atomic::MemScopeTy Scope) {
  _xteam_reduction<_US, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt, Scope);
}
_EXT_ATTR
__kmpc_iteamr_us(_US v, _US *r_p, void (*rf)(_US *, _US),
                 void (*rflds)(_RF_LDS _US *, _RF_LDS _US *), const _US rnv,
                 const uint64_t k) {
  _iteam_reduction(_US, v, r_p, rf, rflds, rnv, k);
}

_EXT_ATTR
__kmpc_xteamr_i(int v, int *r_p, int *tvs, uint32_t *td, void (*rf)(int *, int),
                void (*rflds)(_RF_LDS int *, _RF_LDS int *), const int rnv,
                const uint64_t k, const uint32_t nt,
                ompx::atomic::MemScopeTy Scope) {
  _xteam_reduction<int>(v, r_p, tvs, td, rf, rflds, rnv, k, nt, Scope);
}
_EXT_ATTR
__kmpc_xteamr_i_fast_sum(int v, int *r_p, int *tvs, uint32_t *td,
                         void (*rf)(int *, int),
                         void (*rflds)(_RF_LDS int *, _RF_LDS int *),
                         const int rnv, const uint64_t k, const uint32_t nt,
                         ompx::atomic::MemScopeTy Scope) {
  _xteam_reduction<int, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt, Scope);
}
_EXT_ATTR
__kmpc_iteamr_i(int v, int *r_p, void (*rf)(int *, int),
                void (*rflds)(_RF_LDS int *, _RF_LDS int *), const int rnv,
                const uint64_t k) {
  _iteam_reduction(int, v, r_p, rf, rflds, rnv, k);
}

_EXT_ATTR
__kmpc_xteamr_ui(_UI v, _UI *r_p, _UI *tvs, uint32_t *td,
                 void (*rf)(_UI *, _UI),
                 void (*rflds)(_RF_LDS _UI *, _RF_LDS _UI *), const _UI rnv,
                 const uint64_t k, const uint32_t nt,
                 ompx::atomic::MemScopeTy Scope) {
  _xteam_reduction<_UI>(v, r_p, tvs, td, rf, rflds, rnv, k, nt, Scope);
}
_EXT_ATTR
__kmpc_xteamr_ui_fast_sum(_UI v, _UI *r_p, _UI *tvs, uint32_t *td,
                          void (*rf)(_UI *, _UI),
                          void (*rflds)(_RF_LDS _UI *, _RF_LDS _UI *),
                          const _UI rnv, const uint64_t k, const uint32_t nt,
                          ompx::atomic::MemScopeTy Scope) {
  _xteam_reduction<_UI, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt, Scope);
}
_EXT_ATTR
__kmpc_iteamr_ui(_UI v, _UI *r_p, void (*rf)(_UI *, _UI),
                 void (*rflds)(_RF_LDS _UI *, _RF_LDS _UI *), const _UI rnv,
                 const uint64_t k) {
  _iteam_reduction(_UI, v, r_p, rf, rflds, rnv, k);
}

// Long
_EXT_ATTR
__kmpc_xteamr_l(long v, long *r_p, long *tvs, uint32_t *td,
                void (*rf)(long *, long),
                void (*rflds)(_RF_LDS long *, _RF_LDS long *), const long rnv,
                const uint64_t k, const uint32_t nt,
                ompx::atomic::MemScopeTy Scope) {
  _xteam_reduction<long>(v, r_p, tvs, td, rf, rflds, rnv, k, nt, Scope);
}
_EXT_ATTR
__kmpc_xteamr_l_fast_sum(long v, long *r_p, long *tvs, uint32_t *td,
                         void (*rf)(long *, long),
                         void (*rflds)(_RF_LDS long *, _RF_LDS long *),
                         const long rnv, const uint64_t k, const uint32_t nt,
                         ompx::atomic::MemScopeTy Scope) {
  _xteam_reduction<long, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt, Scope);
}
_EXT_ATTR
__kmpc_iteamr_l(long v, long *r_p, void (*rf)(long *, long),
                void (*rflds)(_RF_LDS long *, _RF_LDS long *), const long rnv,
                const uint64_t k) {
  _iteam_reduction(long, v, r_p, rf, rflds, rnv, k);
}

_EXT_ATTR
__kmpc_xteamr_ul(_UL v, _UL *r_p, _UL *tvs, uint32_t *td,
                 void (*rf)(_UL *, _UL),
                 void (*rflds)(_RF_LDS _UL *, _RF_LDS _UL *), const _UL rnv,
                 const uint64_t k, const uint32_t nt,
                 ompx::atomic::MemScopeTy Scope) {
  _xteam_reduction<_UL>(v, r_p, tvs, td, rf, rflds, rnv, k, nt, Scope);
}
_EXT_ATTR
__kmpc_xteamr_ul_fast_sum(_UL v, _UL *r_p, _UL *tvs, uint32_t *td,
                          void (*rf)(_UL *, _UL),
                          void (*rflds)(_RF_LDS _UL *, _RF_LDS _UL *),
                          const _UL rnv, const uint64_t k, const uint32_t nt,
                          ompx::atomic::MemScopeTy Scope) {
  _xteam_reduction<_UL, true>(v, r_p, tvs, td, rf, rflds, rnv, k, nt, Scope);
}
_EXT_ATTR
__kmpc_iteamr_ul(_UL v, _UL *r_p, void (*rf)(_UL *, _UL),
                 void (*rflds)(_RF_LDS _UL *, _RF_LDS _UL *), const _UL rnv,
                 const uint64_t k) {
  _iteam_reduction(_UL, v, r_p, rf, rflds, rnv, k);
}

//===----------------------------------------------------------------------===//
// Built-in pair reduction functions used as function pointers for
// cross team reduction functions.
//===----------------------------------------------------------------------===//

_EXT_ATTR __kmpc_rfun_sum_d(double *val, double otherval) { *val += otherval; }
_EXT_ATTR __kmpc_rfun_sum_lds_d(_RF_LDS double *val, _RF_LDS double *otherval) {
  *val += *otherval;
}
_EXT_ATTR __kmpc_rfun_sum_f(float *val, float otherval) { *val += otherval; }
_EXT_ATTR __kmpc_rfun_sum_lds_f(_RF_LDS float *val, _RF_LDS float *otherval) {
  *val += *otherval;
}
_EXT_ATTR __kmpc_rfun_sum_h(_Float16 *val, _Float16 otherval) {
  *val += otherval;
}
_EXT_ATTR __kmpc_rfun_sum_lds_h(_RF_LDS _Float16 *val,
                                _RF_LDS _Float16 *otherval) {
  *val += *otherval;
}
_EXT_ATTR __kmpc_rfun_sum_bf(__bf16 *val, __bf16 otherval) { *val += otherval; }
_EXT_ATTR __kmpc_rfun_sum_lds_bf(_RF_LDS __bf16 *val,
                                 _RF_LDS __bf16 *otherval) {
  *val += *otherval;
}
_EXT_ATTR __kmpc_rfun_sum_cd(_CD *val, _CD otherval) { *val += otherval; }
_EXT_ATTR __kmpc_rfun_sum_lds_cd(_RF_LDS _CD *val, _RF_LDS _CD *otherval) {
  *val += *otherval;
}
_EXT_ATTR __kmpc_rfun_sum_cf(_CF *val, _CF otherval) { *val += otherval; }
_EXT_ATTR __kmpc_rfun_sum_lds_cf(_RF_LDS _CF *val, _RF_LDS _CF *otherval) {
  *val += *otherval;
}
_EXT_ATTR __kmpc_rfun_sum_s(short *val, short otherval) { *val += otherval; }
_EXT_ATTR __kmpc_rfun_sum_lds_s(_RF_LDS short *val, _RF_LDS short *otherval) {
  *val += *otherval;
}
_EXT_ATTR __kmpc_rfun_sum_us(_US *val, _US otherval) { *val += otherval; }
_EXT_ATTR __kmpc_rfun_sum_lds_us(_RF_LDS _US *val, _RF_LDS _US *otherval) {
  *val += *otherval;
}
_EXT_ATTR __kmpc_rfun_sum_i(int *val, int otherval) { *val += otherval; }
_EXT_ATTR __kmpc_rfun_sum_lds_i(_RF_LDS int *val, _RF_LDS int *otherval) {
  *val += *otherval;
}
_EXT_ATTR __kmpc_rfun_sum_ui(_UI *val, _UI otherval) { *val += otherval; }
_EXT_ATTR __kmpc_rfun_sum_lds_ui(_RF_LDS _UI *val, _RF_LDS _UI *otherval) {
  *val += *otherval;
}
_EXT_ATTR __kmpc_rfun_sum_l(long *val, long otherval) { *val += otherval; }
_EXT_ATTR __kmpc_rfun_sum_lds_l(_RF_LDS long *val, _RF_LDS long *otherval) {
  *val += *otherval;
}
_EXT_ATTR __kmpc_rfun_sum_ul(_UL *val, _UL otherval) { *val += otherval; }
_EXT_ATTR __kmpc_rfun_sum_lds_ul(_RF_LDS _UL *val, _RF_LDS _UL *otherval) {
  *val += *otherval;
}

_EXT_ATTR __kmpc_rfun_max_d(double *val, double otherval) {
  *val = (otherval > *val) ? otherval : *val;
}
_EXT_ATTR __kmpc_rfun_max_lds_d(_RF_LDS double *val, _RF_LDS double *otherval) {
  *val = (*otherval > *val) ? *otherval : *val;
}
_EXT_ATTR __kmpc_rfun_max_f(float *val, float otherval) {
  *val = (otherval > *val) ? otherval : *val;
}
_EXT_ATTR __kmpc_rfun_max_lds_f(_RF_LDS float *val, _RF_LDS float *otherval) {
  *val = (*otherval > *val) ? *otherval : *val;
}
_EXT_ATTR __kmpc_rfun_max_h(_Float16 *val, _Float16 otherval) {
  *val = (otherval > *val) ? otherval : *val;
}
_EXT_ATTR __kmpc_rfun_max_lds_h(_RF_LDS _Float16 *val,
                                _RF_LDS _Float16 *otherval) {
  *val = (*otherval > *val) ? *otherval : *val;
}
_EXT_ATTR __kmpc_rfun_max_bf(__bf16 *val, __bf16 otherval) {
  *val = (otherval > *val) ? otherval : *val;
}
_EXT_ATTR __kmpc_rfun_max_lds_bf(_RF_LDS __bf16 *val,
                                 _RF_LDS __bf16 *otherval) {
  *val = (*otherval > *val) ? *otherval : *val;
}
_EXT_ATTR __kmpc_rfun_max_s(short *val, short otherval) {
  *val = (otherval > *val) ? otherval : *val;
}
_EXT_ATTR __kmpc_rfun_max_lds_s(_RF_LDS short *val, _RF_LDS short *otherval) {
  *val = (*otherval > *val) ? *otherval : *val;
}
_EXT_ATTR __kmpc_rfun_max_us(_US *val, _US otherval) {
  *val = (otherval > *val) ? otherval : *val;
}
_EXT_ATTR __kmpc_rfun_max_lds_us(_RF_LDS _US *val, _RF_LDS _US *otherval) {
  *val = (*otherval > *val) ? *otherval : *val;
}
_EXT_ATTR __kmpc_rfun_max_i(int *val, int otherval) {
  *val = (otherval > *val) ? otherval : *val;
}
_EXT_ATTR __kmpc_rfun_max_lds_i(_RF_LDS int *val, _RF_LDS int *otherval) {
  *val = (*otherval > *val) ? *otherval : *val;
}
_EXT_ATTR __kmpc_rfun_max_ui(_UI *val, _UI otherval) {
  *val = (otherval > *val) ? otherval : *val;
}
_EXT_ATTR __kmpc_rfun_max_lds_ui(_RF_LDS _UI *val, _RF_LDS _UI *otherval) {
  *val = (*otherval > *val) ? *otherval : *val;
}
_EXT_ATTR __kmpc_rfun_max_l(long *val, long otherval) {
  *val = (otherval > *val) ? otherval : *val;
}
_EXT_ATTR __kmpc_rfun_max_lds_l(_RF_LDS long *val, _RF_LDS long *otherval) {
  *val = (*otherval > *val) ? *otherval : *val;
}
_EXT_ATTR __kmpc_rfun_max_ul(_UL *val, _UL otherval) {
  *val = (otherval > *val) ? otherval : *val;
}
_EXT_ATTR __kmpc_rfun_max_lds_ul(_RF_LDS _UL *val, _RF_LDS _UL *otherval) {
  *val = (*otherval > *val) ? *otherval : *val;
}

_EXT_ATTR __kmpc_rfun_min_d(double *val, double otherval) {
  *val = (otherval < *val) ? otherval : *val;
}
_EXT_ATTR __kmpc_rfun_min_lds_d(_RF_LDS double *val, _RF_LDS double *otherval) {
  *val = (*otherval < *val) ? *otherval : *val;
}
_EXT_ATTR __kmpc_rfun_min_f(float *val, float otherval) {
  *val = (otherval < *val) ? otherval : *val;
}
_EXT_ATTR __kmpc_rfun_min_lds_f(_RF_LDS float *val, _RF_LDS float *otherval) {
  *val = (*otherval < *val) ? *otherval : *val;
}
_EXT_ATTR __kmpc_rfun_min_h(_Float16 *val, _Float16 otherval) {
  *val = (otherval < *val) ? otherval : *val;
}
_EXT_ATTR __kmpc_rfun_min_lds_h(_RF_LDS _Float16 *val,
                                _RF_LDS _Float16 *otherval) {
  *val = (*otherval < *val) ? *otherval : *val;
}
_EXT_ATTR __kmpc_rfun_min_bf(__bf16 *val, __bf16 otherval) {
  *val = (otherval < *val) ? otherval : *val;
}
_EXT_ATTR __kmpc_rfun_min_lds_bf(_RF_LDS __bf16 *val,
                                 _RF_LDS __bf16 *otherval) {
  *val = (*otherval < *val) ? *otherval : *val;
}
_EXT_ATTR __kmpc_rfun_min_s(short *val, short otherval) {
  *val = (otherval < *val) ? otherval : *val;
}
_EXT_ATTR __kmpc_rfun_min_lds_s(_RF_LDS short *val, _RF_LDS short *otherval) {
  *val = (*otherval < *val) ? *otherval : *val;
}
_EXT_ATTR __kmpc_rfun_min_us(_US *val, _US otherval) {
  *val = (otherval < *val) ? otherval : *val;
}
_EXT_ATTR __kmpc_rfun_min_lds_us(_RF_LDS _US *val, _RF_LDS _US *otherval) {
  *val = (*otherval < *val) ? *otherval : *val;
}
_EXT_ATTR __kmpc_rfun_min_i(int *val, int otherval) {
  *val = (otherval < *val) ? otherval : *val;
}
_EXT_ATTR __kmpc_rfun_min_lds_i(_RF_LDS int *val, _RF_LDS int *otherval) {
  *val = (*otherval < *val) ? *otherval : *val;
}
_EXT_ATTR __kmpc_rfun_min_ui(_UI *val, _UI otherval) {
  *val = (otherval < *val) ? otherval : *val;
}
_EXT_ATTR __kmpc_rfun_min_lds_ui(_RF_LDS _UI *val, _RF_LDS _UI *otherval) {
  *val = (*otherval < *val) ? *otherval : *val;
}
_EXT_ATTR __kmpc_rfun_min_l(long *val, long otherval) {
  *val = (otherval < *val) ? otherval : *val;
}
_EXT_ATTR __kmpc_rfun_min_lds_l(_RF_LDS long *val, _RF_LDS long *otherval) {
  *val = (*otherval < *val) ? *otherval : *val;
}
_EXT_ATTR __kmpc_rfun_min_ul(_UL *val, _UL otherval) {
  *val = (otherval < *val) ? otherval : *val;
}
_EXT_ATTR __kmpc_rfun_min_lds_ul(_RF_LDS _UL *val, _RF_LDS _UL *otherval) {
  *val = (*otherval < *val) ? *otherval : *val;
}

#undef _CD
#undef _CF
#undef _US
#undef _UI
#undef _UL
#undef _EXT_ATTR
