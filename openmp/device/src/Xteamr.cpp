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
#include "Debug.h"
#include "DeviceUtils.h"
#include "Interface.h"
#include "Mapping.h"
#include "State.h"

#define _CD double _Complex
#define _CF float _Complex
#define _US unsigned short
#define _UI unsigned int
#define _UL unsigned long
#define _INLINE_ATTR_ __attribute__((flatten, always_inline))
#define _RF_LDS volatile __gpu_local
// Wave size (will be constant-folded since it's known at compile time)
// Should probably be made into constexpr in the future.
#define _WSZ __gpu_num_lanes()
// Maximum number of waves in a thread block
// (1024 / _WSZ = 32 or 16 waves, depending on whether _WSZ is 32 or 64)
#define _MaxNumWaves 32

// Headers for specialized shfl_xor
double xteamr_shfl_xor_d(double var, const int lane_mask, const uint32_t width);
float xteamr_shfl_xor_f(float var, const int lane_mask, const uint32_t width);
int xteamr_shfl_xor_int(int var, const int lane_mask, const uint32_t width);
double _Complex xteamr_shfl_xor_cd(double _Complex var, const int lane_mask,
                                   const uint32_t width);
float _Complex xteamr_shfl_xor_cf(float _Complex var, const int lane_mask,
                                  const uint32_t width);

// Define the arch (amdgcn vs nvptx) variants of shfl
#ifdef __AMDGPU__
int xteamr_shfl_xor_int(int var, const int lane_mask, const uint32_t width) {
  int self = ompx::mapping::getThreadIdInWarp(); // __lane_id();
  int index = self ^ lane_mask;
  index = index >= ((self + width) & ~(width - 1)) ? self : index;
  return __builtin_amdgcn_ds_bpermute(index << 2, var);
}
double xteamr_shfl_xor_d(double var, const int lane_mask,
                         const uint32_t width) {
  static_assert(sizeof(double) == 2 * sizeof(int), "");
  static_assert(sizeof(double) == sizeof(uint64_t), "");

  int tmp[2];
  __builtin_memcpy(tmp, &var, sizeof(tmp));
  tmp[0] = xteamr_shfl_xor_int(tmp[0], lane_mask, width);
  tmp[1] = xteamr_shfl_xor_int(tmp[1], lane_mask, width);

  uint64_t tmp0 =
      (static_cast<uint64_t>(tmp[1]) << 32ull) | static_cast<uint32_t>(tmp[0]);
  double tmp1;
  __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));
  return tmp1;
}
#elif defined(__NVPTX__)
int xteamr_shfl_xor_int(int var, const int lane_mask, const uint32_t width) {
  return __nvvm_shfl_sync_bfly_i32(0xFFFFFFFF, var, lane_mask, 0x1f);
}
double xteamr_shfl_xor_d(double var, int laneMask, const uint32_t width) {
  unsigned lo, hi;
  asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(var));
  hi = xteamr_shfl_xor_int(hi, laneMask, width);
  lo = xteamr_shfl_xor_int(lo, laneMask, width);
  asm volatile("mov.b64 %0, {%1,%2};" : "=d"(var) : "r"(lo), "r"(hi));
  return var;
}
#endif

float xteamr_shfl_xor_f(float var, const int lane_mask, const uint32_t width) {
  union {
    int i;
    unsigned u;
    float f;
  } tmp;
  tmp.f = var;
  tmp.i = xteamr_shfl_xor_int(tmp.i, lane_mask, width);
  return tmp.f;
}
double _Complex xteamr_shfl_xor_cd(double _Complex var, const int lane_mask,
                                   const uint32_t width) {
  __real__(var) = xteamr_shfl_xor_d(__real__(var), lane_mask, width);
  __imag__(var) = xteamr_shfl_xor_d(__imag__(var), lane_mask, width);
  return var;
}
float _Complex xteamr_shfl_xor_cf(float _Complex var, const int lane_mask,
                                  const uint32_t width) {
  __real__(var) = xteamr_shfl_xor_f(__real__(var), lane_mask, width);
  __imag__(var) = xteamr_shfl_xor_f(__imag__(var), lane_mask, width);
  return var;
}

// type specific shfl_xor functions
double xteamr_shfl_xor(double var, const int lane_mask) {
  return xteamr_shfl_xor_d(var, lane_mask, _WSZ);
}
float xteamr_shfl_xor(float var, const int lane_mask) {
  return xteamr_shfl_xor_f(var, lane_mask, _WSZ);
}
float xteamr_shfl_xor(_Float16 var, const int lane_mask) {
  return xteamr_shfl_xor_f(var, lane_mask, _WSZ);
}
float xteamr_shfl_xor(__bf16 var, const int lane_mask) {
  return xteamr_shfl_xor_f(var, lane_mask, _WSZ);
}
double _Complex xteamr_shfl_xor(double _Complex var, const int lane_mask) {
  return xteamr_shfl_xor_cd(var, lane_mask, _WSZ);
}
float _Complex xteamr_shfl_xor(float _Complex var, const int lane_mask) {
  return xteamr_shfl_xor_cf(var, lane_mask, _WSZ);
}
int xteamr_shfl_xor(short var, const int lane_mask) {
  return xteamr_shfl_xor_int(var, lane_mask, _WSZ);
}
unsigned int xteamr_shfl_xor(unsigned short var, const int lane_mask) {
  return xteamr_shfl_xor_int(var, lane_mask, _WSZ);
}
int xteamr_shfl_xor(int var, const int lane_mask) {
  return xteamr_shfl_xor_int(var, lane_mask, _WSZ);
}
unsigned int xteamr_shfl_xor(unsigned int var, const int lane_mask) {
  return xteamr_shfl_xor_int(var, lane_mask, _WSZ);
}
long xteamr_shfl_xor(long var, const int lane_mask) {
  return xteamr_shfl_xor_d(var, lane_mask, _WSZ);
}
unsigned long xteamr_shfl_xor(unsigned long var, const int lane_mask) {
  return xteamr_shfl_xor_d(var, lane_mask, _WSZ);
}

/// Templated internal function used by all extern typed reductions
///
/// \param T Template typename parameter T
/// \param _IS_FAST Template parameter if an atomic add should be used instead
/// of
///         the 1-team-reduction round. Applies to sum reduction currently.
///
/// \param val Input thread local (TLS) value for warp shfl reduce
/// \param r_ptr Pointer to result value, also used in final reduction
/// \param team_vals Global array of team values for this reduction only
/// \param teams_done_ptr Pointer to atomically accessed teams done counter
/// \param _rf Function pointer to TLS pair reduction function
/// \param _rf_lds Function pointer to LDS pair reduction function
/// \param rnv Reduction null value, used for partial waves
/// \param k The iteration value from 0 to (NumTeams*_NUM_THREADS)-1
/// \param NumTeams The number of teams participating in reduction
/// \param Scope The scope of the atomic operation

template <typename T, const bool _IS_FAST = false>
_INLINE_ATTR_ void
_xteam_reduction(T val, T *r_ptr, T *team_vals, uint32_t *teams_done_ptr,
                 void (*_rf)(T *, T), void (*_rf_lds)(_RF_LDS T *, _RF_LDS T *),
                 const T rnv, const uint64_t k, const uint32_t NumTeams,
                 ompx::atomic::MemScopeTy Scope) {

  // More efficient to derive these constants than get from mapped API

  // Must be a power of 2.
  const uint32_t block_size = ompx::mapping::getNumberOfThreadsInBlock();

  const uint32_t number_of_waves = (block_size - 1) / _WSZ + 1;
  const uint32_t omp_thread_num = k % block_size;
  const uint32_t omp_team_num = k / block_size;
  const uint32_t wave_num = omp_thread_num / _WSZ;
  const uint32_t lane_num = omp_thread_num % _WSZ;

  static _RF_LDS T xwave_lds[_MaxNumWaves];

// Cuda may restrict max threads, so clear unused wave values
#ifdef __NVPTX__
  if (number_of_waves == 32) {
    if (omp_thread_num == 0) {
      for (uint32_t i = (omp_get_num_threads() / 32); i < number_of_waves; i++)
        xwave_lds[i] = rnv;
    }
  }
#endif

  // Binary reduce each wave, then copy to xwave_lds[wave_num]
  const uint32_t start_offset = block_size < _WSZ ? block_size / 2 : _WSZ / 2;
  for (unsigned int offset = start_offset; offset > 0; offset >>= 1)
    (*_rf)(&val, xteamr_shfl_xor(val, offset));
  if (lane_num == 0)
    xwave_lds[wave_num] = val;

  // Binary reduce all wave values into wave_lds[0]
  for (unsigned int offset = number_of_waves / 2; offset > 0; offset >>= 1) {
    ompx::synchronize::threadsAligned(ompx::atomic::seq_cst);
    if (omp_thread_num < offset)
      (*_rf_lds)(&(xwave_lds[omp_thread_num]),
                 &(xwave_lds[omp_thread_num + offset]));
  }

  if constexpr (_IS_FAST) {
    if (omp_thread_num == 0)
      ompx::atomic::add(r_ptr, xwave_lds[0], ompx::atomic::seq_cst, Scope);
  } else if (NumTeams == 1) {
    // We're only doing intra-team reduction, team_vals might be nullptr.
    if (omp_thread_num == 0)
      *r_ptr = xwave_lds[0];
    ompx::synchronize::threadsAligned(ompx::atomic::seq_cst);
  } else {
    // No sync needed here from last reduction in LDS loop
    // because we only need xwave_lds[0] correct on thread 0.

    // Save the teams reduced value in team_vals global array
    // and atomically increment teams_done counter.
    static _RF_LDS uint32_t td;
    if (omp_thread_num == 0) {
      team_vals[omp_team_num] = xwave_lds[0];
      td = ompx::atomic::inc(teams_done_ptr, NumTeams - 1u,
                             ompx::atomic::seq_cst,
                             ompx::atomic::MemScopeTy::device);
    }

    // This sync needed so all threads from last team see the shared volatile
    // value td (teams done counter) so they know they are in the last team.
    ompx::synchronize::threadsAligned(ompx::atomic::seq_cst);

    // If td counter reaches NumTeams-1, this is the last team.
    // The team number of this last team is nondeterministic.
    if (td == (NumTeams - 1u)) {

      // All threads from last completed team enter here.
      // All other teams exit the helper function.

      // To use TLS shfl reduce, copy team values to TLS val.
      val = (omp_thread_num < NumTeams) ? team_vals[omp_thread_num] : rnv;

      // Need sync here to prepare for TLS shfl reduce.
      ompx::synchronize::threadsAligned(ompx::atomic::seq_cst);

      // Reduce each wave into xwave_lds[wave_num]
      for (unsigned int offset = start_offset; offset > 0; offset >>= 1)
        (*_rf)(&val, xteamr_shfl_xor(val, offset));
      if (lane_num == 0)
        xwave_lds[wave_num] = val;

      // Binary reduce all wave values into wave_lds[0]
      for (unsigned int offset = number_of_waves / 2; offset > 0;
           offset >>= 1) {
        ompx::synchronize::threadsAligned(ompx::atomic::seq_cst);
        if (omp_thread_num < offset)
          (*_rf_lds)(&(xwave_lds[omp_thread_num]),
                     &(xwave_lds[omp_thread_num + offset]));
      }

      if (omp_thread_num == 0) {
        // Reduce with the original result value.
        val = xwave_lds[0];
        (*_rf)(&val, *r_ptr);

        // If more teams than threads, do non-parallel reduction of extra
        // team_vals. This loop iterates only if NumTeams > block_size.
        for (unsigned int offset = block_size; offset < NumTeams; offset++)
          (*_rf)(&val, team_vals[offset]);

        // Write over the external result value.
        *r_ptr = val;
      }

      // This sync needed to prevent warps in last team from starting
      // if there was another reduction.
      ompx::synchronize::threadsAligned(ompx::atomic::relaxed);
    }
  }
}

/// Internal macro used by extern intra-team reductions
///
/// \param T Template typename parameter T
///
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

//  Calls to these __kmpc extern C functions are created in clang codegen
//  for FORTRAN, c, and C++. They may also be used for simulation and testing.
//  The headers for these extern C functions are in ../include/Interface.h
//  The compiler builds the name based on the data type.
//
#define _EXT_ATTR extern "C" _INLINE_ATTR_ void

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

// Built-in pair reduction functions used as function pointers for
// cross team reduction functions.

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
#undef _INLINE_ATTR_
#undef _RF_LDS
#undef _MaxNumWaves
#undef _WSZ
