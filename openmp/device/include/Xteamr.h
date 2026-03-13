//===---------------- Xteamr.h - OpenMP interface ----------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// DeviceRTL Header file: Xteamr.h
//     External __kmpc headers for cross team reduction functions defined
//     in DeviceRTL/src/Xteamr.cpp. Clang generates a call to one of these
//     functions when it encounter a reduction. The specific function depends
//     on datatype and warpsize. The number of waves must be a power of 2.
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_DEVICERTL_XTEAMR_H
#define OMPTARGET_DEVICERTL_XTEAMR_H

#include "XteamCommon.h"

#define _CD double _Complex
#define _CF float _Complex
#define _US unsigned short
#define _UI unsigned int
#define _UL unsigned long

extern "C" {
/// External cross team reduction (xteamr) helper functions
///
/// The template for name of xteamr helper function is:
/// __kmpc_xteamr_<dtype> where
///    <dtype> is letter(s) representing data type, e.g. d=double.
///    IS_FAST There is an optional template boolean type (defaulting to false)
///    that indicates if an atomic add should be used instead of the last
///    reduction round. This applies to only sum reduction currently.
/// All xteamr helper functions are defined in Xteamr.cpp. They each call the
/// internal templated function _xteam_reduction also defined in Xteamr.cpp.
/// Clang/flang code generation for C, C++, and FORTRAN instantiate a call to
/// a helper function for each reduction used in an OpenMP target region.
///
/// \param v Input thread local reduction value
/// \param r_ptr Pointer to result value
/// \param tvs Global array of team values for this reduction instance
/// \param td Pointer to atomic counter of completed teams
/// \param _rf Function pointer to reduction function (sum,min,max)
/// \param _rf_lds Function pointer to reduction function on LDS memory
/// \param rnv Reduction null value
/// \param k Outer loop iteration value, 0 to numteams*numthreads
/// \param numteams Number of teams
/// \param Scope Memory scope

#define _XTEAMR_DECL(T, TS)                                                    \
  void _XTEAM_EXTERN_ATTR __kmpc_xteamr_##TS(                                  \
      T v, T *r_ptr, T *tvs, uint32_t *td, void (*_rf)(T *, T),                \
      void (*_rf_lds)(_RF_LDS T *, _RF_LDS T *), const T rnv,                  \
      const uint64_t k, const uint32_t numteams,                               \
      ompx::atomic::MemScopeTy Scope = ompx::atomic::system);

#define _XTEAMR_DECL_ALL(T, TS)                                                \
  _XTEAMR_DECL(T, TS);                                                         \
  _XTEAMR_DECL(T, TS##_fast_sum)

_XTEAMR_DECL_ALL(__bf16, bf)
_XTEAMR_DECL_ALL(_Float16, h)
// _XTEAMR_DECL_ALL(_CD, cd)
// _XTEAMR_DECL_ALL(_CF, cf)
_XTEAMR_DECL_ALL(double, d)
_XTEAMR_DECL_ALL(float, f)
_XTEAMR_DECL_ALL(int, i)
_XTEAMR_DECL_ALL(_UI, ui)
_XTEAMR_DECL_ALL(long, l)
_XTEAMR_DECL_ALL(_UL, ul)
_XTEAMR_DECL_ALL(short, s)
_XTEAMR_DECL_ALL(_US, us)

#undef _XTEAMR_DECL
#undef _XTEAMR_DECL_ALL

/// External intra-team reduction (iteamr) helper functions
///
/// The name template for intra-team helper functions is
/// __kmpc_iteamr_<dtype> where
///    <dtype> is letter(s) representing data type, e.g. d=double.
/// All iteamr helper functions are defined in Xteamr.cpp. They each call the
/// internal templated function _iteam_reduction also defined in Xteamr.cpp.
///
/// \param v Input thread local reduction value
/// \param r_ptr Pointer to result value
/// \param _rf Function pointer to reduction function (sum,min,max)
/// \param _rf_lds Function pointer to reduction function on LDS memory
/// \param rnv Reduction null value
/// \param k Outer loop iteration value, 0 to numthreads

#define _ITEAMR_DEF(T, TS)                                                     \
  void _XTEAM_EXTERN_ATTR __kmpc_iteamr_##TS(                                  \
      T v, T *r_ptr, void (*_rf)(T *, T),                                      \
      void (*_rf_lds)(_RF_LDS T *, _RF_LDS T *), const T rnv,                  \
      const uint64_t k);

_ITEAMR_DEF(__bf16, bf)
_ITEAMR_DEF(_Float16, h)
// _ITEAMR_DEF(_CD, cd)
// _ITEAMR_DEF(_CF, cf)
_ITEAMR_DEF(double, d)
_ITEAMR_DEF(float, f)
_ITEAMR_DEF(int, i)
_ITEAMR_DEF(_UI, ui)
_ITEAMR_DEF(long, l)
_ITEAMR_DEF(_UL, ul)
_ITEAMR_DEF(short, s)
_ITEAMR_DEF(_US, us)

#undef _ITEAMR_DEF

/// Built-in pair reduction function, see documentation above.
#define _REDUCTION_FUNCTION(T, OP, TS)                                         \
  void __kmpc_rfun_##OP_##TS(T *val, T otherval);                              \
  void __kmpc_rfun_##OP_lds_##TS(_RF_LDS T *val, _RF_LDS T *otherval);

#define _REDUCTION_FUNCTION_ALL(OP)                                            \
  _REDUCTION_FUNCTION(__bf16, OP, bf)                                          \
  _REDUCTION_FUNCTION(_Float16, OP, h)                                         \
  _REDUCTION_FUNCTION(double, OP, d)                                           \
  _REDUCTION_FUNCTION(float, OP, f)                                            \
  _REDUCTION_FUNCTION(int, OP, i)                                              \
  _REDUCTION_FUNCTION(_UI, OP, ui)                                             \
  _REDUCTION_FUNCTION(long, OP, l)                                             \
  _REDUCTION_FUNCTION(_UL, OP, ul)                                             \
  _REDUCTION_FUNCTION(short, OP, s)                                            \
  _REDUCTION_FUNCTION(_US, OP, us)
// _REDUCTION_FUNCTION(_CD, OP, cd)
// _REDUCTION_FUNCTION(_CF, OP, cf)

_REDUCTION_FUNCTION_ALL(sum)
_REDUCTION_FUNCTION_ALL(max)
_REDUCTION_FUNCTION_ALL(min)

#undef _REDUCTION_FUNCTION
#undef _REDUCTION_FUNCTION_ALL

} // end extern C

#undef _CD
#undef _CF
#undef _US
#undef _UI
#undef _UL

#endif // of ifndef OMPTARGET_DEVICERTL_XTEAMR_H
