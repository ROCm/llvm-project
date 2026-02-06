//===---------------- Xteams.h - OpenMP interface ----------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// DeviceRTL Header file: Xteams.h
//     External __kmpc headers for cross team scan functions are defined
//     in DeviceRTL/src/Xteams.cpp. Clang will generate a call to one
//     of these functions as it encounters the scan directive. The
//     specific function depends on datatype, warpsize, and number of waves
//     in the teamsize. The number of teams should not be more than
//     the teamsize. Teamsize 64 is not supported yet.
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_DEVICERTL_XTEAMS_H
#define OMPTARGET_DEVICERTL_XTEAMS_H
#include "DeviceTypes.h"

#define _CD double _Complex
#define _CF float _Complex
#define _UI unsigned int
#define _UL unsigned long
#define _INLINE_ATTR_ __attribute__((flatten, always_inline))
#define _RF_LDS volatile __gpu_local

extern "C" {
/// External cross team scan (xteams) helper functions
///
/// The template for name of xteams helper function is:
/// __kmpc_xteams_<dtype> where
///    <dtype> is letter(s) representing data type, e.g. d=double
/// All xteams helper functions are defined in Xteams.cpp. They each call the
/// internal templated function _xteam_scan which is defined in Xteams.cpp.
/// Clang code generation for C/C++ shall instantiate a call to a helper
/// function for the operator(addition, max and min) used for a scan directive
/// used in a OpenMP target region.
///
/// \param v Input thread local scanned value
/// \param storage Pointer to a global shared storage used by all the threads
/// \param r_array Pointer to the result scan array (output)
/// \param tvs Global array of team values for this reduction instance
/// (team_vals)
/// \param td Pointer to atomic counter of completed teams (teams_done_ptr)
/// \param _rf Function pointer to reduction function (sum,min,max)
/// \param _rf_lds Function pointer to reduction function on LDS memory
/// \param iv Reduction null value (e.g. 0 for addition)
/// \param k Outer loop iteration value, 0 to numteams*numthreads
/// \param numteams Number of teams
/// Cross team scan (xteams) functions, see documentation above.
void _INLINE_ATTR_
__kmpc_xteams_d(double v, double *storage, double *r_array, double *tvs,
                uint32_t *td, void (*_rf)(double *, double),
                void (*_rf_lds)(_RF_LDS double *, _RF_LDS double *),
                const double iv, const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_
__kmpc_xteams_f(float v, float *storage, float *r_array, float *tvs,
                uint32_t *td, void (*_rf)(float *, float),
                void (*_rf_lds)(_RF_LDS float *, _RF_LDS float *),
                const float iv, const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_ __kmpc_xteams_cd(
    _CD v, _CD *storage, _CD *r_array, _CD *tvs, uint32_t *td,
    void (*_rf)(_CD *, _CD), void (*_rf_lds)(_RF_LDS _CD *, _RF_LDS _CD *),
    const _CD iv, const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_ __kmpc_xteams_cf(
    _CF v, _CF *storage, _CF *r_array, _CF *tvs, uint32_t *td,
    void (*_rf)(_CF *, _CF), void (*_rf_lds)(_RF_LDS _CF *, _RF_LDS _CF *),
    const _CF iv, const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_ __kmpc_xteams_i(
    int v, int *storage, int *r_array, int *tvs, uint32_t *td,
    void (*_rf)(int *, int), void (*_rf_lds)(_RF_LDS int *, _RF_LDS int *),
    const int iv, const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_ __kmpc_xteams_ui(
    _UI v, _UI *storage, _UI *r_array, _UI *tvs, uint32_t *td,
    void (*_rf)(_UI *, _UI), void (*_rf_lds)(_RF_LDS _UI *, _RF_LDS _UI *),
    const _UI iv, const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_ __kmpc_xteams_l(
    long v, long *storage, long *r_array, long *tvs, uint32_t *td,
    void (*_rf)(long *, long), void (*_rf_lds)(_RF_LDS long *, _RF_LDS long *),
    const long iv, const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_ __kmpc_xteams_ul(
    _UL v, _UL *storage, _UL *r_array, _UL *tvs, uint32_t *td,
    void (*_rf)(_UL *, _UL), void (*_rf_lds)(_RF_LDS _UL *, _RF_LDS _UL *),
    const _UL iv, const uint64_t k, const uint32_t numteams);

// Phase Two Entry points
void _INLINE_ATTR_ __kmpc_xteams_phase2_i(int *storage, int segment_size,
                                          int *tvs, int *seg_vals,
                                          void (*rf)(int *, int), const int rnv,
                                          const uint64_t k,
                                          bool is_inclusive_scan);
void _INLINE_ATTR_ __kmpc_xteams_phase2_d(double *storage, int segment_size,
                                          double *tvs, double *seg_vals,
                                          void (*rf)(double *, double),
                                          const double rnv, const uint64_t k,
                                          bool is_inclusive_scan);
void _INLINE_ATTR_ __kmpc_xteams_phase2_l(long *storage, int segment_size,
                                          long *tvs, long *seg_vals,
                                          void (*rf)(long *, long),
                                          const long rnv, const uint64_t k,
                                          bool is_inclusive_scan);
void _INLINE_ATTR_ __kmpc_xteams_phase2_f(float *storage, int segment_size,
                                          float *tvs, float *seg_vals,
                                          void (*rf)(float *, float),
                                          const float rnv, const uint64_t k,
                                          bool is_inclusive_scan);
} // end extern C

#undef _CD
#undef _CF
#undef _UI
#undef _UL
#undef _INLINE_ATTR_
#undef _RF_LDS

#endif // of ifndef OMPTARGET_DEVICERTL_XTEAMS_H
