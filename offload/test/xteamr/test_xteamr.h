// Header file: test_xteamr.h
// Declarations for the xteamr DeviceRTL interface used by the xteamr test.
// The new interface uses a single function per type (__kmpc_xteamr_d, etc.)
// with an extra int Scope parameter, plus _fast_sum and __kmpc_iteamr_
// variants. User apps cannot include DeviceRTL headers, so declarations are
// provided here.

#define _CD double _Complex
#define _CF float _Complex
#define _UI unsigned int
#define _UL unsigned long
#define _INLINE_ATTR_ __attribute__((flatten, always_inline))

#if defined(__AMDGCN__) || defined(__NVPTX__)
#define _XTEAMR_SCOPE __MEMORY_SCOPE_SYSTEM
#else
#define _XTEAMR_SCOPE 0
#endif

#if defined(__AMDGCN__) || defined(__NVPTX__)
extern "C" {
#define _RF_LDS volatile __attribute__((address_space(3)))

// Cross-team reduction
void _INLINE_ATTR_ __kmpc_xteamr_d(double v, double *r_ptr, double *tvs,
                                   uint32_t *td, void (*_rf)(double *, double),
                                   void (*_rf_lds)(_RF_LDS double *,
                                                   _RF_LDS double *),
                                   const double rnv, const uint64_t k,
                                   const uint32_t numteams, int Scope);
void _INLINE_ATTR_ __kmpc_xteamr_f(float v, float *r_ptr, float *tvs,
                                   uint32_t *td, void (*_rf)(float *, float),
                                   void (*_rf_lds)(_RF_LDS float *,
                                                   _RF_LDS float *),
                                   const float rnv, const uint64_t k,
                                   const uint32_t numteams, int Scope);
void _INLINE_ATTR_ __kmpc_xteamr_cd(
    _CD v, _CD *r_ptr, _CD *tvs, uint32_t *td, void (*_rf)(_CD *, _CD),
    void (*_rf_lds)(_RF_LDS _CD *, _RF_LDS _CD *), const _CD rnv,
    const uint64_t k, const uint32_t numteams, int Scope);
void _INLINE_ATTR_ __kmpc_xteamr_cf(
    _CF v, _CF *r_ptr, _CF *tvs, uint32_t *td, void (*_rf)(_CF *, _CF),
    void (*_rf_lds)(_RF_LDS _CF *, _RF_LDS _CF *), const _CF rnv,
    const uint64_t k, const uint32_t numteams, int Scope);
void _INLINE_ATTR_ __kmpc_xteamr_i(
    int v, int *r_ptr, int *tvs, uint32_t *td, void (*_rf)(int *, int),
    void (*_rf_lds)(_RF_LDS int *, _RF_LDS int *), const int rnv,
    const uint64_t k, const uint32_t numteams, int Scope);
void _INLINE_ATTR_ __kmpc_xteamr_ui(
    _UI v, _UI *r_ptr, _UI *tvs, uint32_t *td, void (*_rf)(_UI *, _UI),
    void (*_rf_lds)(_RF_LDS _UI *, _RF_LDS _UI *), const _UI rnv,
    const uint64_t k, const uint32_t numteams, int Scope);
void _INLINE_ATTR_ __kmpc_xteamr_l(
    long v, long *r_ptr, long *tvs, uint32_t *td, void (*_rf)(long *, long),
    void (*_rf_lds)(_RF_LDS long *, _RF_LDS long *), const long rnv,
    const uint64_t k, const uint32_t numteams, int Scope);
void _INLINE_ATTR_ __kmpc_xteamr_ul(
    _UL v, _UL *r_ptr, _UL *tvs, uint32_t *td, void (*_rf)(_UL *, _UL),
    void (*_rf_lds)(_RF_LDS _UL *, _RF_LDS _UL *), const _UL rnv,
    const uint64_t k, const uint32_t numteams, int Scope);

// Fast sum (uses atomic add)
void _INLINE_ATTR_ __kmpc_xteamr_d_fast_sum(
    double v, double *r_ptr, double *tvs, uint32_t *td,
    void (*_rf)(double *, double),
    void (*_rf_lds)(_RF_LDS double *, _RF_LDS double *), const double rnv,
    const uint64_t k, const uint32_t numteams, int Scope);
void _INLINE_ATTR_ __kmpc_xteamr_f_fast_sum(
    float v, float *r_ptr, float *tvs, uint32_t *td,
    void (*_rf)(float *, float),
    void (*_rf_lds)(_RF_LDS float *, _RF_LDS float *), const float rnv,
    const uint64_t k, const uint32_t numteams, int Scope);
void _INLINE_ATTR_ __kmpc_xteamr_cd_fast_sum(
    _CD v, _CD *r_ptr, _CD *tvs, uint32_t *td, void (*_rf)(_CD *, _CD),
    void (*_rf_lds)(_RF_LDS _CD *, _RF_LDS _CD *), const _CD rnv,
    const uint64_t k, const uint32_t numteams, int Scope);
void _INLINE_ATTR_ __kmpc_xteamr_cf_fast_sum(
    _CF v, _CF *r_ptr, _CF *tvs, uint32_t *td, void (*_rf)(_CF *, _CF),
    void (*_rf_lds)(_RF_LDS _CF *, _RF_LDS _CF *), const _CF rnv,
    const uint64_t k, const uint32_t numteams, int Scope);
void _INLINE_ATTR_ __kmpc_xteamr_i_fast_sum(
    int v, int *r_ptr, int *tvs, uint32_t *td, void (*_rf)(int *, int),
    void (*_rf_lds)(_RF_LDS int *, _RF_LDS int *), const int rnv,
    const uint64_t k, const uint32_t numteams, int Scope);
void _INLINE_ATTR_ __kmpc_xteamr_ui_fast_sum(
    _UI v, _UI *r_ptr, _UI *tvs, uint32_t *td, void (*_rf)(_UI *, _UI),
    void (*_rf_lds)(_RF_LDS _UI *, _RF_LDS _UI *), const _UI rnv,
    const uint64_t k, const uint32_t numteams, int Scope);
void _INLINE_ATTR_ __kmpc_xteamr_l_fast_sum(
    long v, long *r_ptr, long *tvs, uint32_t *td, void (*_rf)(long *, long),
    void (*_rf_lds)(_RF_LDS long *, _RF_LDS long *), const long rnv,
    const uint64_t k, const uint32_t numteams, int Scope);
void _INLINE_ATTR_ __kmpc_xteamr_ul_fast_sum(
    _UL v, _UL *r_ptr, _UL *tvs, uint32_t *td, void (*_rf)(_UL *, _UL),
    void (*_rf_lds)(_RF_LDS _UL *, _RF_LDS _UL *), const _UL rnv,
    const uint64_t k, const uint32_t numteams, int Scope);

// Intra-team reduction
void _INLINE_ATTR_ __kmpc_iteamr_d(double v, double *r_ptr,
                                   void (*_rf)(double *, double),
                                   void (*_rf_lds)(_RF_LDS double *,
                                                   _RF_LDS double *),
                                   const double rnv, const uint64_t k);
void _INLINE_ATTR_ __kmpc_iteamr_f(float v, float *r_ptr,
                                   void (*_rf)(float *, float),
                                   void (*_rf_lds)(_RF_LDS float *,
                                                   _RF_LDS float *),
                                   const float rnv, const uint64_t k);
void _INLINE_ATTR_ __kmpc_iteamr_cd(_CD v, _CD *r_ptr, void (*_rf)(_CD *, _CD),
                                    void (*_rf_lds)(_RF_LDS _CD *,
                                                    _RF_LDS _CD *),
                                    const _CD rnv, const uint64_t k);
void _INLINE_ATTR_ __kmpc_iteamr_cf(_CF v, _CF *r_ptr, void (*_rf)(_CF *, _CF),
                                    void (*_rf_lds)(_RF_LDS _CF *,
                                                    _RF_LDS _CF *),
                                    const _CF rnv, const uint64_t k);
void _INLINE_ATTR_ __kmpc_iteamr_i(int v, int *r_ptr, void (*_rf)(int *, int),
                                   void (*_rf_lds)(_RF_LDS int *,
                                                   _RF_LDS int *),
                                   const int rnv, const uint64_t k);
void _INLINE_ATTR_ __kmpc_iteamr_ui(_UI v, _UI *r_ptr, void (*_rf)(_UI *, _UI),
                                    void (*_rf_lds)(_RF_LDS _UI *,
                                                    _RF_LDS _UI *),
                                    const _UI rnv, const uint64_t k);
void _INLINE_ATTR_ __kmpc_iteamr_l(long v, long *r_ptr,
                                   void (*_rf)(long *, long),
                                   void (*_rf_lds)(_RF_LDS long *,
                                                   _RF_LDS long *),
                                   const long rnv, const uint64_t k);
void _INLINE_ATTR_ __kmpc_iteamr_ul(_UL v, _UL *r_ptr, void (*_rf)(_UL *, _UL),
                                    void (*_rf_lds)(_RF_LDS _UL *,
                                                    _RF_LDS _UL *),
                                    const _UL rnv, const uint64_t k);

// rfun declarations (unchanged)
void __kmpc_rfun_sum_d(double *val, double otherval);
void __kmpc_rfun_sum_lds_d(_RF_LDS double *val, _RF_LDS double *otherval);
void __kmpc_rfun_sum_f(float *val, float otherval);
void __kmpc_rfun_sum_lds_f(_RF_LDS float *val, _RF_LDS float *otherval);
void __kmpc_rfun_sum_cd(_CD *val, _CD otherval);
void __kmpc_rfun_sum_lds_cd(_RF_LDS _CD *val, _RF_LDS _CD *otherval);
void __kmpc_rfun_sum_cf(_CF *val, _CF otherval);
void __kmpc_rfun_sum_lds_cf(_RF_LDS _CF *val, _RF_LDS _CF *otherval);
void __kmpc_rfun_sum_i(int *val, int otherval);
void __kmpc_rfun_sum_lds_i(_RF_LDS int *val, _RF_LDS int *otherval);
void __kmpc_rfun_sum_ui(_UI *val, _UI otherval);
void __kmpc_rfun_sum_lds_ui(_RF_LDS _UI *val, _RF_LDS _UI *otherval);
void __kmpc_rfun_sum_l(long *val, long otherval);
void __kmpc_rfun_sum_lds_l(_RF_LDS long *val, _RF_LDS long *otherval);
void __kmpc_rfun_sum_ul(_UL *val, _UL otherval);
void __kmpc_rfun_sum_lds_ul(_RF_LDS _UL *val, _RF_LDS _UL *otherval);
void __kmpc_rfun_max_d(double *val, double otherval);
void __kmpc_rfun_max_lds_d(_RF_LDS double *val, _RF_LDS double *otherval);
void __kmpc_rfun_max_f(float *val, float otherval);
void __kmpc_rfun_max_lds_f(_RF_LDS float *val, _RF_LDS float *otherval);
void __kmpc_rfun_max_i(int *val, int otherval);
void __kmpc_rfun_max_lds_i(_RF_LDS int *val, _RF_LDS int *otherval);
void __kmpc_rfun_max_ui(_UI *val, _UI otherval);
void __kmpc_rfun_max_lds_ui(_RF_LDS _UI *val, _RF_LDS _UI *otherval);
void __kmpc_rfun_max_l(long *val, long otherval);
void __kmpc_rfun_max_lds_l(_RF_LDS long *val, _RF_LDS long *otherval);
void __kmpc_rfun_max_ul(_UL *val, _UL otherval);
void __kmpc_rfun_max_lds_ul(_RF_LDS _UL *val, _RF_LDS _UL *otherval);
void __kmpc_rfun_min_d(double *val, double otherval);
void __kmpc_rfun_min_lds_d(_RF_LDS double *val, _RF_LDS double *otherval);
void __kmpc_rfun_min_f(float *val, float otherval);
void __kmpc_rfun_min_lds_f(_RF_LDS float *val, _RF_LDS float *otherval);
void __kmpc_rfun_min_i(int *val, int otherval);
void __kmpc_rfun_min_lds_i(_RF_LDS int *val, _RF_LDS int *otherval);
void __kmpc_rfun_min_ui(_UI *val, _UI otherval);
void __kmpc_rfun_min_lds_ui(_RF_LDS _UI *val, _RF_LDS _UI *otherval);
void __kmpc_rfun_min_l(long *val, long otherval);
void __kmpc_rfun_min_lds_l(_RF_LDS long *val, _RF_LDS long *otherval);
void __kmpc_rfun_min_ul(_UL *val, _UL otherval);
void __kmpc_rfun_min_lds_ul(_RF_LDS _UL *val, _RF_LDS _UL *otherval);

#undef _RF_LDS
int __kmpc_get_warp_size();
} // end extern C

#else

// For host compilation, define null stub functions for host linking.
#include <cstdio>
extern "C" {
#undef _RF_LDS
#define _RF_LDS

// Cross-team reduction stubs
void __kmpc_xteamr_d(double, double *, double *, uint32_t *,
                     void (*)(double *, double),
                     void (*)(_RF_LDS double *, _RF_LDS double *), const double,
                     const uint64_t, const uint32_t, int) {}
void __kmpc_xteamr_f(float, float *, float *, uint32_t *,
                     void (*)(float *, float),
                     void (*)(_RF_LDS float *, _RF_LDS float *), const float,
                     const uint64_t, const uint32_t, int) {}
void __kmpc_xteamr_cd(_CD, _CD *, _CD *, uint32_t *, void (*)(_CD *, _CD),
                      void (*)(_RF_LDS _CD *, _RF_LDS _CD *), const _CD,
                      const uint64_t, const uint32_t, int) {}
void __kmpc_xteamr_cf(_CF, _CF *, _CF *, uint32_t *, void (*)(_CF *, _CF),
                      void (*)(_RF_LDS _CF *, _RF_LDS _CF *), const _CF,
                      const uint64_t, const uint32_t, int) {}
void __kmpc_xteamr_i(int, int *, int *, uint32_t *, void (*)(int *, int),
                     void (*)(_RF_LDS int *, _RF_LDS int *), const int,
                     const uint64_t, const uint32_t, int) {}
void __kmpc_xteamr_ui(_UI, _UI *, _UI *, uint32_t *, void (*)(_UI *, _UI),
                      void (*)(_RF_LDS _UI *, _RF_LDS _UI *), const _UI,
                      const uint64_t, const uint32_t, int) {}
void __kmpc_xteamr_l(long, long *, long *, uint32_t *, void (*)(long *, long),
                     void (*)(_RF_LDS long *, _RF_LDS long *), const long,
                     const uint64_t, const uint32_t, int) {}
void __kmpc_xteamr_ul(_UL, _UL *, _UL *, uint32_t *, void (*)(_UL *, _UL),
                      void (*)(_RF_LDS _UL *, _RF_LDS _UL *), const _UL,
                      const uint64_t, const uint32_t, int) {}

// Fast sum stubs
void __kmpc_xteamr_d_fast_sum(double, double *, double *, uint32_t *,
                              void (*)(double *, double),
                              void (*)(_RF_LDS double *, _RF_LDS double *),
                              const double, const uint64_t, const uint32_t,
                              int) {}
void __kmpc_xteamr_f_fast_sum(float, float *, float *, uint32_t *,
                              void (*)(float *, float),
                              void (*)(_RF_LDS float *, _RF_LDS float *),
                              const float, const uint64_t, const uint32_t,
                              int) {}
void __kmpc_xteamr_cd_fast_sum(_CD, _CD *, _CD *, uint32_t *,
                               void (*)(_CD *, _CD),
                               void (*)(_RF_LDS _CD *, _RF_LDS _CD *),
                               const _CD, const uint64_t, const uint32_t, int) {
}
void __kmpc_xteamr_cf_fast_sum(_CF, _CF *, _CF *, uint32_t *,
                               void (*)(_CF *, _CF),
                               void (*)(_RF_LDS _CF *, _RF_LDS _CF *),
                               const _CF, const uint64_t, const uint32_t, int) {
}
void __kmpc_xteamr_i_fast_sum(int, int *, int *, uint32_t *,
                              void (*)(int *, int),
                              void (*)(_RF_LDS int *, _RF_LDS int *), const int,
                              const uint64_t, const uint32_t, int) {}
void __kmpc_xteamr_ui_fast_sum(_UI, _UI *, _UI *, uint32_t *,
                               void (*)(_UI *, _UI),
                               void (*)(_RF_LDS _UI *, _RF_LDS _UI *),
                               const _UI, const uint64_t, const uint32_t, int) {
}
void __kmpc_xteamr_l_fast_sum(long, long *, long *, uint32_t *,
                              void (*)(long *, long),
                              void (*)(_RF_LDS long *, _RF_LDS long *),
                              const long, const uint64_t, const uint32_t, int) {
}
void __kmpc_xteamr_ul_fast_sum(_UL, _UL *, _UL *, uint32_t *,
                               void (*)(_UL *, _UL),
                               void (*)(_RF_LDS _UL *, _RF_LDS _UL *),
                               const _UL, const uint64_t, const uint32_t, int) {
}

// Intra-team reduction stubs
void __kmpc_iteamr_d(double, double *, void (*)(double *, double),
                     void (*)(_RF_LDS double *, _RF_LDS double *), const double,
                     const uint64_t) {}
void __kmpc_iteamr_f(float, float *, void (*)(float *, float),
                     void (*)(_RF_LDS float *, _RF_LDS float *), const float,
                     const uint64_t) {}
void __kmpc_iteamr_cd(_CD, _CD *, void (*)(_CD *, _CD),
                      void (*)(_RF_LDS _CD *, _RF_LDS _CD *), const _CD,
                      const uint64_t) {}
void __kmpc_iteamr_cf(_CF, _CF *, void (*)(_CF *, _CF),
                      void (*)(_RF_LDS _CF *, _RF_LDS _CF *), const _CF,
                      const uint64_t) {}
void __kmpc_iteamr_i(int, int *, void (*)(int *, int),
                     void (*)(_RF_LDS int *, _RF_LDS int *), const int,
                     const uint64_t) {}
void __kmpc_iteamr_ui(_UI, _UI *, void (*)(_UI *, _UI),
                      void (*)(_RF_LDS _UI *, _RF_LDS _UI *), const _UI,
                      const uint64_t) {}
void __kmpc_iteamr_l(long, long *, void (*)(long *, long),
                     void (*)(_RF_LDS long *, _RF_LDS long *), const long,
                     const uint64_t) {}
void __kmpc_iteamr_ul(_UL, _UL *, void (*)(_UL *, _UL),
                      void (*)(_RF_LDS _UL *, _RF_LDS _UL *), const _UL,
                      const uint64_t) {}

// rfun stubs (unchanged)
void __kmpc_rfun_sum_d(double *val, double otherval) {}
void __kmpc_rfun_sum_lds_d(_RF_LDS double *val, _RF_LDS double *otherval) {}
void __kmpc_rfun_sum_f(float *val, float otherval) {}
void __kmpc_rfun_sum_lds_f(_RF_LDS float *val, _RF_LDS float *otherval) {}
void __kmpc_rfun_sum_cd(_CD *val, _CD otherval) {}
void __kmpc_rfun_sum_lds_cd(_RF_LDS _CD *val, _RF_LDS _CD *otherval) {}
void __kmpc_rfun_sum_cf(_CF *val, _CF otherval) {}
void __kmpc_rfun_sum_lds_cf(_RF_LDS _CF *val, _RF_LDS _CF *otherval) {}
void __kmpc_rfun_sum_i(int *val, int otherval) {}
void __kmpc_rfun_sum_lds_i(_RF_LDS int *val, _RF_LDS int *otherval) {}
void __kmpc_rfun_sum_ui(_UI *val, _UI otherval) {}
void __kmpc_rfun_sum_lds_ui(_RF_LDS _UI *val, _RF_LDS _UI *otherval) {}
void __kmpc_rfun_sum_l(long *val, long otherval) {}
void __kmpc_rfun_sum_lds_l(_RF_LDS long *val, _RF_LDS long *otherval) {}
void __kmpc_rfun_sum_ul(_UL *val, _UL otherval) {}
void __kmpc_rfun_sum_lds_ul(_RF_LDS _UL *val, _RF_LDS _UL *otherval) {}
void __kmpc_rfun_max_d(double *val, double otherval) {}
void __kmpc_rfun_max_lds_d(_RF_LDS double *val, _RF_LDS double *otherval) {}
void __kmpc_rfun_max_f(float *val, float otherval) {}
void __kmpc_rfun_max_lds_f(_RF_LDS float *val, _RF_LDS float *otherval) {}
void __kmpc_rfun_max_i(int *val, int otherval) {}
void __kmpc_rfun_max_lds_i(_RF_LDS int *val, _RF_LDS int *otherval) {}
void __kmpc_rfun_max_ui(_UI *val, _UI otherval) {}
void __kmpc_rfun_max_lds_ui(_RF_LDS _UI *val, _RF_LDS _UI *otherval) {}
void __kmpc_rfun_max_l(long *val, long otherval) {}
void __kmpc_rfun_max_lds_l(_RF_LDS long *val, _RF_LDS long *otherval) {}
void __kmpc_rfun_max_ul(_UL *val, _UL otherval) {}
void __kmpc_rfun_max_lds_ul(_RF_LDS _UL *val, _RF_LDS _UL *otherval) {}
void __kmpc_rfun_min_d(double *val, double otherval) {}
void __kmpc_rfun_min_lds_d(_RF_LDS double *val, _RF_LDS double *otherval) {}
void __kmpc_rfun_min_f(float *val, float otherval) {}
void __kmpc_rfun_min_lds_f(_RF_LDS float *val, _RF_LDS float *otherval) {}
void __kmpc_rfun_min_i(int *val, int otherval) {}
void __kmpc_rfun_min_lds_i(_RF_LDS int *val, _RF_LDS int *otherval) {}
void __kmpc_rfun_min_ui(_UI *val, _UI otherval) {}
void __kmpc_rfun_min_lds_ui(_RF_LDS _UI *val, _RF_LDS _UI *otherval) {}
void __kmpc_rfun_min_l(long *val, long otherval) {}
void __kmpc_rfun_min_lds_l(_RF_LDS long *val, _RF_LDS long *otherval) {}
void __kmpc_rfun_min_ul(_UL *val, _UL otherval) {}
void __kmpc_rfun_min_lds_ul(_RF_LDS _UL *val, _RF_LDS _UL *otherval) {}

#undef _RF_LDS
int __kmpc_get_warp_size() {
  printf("ERROR: executing __kmpc_get_warp_size on host\n");
  return -1;
}
} // end extern C

#endif

// Overloaded helper functions that wrap the extern DeviceRTL calls.
// These are used by the xteamr test framework to invoke the reduction
// functions.

// _overload_to_extern_sum
void _INLINE_ATTR_ _overload_to_extern_sum(double val, double *rv, double *tvs,
                                           uint32_t *td, const double iv,
                                           const uint64_t k,
                                           const uint32_t numteams) {
  __kmpc_xteamr_d(val, rv, tvs, td, __kmpc_rfun_sum_d, __kmpc_rfun_sum_lds_d,
                  iv, k, numteams, _XTEAMR_SCOPE);
}
void _INLINE_ATTR_ _overload_to_extern_sum(float val, float *rv, float *tvs,
                                           uint32_t *td, const float iv,
                                           const uint64_t k,
                                           const uint32_t numteams) {
  __kmpc_xteamr_f(val, rv, tvs, td, __kmpc_rfun_sum_f, __kmpc_rfun_sum_lds_f,
                  iv, k, numteams, _XTEAMR_SCOPE);
}
void _INLINE_ATTR_ _overload_to_extern_sum(_CD val, _CD *rv, _CD *tvs,
                                           uint32_t *td, const _CD iv,
                                           const uint64_t k,
                                           const uint32_t numteams) {
  __kmpc_xteamr_cd(val, rv, tvs, td, __kmpc_rfun_sum_cd, __kmpc_rfun_sum_lds_cd,
                   iv, k, numteams, _XTEAMR_SCOPE);
}
void _INLINE_ATTR_ _overload_to_extern_sum(_CF val, _CF *rv, _CF *tvs,
                                           uint32_t *td, const _CF iv,
                                           const uint64_t k,
                                           const uint32_t numteams) {
  __kmpc_xteamr_cf(val, rv, tvs, td, __kmpc_rfun_sum_cf, __kmpc_rfun_sum_lds_cf,
                   iv, k, numteams, _XTEAMR_SCOPE);
}
void _INLINE_ATTR_ _overload_to_extern_sum(int val, int *rv, int *tvs,
                                           uint32_t *td, const int iv,
                                           const uint64_t k,
                                           const uint32_t numteams) {
  __kmpc_xteamr_i(val, rv, tvs, td, __kmpc_rfun_sum_i, __kmpc_rfun_sum_lds_i,
                  iv, k, numteams, _XTEAMR_SCOPE);
}
void _INLINE_ATTR_ _overload_to_extern_sum(_UI val, _UI *rv, _UI *tvs,
                                           uint32_t *td, const _UI iv,
                                           const uint64_t k,
                                           const uint32_t numteams) {
  __kmpc_xteamr_ui(val, rv, tvs, td, __kmpc_rfun_sum_ui, __kmpc_rfun_sum_lds_ui,
                   iv, k, numteams, _XTEAMR_SCOPE);
}
void _INLINE_ATTR_ _overload_to_extern_sum(long val, long *rv, long *tvs,
                                           uint32_t *td, const long iv,
                                           const uint64_t k,
                                           const uint32_t numteams) {
  __kmpc_xteamr_l(val, rv, tvs, td, __kmpc_rfun_sum_l, __kmpc_rfun_sum_lds_l,
                  iv, k, numteams, _XTEAMR_SCOPE);
}
void _INLINE_ATTR_ _overload_to_extern_sum(_UL val, _UL *rv, _UL *tvs,
                                           uint32_t *td, const _UL iv,
                                           const uint64_t k,
                                           const uint32_t numteams) {
  __kmpc_xteamr_ul(val, rv, tvs, td, __kmpc_rfun_sum_ul, __kmpc_rfun_sum_lds_ul,
                   iv, k, numteams, _XTEAMR_SCOPE);
}

// _overload_to_extern_max
void _INLINE_ATTR_ _overload_to_extern_max(double val, double *rv, double *tvs,
                                           uint32_t *td, const double iv,
                                           const uint64_t k,
                                           const uint32_t numteams) {
  __kmpc_xteamr_d(val, rv, tvs, td, __kmpc_rfun_max_d, __kmpc_rfun_max_lds_d,
                  iv, k, numteams, _XTEAMR_SCOPE);
}
void _INLINE_ATTR_ _overload_to_extern_max(float val, float *rv, float *tvs,
                                           uint32_t *td, const float iv,
                                           const uint64_t k,
                                           const uint32_t numteams) {
  __kmpc_xteamr_f(val, rv, tvs, td, __kmpc_rfun_max_f, __kmpc_rfun_max_lds_f,
                  iv, k, numteams, _XTEAMR_SCOPE);
}
void _INLINE_ATTR_ _overload_to_extern_max(int val, int *rv, int *tvs,
                                           uint32_t *td, const int iv,
                                           const uint64_t k,
                                           const uint32_t numteams) {
  __kmpc_xteamr_i(val, rv, tvs, td, __kmpc_rfun_max_i, __kmpc_rfun_max_lds_i,
                  iv, k, numteams, _XTEAMR_SCOPE);
}
void _INLINE_ATTR_ _overload_to_extern_max(_UI val, _UI *rv, _UI *tvs,
                                           uint32_t *td, const _UI iv,
                                           const uint64_t k,
                                           const uint32_t numteams) {
  __kmpc_xteamr_ui(val, rv, tvs, td, __kmpc_rfun_max_ui, __kmpc_rfun_max_lds_ui,
                   iv, k, numteams, _XTEAMR_SCOPE);
}
void _INLINE_ATTR_ _overload_to_extern_max(long val, long *rv, long *tvs,
                                           uint32_t *td, const long iv,
                                           const uint64_t k,
                                           const uint32_t numteams) {
  __kmpc_xteamr_l(val, rv, tvs, td, __kmpc_rfun_max_l, __kmpc_rfun_max_lds_l,
                  iv, k, numteams, _XTEAMR_SCOPE);
}
void _INLINE_ATTR_ _overload_to_extern_max(_UL val, _UL *rv, _UL *tvs,
                                           uint32_t *td, const _UL iv,
                                           const uint64_t k,
                                           const uint32_t numteams) {
  __kmpc_xteamr_ul(val, rv, tvs, td, __kmpc_rfun_max_ul, __kmpc_rfun_max_lds_ul,
                   iv, k, numteams, _XTEAMR_SCOPE);
}

// _overload_to_extern_min
void _INLINE_ATTR_ _overload_to_extern_min(double val, double *rv, double *tvs,
                                           uint32_t *td, const double iv,
                                           const uint64_t k,
                                           const uint32_t numteams) {
  __kmpc_xteamr_d(val, rv, tvs, td, __kmpc_rfun_min_d, __kmpc_rfun_min_lds_d,
                  iv, k, numteams, _XTEAMR_SCOPE);
}
void _INLINE_ATTR_ _overload_to_extern_min(float val, float *rv, float *tvs,
                                           uint32_t *td, const float iv,
                                           const uint64_t k,
                                           const uint32_t numteams) {
  __kmpc_xteamr_f(val, rv, tvs, td, __kmpc_rfun_min_f, __kmpc_rfun_min_lds_f,
                  iv, k, numteams, _XTEAMR_SCOPE);
}
void _INLINE_ATTR_ _overload_to_extern_min(int val, int *rv, int *tvs,
                                           uint32_t *td, const int iv,
                                           const uint64_t k,
                                           const uint32_t numteams) {
  __kmpc_xteamr_i(val, rv, tvs, td, __kmpc_rfun_min_i, __kmpc_rfun_min_lds_i,
                  iv, k, numteams, _XTEAMR_SCOPE);
}
void _INLINE_ATTR_ _overload_to_extern_min(_UI val, _UI *rv, _UI *tvs,
                                           uint32_t *td, const _UI iv,
                                           const uint64_t k,
                                           const uint32_t numteams) {
  __kmpc_xteamr_ui(val, rv, tvs, td, __kmpc_rfun_min_ui, __kmpc_rfun_min_lds_ui,
                   iv, k, numteams, _XTEAMR_SCOPE);
}
void _INLINE_ATTR_ _overload_to_extern_min(long val, long *rv, long *tvs,
                                           uint32_t *td, const long iv,
                                           const uint64_t k,
                                           const uint32_t numteams) {
  __kmpc_xteamr_l(val, rv, tvs, td, __kmpc_rfun_min_l, __kmpc_rfun_min_lds_l,
                  iv, k, numteams, _XTEAMR_SCOPE);
}
void _INLINE_ATTR_ _overload_to_extern_min(_UL val, _UL *rv, _UL *tvs,
                                           uint32_t *td, const _UL iv,
                                           const uint64_t k,
                                           const uint32_t numteams) {
  __kmpc_xteamr_ul(val, rv, tvs, td, __kmpc_rfun_min_ul, __kmpc_rfun_min_lds_ul,
                   iv, k, numteams, _XTEAMR_SCOPE);
}

#undef _CD
#undef _CF
#undef _UI
#undef _UL
#undef _INLINE_ATTR_
#undef _XTEAMR_SCOPE
