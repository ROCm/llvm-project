/*=============================== test_xteams.h -=============================//
 *
 * Headerfile for testing the Cross-Team Scan Implementation in the DeviceRTL.
 * Also contains headers for the kmpc_ functions defined in the DeviceRTL/src/
 * Xteams.cpp.
 *
 * New single-pass scan interface (decoupled look-back algorithm).
 *
//===----------------------------------------------------------------------===*/

#include "../xteamr/test_xteamr.h" // include reduction helper functions rfun_*

#define _CD double _Complex
#define _CF float _Complex
#define _UI unsigned int
#define _UL unsigned long
#define _INLINE_ATTR_ __attribute__((flatten, always_inline))

// Headers for extern xteams functions defined in libomptarget DeviceRTL
// are defined here in the test header file because user apps cannot include
// the DeviceRTL Xteams.h header file.

#if defined(__AMDGCN__) || defined(__NVPTX__)
extern "C" {
double _INLINE_ATTR_ __kmpc_xteams_d(double v, uint32_t *status,
                                     double *values,
                                     void (*rf)(double *, double),
                                     const double rnv, const uint64_t k,
                                     const uint64_t n, bool is_inclusive);
float _INLINE_ATTR_ __kmpc_xteams_f(float v, uint32_t *status,
                                    float *values,
                                    void (*rf)(float *, float), const float rnv,
                                    const uint64_t k, const uint64_t n,
                                    bool is_inclusive);
_CD _INLINE_ATTR_ __kmpc_xteams_cd(_CD v, uint32_t *status,
                                   _CD *values,
                                   void (*rf)(_CD *, _CD), const _CD rnv,
                                   const uint64_t k, const uint64_t n,
                                   bool is_inclusive);
_CF _INLINE_ATTR_ __kmpc_xteams_cf(_CF v, uint32_t *status,
                                   _CF *values,
                                   void (*rf)(_CF *, _CF), const _CF rnv,
                                   const uint64_t k, const uint64_t n,
                                   bool is_inclusive);
int _INLINE_ATTR_ __kmpc_xteams_i(int v, uint32_t *status,
                                  int *values,
                                  void (*rf)(int *, int), const int rnv,
                                  const uint64_t k, const uint64_t n,
                                  bool is_inclusive);
_UI _INLINE_ATTR_ __kmpc_xteams_ui(_UI v, uint32_t *status,
                                   _UI *values,
                                   void (*rf)(_UI *, _UI), const _UI rnv,
                                   const uint64_t k, const uint64_t n,
                                   bool is_inclusive);
long _INLINE_ATTR_ __kmpc_xteams_l(long v, uint32_t *status,
                                   long *values,
                                   void (*rf)(long *, long), const long rnv,
                                   const uint64_t k, const uint64_t n,
                                   bool is_inclusive);
_UL _INLINE_ATTR_ __kmpc_xteams_ul(_UL v, uint32_t *status,
                                   _UL *values,
                                   void (*rf)(_UL *, _UL), const _UL rnv,
                                   const uint64_t k, const uint64_t n,
                                   bool is_inclusive);
} // end extern C

#else

// For host compilation, define null functions for host linking.

extern "C" {
double __kmpc_xteams_d(double v, uint32_t *status, double *values,
                       void (*rf)(double *, double),
                       const double rnv, const uint64_t k, const uint64_t n,
                       bool is_inclusive) { return 0; }
float __kmpc_xteams_f(float v, uint32_t *status, float *values,
                      void (*rf)(float *, float), const float rnv,
                      const uint64_t k, const uint64_t n,
                      bool is_inclusive) { return 0; }
_CD __kmpc_xteams_cd(_CD v, uint32_t *status, _CD *values,
                     void (*rf)(_CD *, _CD), const _CD rnv,
                     const uint64_t k, const uint64_t n,
                     bool is_inclusive) { return 0; }
_CF __kmpc_xteams_cf(_CF v, uint32_t *status, _CF *values,
                     void (*rf)(_CF *, _CF), const _CF rnv,
                     const uint64_t k, const uint64_t n,
                     bool is_inclusive) { return 0; }
int __kmpc_xteams_i(int v, uint32_t *status, int *values,
                    void (*rf)(int *, int), const int rnv,
                    const uint64_t k, const uint64_t n,
                    bool is_inclusive) { return 0; }
_UI __kmpc_xteams_ui(_UI v, uint32_t *status, _UI *values,
                     void (*rf)(_UI *, _UI), const _UI rnv,
                     const uint64_t k, const uint64_t n,
                     bool is_inclusive) { return 0; }
long __kmpc_xteams_l(long v, uint32_t *status, long *values,
                     void (*rf)(long *, long), const long rnv,
                     const uint64_t k, const uint64_t n,
                     bool is_inclusive) { return 0; }
_UL __kmpc_xteams_ul(_UL v, uint32_t *status, _UL *values,
                     void (*rf)(_UL *, _UL), const _UL rnv,
                     const uint64_t k, const uint64_t n,
                     bool is_inclusive) { return 0; }
} // end extern C

#endif

// Overloaded helper functions for this test framework (xteams.cpp) to invoke
// the extern DeviceRTL scan functions.

// _overload_to_extern_scan_sum - sum reduction scan
double _INLINE_ATTR_ _overload_to_extern_scan_sum(
    double val, uint32_t *status, double *values,
    const double rnv, const uint64_t k, const uint64_t n, bool is_inclusive) {
  return __kmpc_xteams_d(val, status, values, __kmpc_rfun_sum_d, rnv, k,
                         n, is_inclusive);
}
float _INLINE_ATTR_ _overload_to_extern_scan_sum(
    float val, uint32_t *status, float *values,
    const float rnv, const uint64_t k, const uint64_t n, bool is_inclusive) {
  return __kmpc_xteams_f(val, status, values, __kmpc_rfun_sum_f, rnv, k,
                         n, is_inclusive);
}
_CD _INLINE_ATTR_ _overload_to_extern_scan_sum(
    _CD val, uint32_t *status, _CD *values,
    const _CD rnv, const uint64_t k, const uint64_t n, bool is_inclusive) {
  return __kmpc_xteams_cd(val, status, values, __kmpc_rfun_sum_cd, rnv, k,
                          n, is_inclusive);
}
_CF _INLINE_ATTR_ _overload_to_extern_scan_sum(
    _CF val, uint32_t *status, _CF *values,
    const _CF rnv, const uint64_t k, const uint64_t n, bool is_inclusive) {
  return __kmpc_xteams_cf(val, status, values, __kmpc_rfun_sum_cf, rnv, k,
                          n, is_inclusive);
}
int _INLINE_ATTR_ _overload_to_extern_scan_sum(
    int val, uint32_t *status, int *values,
    const int rnv, const uint64_t k, const uint64_t n, bool is_inclusive) {
  return __kmpc_xteams_i(val, status, values, __kmpc_rfun_sum_i, rnv, k,
                         n, is_inclusive);
}
_UI _INLINE_ATTR_ _overload_to_extern_scan_sum(
    _UI val, uint32_t *status, _UI *values,
    const _UI rnv, const uint64_t k, const uint64_t n, bool is_inclusive) {
  return __kmpc_xteams_ui(val, status, values, __kmpc_rfun_sum_ui, rnv, k,
                          n, is_inclusive);
}
long _INLINE_ATTR_ _overload_to_extern_scan_sum(
    long val, uint32_t *status, long *values,
    const long rnv, const uint64_t k, const uint64_t n, bool is_inclusive) {
  return __kmpc_xteams_l(val, status, values, __kmpc_rfun_sum_l, rnv, k,
                         n, is_inclusive);
}
_UL _INLINE_ATTR_ _overload_to_extern_scan_sum(
    _UL val, uint32_t *status, _UL *values,
    const _UL rnv, const uint64_t k, const uint64_t n, bool is_inclusive) {
  return __kmpc_xteams_ul(val, status, values, __kmpc_rfun_sum_ul, rnv, k,
                          n, is_inclusive);
}

// _overload_to_extern_scan_max - max reduction scan
double _INLINE_ATTR_ _overload_to_extern_scan_max(
    double val, uint32_t *status, double *values,
    const double rnv, const uint64_t k, const uint64_t n, bool is_inclusive) {
  return __kmpc_xteams_d(val, status, values, __kmpc_rfun_max_d, rnv, k,
                         n, is_inclusive);
}
float _INLINE_ATTR_ _overload_to_extern_scan_max(
    float val, uint32_t *status, float *values,
    const float rnv, const uint64_t k, const uint64_t n, bool is_inclusive) {
  return __kmpc_xteams_f(val, status, values, __kmpc_rfun_max_f, rnv, k,
                         n, is_inclusive);
}
int _INLINE_ATTR_ _overload_to_extern_scan_max(
    int val, uint32_t *status, int *values,
    const int rnv, const uint64_t k, const uint64_t n, bool is_inclusive) {
  return __kmpc_xteams_i(val, status, values, __kmpc_rfun_max_i, rnv, k,
                         n, is_inclusive);
}
_UI _INLINE_ATTR_ _overload_to_extern_scan_max(
    _UI val, uint32_t *status, _UI *values,
    const _UI rnv, const uint64_t k, const uint64_t n, bool is_inclusive) {
  return __kmpc_xteams_ui(val, status, values, __kmpc_rfun_max_ui, rnv, k,
                          n, is_inclusive);
}
long _INLINE_ATTR_ _overload_to_extern_scan_max(
    long val, uint32_t *status, long *values,
    const long rnv, const uint64_t k, const uint64_t n, bool is_inclusive) {
  return __kmpc_xteams_l(val, status, values, __kmpc_rfun_max_l, rnv, k,
                         n, is_inclusive);
}
_UL _INLINE_ATTR_ _overload_to_extern_scan_max(
    _UL val, uint32_t *status, _UL *values,
    const _UL rnv, const uint64_t k, const uint64_t n, bool is_inclusive) {
  return __kmpc_xteams_ul(val, status, values, __kmpc_rfun_max_ul, rnv, k,
                          n, is_inclusive);
}

// _overload_to_extern_scan_min - min reduction scan
double _INLINE_ATTR_ _overload_to_extern_scan_min(
    double val, uint32_t *status, double *values,
    const double rnv, const uint64_t k, const uint64_t n, bool is_inclusive) {
  return __kmpc_xteams_d(val, status, values, __kmpc_rfun_min_d, rnv, k,
                         n, is_inclusive);
}
float _INLINE_ATTR_ _overload_to_extern_scan_min(
    float val, uint32_t *status, float *values,
    const float rnv, const uint64_t k, const uint64_t n, bool is_inclusive) {
  return __kmpc_xteams_f(val, status, values, __kmpc_rfun_min_f, rnv, k,
                         n, is_inclusive);
}
int _INLINE_ATTR_ _overload_to_extern_scan_min(
    int val, uint32_t *status, int *values,
    const int rnv, const uint64_t k, const uint64_t n, bool is_inclusive) {
  return __kmpc_xteams_i(val, status, values, __kmpc_rfun_min_i, rnv, k,
                         n, is_inclusive);
}
_UI _INLINE_ATTR_ _overload_to_extern_scan_min(
    _UI val, uint32_t *status, _UI *values,
    const _UI rnv, const uint64_t k, const uint64_t n, bool is_inclusive) {
  return __kmpc_xteams_ui(val, status, values, __kmpc_rfun_min_ui, rnv, k,
                          n, is_inclusive);
}
long _INLINE_ATTR_ _overload_to_extern_scan_min(
    long val, uint32_t *status, long *values,
    const long rnv, const uint64_t k, const uint64_t n, bool is_inclusive) {
  return __kmpc_xteams_l(val, status, values, __kmpc_rfun_min_l, rnv, k,
                         n, is_inclusive);
}
_UL _INLINE_ATTR_ _overload_to_extern_scan_min(
    _UL val, uint32_t *status, _UL *values,
    const _UL rnv, const uint64_t k, const uint64_t n, bool is_inclusive) {
  return __kmpc_xteams_ul(val, status, values, __kmpc_rfun_min_ul, rnv, k,
                          n, is_inclusive);
}

#undef _CD
#undef _CF
#undef _UI
#undef _UL
#undef _INLINE_ATTR_
