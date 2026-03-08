/*=============================== test_xteams.h -=============================//
 *
 * Headerfile for testing the Cross-Team Scan Implementation in the DeviceRTL.
 * Also contains headers for the kmpc_ functions defined in the DeviceRTL/src/
 * Xteams.cpp.
 *
 * New single-pass scan interface (decoupled look-back algorithm).
 *
//===----------------------------------------------------------------------===*/

#include <type_traits>

#include "../xteamr/test_xteamr.h" // include reduction helper functions rfun_*

#define _CD double _Complex
#define _CF float _Complex
#define _UI unsigned int
#define _UL unsigned long
#define _INLINE_ATTR_ __attribute__((flatten, always_inline))

// Extern xteams functions defined in the device runtime are declared/defined
// here in the test header file because user apps cannot include the DeviceRTL
// Xteams.h header file.

#define _XTEAMS_FUNC(T, TS, ATTR, BODY)                                        \
  ATTR void __kmpc_xteams_##TS(T v, T *result, uint32_t *status,               \
                               T *aggregates, T *prefixes, void (*rf)(T *, T), \
                               const T rnv, const uint64_t k,                  \
                               bool is_inclusive) BODY

#if defined(__AMDGCN__) || defined(__NVPTX__)
extern "C" {
_XTEAMS_FUNC(double, d, _INLINE_ATTR_, ;)
_XTEAMS_FUNC(float, f, _INLINE_ATTR_, ;)
_XTEAMS_FUNC(_CD, cd, _INLINE_ATTR_, ;)
_XTEAMS_FUNC(_CF, cf, _INLINE_ATTR_, ;)
_XTEAMS_FUNC(int, i, _INLINE_ATTR_, ;)
_XTEAMS_FUNC(_UI, ui, _INLINE_ATTR_, ;)
_XTEAMS_FUNC(long, l, _INLINE_ATTR_, ;)
_XTEAMS_FUNC(_UL, ul, _INLINE_ATTR_, ;)
} // end extern C

#else

// For host compilation, define null functions for host linking.

extern "C" {
_XTEAMS_FUNC(double, d, , {})
_XTEAMS_FUNC(float, f, , {})
_XTEAMS_FUNC(_CD, cd, , {})
_XTEAMS_FUNC(_CF, cf, , {})
_XTEAMS_FUNC(int, i, , {})
_XTEAMS_FUNC(_UI, ui, , {})
_XTEAMS_FUNC(long, l, , {})
_XTEAMS_FUNC(_UL, ul, , {})
} // end extern C

#endif

#undef _XTEAMS_FUNC

// Get the correct extern DeviceRTL scan functions based on the type.
template <typename T> constexpr auto get_kmpc_xteams_func() {
  if constexpr (std::is_same_v<T, double>) {
    return __kmpc_xteams_d;
  } else if constexpr (std::is_same_v<T, float>) {
    return __kmpc_xteams_f;
  } else if constexpr (std::is_same_v<T, _CD>) {
    return __kmpc_xteams_cd;
  } else if constexpr (std::is_same_v<T, _CF>) {
    return __kmpc_xteams_cf;
  } else if constexpr (std::is_same_v<T, int>) {
    return __kmpc_xteams_i;
  } else if constexpr (std::is_same_v<T, _UI>) {
    return __kmpc_xteams_ui;
  } else if constexpr (std::is_same_v<T, long>) {
    return __kmpc_xteams_l;
  } else if constexpr (std::is_same_v<T, _UL>) {
    return __kmpc_xteams_ul;
  } else {
    static_assert(false, "Unsupported type");
  }
}

#undef _CD
#undef _CF
#undef _UI
#undef _UL
#undef _INLINE_ATTR_
