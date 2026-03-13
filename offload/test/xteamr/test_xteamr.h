// Header file: test_xteamr.h
// Declarations for the xteamr DeviceRTL interface used by the xteamr test.
// The new interface uses a single function per type (__kmpc_xteamr_d, etc.)
// with an extra int Scope parameter, plus _fast_sum and __kmpc_iteamr_
// variants. User apps cannot include DeviceRTL headers, so declarations are
// provided here.

#include <cstdint>
#include <type_traits>

#define _UI unsigned int
#define _UL unsigned long
#define _INLINE_ATTR_ __attribute__((flatten, always_inline))

#if defined(__AMDGCN__) || defined(__NVPTX__)
#define _XTEAMR_SCOPE __MEMORY_SCOPE_SYSTEM
#else
#define _XTEAMR_SCOPE 0
#endif

#define _XTEAMR_FUNC(T, TS, ATTR, BODY)                                        \
  ATTR void __kmpc_xteamr_##TS(                                                \
      T v, T *r_ptr, T *tvs, uint32_t *td, void (*_rf)(T *, T),                \
      void (*_rf_lds)(_RF_LDS T *, _RF_LDS T *), const T rnv,                  \
      const uint64_t k, const uint32_t numteams, int Scope) BODY

/// Built-in pair reduction function, see documentation above.
#define _REDUCTION_FUNC(T, OP, TS, BODY)                                       \
  void __kmpc_rfun_##OP##_##TS(T *val, T otherval) BODY;                       \
  void __kmpc_rfun_##OP##_lds_##TS(_RF_LDS T *val, _RF_LDS T *otherval) BODY

#define _REDUCTION_FUNC_ALL(OP, BODY)                                          \
  _REDUCTION_FUNC(double, OP, d, BODY)                                         \
  _REDUCTION_FUNC(float, OP, f, BODY)                                          \
  _REDUCTION_FUNC(int, OP, i, BODY)                                            \
  _REDUCTION_FUNC(_UI, OP, ui, BODY)                                           \
  _REDUCTION_FUNC(long, OP, l, BODY)                                           \
  _REDUCTION_FUNC(_UL, OP, ul, BODY)

#if defined(__AMDGCN__) || defined(__NVPTX__)
extern "C" {
#define _RF_LDS volatile __attribute__((address_space(3)))

// Cross-team reduction
_XTEAMR_FUNC(double, d, _INLINE_ATTR_, ;)
_XTEAMR_FUNC(float, f, _INLINE_ATTR_, ;)
_XTEAMR_FUNC(int, i, _INLINE_ATTR_, ;)
_XTEAMR_FUNC(_UI, ui, _INLINE_ATTR_, ;)
_XTEAMR_FUNC(long, l, _INLINE_ATTR_, ;)
_XTEAMR_FUNC(_UL, ul, _INLINE_ATTR_, ;)

// Fast sum (uses atomic add)
_XTEAMR_FUNC(double, d_fast_sum, _INLINE_ATTR_, ;)
_XTEAMR_FUNC(float, f_fast_sum, _INLINE_ATTR_, ;)
_XTEAMR_FUNC(int, i_fast_sum, _INLINE_ATTR_, ;)
_XTEAMR_FUNC(_UI, ui_fast_sum, _INLINE_ATTR_, ;)
_XTEAMR_FUNC(long, l_fast_sum, _INLINE_ATTR_, ;)
_XTEAMR_FUNC(_UL, ul_fast_sum, _INLINE_ATTR_, ;)

// rfun declarations
_REDUCTION_FUNC_ALL(sum, ;)
_REDUCTION_FUNC_ALL(max, ;)
_REDUCTION_FUNC_ALL(min, ;)

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
_XTEAMR_FUNC(double, d, _INLINE_ATTR_, {})
_XTEAMR_FUNC(float, f, _INLINE_ATTR_, {})
_XTEAMR_FUNC(int, i, _INLINE_ATTR_, {})
_XTEAMR_FUNC(_UI, ui, _INLINE_ATTR_, {})
_XTEAMR_FUNC(long, l, _INLINE_ATTR_, {})
_XTEAMR_FUNC(_UL, ul, _INLINE_ATTR_, {})

// Fast sum stubs
_XTEAMR_FUNC(double, d_fast_sum, _INLINE_ATTR_, {})
_XTEAMR_FUNC(float, f_fast_sum, _INLINE_ATTR_, {})
_XTEAMR_FUNC(int, i_fast_sum, _INLINE_ATTR_, {})
_XTEAMR_FUNC(_UI, ui_fast_sum, _INLINE_ATTR_, {})
_XTEAMR_FUNC(long, l_fast_sum, _INLINE_ATTR_, {})
_XTEAMR_FUNC(_UL, ul_fast_sum, _INLINE_ATTR_, {})

// rfun stubs (unchanged)
_REDUCTION_FUNC_ALL(sum, {})
_REDUCTION_FUNC_ALL(max, {})
_REDUCTION_FUNC_ALL(min, {})

#undef _RF_LDS
int __kmpc_get_warp_size() {
  printf("ERROR: executing __kmpc_get_warp_size on host\n");
  return -1;
}
} // end extern C

#endif

#undef _XTEAMR_FUNC
#undef _REDUCTION_FUNC
#undef _REDUCTION_FUNC_ALL

template <typename T> constexpr auto get_kmpc_xteamr_func() {
  if constexpr (std::is_same_v<T, double>) {
    return __kmpc_xteamr_d;
  } else if constexpr (std::is_same_v<T, float>) {
    return __kmpc_xteamr_f;
  } else if constexpr (std::is_same_v<T, int>) {
    return __kmpc_xteamr_i;
  } else if constexpr (std::is_same_v<T, _UI>) {
    return __kmpc_xteamr_ui;
  } else if constexpr (std::is_same_v<T, long>) {
    return __kmpc_xteamr_l;
  } else if constexpr (std::is_same_v<T, _UL>) {
    return __kmpc_xteamr_ul;
  } else {
    static_assert(false, "Unsupported type");
  }
}

template <typename T> constexpr auto get_kmpc_rfun_sum_func() {
  if constexpr (std::is_same_v<T, double>) {
    return __kmpc_rfun_sum_d;
  } else if constexpr (std::is_same_v<T, float>) {
    return __kmpc_rfun_sum_f;
  } else if constexpr (std::is_same_v<T, int>) {
    return __kmpc_rfun_sum_i;
  } else if constexpr (std::is_same_v<T, _UI>) {
    return __kmpc_rfun_sum_ui;
  } else if constexpr (std::is_same_v<T, long>) {
    return __kmpc_rfun_sum_l;
  } else if constexpr (std::is_same_v<T, _UL>) {
    return __kmpc_rfun_sum_ul;
  } else {
    static_assert(false, "Unsupported type");
  }
}

template <typename T> constexpr auto get_kmpc_rfun_max_func() {

  if constexpr (std::is_same_v<T, double>) {
    return __kmpc_rfun_max_d;
  } else if constexpr (std::is_same_v<T, float>) {
    return __kmpc_rfun_max_f;
  } else if constexpr (std::is_same_v<T, int>) {
    return __kmpc_rfun_max_i;
  } else if constexpr (std::is_same_v<T, _UI>) {
    return __kmpc_rfun_max_ui;
  } else if constexpr (std::is_same_v<T, long>) {
    return __kmpc_rfun_max_l;
  } else if constexpr (std::is_same_v<T, _UL>) {
    return __kmpc_rfun_max_ul;
  } else {
    static_assert(false, "Unsupported type");
  }
}

template <typename T> constexpr auto get_kmpc_rfun_min_func() {
  if constexpr (std::is_same_v<T, double>) {
    return __kmpc_rfun_min_d;
  } else if constexpr (std::is_same_v<T, float>) {
    return __kmpc_rfun_min_f;
  } else if constexpr (std::is_same_v<T, int>) {
    return __kmpc_rfun_min_i;
  } else if constexpr (std::is_same_v<T, _UI>) {
    return __kmpc_rfun_min_ui;
  } else if constexpr (std::is_same_v<T, long>) {
    return __kmpc_rfun_min_l;
  } else if constexpr (std::is_same_v<T, _UL>) {
    return __kmpc_rfun_min_ul;
  } else {
    static_assert(false, "Unsupported type");
  }
}

template <typename T> constexpr auto get_kmpc_rfun_sum_lds_func() {
  if constexpr (std::is_same_v<T, double>) {
    return __kmpc_rfun_sum_lds_d;
  } else if constexpr (std::is_same_v<T, float>) {
    return __kmpc_rfun_sum_lds_f;
  } else if constexpr (std::is_same_v<T, int>) {
    return __kmpc_rfun_sum_lds_i;
  } else if constexpr (std::is_same_v<T, _UI>) {
    return __kmpc_rfun_sum_lds_ui;
  } else if constexpr (std::is_same_v<T, long>) {
    return __kmpc_rfun_sum_lds_l;
  } else if constexpr (std::is_same_v<T, _UL>) {
    return __kmpc_rfun_sum_lds_ul;
  } else {
    static_assert(false, "Unsupported type");
  }
}

template <typename T> constexpr auto get_kmpc_rfun_max_lds_func() {

  if constexpr (std::is_same_v<T, double>) {
    return __kmpc_rfun_max_lds_d;
  } else if constexpr (std::is_same_v<T, float>) {
    return __kmpc_rfun_max_lds_f;
  } else if constexpr (std::is_same_v<T, int>) {
    return __kmpc_rfun_max_lds_i;
  } else if constexpr (std::is_same_v<T, _UI>) {
    return __kmpc_rfun_max_lds_ui;
  } else if constexpr (std::is_same_v<T, long>) {
    return __kmpc_rfun_max_lds_l;
  } else if constexpr (std::is_same_v<T, _UL>) {
    return __kmpc_rfun_max_lds_ul;
  } else {
    static_assert(false, "Unsupported type");
  }
}

template <typename T> constexpr auto get_kmpc_rfun_min_lds_func() {
  if constexpr (std::is_same_v<T, double>) {
    return __kmpc_rfun_min_lds_d;
  } else if constexpr (std::is_same_v<T, float>) {
    return __kmpc_rfun_min_lds_f;
  } else if constexpr (std::is_same_v<T, int>) {
    return __kmpc_rfun_min_lds_i;
  } else if constexpr (std::is_same_v<T, _UI>) {
    return __kmpc_rfun_min_lds_ui;
  } else if constexpr (std::is_same_v<T, long>) {
    return __kmpc_rfun_min_lds_l;
  } else if constexpr (std::is_same_v<T, _UL>) {
    return __kmpc_rfun_min_lds_ul;
  } else {
    static_assert(false, "Unsupported type");
  }
}

#undef _UI
#undef _UL
#undef _INLINE_ATTR_
