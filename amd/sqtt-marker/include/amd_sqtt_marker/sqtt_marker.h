/*===-- sqtt_marker.h - AMD SQTT device marker API --------------*- C -*-===
 *
 * Part of AMD SQTT Marker, under the MIT License. See
 * amd/sqtt-marker/LICENSE.txt for license information.
 * SPDX-License-Identifier: MIT
 *
 *===----------------------------------------------------------------------===*/

#ifndef AMD_SQTT_MARKER_H
#define AMD_SQTT_MARKER_H

#include <stdint.h>

#ifndef AMD_SQTT_MARKER_ENABLE
#define AMD_SQTT_MARKER_ENABLE 0
#endif

/*
 * Marker calls compile to no-ops unless AMD_SQTT_MARKER_ENABLE is nonzero.
 * String markers require the SQTT marker pass plugin. ID markers emit the
 * legacy two-flag-bit marker encoding directly.
 */

#define AMD_SQTT_MARKER_FLAG_EXIT_PREV UINT32_C(1)
#define AMD_SQTT_MARKER_FLAG_ENTER (UINT32_C(1) << 1)

#if defined(__AMDGCN__) || defined(__HIP_DEVICE_COMPILE__)

#if defined(__HIP__)
#define AMD_SQTT_MARKER_DEVICE __attribute__((device))
#else
#define AMD_SQTT_MARKER_DEVICE
#endif
#define AMD_SQTT_MARKER_INLINE                                                 \
  static AMD_SQTT_MARKER_DEVICE __attribute__((always_inline)) inline

#if !AMD_SQTT_MARKER_ENABLE

/** Opens the named scope. */
AMD_SQTT_MARKER_INLINE void sqtt_marker_enter(const char *name) { (void)name; }
/** Pops without validating the marker ID; name may be null or empty. */
AMD_SQTT_MARKER_INLINE void sqtt_marker_exit(const char *name) { (void)name; }
/** Emits a named point event. */
AMD_SQTT_MARKER_INLINE void sqtt_marker_point(const char *name) { (void)name; }
/** Emits a named point event followed by one 32-bit payload. */
AMD_SQTT_MARKER_INLINE void sqtt_marker_data(const char *name, uint32_t data) {
  (void)name;
  (void)data;
}
/** Opens the scope identified by id. */
AMD_SQTT_MARKER_INLINE void sqtt_marker_enter_id(uint32_t id) { (void)id; }
/** Closes the current scope. */
AMD_SQTT_MARKER_INLINE void sqtt_marker_exit_id(uint32_t id) { (void)id; }
/** Emits a point event identified by id. */
AMD_SQTT_MARKER_INLINE void sqtt_marker_point_id(uint32_t id) { (void)id; }

#else

#ifdef __cplusplus
extern "C" {
#endif
/** Opens the named scope. */
AMD_SQTT_MARKER_DEVICE void sqtt_marker_enter(const char *);
/** Pops without validating the marker ID; name may be null or empty. */
AMD_SQTT_MARKER_DEVICE void sqtt_marker_exit(const char *);
/** Emits a named point event. */
AMD_SQTT_MARKER_DEVICE void sqtt_marker_point(const char *);
/** Emits a named point event followed by one 32-bit payload. */
AMD_SQTT_MARKER_DEVICE void sqtt_marker_data(const char *, uint32_t);
#ifdef __cplusplus
}
#endif

/* M0-to-trace delay: s_nop 0 on gfx9, four cycles (s_nop 3) on gfx10+. */
#if defined(__GFX9__)
#define AMD_SQTT_MARKER_M0_NOP 0
#else
#define AMD_SQTT_MARKER_M0_NOP 3
#endif

AMD_SQTT_MARKER_INLINE void amd_sqtt_marker_emit_id(uint32_t value) {
  uint32_t scalar_value = __builtin_amdgcn_readfirstlane(value);
  uint32_t m0_value;
  __asm__ volatile("s_mov_b32 m0, %1\n"
                   "s_nop %2\n"
                   "s_ttracedata"
                   : "={m0}"(m0_value)
                   : "s"(scalar_value), "i"(AMD_SQTT_MARKER_M0_NOP));
  (void)m0_value;
}

#undef AMD_SQTT_MARKER_M0_NOP

/** Opens the scope identified by id. */
AMD_SQTT_MARKER_INLINE void sqtt_marker_enter_id(uint32_t id) {
  __builtin_amdgcn_sched_barrier(0);
  amd_sqtt_marker_emit_id((id << 2) | AMD_SQTT_MARKER_FLAG_ENTER);
  __builtin_amdgcn_sched_barrier(0);
}
/** Closes the current scope. */
AMD_SQTT_MARKER_INLINE void sqtt_marker_exit_id(uint32_t id) {
  (void)id;
  __builtin_amdgcn_sched_barrier(0);
  amd_sqtt_marker_emit_id(AMD_SQTT_MARKER_FLAG_EXIT_PREV);
  __builtin_amdgcn_sched_barrier(0);
}
/** Emits a point event identified by id. */
AMD_SQTT_MARKER_INLINE void sqtt_marker_point_id(uint32_t id) {
  __builtin_amdgcn_sched_barrier(0);
  amd_sqtt_marker_emit_id(id << 2);
  __builtin_amdgcn_sched_barrier(0);
}

#endif

#undef AMD_SQTT_MARKER_INLINE
#undef AMD_SQTT_MARKER_DEVICE

#endif

#endif
