// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#ifndef AMD_SQTT_MARKER_H
#define AMD_SQTT_MARKER_H

#include <stdint.h>

#ifndef AMD_SQTT_MARKER_ENABLE
#define AMD_SQTT_MARKER_ENABLE 0
#endif

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

AMD_SQTT_MARKER_INLINE void amd_sqtt_marker_enter_string(const char *name) {
  (void)name;
}
AMD_SQTT_MARKER_INLINE void amd_sqtt_marker_exit_string(const char *name) {
  (void)name;
}
AMD_SQTT_MARKER_INLINE void amd_sqtt_marker_point_string(const char *name) {
  (void)name;
}
AMD_SQTT_MARKER_INLINE void amd_sqtt_marker_data_string(const char *name,
                                                        uint32_t data) {
  (void)name;
  (void)data;
}
AMD_SQTT_MARKER_INLINE void amd_sqtt_marker_enter_id(uint32_t id) { (void)id; }
AMD_SQTT_MARKER_INLINE void amd_sqtt_marker_exit(void) {}
AMD_SQTT_MARKER_INLINE void amd_sqtt_marker_point_id(uint32_t id) { (void)id; }

#else

#ifdef __cplusplus
extern "C" {
#endif
AMD_SQTT_MARKER_DEVICE void __sqtt_named_marker_enter(const char *);
AMD_SQTT_MARKER_DEVICE void __sqtt_named_marker_exit(const char *);
AMD_SQTT_MARKER_DEVICE void __sqtt_named_marker_point(const char *);
AMD_SQTT_MARKER_DEVICE void __sqtt_named_marker_data(const char *, uint32_t);
#ifdef __cplusplus
}
#endif

AMD_SQTT_MARKER_INLINE void amd_sqtt_marker_enter_string(const char *name) {
  __sqtt_named_marker_enter(name);
}
AMD_SQTT_MARKER_INLINE void amd_sqtt_marker_exit_string(const char *name) {
  __sqtt_named_marker_exit(name);
}
AMD_SQTT_MARKER_INLINE void amd_sqtt_marker_point_string(const char *name) {
  __sqtt_named_marker_point(name);
}
AMD_SQTT_MARKER_INLINE void amd_sqtt_marker_data_string(const char *name,
                                                        uint32_t data) {
  __sqtt_named_marker_data(name, data);
}

AMD_SQTT_MARKER_INLINE void amd_sqtt_marker_enter_id(uint32_t id) {
  __builtin_amdgcn_sched_barrier(0);
  __builtin_amdgcn_s_ttracedata((int)((id << 2) | AMD_SQTT_MARKER_FLAG_ENTER));
  __builtin_amdgcn_sched_barrier(0);
}
AMD_SQTT_MARKER_INLINE void amd_sqtt_marker_exit(void) {
  __builtin_amdgcn_sched_barrier(0);
  __builtin_amdgcn_s_ttracedata((int)AMD_SQTT_MARKER_FLAG_EXIT_PREV);
  __builtin_amdgcn_sched_barrier(0);
}
AMD_SQTT_MARKER_INLINE void amd_sqtt_marker_point_id(uint32_t id) {
  __builtin_amdgcn_sched_barrier(0);
  __builtin_amdgcn_s_ttracedata((int)(id << 2));
  __builtin_amdgcn_sched_barrier(0);
}

#endif

#undef AMD_SQTT_MARKER_INLINE
#undef AMD_SQTT_MARKER_DEVICE

#endif

#endif
