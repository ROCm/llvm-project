//===-------- XteamCommon.h - Shared cross-team primitives -------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains shared primitives for cross-team reductions and scans.
// These primitives provide optimized wave-level and block-level operations
// that can be used by both Xteamr.cpp (reductions) and Xteams.cpp (scans).
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_DEVICERTL_XTEAM_COMMON_H
#define OMPTARGET_DEVICERTL_XTEAM_COMMON_H

#include "DeviceTypes.h"
#include "Mapping.h"
#include "Synchronization.h"

//===----------------------------------------------------------------------===//
// Common macros and constants
//===----------------------------------------------------------------------===//

#define _XTEAM_RF_LDS volatile __gpu_local
#define _RF_LDS _XTEAM_RF_LDS // Alias for backward compatibility
#define _XTEAM_INLINE_ATTR inline __attribute__((flatten, always_inline))
#define _XTEAM_EXTERN_ATTR __attribute__((flatten, always_inline))

// Wave size - will be constant-folded since it's known at compile time
#define _XTEAM_WARP_SIZE __gpu_num_lanes()

// Maximum number of waves in a thread block (1024 / warp_size)
#define _XTEAM_MAX_NUM_WAVES 32

// Maximum threads per block (conservative, works for both wave32 and wave64)
#define _XTEAM_MAX_THREADS_PER_BLOCK (_XTEAM_MAX_NUM_WAVES * 64)

namespace xteam {

using namespace ompx;

//===----------------------------------------------------------------------===//
// Architecture-specific shuffle primitives
//===----------------------------------------------------------------------===//

/// Shuffle XOR - exchanges values between lanes using XOR of lane IDs
/// Used for butterfly reduction patterns
#ifdef __AMDGPU__
_XTEAM_INLINE_ATTR
int shfl_xor_int(int var, int lane_mask, uint32_t width) {
  int self = mapping::getThreadIdInWarp();
  int index = self ^ lane_mask;
  index = index >= ((self + width) & ~(width - 1)) ? self : index;
  return __builtin_amdgcn_ds_bpermute(index << 2, var);
}

_XTEAM_INLINE_ATTR
int shfl_up_int(int var, int offset, uint32_t width) {
  int self = mapping::getThreadIdInWarp();
  int index = self - offset;
  // Clamp to wave boundary - if index is negative, use self (identity)
  index = (index < (int)(self & ~(width - 1))) ? self : index;
  return __builtin_amdgcn_ds_bpermute(index << 2, var);
}

#elif defined(__NVPTX__)
_XTEAM_INLINE_ATTR
int shfl_xor_int(int var, int lane_mask, uint32_t width) {
  return __nvvm_shfl_sync_bfly_i32(0xFFFFFFFF, var, lane_mask, 0x1f);
}

_XTEAM_INLINE_ATTR
int shfl_up_int(int var, int offset, uint32_t width) {
  return __nvvm_shfl_sync_up_i32(0xFFFFFFFF, var, offset, 0);
}
#endif

/// Double shuffle using two int shuffles
_XTEAM_INLINE_ATTR
double shfl_xor_double(double var, int lane_mask, uint32_t width) {
  static_assert(sizeof(double) == 2 * sizeof(int), "");
  static_assert(sizeof(double) == sizeof(uint64_t), "");

  int tmp[2];
  __builtin_memcpy(tmp, &var, sizeof(tmp));
  tmp[0] = shfl_xor_int(tmp[0], lane_mask, width);
  tmp[1] = shfl_xor_int(tmp[1], lane_mask, width);

  uint64_t tmp0 =
      (static_cast<uint64_t>(tmp[1]) << 32ull) | static_cast<uint32_t>(tmp[0]);
  double result;
  __builtin_memcpy(&result, &tmp0, sizeof(result));
  return result;
}

_XTEAM_INLINE_ATTR
double shfl_up_double(double var, int offset, uint32_t width) {
  static_assert(sizeof(double) == 2 * sizeof(int), "");
  static_assert(sizeof(double) == sizeof(uint64_t), "");

  int tmp[2];
  __builtin_memcpy(tmp, &var, sizeof(tmp));
  tmp[0] = shfl_up_int(tmp[0], offset, width);
  tmp[1] = shfl_up_int(tmp[1], offset, width);

  uint64_t tmp0 =
      (static_cast<uint64_t>(tmp[1]) << 32ull) | static_cast<uint32_t>(tmp[0]);
  double result;
  __builtin_memcpy(&result, &tmp0, sizeof(result));
  return result;
}

/// Float shuffle using int shuffle with bit casting
_XTEAM_INLINE_ATTR
float shfl_xor_float(float var, int lane_mask, uint32_t width) {
  // using a union here would be undefined behavior
  int tmp;
  __builtin_memcpy(&tmp, &var, sizeof(tmp));
  tmp = shfl_xor_int(tmp, lane_mask, width);
  float result;
  __builtin_memcpy(&result, &tmp, sizeof(result));
  return result;
}

_XTEAM_INLINE_ATTR
float shfl_up_float(float var, int offset, uint32_t width) {
  // using a union here would be undefined behavior
  int tmp;
  __builtin_memcpy(&tmp, &var, sizeof(tmp));
  tmp = shfl_up_int(tmp, offset, width);
  float result;
  __builtin_memcpy(&result, &tmp, sizeof(result));
  return result;
}

/// Complex type shuffles
_XTEAM_INLINE_ATTR
double _Complex shfl_xor_cd(double _Complex var, int lane_mask,
                            uint32_t width) {
  __real__(var) = shfl_xor_double(__real__(var), lane_mask, width);
  __imag__(var) = shfl_xor_double(__imag__(var), lane_mask, width);
  return var;
}

_XTEAM_INLINE_ATTR
double _Complex shfl_up_cd(double _Complex var, int offset, uint32_t width) {
  __real__(var) = shfl_up_double(__real__(var), offset, width);
  __imag__(var) = shfl_up_double(__imag__(var), offset, width);
  return var;
}

_XTEAM_INLINE_ATTR
float _Complex shfl_xor_cf(float _Complex var, int lane_mask, uint32_t width) {
  __real__(var) = shfl_xor_float(__real__(var), lane_mask, width);
  __imag__(var) = shfl_xor_float(__imag__(var), lane_mask, width);
  return var;
}

_XTEAM_INLINE_ATTR
float _Complex shfl_up_cf(float _Complex var, int offset, uint32_t width) {
  __real__(var) = shfl_up_float(__real__(var), offset, width);
  __imag__(var) = shfl_up_float(__imag__(var), offset, width);
  return var;
}

//===----------------------------------------------------------------------===//
// Type-generic shuffle wrappers using overloading
//===----------------------------------------------------------------------===//

// XOR shuffles for reduction (butterfly pattern)
_XTEAM_INLINE_ATTR double shfl_xor(double var, int lane_mask) {
  return shfl_xor_double(var, lane_mask, _XTEAM_WARP_SIZE);
}
_XTEAM_INLINE_ATTR float shfl_xor(float var, int lane_mask) {
  return shfl_xor_float(var, lane_mask, _XTEAM_WARP_SIZE);
}
_XTEAM_INLINE_ATTR int shfl_xor(int var, int lane_mask) {
  return shfl_xor_int(var, lane_mask, _XTEAM_WARP_SIZE);
}
_XTEAM_INLINE_ATTR unsigned int shfl_xor(unsigned int var, int lane_mask) {
  return shfl_xor_int(var, lane_mask, _XTEAM_WARP_SIZE);
}
_XTEAM_INLINE_ATTR long shfl_xor(long var, int lane_mask) {
  return shfl_xor_double(var, lane_mask, _XTEAM_WARP_SIZE);
}
_XTEAM_INLINE_ATTR unsigned long shfl_xor(unsigned long var, int lane_mask) {
  return shfl_xor_double(var, lane_mask, _XTEAM_WARP_SIZE);
}
_XTEAM_INLINE_ATTR short shfl_xor(short var, int lane_mask) {
  return shfl_xor_int(var, lane_mask, _XTEAM_WARP_SIZE);
}
_XTEAM_INLINE_ATTR unsigned short shfl_xor(unsigned short var, int lane_mask) {
  return shfl_xor_int(var, lane_mask, _XTEAM_WARP_SIZE);
}
_XTEAM_INLINE_ATTR _Float16 shfl_xor(_Float16 var, int lane_mask) {
  return shfl_xor_float(var, lane_mask, _XTEAM_WARP_SIZE);
}
_XTEAM_INLINE_ATTR __bf16 shfl_xor(__bf16 var, int lane_mask) {
  return shfl_xor_float(var, lane_mask, _XTEAM_WARP_SIZE);
}
_XTEAM_INLINE_ATTR double _Complex shfl_xor(double _Complex var,
                                            int lane_mask) {
  return shfl_xor_cd(var, lane_mask, _XTEAM_WARP_SIZE);
}
_XTEAM_INLINE_ATTR float _Complex shfl_xor(float _Complex var, int lane_mask) {
  return shfl_xor_cf(var, lane_mask, _XTEAM_WARP_SIZE);
}

// UP shuffles for scan (prefix pattern)
_XTEAM_INLINE_ATTR double shfl_up(double var, int offset) {
  return shfl_up_double(var, offset, _XTEAM_WARP_SIZE);
}
_XTEAM_INLINE_ATTR float shfl_up(float var, int offset) {
  return shfl_up_float(var, offset, _XTEAM_WARP_SIZE);
}
_XTEAM_INLINE_ATTR int shfl_up(int var, int offset) {
  return shfl_up_int(var, offset, _XTEAM_WARP_SIZE);
}
_XTEAM_INLINE_ATTR unsigned int shfl_up(unsigned int var, int offset) {
  return shfl_up_int(var, offset, _XTEAM_WARP_SIZE);
}
_XTEAM_INLINE_ATTR long shfl_up(long var, int offset) {
  return shfl_up_double(var, offset, _XTEAM_WARP_SIZE);
}
_XTEAM_INLINE_ATTR unsigned long shfl_up(unsigned long var, int offset) {
  return shfl_up_double(var, offset, _XTEAM_WARP_SIZE);
}
_XTEAM_INLINE_ATTR short shfl_up(short var, int offset) {
  return shfl_up_int(var, offset, _XTEAM_WARP_SIZE);
}
_XTEAM_INLINE_ATTR unsigned short shfl_up(unsigned short var, int offset) {
  return shfl_up_int(var, offset, _XTEAM_WARP_SIZE);
}
_XTEAM_INLINE_ATTR _Float16 shfl_up(_Float16 var, int offset) {
  return shfl_up_float(var, offset, _XTEAM_WARP_SIZE);
}
_XTEAM_INLINE_ATTR __bf16 shfl_up(__bf16 var, int offset) {
  return shfl_up_float(var, offset, _XTEAM_WARP_SIZE);
}
_XTEAM_INLINE_ATTR double _Complex shfl_up(double _Complex var, int offset) {
  return shfl_up_cd(var, offset, _XTEAM_WARP_SIZE);
}
_XTEAM_INLINE_ATTR float _Complex shfl_up(float _Complex var, int offset) {
  return shfl_up_cf(var, offset, _XTEAM_WARP_SIZE);
}

//===----------------------------------------------------------------------===//
// Wave-level primitives
//===----------------------------------------------------------------------===//

/// Intra-wave reduction using butterfly pattern (shfl_xor)
/// Reduces all values in a wave to a single value in lane 0
template <typename T>
_XTEAM_INLINE_ATTR T wave_reduce(T val, void (*_rf)(T *, T),
                                 uint32_t block_size) {
  // If block is smaller than warp, start with block_size/2 to avoid
  // shuffling with inactive lanes
  const uint32_t start_offset =
      block_size < _XTEAM_WARP_SIZE ? block_size / 2 : _XTEAM_WARP_SIZE / 2;
  for (unsigned offset = start_offset; offset > 0; offset >>= 1)
    (*_rf)(&val, shfl_xor(val, offset));
  return val;
}

/// Intra-wave scan (inclusive or exclusive) using Kogge-Stone pattern (shfl_up)
/// Each lane gets the prefix sum of all lanes up to and including itself
/// (inclusive) or the prefix sum of all lanes before itself (exclusive).
/// \param val The input value for this lane
/// \param _rf The reduction function
/// \param rnv Reduction null value (used for exclusive scan)
/// \param num_elements Number of active elements
template <typename T, bool is_inclusive_scan>
_XTEAM_INLINE_ATTR T wave_scan(T val, void (*_rf)(T *, T), const T rnv,
                               uint32_t num_elements) {
  const uint32_t lane = mapping::getThreadIdInWarp();

  // Determine the scan limit
  const uint32_t limit =
      num_elements < _XTEAM_WARP_SIZE ? num_elements : _XTEAM_WARP_SIZE;

  // First do inclusive scan
  for (unsigned offset = 1; offset < limit; offset <<= 1) {
    T other = shfl_up(val, offset);
    if (lane >= offset)
      (*_rf)(&val, other);
  }
  if constexpr (is_inclusive_scan)
    return val;
  // Shift right by one lane for exclusive scan
  T result = shfl_up(val, 1);
  return (lane == 0) ? rnv : result;
}

/// Convenience aliases for wave_scan
template <typename T>
_XTEAM_INLINE_ATTR T wave_inclusive_scan(T val, void (*_rf)(T *, T),
                                         uint32_t num_elements) {
  return wave_scan<T, true>(val, _rf, T(), num_elements);
}

template <typename T>
_XTEAM_INLINE_ATTR T wave_exclusive_scan(T val, void (*_rf)(T *, T),
                                         const T rnv, uint32_t num_elements) {
  return wave_scan<T, false>(val, _rf, rnv, num_elements);
}

//===----------------------------------------------------------------------===//
// Block-level primitives
//===----------------------------------------------------------------------===//

/// Block-level reduction: wave reduce → LDS → single value
/// Returns the reduced value (valid *only* in thread 0)
template <typename T>
_XTEAM_INLINE_ATTR T block_reduce(T val, void (*_rf)(T *, T),
                                  void (*_rf_lds)(_XTEAM_RF_LDS T *,
                                                  _XTEAM_RF_LDS T *),
                                  const T rnv, _XTEAM_RF_LDS T *wave_lds) {
  const uint32_t block_size = mapping::getNumberOfThreadsInBlock();
  const uint32_t num_waves =
      (block_size + _XTEAM_WARP_SIZE - 1) / _XTEAM_WARP_SIZE;
  const uint32_t lane_num = mapping::getThreadIdInWarp();
  const uint32_t tid = mapping::getThreadIdInBlock();

  // Step 1: Intra-wave reduction using shuffles (no memory access)
  val = wave_reduce(val, _rf, block_size);

  // Step 2: Lane 0 of each wave stores result to LDS
  if (lane_num == 0) {
    const uint32_t wave_num = tid / _XTEAM_WARP_SIZE;
    wave_lds[wave_num] = val;
  }

  // Step 3: Reduce wave results in LDS
  for (unsigned offset = num_waves / 2; offset > 0; offset >>= 1) {
    synchronize::threadsAligned(atomic::acq_rel);
    if (tid < offset)
      (*_rf_lds)(&wave_lds[tid], &wave_lds[tid + offset]);
  }

  // We only need the return value in thread 0, so no need to synchronize all
  // threads here.
  return wave_lds[0];
}

/// Block-level inclusive scan: wave scan → LDS → full prefix sums
/// Each thread gets its inclusive prefix sum across the entire block
template <typename T>
_XTEAM_INLINE_ATTR T block_inclusive_scan(T val, void (*_rf)(T *, T),
                                          const T rnv,
                                          _XTEAM_RF_LDS T *wave_totals) {
  const uint32_t block_size = mapping::getNumberOfThreadsInBlock();
  const uint32_t num_waves =
      (block_size + _XTEAM_WARP_SIZE - 1) / _XTEAM_WARP_SIZE;
  const uint32_t wave_num = mapping::getThreadIdInBlock() / _XTEAM_WARP_SIZE;
  const uint32_t lane_num = mapping::getThreadIdInWarp();

  // Step 1: Intra-wave inclusive scan using shuffles (no memory access)
  val = wave_inclusive_scan(val, _rf, block_size);

  // Step 2: Last lane of each wave stores wave total to LDS
  if (lane_num == _XTEAM_WARP_SIZE - 1)
    wave_totals[wave_num] = val;
  synchronize::threadsAligned(atomic::relaxed);

  // Step 3: First wave scans the wave totals
  if (wave_num == 0 && lane_num < num_waves) {
    T wt = wave_totals[lane_num];
    // Scan wave totals using the same wave scan primitive
    for (unsigned offset = 1; offset < num_waves; offset <<= 1) {
      T other = shfl_up(wt, offset);
      if (lane_num >= offset)
        (*_rf)(&wt, other);
    }
    wave_totals[lane_num] = wt;
  }
  synchronize::threadsAligned(atomic::relaxed);

  // Step 4: Add prefix from previous waves to each thread's value
  if (wave_num > 0)
    (*_rf)(&val, wave_totals[wave_num - 1]);

  return val;
}

/// Block-level exclusive scan
/// Each thread gets the prefix sum of all threads before it (thread 0 gets rnv)
template <typename T>
_XTEAM_INLINE_ATTR T block_exclusive_scan(T val, void (*_rf)(T *, T),
                                          const T rnv,
                                          _XTEAM_RF_LDS T *wave_totals) {
  const uint32_t block_size = mapping::getNumberOfThreadsInBlock();
  const uint32_t num_waves =
      (block_size + _XTEAM_WARP_SIZE - 1) / _XTEAM_WARP_SIZE;
  const uint32_t wave_num = mapping::getThreadIdInBlock() / _XTEAM_WARP_SIZE;
  const uint32_t lane_num = mapping::getThreadIdInWarp();

  // Step 1: Intra-wave inclusive scan first
  T inclusive_val = wave_inclusive_scan(val, _rf, block_size);

  // Step 2: Last lane stores wave total
  if (lane_num == _XTEAM_WARP_SIZE - 1)
    wave_totals[wave_num] = inclusive_val;
  synchronize::threadsAligned(atomic::relaxed);

  // Step 3: Exclusive scan of wave totals
  if (wave_num == 0 && lane_num < num_waves) {
    T wt = wave_totals[lane_num];
    for (unsigned offset = 1; offset < num_waves; offset <<= 1) {
      T other = shfl_up(wt, offset);
      if (lane_num >= offset)
        (*_rf)(&wt, other);
    }
    // Shift to make exclusive
    T exclusive_wt = shfl_up(wt, 1);
    wave_totals[lane_num] = (lane_num == 0) ? rnv : exclusive_wt;
  }
  synchronize::threadsAligned(atomic::relaxed);

  // Step 4: Convert to exclusive and add prefix from previous waves
  T exclusive_val = shfl_up(inclusive_val, 1);
  exclusive_val = (lane_num == 0) ? rnv : exclusive_val;
  if (wave_num > 0)
    (*_rf)(&exclusive_val, wave_totals[wave_num]);

  return exclusive_val;
}

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

/// Returns true if num is an odd power of two (2^1, 2^3, 2^5, ...)
_XTEAM_INLINE_ATTR
bool is_odd_power(uint32_t num) {
  bool is_odd = false;
  while (num != 1) {
    num >>= 1;
    is_odd = !is_odd;
  }
  return is_odd;
}

/// Returns the smallest power of two >= num
_XTEAM_INLINE_ATTR
uint32_t ceil_to_power_of_two(uint32_t num) {
  uint32_t ceil_num = 1;
  while (ceil_num < num)
    ceil_num <<= 1;
  return ceil_num;
}

} // namespace xteam

#endif // OMPTARGET_DEVICERTL_XTEAM_COMMON_H
