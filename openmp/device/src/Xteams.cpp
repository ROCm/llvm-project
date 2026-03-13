//===---- Xteams.cpp - Cross team scan --------------------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements cross-team scan using the decoupled look-back algorithm.
// (single-pass algorithm)
//
// References:
// - Merrill & Garland, "Single-pass Parallel Prefix Scan with Decoupled
//   Look-back", 2016
//   https://research.nvidia.com/sites/default/files/pubs/2016-03_Single-pass-Parallel-Prefix/nvr-2016-002.pdf
//
//===----------------------------------------------------------------------===//

#include "Xteams.h"
#include "Mapping.h"
#include "Synchronization.h"

using namespace ompx;

//===----------------------------------------------------------------------===//
// Block state for decoupled look-back
//===----------------------------------------------------------------------===//

namespace {

/// Status values for block state (stored in separate block_status array)
enum BlockStatus : uint32_t {
  BLOCK_INVALID = 0, // Block hasn't started processing
  BLOCK_PARTIAL = 1, // Block has computed local aggregate, not final prefix
  BLOCK_COMPLETE = 2 // Block has computed final inclusive prefix
};

#define load_relaxed_device(status_ptr)                                        \
  atomic::load(status_ptr, atomic::relaxed, atomic::MemScopeTy::device)
#define store_relaxed_device(status_ptr, status)                               \
  atomic::store(status_ptr, status, atomic::relaxed, atomic::MemScopeTy::device)

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Decoupled look-back scan implementation
//===----------------------------------------------------------------------===//

/// Single-pass cross-team scan using decoupled look-back algorithm
///
/// This algorithm allows each block to complete its portion of the scan
/// as soon as its predecessors are ready, without waiting for all blocks.
///
/// Memory layout:
/// - block_status[NumTeams + 1]: Status of each block
/// (INVALID/PARTIAL/COMPLETE)
///     The extra entry is an atomic done-counter for self-reset.
/// - block_aggregates[NumTeams]: Written once at PARTIAL, never overwritten.
/// - block_prefixes[NumTeams]: Written once when transitioning to COMPLETE.
///   Using separate arrays eliminates the TOCTOU race that occurs when a
///   single location is overwritten during PARTIAL-to-COMPLETE transitions.
///
/// \param val Input thread local value (use rnv for out-of-bounds threads)
/// \param result_array Output array for per-thread scan results (size: Grid)
/// \param block_status Array of block status values (size: NumTeams + 1)
/// \param block_aggregates Array for per-block aggregates (size: NumTeams)
/// \param block_prefixes Array for per-block inclusive prefixes (size:
/// NumTeams)
/// \param _rf Function pointer to reduction function
/// \param rnv Reduction null value (identity element)
/// \param k Global thread index
/// \param is_inclusive True for inclusive scan, false for exclusive
///
/// Note:
/// - block=team and warp=wave.
/// - callers must pass rnv for out-of-bounds threads (k >= actual element
/// count).
///
template <typename T>
__attribute__((flatten, always_inline)) void
_xteam_scan(T val, T *result_array, uint32_t *block_status, T *block_aggregates,
            T *block_prefixes, void (*_rf)(T *, T), const T rnv,
            const uint64_t k, bool is_inclusive) {

  const uint32_t block_size = mapping::getNumberOfThreadsInBlock();
  const uint32_t num_waves =
      (block_size + _XTEAM_WARP_SIZE - 1) / _XTEAM_WARP_SIZE;

  // Derive thread/team IDs from k (logical iteration index)
  // This is consistent with how the reduction code handles it
  const uint32_t omp_thread_num = k % block_size; // Thread ID within team
  const uint32_t omp_team_num = k / block_size;   // Team ID
  const uint32_t wave_num = omp_thread_num / _XTEAM_WARP_SIZE;
  const uint32_t lane_num = omp_thread_num % _XTEAM_WARP_SIZE;

  // LDS for wave totals during block scan
  static _RF_LDS T wave_totals[_XTEAM_MAX_NUM_WAVES];
  // LDS for broadcasting prefix to all threads
  static _RF_LDS T block_prefix_lds;

  // =========================================================================
  // Step 1: Compute block-level scan (inclusive or exclusive)
  // =========================================================================

  // Intra-wave inclusive scan (always inclusive, needed for wave totals)
  // Callers must pass rnv for out-of-bounds threads (k >= num_elements).
  T local_inclusive = xteam::wave_inclusive_scan(val, _rf, block_size);

  // Derive per-thread scan value (exclusive = shift inclusive right by 1 lane)
  T local_scan;
  if (is_inclusive) {
    local_scan = local_inclusive;
  } else {
    local_scan = xteam::shfl_up(local_inclusive, 1);
    if (lane_num == 0)
      local_scan = rnv;
  }

  // Cross-wave scan within block (wave totals always use inclusive values)
  if (lane_num == _XTEAM_WARP_SIZE - 1)
    wave_totals[wave_num] = local_inclusive;
  synchronize::threadsAligned(atomic::relaxed);

  // First wave scans wave totals
  if (wave_num == 0) {
    T wt = (lane_num < num_waves) ? wave_totals[lane_num] : rnv;
    wt = xteam::wave_inclusive_scan(wt, _rf, num_waves);
    if (lane_num < num_waves)
      wave_totals[lane_num] = wt;
  }
  synchronize::threadsAligned(atomic::relaxed);

  // Add prefix from previous waves
  if (wave_num > 0)
    (*_rf)(&local_scan, wave_totals[wave_num - 1]);

  // Block aggregate is the last thread's inclusive scan value
  T block_aggregate = wave_totals[num_waves - 1];

  // =========================================================================
  // Step 2: Publish our aggregate and look back at predecessors
  // =========================================================================

  T prefix_from_predecessors = rnv;

  if (omp_thread_num == 0) {
    if (omp_team_num == 0) {
      // Block 0 has no predecessors - immediately complete
      block_prefixes[0] = block_aggregate;
      fence::kernel(atomic::release);
      store_relaxed_device(&block_status[0], BLOCK_COMPLETE);
    } else {
      // Publish our aggregate with PARTIAL status
      block_aggregates[omp_team_num] = block_aggregate;
      fence::kernel(atomic::release);
      store_relaxed_device(&block_status[omp_team_num], BLOCK_PARTIAL);

      // Look back at predecessor blocks.
      // Aggregates and prefixes are in separate arrays, so no TOCTOU race:
      // block_aggregates[b] is written once (at PARTIAL) and never changed.
      // block_prefixes[b] is written once (at COMPLETE) in a separate location.
      int pred = omp_team_num - 1;

      while (pred >= 0) {
        uint32_t pred_status;
        do {
          pred_status = load_relaxed_device(&block_status[pred]);
        } while (pred_status == BLOCK_INVALID);
        fence::kernel(atomic::acquire);

        if (pred_status == BLOCK_COMPLETE) {
          T pred_val = block_prefixes[pred];
          (*_rf)(&prefix_from_predecessors, pred_val);
          break;
        }

        // PARTIAL: accumulate aggregate and continue looking back
        T pred_val = block_aggregates[pred];
        (*_rf)(&prefix_from_predecessors, pred_val);
        pred--;
      }

      // Compute our inclusive prefix and mark complete
      T our_prefix = prefix_from_predecessors;
      (*_rf)(&our_prefix, block_aggregate);
      block_prefixes[omp_team_num] = our_prefix;
      fence::kernel(atomic::release);
      store_relaxed_device(&block_status[omp_team_num], BLOCK_COMPLETE);

      // Broadcast prefix to all threads via LDS
      block_prefix_lds = prefix_from_predecessors;
    }
  }

  // All threads wait for thread 0 to complete look-back
  synchronize::threadsAligned(atomic::relaxed);

  // =========================================================================
  // Step 3: Compute final result for each thread
  // =========================================================================

  // Get prefix from predecessors (broadcast from thread 0)
  if (omp_team_num > 0)
    prefix_from_predecessors = block_prefix_lds;

  // Compute final scan value (inclusive/exclusive already resolved in Step 1)
  T final_value = local_scan;
  if (omp_team_num > 0)
    (*_rf)(&final_value, prefix_from_predecessors);

  // =========================================================================
  // (Step 4: Self-reset block status for next invocation)
  // Would be useful if we would have multiple invocations of this function in
  // the same kernel or re-use the block status allocation for multiple kernels.
  // Since that's not the case at the moment, we'll skip it for now.
  // =========================================================================

  result_array[k] = final_value;
}

//===----------------------------------------------------------------------===//
// Extern C wrapper functions
//===----------------------------------------------------------------------===//

#define _CD double _Complex
#define _CF float _Complex
#define _UI unsigned int
#define _UL unsigned long

// Single-pass scan functions using decoupled look-back
#define _XTEAMS_DEF(T, TS)                                                     \
  extern "C" _XTEAM_EXTERN_ATTR void __kmpc_xteams_##TS(                       \
      T v, T *result, uint32_t *status, T *aggregates, T *prefixes,            \
      void (*rf)(T *, T), const T rnv, const uint64_t k, bool is_inclusive) {  \
    _xteam_scan<T>(v, result, status, aggregates, prefixes, rf, rnv, k,        \
                   is_inclusive);                                              \
  }

_XTEAMS_DEF(_CD, cd)
_XTEAMS_DEF(_CF, cf)
_XTEAMS_DEF(double, d)
_XTEAMS_DEF(float, f)
_XTEAMS_DEF(int, i)
_XTEAMS_DEF(_UI, ui)
_XTEAMS_DEF(long, l)
_XTEAMS_DEF(_UL, ul)

#undef _XTEAMS_DEF

#undef _CF
#undef _CD
#undef _UI
#undef _UL
