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
// - rocPRIM / CUB implementations
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

/// The status array is separate from the value array to simplify atomics.
/// The status is updated AFTER the value is written, with appropriate fences.

/// Atomically load block status with relaxed ordering (device scope).
/// Ordering is provided by the standalone fence::kernel(acquire) calls that
/// follow status reads -- those invalidate the per-CU L1 cache so subsequent
/// non-atomic reads (e.g. block_values[]) see data flushed to L2 by the
/// writer's release fence.
#define load_block_status(status_ptr)                                          \
  atomic::load(status_ptr, atomic::relaxed, atomic::MemScopeTy::device)

/// Atomically store block status with relaxed ordering (device scope).
/// Ordering is provided by the standalone fence::kernel(release) calls that
/// precede status writes -- those flush the per-CU L1 dirty lines to L2 so
/// other CUs can see prior non-atomic writes (e.g. block_values[] = ...).
#define store_block_status(status_ptr, status)                                 \
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
/// - block_status[NumTeams + 1]: Status of each block (INVALID/PARTIAL/COMPLETE)
///     The extra entry is an atomic done-counter for self-reset.
/// - block_values[NumTeams]: Holds the aggregate while PARTIAL, overwritten
///     with the inclusive prefix when transitioning to COMPLETE.
///
/// \param val Input thread local value (use rnv for out-of-bounds threads)
/// \param result_array Output array for final scan results
/// \param block_status Array of block status values
/// \param block_values Shared array for aggregates (PARTIAL) and prefixes (COMPLETE)
/// \param _rf Function pointer to reduction function
/// \param rnv Reduction null value (identity element)
/// \param k Global thread index
/// \param num_elements Total number of elements in the scan (N)
/// \param is_inclusive True for inclusive scan, false for exclusive
///
/// Note that block=team and warp=wave.
/// Threads with k >= num_elements use rnv as their input value and do not
/// write to result_array, but still participate in the look-back protocol.
///
template <typename T>
__attribute__((flatten, always_inline)) void
_xteam_scan(T val, T *result_array, uint32_t *block_status, T *block_values,
            void (*_rf)(T *, T), const T rnv,
            const uint64_t k, const uint64_t num_elements,
            bool is_inclusive) {

  const uint32_t block_size = mapping::getNumberOfThreadsInBlock();
  const uint32_t warp_size = _XTEAM_WARP_SIZE;
  const uint32_t num_waves = (block_size + warp_size - 1) / warp_size;

  // Derive thread/team IDs from k (logical iteration index)
  // This is consistent with how the reduction code handles it
  const uint32_t omp_thread_num = k % block_size; // Thread ID within team
  const uint32_t omp_team_num = k / block_size;   // Team ID
  const uint32_t wave_num = omp_thread_num / warp_size;
  const uint32_t lane_num = omp_thread_num % warp_size;

  // LDS for wave totals during block scan
  static _RF_LDS T wave_totals[_XTEAM_MAX_NUM_WAVES];
  // LDS for broadcasting prefix to all threads
  static _RF_LDS T block_prefix_lds;

  // =========================================================================
  // Step 1: Compute local inclusive scan within this block
  // =========================================================================

  // Out-of-bounds threads use identity element so they don't affect the scan
  const T scan_input = (k < num_elements) ? val : rnv;

  // Intra-wave inclusive scan using shuffles
  T local_scan = xteam::wave_inclusive_scan(scan_input, _rf);

  // Cross-wave scan within block
  if (lane_num == warp_size - 1)
    wave_totals[wave_num] = local_scan;
  synchronize::threadsAligned(atomic::acq_rel);

  // First wave scans wave totals
  if (wave_num == 0) {
    T wt = (lane_num < num_waves) ? wave_totals[lane_num] : rnv;
    wt = xteam::wave_inclusive_scan(wt, _rf, num_waves);
    if (lane_num < num_waves)
      wave_totals[lane_num] = wt;
  }
  synchronize::threadsAligned(atomic::acq_rel);

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
      block_values[0] = block_aggregate;
      fence::kernel(atomic::release);
      store_block_status(&block_status[0], BLOCK_COMPLETE);
    } else {
      // Publish our aggregate with PARTIAL status
      block_values[omp_team_num] = block_aggregate;
      fence::kernel(atomic::release);
      store_block_status(&block_status[omp_team_num], BLOCK_PARTIAL);

      // Look back at predecessor blocks.
      // Because block_values[] is shared for both aggregates (PARTIAL) and
      // inclusive prefixes (COMPLETE), a predecessor can overwrite its
      // aggregate with its prefix between our status read and value read.
      // We re-check the status after reading the value to detect this.
      int pred = omp_team_num - 1;

      while (pred >= 0) {
        uint32_t pred_status;
        do {
          pred_status = load_block_status(&block_status[pred]);
        } while (pred_status == BLOCK_INVALID);

        fence::kernel(atomic::acquire);

        if (pred_status == BLOCK_COMPLETE) {
          T pred_val = block_values[pred];
          (*_rf)(&prefix_from_predecessors, pred_val);
          break;
        }

        // PARTIAL: read aggregate, then verify status hasn't changed
        T pred_val = block_values[pred];
        fence::kernel(atomic::acquire);
        pred_status = load_block_status(&block_status[pred]);
        if (pred_status == BLOCK_COMPLETE) {
          // Block transitioned; re-read to get the inclusive prefix
          pred_val = block_values[pred];
          (*_rf)(&prefix_from_predecessors, pred_val);
          break;
        }

        (*_rf)(&prefix_from_predecessors, pred_val);
        pred--;
      }

      // Compute our inclusive prefix and mark complete
      T our_prefix = prefix_from_predecessors;
      (*_rf)(&our_prefix, block_aggregate);
      block_values[omp_team_num] = our_prefix;
      fence::kernel(atomic::release);
      store_block_status(&block_status[omp_team_num], BLOCK_COMPLETE);

      // Broadcast prefix to all threads via LDS
      block_prefix_lds = prefix_from_predecessors;
    }
  }

  // All threads wait for thread 0 to complete look-back
  synchronize::threadsAligned(atomic::acq_rel);

  // =========================================================================
  // Step 3: Compute final result for each thread
  // =========================================================================

  // Get prefix from predecessors (broadcast from thread 0)
  if (omp_team_num > 0)
    prefix_from_predecessors = block_prefix_lds;

  // Compute final scan value
  T final_value;
  if (is_inclusive) {
    // Inclusive: result = local_scan + prefix_from_predecessors
    final_value = local_scan;
    if (omp_team_num > 0)
      (*_rf)(&final_value, prefix_from_predecessors);
  } else {
    // Exclusive: result = prefix_from_predecessors + local_exclusive_scan
    // local_exclusive_scan = shift local_scan right by 1
    T local_exclusive = xteam::shfl_up(local_scan, 1);
    if (lane_num == 0) {
      // First lane of each wave gets from previous wave or prefix
      if (wave_num == 0)
        local_exclusive = prefix_from_predecessors;
      else {
        local_exclusive = wave_totals[wave_num - 1];
        if (omp_team_num > 0)
          (*_rf)(&local_exclusive, prefix_from_predecessors);
      }
    } else if (omp_team_num > 0) {
      (*_rf)(&local_exclusive, prefix_from_predecessors);
    }
    final_value = local_exclusive;
  }

  // Store final result (only for valid threads)
  if (k < num_elements)
    result_array[k] = final_value;

  // =========================================================================
  // Step 4: Self-reset block status for next invocation
  // =========================================================================
  // The last block to finish resets all status entries to BLOCK_INVALID (0),
  // eliminating the need for a host-side memcpy between scan invocations.
  // Requires block_status to have NumBlocks + 1 entries; the extra entry
  // at index NumBlocks serves as an atomic done-counter.

  synchronize::threadsAligned(atomic::acq_rel);

  if (omp_thread_num == 0) {
    const uint32_t num_blocks = mapping::getNumberOfBlocksInKernel();
    uint32_t done = atomic::add(&block_status[num_blocks], 1u,
                                atomic::relaxed,
                                atomic::MemScopeTy::device);
    if (done + 1 == num_blocks) {
      // Last block: reset all status entries and the counter for next use
      for (uint32_t i = 0; i <= num_blocks; i++)
        block_status[i] = 0;
    }
  }
}

//===----------------------------------------------------------------------===//
// Extern C wrapper functions
//===----------------------------------------------------------------------===//

#define _EXT_ATTR extern "C" _XTEAM_EXTERN_ATTR void
#define _CD double _Complex
#define _CF float _Complex
#define _UI unsigned int
#define _UL unsigned long

// Single-pass scan functions using decoupled look-back
_EXT_ATTR
__kmpc_xteams_d(double v, double *result, uint32_t *status, double *values,
                void (*rf)(double *, double), const double rnv,
                const uint64_t k, const uint64_t n, bool is_inclusive) {
  _xteam_scan(v, result, status, values, rf, rnv, k, n, is_inclusive);
}

_EXT_ATTR
__kmpc_xteams_f(float v, float *result, uint32_t *status, float *values,
                void (*rf)(float *, float), const float rnv,
                const uint64_t k, const uint64_t n, bool is_inclusive) {
  _xteam_scan(v, result, status, values, rf, rnv, k, n, is_inclusive);
}

_EXT_ATTR
__kmpc_xteams_i(int v, int *result, uint32_t *status, int *values,
                void (*rf)(int *, int), const int rnv, const uint64_t k,
                const uint64_t n, bool is_inclusive) {
  _xteam_scan(v, result, status, values, rf, rnv, k, n, is_inclusive);
}

_EXT_ATTR
__kmpc_xteams_ui(_UI v, _UI *result, uint32_t *status, _UI *values,
                 void (*rf)(_UI *, _UI), const _UI rnv, const uint64_t k,
                 const uint64_t n, bool is_inclusive) {
  _xteam_scan(v, result, status, values, rf, rnv, k, n, is_inclusive);
}

_EXT_ATTR
__kmpc_xteams_l(long v, long *result, uint32_t *status, long *values,
                void (*rf)(long *, long), const long rnv, const uint64_t k,
                const uint64_t n, bool is_inclusive) {
  _xteam_scan(v, result, status, values, rf, rnv, k, n, is_inclusive);
}

_EXT_ATTR
__kmpc_xteams_ul(_UL v, _UL *result, uint32_t *status, _UL *values,
                 void (*rf)(_UL *, _UL), const _UL rnv, const uint64_t k,
                 const uint64_t n, bool is_inclusive) {
  _xteam_scan(v, result, status, values, rf, rnv, k, n, is_inclusive);
}

_EXT_ATTR
__kmpc_xteams_cd(_CD v, _CD *result, uint32_t *status, _CD *values,
                 void (*rf)(_CD *, _CD), const _CD rnv, const uint64_t k,
                 const uint64_t n, bool is_inclusive) {
  _xteam_scan(v, result, status, values, rf, rnv, k, n, is_inclusive);
}

_EXT_ATTR
__kmpc_xteams_cf(_CF v, _CF *result, uint32_t *status, _CF *values,
                 void (*rf)(_CF *, _CF), const _CF rnv, const uint64_t k,
                 const uint64_t n, bool is_inclusive) {
  _xteam_scan(v, result, status, values, rf, rnv, k, n, is_inclusive);
}

#undef _CF
#undef _CD
#undef _UI
#undef _UL
#undef _EXT_ATTR
