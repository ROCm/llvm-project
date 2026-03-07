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
/// \param result_array Output array for per-thread scan results (size >=
/// num_elements)
/// \param block_status Array of block status values
/// \param block_aggregates Array for per-block aggregates (size: NumTeams)
/// \param block_prefixes Array for per-block inclusive prefixes (size:
/// NumTeams)
/// \param _rf Function pointer to reduction function
/// \param rnv Reduction null value (identity element)
/// \param k Global thread index
///
/// Note:
/// - block=team and warp=wave.
/// - callers must pass rnv for out-of-bounds threads (k >= actual element
/// count).
/// - this always calculates the exclusive scan; inclusiveness/exclusiveness
///   is handled by the caller when writing to the output array.
///
template <typename T>
__attribute__((flatten, always_inline)) void
_xteam_scan(T val, T *result_array, uint32_t *block_status, T *block_aggregates,
            T *block_prefixes, void (*_rf)(T *, T), const T rnv,
            const uint64_t k) {

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
  // Step 1: Compute local inclusive scan within this block
  // =========================================================================

  // Intra-wave inclusive scan using shuffles
  // Callers must pass rnv for out-of-bounds threads (k >= num_elements).
  T local_scan = xteam::wave_inclusive_scan(val, _rf, block_size);

  // Cross-wave scan within block
  if (lane_num == _XTEAM_WARP_SIZE - 1)
    wave_totals[wave_num] = local_scan;
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

  // Compute final scan value
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
  T final_value = local_exclusive;

  // =========================================================================
  // Step 4: Self-reset block status for next invocation
  // =========================================================================
  // The last block to finish resets all status entries to BLOCK_INVALID (0),
  // eliminating the need for a host-side memcpy between scan invocations.
  // Requires block_status to have NumBlocks + 1 entries; the extra entry
  // at index NumBlocks serves as an atomic done-counter.

  synchronize::threadsAligned(atomic::relaxed);

  if (omp_thread_num == 0) {
    const uint32_t num_blocks = mapping::getNumberOfBlocksInKernel();
    uint32_t done = atomic::add(&block_status[num_blocks], 1u, atomic::relaxed,
                                atomic::MemScopeTy::device);
    if (done + 1 == num_blocks) {
      // Last block: reset all status entries and the counter for next use
      for (uint32_t i = 0; i <= num_blocks; i++)
        block_status[i] = BLOCK_INVALID;
    }
  }

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
extern "C" _XTEAM_EXTERN_ATTR void
__kmpc_xteams_d(double v, double *result, uint32_t *status, double *aggregates,
                double *prefixes, void (*rf)(double *, double),
                const double rnv, const uint64_t k) {
  _xteam_scan(v, result, status, aggregates, prefixes, rf, rnv, k);
}

extern "C" _XTEAM_EXTERN_ATTR void
__kmpc_xteams_f(float v, float *result, uint32_t *status, float *aggregates,
                float *prefixes, void (*rf)(float *, float), const float rnv,
                const uint64_t k) {
  _xteam_scan(v, result, status, aggregates, prefixes, rf, rnv, k);
}

extern "C" _XTEAM_EXTERN_ATTR void
__kmpc_xteams_i(int v, int *result, uint32_t *status, int *aggregates,
                int *prefixes, void (*rf)(int *, int), const int rnv,
                const uint64_t k) {
  _xteam_scan(v, result, status, aggregates, prefixes, rf, rnv, k);
}

extern "C" _XTEAM_EXTERN_ATTR void
__kmpc_xteams_ui(_UI v, _UI *result, uint32_t *status, _UI *aggregates,
                 _UI *prefixes, void (*rf)(_UI *, _UI), const _UI rnv,
                 const uint64_t k) {
  _xteam_scan(v, result, status, aggregates, prefixes, rf, rnv, k);
}

extern "C" _XTEAM_EXTERN_ATTR void
__kmpc_xteams_l(long v, long *result, uint32_t *status, long *aggregates,
                long *prefixes, void (*rf)(long *, long), const long rnv,
                const uint64_t k) {
  _xteam_scan(v, result, status, aggregates, prefixes, rf, rnv, k);
}

extern "C" _XTEAM_EXTERN_ATTR void
__kmpc_xteams_ul(_UL v, _UL *result, uint32_t *status, _UL *aggregates,
                 _UL *prefixes, void (*rf)(_UL *, _UL), const _UL rnv,
                 const uint64_t k) {
  _xteam_scan(v, result, status, aggregates, prefixes, rf, rnv, k);
}

extern "C" _XTEAM_EXTERN_ATTR void
__kmpc_xteams_cd(_CD v, _CD *result, uint32_t *status, _CD *aggregates,
                 _CD *prefixes, void (*rf)(_CD *, _CD), const _CD rnv,
                 const uint64_t k) {
  _xteam_scan(v, result, status, aggregates, prefixes, rf, rnv, k);
}

extern "C" _XTEAM_EXTERN_ATTR void
__kmpc_xteams_cf(_CF v, _CF *result, uint32_t *status, _CF *aggregates,
                 _CF *prefixes, void (*rf)(_CF *, _CF), const _CF rnv,
                 const uint64_t k) {
  _xteam_scan(v, result, status, aggregates, prefixes, rf, rnv, k);
}

#undef _CF
#undef _CD
#undef _UI
#undef _UL
