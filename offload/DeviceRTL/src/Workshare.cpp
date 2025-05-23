//===----- Workshare.cpp -  OpenMP workshare implementation ------ C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of the KMPC interface
// for the loop construct plus other worksharing constructs that use the same
// interface as loops.
//
//===----------------------------------------------------------------------===//

#include "Workshare.h"
#include "Debug.h"
#include "DeviceTypes.h"
#include "DeviceUtils.h"
#include "Interface.h"
#include "Mapping.h"
#include "State.h"
#include "Synchronization.h"

using namespace ompx;

// TODO:
struct DynamicScheduleTracker {
  int64_t Chunk;
  int64_t LoopUpperBound;
  int64_t NextLowerBound;
  int64_t Stride;
  kmp_sched_t ScheduleType;
  DynamicScheduleTracker *NextDST;
};

#define ASSERT0(...)

// used by the library for the interface with the app
#define DISPATCH_FINISHED 0
#define DISPATCH_NOTFINISHED 1

// used by dynamic scheduling
#define FINISHED 0
#define NOT_FINISHED 1
#define LAST_CHUNK 2

// TODO: This variable is a hack inherited from the old runtime.
[[clang::loader_uninitialized]] static Local<uint64_t> Cnt;

template <typename T, typename ST> struct omptarget_nvptx_LoopSupport {
  ////////////////////////////////////////////////////////////////////////////////
  // Loop with static scheduling with chunk

  // Generic implementation of OMP loop scheduling with static policy
  /*! \brief Calculate initial bounds for static loop and stride
   *  @param[in] loc location in code of the call (not used here)
   *  @param[in] global_tid global thread id
   *  @param[in] schetype type of scheduling (see omptarget-nvptx.h)
   *  @param[in] plastiter pointer to last iteration
   *  @param[in,out] pointer to loop lower bound. it will contain value of
   *  lower bound of first chunk
   *  @param[in,out] pointer to loop upper bound. It will contain value of
   *  upper bound of first chunk
   *  @param[in,out] pointer to loop stride. It will contain value of stride
   *  between two successive chunks executed by the same thread
   *  @param[in] loop increment bump
   *  @param[in] chunk size
   */

  // helper function for static chunk
  static void ForStaticChunk(int &last, T &lb, T &ub, ST &stride, ST chunk,
                             T entityId, T numberOfEntities) {
    // each thread executes multiple chunks all of the same size, except
    // the last one
    // distance between two successive chunks
    stride = numberOfEntities * chunk;
    lb = lb + entityId * chunk;
    T inputUb = ub;
    ub = lb + chunk - 1; // Clang uses i <= ub
    // Say ub' is the beginning of the last chunk. Then who ever has a
    // lower bound plus a multiple of the increment equal to ub' is
    // the last one.
    T beginingLastChunk = inputUb - (inputUb % chunk);
    last = ((beginingLastChunk - lb) % stride) == 0;
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Loop with static scheduling without chunk

  // helper function for static no chunk
  static void ForStaticNoChunk(int &last, T &lb, T &ub, ST &stride, ST &chunk,
                               T entityId, T numberOfEntities) {
    // No chunk size specified.  Each thread or warp gets at most one
    // chunk; chunks are all almost of equal size
    T loopSize = ub - lb + 1;

    chunk = loopSize / numberOfEntities;
    T leftOver = loopSize - chunk * numberOfEntities;

    if (entityId < leftOver) {
      chunk++;
      lb = lb + entityId * chunk;
    } else {
      lb = lb + entityId * chunk + leftOver;
    }

    T inputUb = ub;
    ub = lb + chunk - 1; // Clang uses i <= ub
    last = lb <= inputUb && inputUb <= ub;
    stride = loopSize; // make sure we only do 1 chunk per warp
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Support for Static Init

  static void for_static_init(int32_t, int32_t schedtype, int32_t *plastiter,
                              T *plower, T *pupper, ST *pstride, ST chunk,
                              bool IsSPMDExecutionMode) {
    int32_t gtid = omp_get_thread_num();
    int numberOfActiveOMPThreads = omp_get_num_threads();

    // All warps that are in excess of the maximum requested, do
    // not execute the loop
    ASSERT0(LT_FUSSY, gtid < numberOfActiveOMPThreads,
            "current thread is not needed here; error");

    // copy
    int lastiter = 0;
    T lb = *plower;
    T ub = *pupper;
    ST stride = *pstride;

    // init
    switch (SCHEDULE_WITHOUT_MODIFIERS(schedtype)) {
    case kmp_sched_static_chunk: {
      if (chunk > 0) {
        ForStaticChunk(lastiter, lb, ub, stride, chunk, gtid,
                       numberOfActiveOMPThreads);
        break;
      }
      [[fallthrough]];
    } // note: if chunk <=0, use nochunk
    case kmp_sched_static_balanced_chunk: {
      if (chunk > 0) {
        // round up to make sure the chunk is enough to cover all iterations
        T tripCount = ub - lb + 1; // +1 because ub is inclusive
        T span = (tripCount + numberOfActiveOMPThreads - 1) /
                 numberOfActiveOMPThreads;
        // perform chunk adjustment
        chunk = (span + chunk - 1) & ~(chunk - 1);

        ASSERT0(LT_FUSSY, ub >= lb, "ub must be >= lb.");
        T oldUb = ub;
        ForStaticChunk(lastiter, lb, ub, stride, chunk, gtid,
                       numberOfActiveOMPThreads);
        if (ub > oldUb)
          ub = oldUb;
        break;
      }
      [[fallthrough]];
    } // note: if chunk <=0, use nochunk
    case kmp_sched_static_nochunk: {
      ForStaticNoChunk(lastiter, lb, ub, stride, chunk, gtid,
                       numberOfActiveOMPThreads);
      break;
    }
    case kmp_sched_distr_static_chunk: {
      if (chunk > 0) {
        ForStaticChunk(lastiter, lb, ub, stride, chunk, omp_get_team_num(),
                       omp_get_num_teams());
        break;
      }
      [[fallthrough]];
    } // note: if chunk <=0, use nochunk
    case kmp_sched_distr_static_nochunk: {
      ForStaticNoChunk(lastiter, lb, ub, stride, chunk, omp_get_team_num(),
                       omp_get_num_teams());
      break;
    }
    case kmp_sched_distr_static_chunk_sched_static_chunkone: {
      ForStaticChunk(lastiter, lb, ub, stride, chunk,
                     numberOfActiveOMPThreads * omp_get_team_num() + gtid,
                     omp_get_num_teams() * numberOfActiveOMPThreads);
      break;
    }
    default: {
      // ASSERT(LT_FUSSY, 0, "unknown schedtype %d", (int)schedtype);
      ForStaticChunk(lastiter, lb, ub, stride, chunk, gtid,
                     numberOfActiveOMPThreads);
      break;
    }
    }
    // copy back
    *plastiter = lastiter;
    *plower = lb;
    *pupper = ub;
    *pstride = stride;
  }

  /// static init function that takes into account multi-device execution
  static void for_static_init_md(int32_t global_tid, int32_t schedtype,
                                 int32_t *plastiter, T *plower_md, T *pupper_md,
                                 T *plower, T *pupper, ST *pstride, ST chunk,
                                 bool IsSPMDExecutionMode) {
    T multi_device_lb;
    multi_device_lb = *plower_md;
    T multi_device_ub;
    multi_device_ub = *pupper_md;

    for_static_init(global_tid, schedtype, plastiter, &multi_device_lb,
                    &multi_device_ub, pstride, chunk, IsSPMDExecutionMode);

    // Perform post static init adjustment of LB and UB
    *plower = multi_device_lb;
    *pupper = multi_device_ub;
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Support for dispatch Init

  static int OrderedSchedule(kmp_sched_t schedule) {
    return schedule >= kmp_sched_ordered_first &&
           schedule <= kmp_sched_ordered_last;
  }

  static void dispatch_init(IdentTy *loc, int32_t threadId,
                            kmp_sched_t schedule, T lb, T ub, ST st, ST chunk,
                            DynamicScheduleTracker *DST) {
    int tid = mapping::getThreadIdInBlock();
    T tnum = omp_get_num_threads();
    T tripCount = ub - lb + 1; // +1 because ub is inclusive
    ASSERT0(LT_FUSSY, threadId < tnum,
            "current thread is not needed here; error");

    /* Currently just ignore the monotonic and non-monotonic modifiers
     * (the compiler isn't producing them * yet anyway).
     * When it is we'll want to look at them somewhere here and use that
     * information to add to our schedule choice. We shouldn't need to pass
     * them on, they merely affect which schedule we can legally choose for
     * various dynamic cases. (In particular, whether or not a stealing scheme
     * is legal).
     */
    schedule = SCHEDULE_WITHOUT_MODIFIERS(schedule);

    // Process schedule.
    if (tnum == 1 || tripCount <= 1 || OrderedSchedule(schedule)) {
      if (OrderedSchedule(schedule))
        __kmpc_barrier(loc, threadId);
      schedule = kmp_sched_static_chunk;
      chunk = tripCount; // one thread gets the whole loop
    } else if (schedule == kmp_sched_runtime) {
      // process runtime
      omp_sched_t rtSched;
      int ChunkInt;
      omp_get_schedule(&rtSched, &ChunkInt);
      chunk = ChunkInt;
      switch (rtSched) {
      case omp_sched_static: {
        if (chunk > 0)
          schedule = kmp_sched_static_chunk;
        else
          schedule = kmp_sched_static_nochunk;
        break;
      }
      case omp_sched_auto: {
        schedule = kmp_sched_static_chunk;
        chunk = 1;
        break;
      }
      case omp_sched_dynamic:
      case omp_sched_guided: {
        schedule = kmp_sched_dynamic;
        break;
      }
      }
    } else if (schedule == kmp_sched_auto) {
      schedule = kmp_sched_static_chunk;
      chunk = 1;
    } else {
      // ASSERT(LT_FUSSY,
      //        schedule == kmp_sched_dynamic || schedule == kmp_sched_guided,
      //        "unknown schedule %d & chunk %lld\n", (int)schedule,
      //        (long long)chunk);
    }

    // init schedules
    if (schedule == kmp_sched_static_chunk) {
      ASSERT0(LT_FUSSY, chunk > 0, "bad chunk value");
      // save sched state
      DST->ScheduleType = schedule;
      // save ub
      DST->LoopUpperBound = ub;
      // compute static chunk
      ST stride;
      int lastiter = 0;
      ForStaticChunk(lastiter, lb, ub, stride, chunk, threadId, tnum);
      // save computed params
      DST->Chunk = chunk;
      DST->NextLowerBound = lb;
      DST->Stride = stride;
    } else if (schedule == kmp_sched_static_balanced_chunk) {
      ASSERT0(LT_FUSSY, chunk > 0, "bad chunk value");
      // save sched state
      DST->ScheduleType = schedule;
      // save ub
      DST->LoopUpperBound = ub;
      // compute static chunk
      ST stride;
      int lastiter = 0;
      // round up to make sure the chunk is enough to cover all iterations
      T span = (tripCount + tnum - 1) / tnum;
      // perform chunk adjustment
      chunk = (span + chunk - 1) & ~(chunk - 1);

      T oldUb = ub;
      ForStaticChunk(lastiter, lb, ub, stride, chunk, threadId, tnum);
      ASSERT0(LT_FUSSY, ub >= lb, "ub must be >= lb.");
      if (ub > oldUb)
        ub = oldUb;
      // save computed params
      DST->Chunk = chunk;
      DST->NextLowerBound = lb;
      DST->Stride = stride;
    } else if (schedule == kmp_sched_static_nochunk) {
      ASSERT0(LT_FUSSY, chunk == 0, "bad chunk value");
      // save sched state
      DST->ScheduleType = schedule;
      // save ub
      DST->LoopUpperBound = ub;
      // compute static chunk
      ST stride;
      int lastiter = 0;
      ForStaticNoChunk(lastiter, lb, ub, stride, chunk, threadId, tnum);
      // save computed params
      DST->Chunk = chunk;
      DST->NextLowerBound = lb;
      DST->Stride = stride;
    } else if (schedule == kmp_sched_dynamic || schedule == kmp_sched_guided) {
      // save data
      DST->ScheduleType = schedule;
      if (chunk < 1)
        chunk = 1;
      DST->Chunk = chunk;
      DST->LoopUpperBound = ub;
      DST->NextLowerBound = lb;
      __kmpc_barrier(loc, threadId);
      if (tid == 0) {
        Cnt = 0;
        fence::team(atomic::seq_cst);
      }
      __kmpc_barrier(loc, threadId);
    }
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Support for dispatch next

  static uint64_t NextIter() {
    __kmpc_impl_lanemask_t active = mapping::activemask();
    uint32_t leader = utils::ffs(active) - 1;
    uint32_t change = utils::popc(active);
    __kmpc_impl_lanemask_t lane_mask_lt = mapping::lanemaskLT();
    unsigned int rank = utils::popc(active & lane_mask_lt);
    uint64_t warp_res = 0;
    if (rank == 0) {
      warp_res = atomic::add(&Cnt, change, atomic::seq_cst);
    }
    warp_res = utils::shuffle(active, warp_res, leader, mapping::getWarpSize());
    return warp_res + rank;
  }

  static int DynamicNextChunk(T &lb, T &ub, T chunkSize, T loopLowerBound,
                              T loopUpperBound) {
    T N = NextIter();
    lb = loopLowerBound + N * chunkSize;
    ub = lb + chunkSize - 1; // Clang uses i <= ub

    // 3 result cases:
    //  a. lb and ub < loopUpperBound --> NOT_FINISHED
    //  b. lb < loopUpperBound and ub >= loopUpperBound: last chunk -->
    //  NOT_FINISHED
    //  c. lb and ub >= loopUpperBound: empty chunk --> FINISHED
    // a.
    if (lb <= loopUpperBound && ub < loopUpperBound) {
      return NOT_FINISHED;
    }
    // b.
    if (lb <= loopUpperBound) {
      ub = loopUpperBound;
      return LAST_CHUNK;
    }
    // c. if we are here, we are in case 'c'
    lb = loopUpperBound + 2;
    ub = loopUpperBound + 1;
    return FINISHED;
  }

  static int dispatch_next(IdentTy *loc, int32_t gtid, int32_t *plast,
                           T *plower, T *pupper, ST *pstride,
                           DynamicScheduleTracker *DST) {
    // ID of a thread in its own warp

    // automatically selects thread or warp ID based on selected implementation
    ASSERT0(LT_FUSSY, gtid < omp_get_num_threads(),
            "current thread is not needed here; error");
    // retrieve schedule
    kmp_sched_t schedule = DST->ScheduleType;

    // xxx reduce to one
    if (schedule == kmp_sched_static_chunk ||
        schedule == kmp_sched_static_nochunk) {
      T myLb = DST->NextLowerBound;
      T ub = DST->LoopUpperBound;
      // finished?
      if (myLb > ub) {
        return DISPATCH_FINISHED;
      }
      // not finished, save current bounds
      ST chunk = DST->Chunk;
      *plower = myLb;
      T myUb = myLb + chunk - 1; // Clang uses i <= ub
      if (myUb > ub)
        myUb = ub;
      *pupper = myUb;
      *plast = (int32_t)(myUb == ub);

      // increment next lower bound by the stride
      ST stride = DST->Stride;
      DST->NextLowerBound = myLb + stride;
      return DISPATCH_NOTFINISHED;
    }
    ASSERT0(LT_FUSSY,
            schedule == kmp_sched_dynamic || schedule == kmp_sched_guided,
            "bad sched");
    T myLb, myUb;
    int finished = DynamicNextChunk(myLb, myUb, DST->Chunk, DST->NextLowerBound,
                                    DST->LoopUpperBound);

    if (finished == FINISHED)
      return DISPATCH_FINISHED;

    // not finished (either not finished or last chunk)
    *plast = (int32_t)(finished == LAST_CHUNK);
    *plower = myLb;
    *pupper = myUb;
    *pstride = 1;

    return DISPATCH_NOTFINISHED;
  }

  static void dispatch_fini() {
    // nothing
  }

  ////////////////////////////////////////////////////////////////////////////////
  // end of template class that encapsulate all the helper functions
  ////////////////////////////////////////////////////////////////////////////////
};

////////////////////////////////////////////////////////////////////////////////
// KMP interface implementation (dyn loops)
////////////////////////////////////////////////////////////////////////////////

// TODO: Expand the dispatch API to take a DST pointer which can then be
//       allocated properly without malloc.
// For now, each team will contain an LDS pointer (ThreadDST) to a global array
// of references to the DST structs allocated (in global memory) for each thread
// in the team. The global memory array is allocated during the init phase if it
// was not allocated already and will be deallocated when the dispatch phase
// ends:
//
//  __kmpc_dispatch_init
//
//  ** Dispatch loop **
//
//  __kmpc_dispatch_deinit
//
[[clang::loader_uninitialized]] static Local<DynamicScheduleTracker **>
    ThreadDST;

// Create a new DST, link the current one, and define the new as current.
static DynamicScheduleTracker *pushDST() {
  int32_t ThreadIndex = mapping::getThreadIdInBlock();
  // Each block will allocate an array of pointers to DST structs. The array is
  // equal in length to the number of threads in that block.
  if (!ThreadDST) {
    // Allocate global memory array of pointers to DST structs:
    if (mapping::isMainThreadInGenericMode() || ThreadIndex == 0)
      ThreadDST = static_cast<DynamicScheduleTracker **>(
          memory::allocGlobal(mapping::getNumberOfThreadsInBlock() *
                                  sizeof(DynamicScheduleTracker *),
                              "new ThreadDST array"));
    synchronize::threads(atomic::seq_cst);

    // Initialize the array pointers:
    ThreadDST[ThreadIndex] = nullptr;
  }

  // Create a DST struct for the current thread:
  DynamicScheduleTracker *NewDST = static_cast<DynamicScheduleTracker *>(
      memory::allocGlobal(sizeof(DynamicScheduleTracker), "new DST"));
  *NewDST = DynamicScheduleTracker({0});

  // Add the new DST struct to the array of DST structs:
  NewDST->NextDST = ThreadDST[ThreadIndex];
  ThreadDST[ThreadIndex] = NewDST;
  return NewDST;
}

// Return the current DST.
static DynamicScheduleTracker *peekDST() {
  return ThreadDST[mapping::getThreadIdInBlock()];
}

// Pop the current DST and restore the last one.
static void popDST() {
  int32_t ThreadIndex = mapping::getThreadIdInBlock();
  DynamicScheduleTracker *CurrentDST = ThreadDST[ThreadIndex];
  DynamicScheduleTracker *OldDST = CurrentDST->NextDST;
  memory::freeGlobal(CurrentDST, "remove DST");
  ThreadDST[ThreadIndex] = OldDST;

  // Check if we need to deallocate the global array. Ensure all threads
  // in the block have finished deallocating the individual DSTs.
  synchronize::threads(atomic::seq_cst);
  if (!ThreadDST[ThreadIndex] && !ThreadIndex) {
    memory::freeGlobal(ThreadDST, "remove ThreadDST array");
    ThreadDST = nullptr;
  }
  synchronize::threads(atomic::seq_cst);
}

void workshare::init(bool IsSPMD) {
  if (mapping::isInitialThreadInLevel0(IsSPMD))
    ThreadDST = nullptr;
}

extern "C" {

// init
void __kmpc_dispatch_init_4(IdentTy *loc, int32_t tid, int32_t schedule,
                            int32_t lb, int32_t ub, int32_t st, int32_t chunk) {
  DynamicScheduleTracker *DST = pushDST();
  omptarget_nvptx_LoopSupport<int32_t, int32_t>::dispatch_init(
      loc, tid, (kmp_sched_t)schedule, lb, ub, st, chunk, DST);
}

void __kmpc_dispatch_init_4u(IdentTy *loc, int32_t tid, int32_t schedule,
                             uint32_t lb, uint32_t ub, int32_t st,
                             int32_t chunk) {
  DynamicScheduleTracker *DST = pushDST();
  omptarget_nvptx_LoopSupport<uint32_t, int32_t>::dispatch_init(
      loc, tid, (kmp_sched_t)schedule, lb, ub, st, chunk, DST);
}

void __kmpc_dispatch_init_8(IdentTy *loc, int32_t tid, int32_t schedule,
                            int64_t lb, int64_t ub, int64_t st, int64_t chunk) {
  DynamicScheduleTracker *DST = pushDST();
  omptarget_nvptx_LoopSupport<int64_t, int64_t>::dispatch_init(
      loc, tid, (kmp_sched_t)schedule, lb, ub, st, chunk, DST);
}

void __kmpc_dispatch_init_8u(IdentTy *loc, int32_t tid, int32_t schedule,
                             uint64_t lb, uint64_t ub, int64_t st,
                             int64_t chunk) {
  DynamicScheduleTracker *DST = pushDST();
  omptarget_nvptx_LoopSupport<uint64_t, int64_t>::dispatch_init(
      loc, tid, (kmp_sched_t)schedule, lb, ub, st, chunk, DST);
}

// next
int __kmpc_dispatch_next_4(IdentTy *loc, int32_t tid, int32_t *p_last,
                           int32_t *p_lb, int32_t *p_ub, int32_t *p_st) {
  DynamicScheduleTracker *DST = peekDST();
  return omptarget_nvptx_LoopSupport<int32_t, int32_t>::dispatch_next(
      loc, tid, p_last, p_lb, p_ub, p_st, DST);
}

int __kmpc_dispatch_next_4u(IdentTy *loc, int32_t tid, int32_t *p_last,
                            uint32_t *p_lb, uint32_t *p_ub, int32_t *p_st) {
  DynamicScheduleTracker *DST = peekDST();
  return omptarget_nvptx_LoopSupport<uint32_t, int32_t>::dispatch_next(
      loc, tid, p_last, p_lb, p_ub, p_st, DST);
}

int __kmpc_dispatch_next_8(IdentTy *loc, int32_t tid, int32_t *p_last,
                           int64_t *p_lb, int64_t *p_ub, int64_t *p_st) {
  DynamicScheduleTracker *DST = peekDST();
  return omptarget_nvptx_LoopSupport<int64_t, int64_t>::dispatch_next(
      loc, tid, p_last, p_lb, p_ub, p_st, DST);
}

int __kmpc_dispatch_next_8u(IdentTy *loc, int32_t tid, int32_t *p_last,
                            uint64_t *p_lb, uint64_t *p_ub, int64_t *p_st) {
  DynamicScheduleTracker *DST = peekDST();
  return omptarget_nvptx_LoopSupport<uint64_t, int64_t>::dispatch_next(
      loc, tid, p_last, p_lb, p_ub, p_st, DST);
}

// fini
void __kmpc_dispatch_fini_4(IdentTy *loc, int32_t tid) {
  omptarget_nvptx_LoopSupport<int32_t, int32_t>::dispatch_fini();
}

void __kmpc_dispatch_fini_4u(IdentTy *loc, int32_t tid) {
  omptarget_nvptx_LoopSupport<uint32_t, int32_t>::dispatch_fini();
}

void __kmpc_dispatch_fini_8(IdentTy *loc, int32_t tid) {
  omptarget_nvptx_LoopSupport<int64_t, int64_t>::dispatch_fini();
}

void __kmpc_dispatch_fini_8u(IdentTy *loc, int32_t tid) {
  omptarget_nvptx_LoopSupport<uint64_t, int64_t>::dispatch_fini();
}

// deinit
void __kmpc_dispatch_deinit(IdentTy *loc, int32_t tid) { popDST(); }

////////////////////////////////////////////////////////////////////////////////
// KMP interface implementation (static loops) for multi-device
////////////////////////////////////////////////////////////////////////////////

void __kmpc_distribute_static_init_multi_device_4(
    IdentTy *loc, int32_t global_tid, int32_t schedtype, int32_t *plastiter,
    int32_t *plower_md, int32_t *pupper_md, int32_t *plower, int32_t *pupper,
    int32_t *pstride, int32_t incr, int32_t chunk) {
  omptarget_nvptx_LoopSupport<int32_t, int32_t>::for_static_init_md(
      global_tid, schedtype, plastiter, plower_md, pupper_md, plower, pupper,
      pstride, chunk, mapping::isSPMDMode());
}

void __kmpc_distribute_static_init_multi_device_4u(
    IdentTy *loc, int32_t global_tid, int32_t schedtype, int32_t *plastiter,
    uint32_t *plower_md, uint32_t *pupper_md, uint32_t *plower,
    uint32_t *pupper, int32_t *pstride, int32_t incr, int32_t chunk) {
  omptarget_nvptx_LoopSupport<uint32_t, int32_t>::for_static_init_md(
      global_tid, schedtype, plastiter, plower_md, pupper_md, plower, pupper,
      pstride, chunk, mapping::isSPMDMode());
}

void __kmpc_distribute_static_init_multi_device_8(
    IdentTy *loc, int32_t global_tid, int32_t schedtype, int32_t *plastiter,
    int64_t *plower_md, int64_t *pupper_md, int64_t *plower, int64_t *pupper,
    int64_t *pstride, int64_t incr, int64_t chunk) {
  omptarget_nvptx_LoopSupport<int64_t, int64_t>::for_static_init_md(
      global_tid, schedtype, plastiter, plower_md, pupper_md, plower, pupper,
      pstride, chunk, mapping::isSPMDMode());
}

void __kmpc_distribute_static_init_multi_device_8u(
    IdentTy *loc, int32_t global_tid, int32_t schedtype, int32_t *plastiter,
    uint64_t *plower_md, uint64_t *pupper_md, uint64_t *plower,
    uint64_t *pupper, int64_t *pstride, int64_t incr, int64_t chunk) {
  omptarget_nvptx_LoopSupport<uint64_t, int64_t>::for_static_init_md(
      global_tid, schedtype, plastiter, plower_md, pupper_md, plower, pupper,
      pstride, chunk, mapping::isSPMDMode());
}

////////////////////////////////////////////////////////////////////////////////
// KMP interface implementation (static loops)
////////////////////////////////////////////////////////////////////////////////

void __kmpc_for_static_init_4(IdentTy *loc, int32_t global_tid,
                              int32_t schedtype, int32_t *plastiter,
                              int32_t *plower, int32_t *pupper,
                              int32_t *pstride, int32_t incr, int32_t chunk) {
  omptarget_nvptx_LoopSupport<int32_t, int32_t>::for_static_init(
      global_tid, schedtype, plastiter, plower, pupper, pstride, chunk,
      mapping::isSPMDMode());
}

void __kmpc_for_static_init_4u(IdentTy *loc, int32_t global_tid,
                               int32_t schedtype, int32_t *plastiter,
                               uint32_t *plower, uint32_t *pupper,
                               int32_t *pstride, int32_t incr, int32_t chunk) {
  omptarget_nvptx_LoopSupport<uint32_t, int32_t>::for_static_init(
      global_tid, schedtype, plastiter, plower, pupper, pstride, chunk,
      mapping::isSPMDMode());
}

void __kmpc_for_static_init_8(IdentTy *loc, int32_t global_tid,
                              int32_t schedtype, int32_t *plastiter,
                              int64_t *plower, int64_t *pupper,
                              int64_t *pstride, int64_t incr, int64_t chunk) {
  omptarget_nvptx_LoopSupport<int64_t, int64_t>::for_static_init(
      global_tid, schedtype, plastiter, plower, pupper, pstride, chunk,
      mapping::isSPMDMode());
}

void __kmpc_for_static_init_8u(IdentTy *loc, int32_t global_tid,
                               int32_t schedtype, int32_t *plastiter,
                               uint64_t *plower, uint64_t *pupper,
                               int64_t *pstride, int64_t incr, int64_t chunk) {
  omptarget_nvptx_LoopSupport<uint64_t, int64_t>::for_static_init(
      global_tid, schedtype, plastiter, plower, pupper, pstride, chunk,
      mapping::isSPMDMode());
}

void __kmpc_distribute_static_init_4(IdentTy *loc, int32_t global_tid,
                                     int32_t schedtype, int32_t *plastiter,
                                     int32_t *plower, int32_t *pupper,
                                     int32_t *pstride, int32_t incr,
                                     int32_t chunk) {
  omptarget_nvptx_LoopSupport<int32_t, int32_t>::for_static_init(
      global_tid, schedtype, plastiter, plower, pupper, pstride, chunk,
      mapping::isSPMDMode());
}

void __kmpc_distribute_static_init_4u(IdentTy *loc, int32_t global_tid,
                                      int32_t schedtype, int32_t *plastiter,
                                      uint32_t *plower, uint32_t *pupper,
                                      int32_t *pstride, int32_t incr,
                                      int32_t chunk) {
  omptarget_nvptx_LoopSupport<uint32_t, int32_t>::for_static_init(
      global_tid, schedtype, plastiter, plower, pupper, pstride, chunk,
      mapping::isSPMDMode());
}

void __kmpc_distribute_static_init_8(IdentTy *loc, int32_t global_tid,
                                     int32_t schedtype, int32_t *plastiter,
                                     int64_t *plower, int64_t *pupper,
                                     int64_t *pstride, int64_t incr,
                                     int64_t chunk) {
  omptarget_nvptx_LoopSupport<int64_t, int64_t>::for_static_init(
      global_tid, schedtype, plastiter, plower, pupper, pstride, chunk,
      mapping::isSPMDMode());
}

void __kmpc_distribute_static_init_8u(IdentTy *loc, int32_t global_tid,
                                      int32_t schedtype, int32_t *plastiter,
                                      uint64_t *plower, uint64_t *pupper,
                                      int64_t *pstride, int64_t incr,
                                      int64_t chunk) {
  omptarget_nvptx_LoopSupport<uint64_t, int64_t>::for_static_init(
      global_tid, schedtype, plastiter, plower, pupper, pstride, chunk,
      mapping::isSPMDMode());
}

void __kmpc_for_static_fini(IdentTy *loc, int32_t global_tid) {}

void __kmpc_distribute_static_fini(IdentTy *loc, int32_t global_tid) {}
}

namespace ompx {

/// Helper class to hide the generic loop nest and provide the template argument
/// throughout.
template <typename Ty> class StaticLoopChunker {

  /// Generic loop nest that handles block and/or thread distribution in the
  /// absence of user specified chunk sizes. This implicitly picks a block chunk
  /// size equal to the number of threads in the block and a thread chunk size
  /// equal to one. In contrast to the chunked version we can get away with a
  /// single loop in this case
  static void NormalizedLoopNestNoChunk(void (*LoopBody)(Ty, void *), void *Arg,
                                        Ty NumBlocks, Ty BId, Ty NumThreads,
                                        Ty TId, Ty NumIters,
                                        bool OneIterationPerThread) {
    Ty KernelIteration = NumBlocks * NumThreads;

    // Start index in the normalized space.
    Ty IV = BId * NumThreads + TId;
    ASSERT(IV >= 0, "Bad index");

    // Cover the entire iteration space, assumptions in the caller might allow
    // to simplify this loop to a conditional.
    if (IV < NumIters) {
      do {

        // Execute the loop body.
        LoopBody(IV, Arg);

        // Every thread executed one block and thread chunk now.
        IV += KernelIteration;

        if (OneIterationPerThread)
          return;

      } while (IV < NumIters);
    }
  }

  /// Generic loop nest that handles block and/or thread distribution in the
  /// presence of user specified chunk sizes (for at least one of them).
  static void NormalizedLoopNestChunked(void (*LoopBody)(Ty, void *), void *Arg,
                                        Ty BlockChunk, Ty NumBlocks, Ty BId,
                                        Ty ThreadChunk, Ty NumThreads, Ty TId,
                                        Ty NumIters,
                                        bool OneIterationPerThread) {
    Ty KernelIteration = NumBlocks * BlockChunk;

    // Start index in the chunked space.
    Ty IV = BId * BlockChunk + TId;
    ASSERT(IV >= 0, "Bad index");

    // Cover the entire iteration space, assumptions in the caller might allow
    // to simplify this loop to a conditional.
    do {

      Ty BlockChunkLeft =
          BlockChunk >= TId * ThreadChunk ? BlockChunk - TId * ThreadChunk : 0;
      Ty ThreadChunkLeft =
          ThreadChunk <= BlockChunkLeft ? ThreadChunk : BlockChunkLeft;

      while (ThreadChunkLeft--) {

        // Given the blocking it's hard to keep track of what to execute.
        if (IV >= NumIters)
          return;

        // Execute the loop body.
        LoopBody(IV, Arg);

        if (OneIterationPerThread)
          return;

        ++IV;
      }

      IV += KernelIteration;

    } while (IV < NumIters);
  }

public:
  /// Worksharing `for`-loop.
  static void For(IdentTy *Loc, void (*LoopBody)(Ty, void *), void *Arg,
                  Ty NumIters, Ty NumThreads, Ty ThreadChunk) {
    ASSERT(NumIters >= 0, "Bad iteration count");
    ASSERT(ThreadChunk >= 0, "Bad thread count");

    // All threads need to participate but we don't know if we are in a
    // parallel at all or if the user might have used a `num_threads` clause
    // on the parallel and reduced the number compared to the block size.
    // Since nested parallels are possible too we need to get the thread id
    // from the `omp` getter and not the mapping directly.
    Ty TId = omp_get_thread_num();

    // There are no blocks involved here.
    Ty BlockChunk = 0;
    Ty NumBlocks = 1;
    Ty BId = 0;

    // If the thread chunk is not specified we pick a default now.
    if (ThreadChunk == 0)
      ThreadChunk = 1;

    // If we know we have more threads than iterations we can indicate that to
    // avoid an outer loop.
    bool OneIterationPerThread = false;
    if (config::getAssumeThreadsOversubscription()) {
      ASSERT(NumThreads >= NumIters, "Broken assumption");
      OneIterationPerThread = true;
    }

    if (ThreadChunk != 1)
      NormalizedLoopNestChunked(LoopBody, Arg, BlockChunk, NumBlocks, BId,
                                ThreadChunk, NumThreads, TId, NumIters,
                                OneIterationPerThread);
    else
      NormalizedLoopNestNoChunk(LoopBody, Arg, NumBlocks, BId, NumThreads, TId,
                                NumIters, OneIterationPerThread);
  }

  /// Worksharing `distribute`-loop.
  static void Distribute(IdentTy *Loc, void (*LoopBody)(Ty, void *), void *Arg,
                         Ty NumIters, Ty BlockChunk) {
    ASSERT(icv::Level == 0, "Bad distribute");
    ASSERT(icv::ActiveLevel == 0, "Bad distribute");
    ASSERT(state::ParallelRegionFn == nullptr, "Bad distribute");
    ASSERT(state::ParallelTeamSize == 1, "Bad distribute");

    ASSERT(NumIters >= 0, "Bad iteration count");
    ASSERT(BlockChunk >= 0, "Bad block count");

    // There are no threads involved here.
    Ty ThreadChunk = 0;
    Ty NumThreads = 1;
    Ty TId = 0;

    // All teams need to participate.
    Ty NumBlocks = mapping::getNumberOfBlocksInKernel();
    Ty BId = mapping::getBlockIdInKernel();

    // If the block chunk is not specified we pick a default now.
    if (BlockChunk == 0)
      BlockChunk = NumThreads;

    // If we know we have more blocks than iterations we can indicate that to
    // avoid an outer loop.
    bool OneIterationPerThread = false;
    if (config::getAssumeTeamsOversubscription()) {
      ASSERT(NumBlocks >= NumIters, "Broken assumption");
      OneIterationPerThread = true;
    }

    if (BlockChunk != NumThreads)
      NormalizedLoopNestChunked(LoopBody, Arg, BlockChunk, NumBlocks, BId,
                                ThreadChunk, NumThreads, TId, NumIters,
                                OneIterationPerThread);
    else
      NormalizedLoopNestNoChunk(LoopBody, Arg, NumBlocks, BId, NumThreads, TId,
                                NumIters, OneIterationPerThread);

    ASSERT(icv::Level == 0, "Bad distribute");
    ASSERT(icv::ActiveLevel == 0, "Bad distribute");
    ASSERT(state::ParallelRegionFn == nullptr, "Bad distribute");
    ASSERT(state::ParallelTeamSize == 1, "Bad distribute");
  }

  /// Worksharing `distribute parallel for`-loop.
  static void DistributeFor(IdentTy *Loc, void (*LoopBody)(Ty, void *),
                            void *Arg, Ty NumIters, Ty NumThreads,
                            Ty BlockChunk, Ty ThreadChunk) {
    ASSERT(icv::Level == 1, "Bad distribute");
    ASSERT(icv::ActiveLevel == 1, "Bad distribute");
    ASSERT(state::ParallelRegionFn == nullptr, "Bad distribute");

    ASSERT(NumIters >= 0, "Bad iteration count");
    ASSERT(BlockChunk >= 0, "Bad block count");
    ASSERT(ThreadChunk >= 0, "Bad thread count");

    // All threads need to participate but the user might have used a
    // `num_threads` clause on the parallel and reduced the number compared to
    // the block size.
    Ty TId = mapping::getThreadIdInBlock();

    // All teams need to participate.
    Ty NumBlocks = mapping::getNumberOfBlocksInKernel();
    Ty BId = mapping::getBlockIdInKernel();

    // If the block chunk is not specified we pick a default now.
    if (BlockChunk == 0)
      BlockChunk = NumThreads;

    // If the thread chunk is not specified we pick a default now.
    if (ThreadChunk == 0)
      ThreadChunk = 1;

    // If we know we have more threads (across all blocks) than iterations we
    // can indicate that to avoid an outer loop.
    bool OneIterationPerThread = false;
    if (config::getAssumeTeamsOversubscription() &
        config::getAssumeThreadsOversubscription()) {
      OneIterationPerThread = true;
      ASSERT(NumBlocks * NumThreads >= NumIters, "Broken assumption");
    }

    if (BlockChunk != NumThreads || ThreadChunk != 1)
      NormalizedLoopNestChunked(LoopBody, Arg, BlockChunk, NumBlocks, BId,
                                ThreadChunk, NumThreads, TId, NumIters,
                                OneIterationPerThread);
    else
      NormalizedLoopNestNoChunk(LoopBody, Arg, NumBlocks, BId, NumThreads, TId,
                                NumIters, OneIterationPerThread);

    ASSERT(icv::Level == 1, "Bad distribute");
    ASSERT(icv::ActiveLevel == 1, "Bad distribute");
    ASSERT(state::ParallelRegionFn == nullptr, "Bad distribute");
  }
};

} // namespace ompx

#define OMP_LOOP_ENTRY(BW, TY)                                                 \
  [[gnu::flatten, clang::always_inline]] void                                  \
      __kmpc_distribute_for_static_loop##BW(                                   \
          IdentTy *loc, void (*fn)(TY, void *), void *arg, TY num_iters,       \
          TY num_threads, TY block_chunk, TY thread_chunk) {                   \
    ompx::StaticLoopChunker<TY>::DistributeFor(                                \
        loc, fn, arg, num_iters, num_threads, block_chunk, thread_chunk);      \
  }                                                                            \
  [[gnu::flatten, clang::always_inline]] void                                  \
      __kmpc_distribute_static_loop##BW(IdentTy *loc, void (*fn)(TY, void *),  \
                                        void *arg, TY num_iters,               \
                                        TY block_chunk) {                      \
    ompx::StaticLoopChunker<TY>::Distribute(loc, fn, arg, num_iters,           \
                                            block_chunk);                      \
  }                                                                            \
  [[gnu::flatten, clang::always_inline]] void __kmpc_for_static_loop##BW(      \
      IdentTy *loc, void (*fn)(TY, void *), void *arg, TY num_iters,           \
      TY num_threads, TY thread_chunk) {                                       \
    ompx::StaticLoopChunker<TY>::For(loc, fn, arg, num_iters, num_threads,     \
                                     thread_chunk);                            \
  }

extern "C" {
OMP_LOOP_ENTRY(_4, int32_t)
OMP_LOOP_ENTRY(_4u, uint32_t)
OMP_LOOP_ENTRY(_8, int64_t)
OMP_LOOP_ENTRY(_8u, uint64_t)
}
