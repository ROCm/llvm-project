/*===- InstrProfilingPlatformGPU.c - GPU profiling support ----------------===*\
|*
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
|* See https://llvm.org/LICENSE.txt for license information.
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
|*
\*===----------------------------------------------------------------------===*/

// GPU-specific profiling functions for AMDGPU and NVPTX targets. This file
// provides:
//
// Platform plumbing (section boundaries, binary IDs, VNodes) are handled by
// InstrProfilingPlatformLinux.c via the COMPILER_RT_PROFILE_BAREMETAL path.

#if defined(__NVPTX__) || defined(__AMDGPU__)

#include "InstrProfiling.h"
#include <gpuintrin.h>
#include <stdint.h>

// Symbols exported to the GPU runtime need to be visible in the .dynsym table.
#define COMPILER_RT_GPU_VISIBILITY __attribute__((visibility("protected")))

// Indicates that the current wave is fully occupied.
static int is_uniform(uint64_t mask) {
  const uint64_t uniform_mask = ~0ull >> (64 - __gpu_num_lanes());
  return mask == uniform_mask;
}

// Wave-cooperative counter increment. The instrumentation pass emits calls to
// this in place of the default non-atomic load/add/store or atomicrmw sequence.
// The optional uniform counter allows calculating wave uniformity if present.
COMPILER_RT_VISIBILITY void __llvm_profile_instrument_gpu(uint64_t *counter,
                                                          uint64_t *uniform,
                                                          uint64_t step) {
  uint64_t mask = __gpu_lane_mask();
  if (__gpu_is_first_in_lane(mask)) {
    __scoped_atomic_fetch_add(counter, step * __builtin_popcountg(mask),
                              __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
    if (uniform && is_uniform(mask))
      __scoped_atomic_fetch_add(uniform, step * __builtin_popcountg(mask),
                                __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
  }
}

// Block-level sampling for offload PGO. For GPU kernels with stationary
// behavior (where all thread blocks execute the same code paths regardless of
// block ID), partial sampling significantly reduces instrumentation overhead
// without losing PGO performance gains.
//
// Returns 1 if this block should be instrumented, 0 to skip. Samples by
// matching lower bits of the linearized 3D block ID to zero.
//   sampling_bits=0: all blocks (100%)
//   sampling_bits=3: every 8th block (12.5%, default)
COMPILER_RT_VISIBILITY int __llvm_profile_sampling_gpu(uint32_t sampling_bits) {
  if (sampling_bits == 0)
    return 1;

  uint32_t gdx = __gpu_num_blocks_x();
  uint32_t gdy = __gpu_num_blocks_y();
  uint32_t block_id = __gpu_block_id_x() + __gpu_block_id_y() * gdx +
                      __gpu_block_id_z() * gdx * gdy;

  uint32_t mask = (1u << sampling_bits) - 1;
  return (block_id & mask) == 0;
}

#if defined(__AMDGPU__)
__attribute__((weak)) const int __oclc_ABI_version = 600;

#define PROF_NAME_START INSTR_PROF_SECT_START(INSTR_PROF_NAME_COMMON)
#define PROF_NAME_STOP INSTR_PROF_SECT_STOP(INSTR_PROF_NAME_COMMON)
#define PROF_CNTS_START INSTR_PROF_SECT_START(INSTR_PROF_CNTS_COMMON)
#define PROF_CNTS_STOP INSTR_PROF_SECT_STOP(INSTR_PROF_CNTS_COMMON)
#define PROF_DATA_START INSTR_PROF_SECT_START(INSTR_PROF_DATA_COMMON)
#define PROF_DATA_STOP INSTR_PROF_SECT_STOP(INSTR_PROF_DATA_COMMON)
#define PROF_UCNTS_START INSTR_PROF_SECT_START(INSTR_PROF_UCNTS_COMMON)
#define PROF_UCNTS_STOP INSTR_PROF_SECT_STOP(INSTR_PROF_UCNTS_COMMON)

extern char PROF_NAME_START[] COMPILER_RT_VISIBILITY COMPILER_RT_WEAK;
extern char PROF_NAME_STOP[] COMPILER_RT_VISIBILITY COMPILER_RT_WEAK;
extern char PROF_CNTS_START[] COMPILER_RT_VISIBILITY COMPILER_RT_WEAK;
extern char PROF_CNTS_STOP[] COMPILER_RT_VISIBILITY COMPILER_RT_WEAK;
extern __llvm_profile_data PROF_DATA_START[] COMPILER_RT_VISIBILITY
    COMPILER_RT_WEAK;
extern __llvm_profile_data PROF_DATA_STOP[] COMPILER_RT_VISIBILITY
    COMPILER_RT_WEAK;
extern char PROF_UCNTS_START[] COMPILER_RT_VISIBILITY COMPILER_RT_WEAK;
extern char PROF_UCNTS_STOP[] COMPILER_RT_VISIBILITY COMPILER_RT_WEAK;

COMPILER_RT_GPU_VISIBILITY
__llvm_profile_gpu_sections INSTR_PROF_SECT_BOUNDS_TABLE = {
    PROF_NAME_START,  PROF_NAME_STOP,  PROF_CNTS_START,
    PROF_CNTS_STOP,   PROF_DATA_START, PROF_DATA_STOP,
    PROF_UCNTS_START, PROF_UCNTS_STOP, &INSTR_PROF_RAW_VERSION_VAR};

#elif defined(__NVPTX__)

COMPILER_RT_GPU_VISIBILITY
__llvm_profile_gpu_sections INSTR_PROF_SECT_BOUNDS_TABLE = {
    NULL, NULL, NULL,
    NULL, NULL, NULL,
    NULL, NULL, &INSTR_PROF_RAW_VERSION_VAR};
#endif

#endif
