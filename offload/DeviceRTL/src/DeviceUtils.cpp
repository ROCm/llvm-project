//===------- Utils.cpp - OpenMP device runtime utility functions -- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "DeviceUtils.h"

#include "Debug.h"
#include "Interface.h"
#include "Mapping.h"
#include "gpuintrin.h"

using namespace ompx;

extern "C" [[gnu::weak]] int IsSPMDMode;

/// Helper to keep code alive without introducing a performance penalty.
extern "C" __attribute__((weak, optnone, cold, used, retain)) void
__keep_alive() {
  __kmpc_get_hardware_thread_id_in_block();
  __kmpc_get_hardware_num_threads_in_block();
  __kmpc_get_warp_size();
  __kmpc_barrier_simple_spmd(nullptr, IsSPMDMode);
  __kmpc_barrier_simple_generic(nullptr, IsSPMDMode);
}

uint64_t utils::pack(uint32_t LowBits, uint32_t HighBits) {
  return (((uint64_t)HighBits) << 32) | (uint64_t)LowBits;
}

void utils::unpack(uint64_t Val, uint32_t &LowBits, uint32_t &HighBits) {
  static_assert(sizeof(unsigned long) == 8, "");
  LowBits = static_cast<uint32_t>(Val & 0x00000000FFFFFFFFUL);
  HighBits = static_cast<uint32_t>((Val & 0xFFFFFFFF00000000UL) >> 32);
}

int32_t utils::shuffle(uint64_t Mask, int32_t Var, int32_t SrcLane,
                       int32_t Width) {
  return __gpu_shuffle_idx_u32(Mask, SrcLane, Var, Width);
}

int32_t utils::shuffleDown(uint64_t Mask, int32_t Var, uint32_t Delta,
                           int32_t Width) {
  int32_t Self = mapping::getThreadIdInWarp();
  int32_t Index = (Delta + (Self & (Width - 1))) >= Width ? Self : Self + Delta;
  return __gpu_shuffle_idx_u64(Mask, Index, Var, Width);
}

int64_t utils::shuffleDown(uint64_t Mask, int64_t Var, uint32_t Delta,
                           int32_t Width) {
  int32_t Self = mapping::getThreadIdInWarp();
  int32_t Index = (Delta + (Self & (Width - 1))) >= Width ? Self : Self + Delta;
  return __gpu_shuffle_idx_u64(Mask, Index, Var, Width);
}

uint64_t utils::ballotSync(uint64_t Mask, int32_t Pred) {
  return __gpu_ballot(Mask, Pred);
}

bool utils::isSharedMemPtr(void *Ptr) { return __gpu_is_ptr_local(Ptr); }

extern "C" {
int32_t __kmpc_shuffle_int32(int32_t Val, int16_t Delta, int16_t SrcLane) {
  return utils::shuffleDown(lanes::All, Val, Delta, SrcLane);
}

int64_t __kmpc_shuffle_int64(int64_t Val, int16_t Delta, int16_t Width) {
  return utils::shuffleDown(lanes::All, Val, Delta, Width);
}
}
