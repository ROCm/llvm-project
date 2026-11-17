//===-- OpenMP/AMDGPUSupport.h - AMDGPU-specific device support -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Downstream: liboffload helpers for AMDGPU-specific features that the
// liboffload API doesn't expose. Gate their use with the BackendConfig flags.
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_OPENMP_AMDGPUSUPPORT_H
#define OMPTARGET_OPENMP_AMDGPUSUPPORT_H

#include "OffloadAPI.h"

#include <cstdint>

struct __tgt_device_image;

namespace llvm::offload::tmp {

/// Report why \p Image is incompatible with \p Platform's devices (XNACK).
void __ol_tgt_CheckInvalidImage(ol_platform_handle_t Platform,
                                __tgt_device_image *Image);

/// Whether \p Device is an APU.
bool __ol_tgt_IsAPU(ol_device_handle_t Device);

/// Diagnose the zero-copy configuration of \p Device.
void __ol_tgt_ZeroCopySanityChecksAndDiag(ol_device_handle_t Device,
                                          bool IsUnifiedSharedMemory,
                                          bool IsAutoZeroCopy,
                                          bool IsEagerMaps);

/// Whether \p Device is a gfx90a that maps memory as coarse grain under USM.
bool __ol_tgt_IsGfx90aCoarseGrainUsmMapEnabled(ol_device_handle_t Device);

/// Switch the host memory region [\p Ptr, \p Ptr + \p Size) to coarse grain.
void __ol_tgt_SetCoarseGrainMemRegion(ol_device_handle_t Device, void *Ptr,
                                      int64_t Size);

/// Record [\p Ptr, \p Ptr + \p Size) as coarse grain, and switch it if
/// \p SetAttr.
void __ol_tgt_SetCoarseGrainMem(ol_device_handle_t Device, const void *Ptr,
                                int64_t Size, bool SetAttr);

/// Whether [\p Ptr, \p Ptr + \p Size) is coarse grain memory.
int32_t __ol_tgt_QueryCoarseGrainMemRegion(ol_device_handle_t Device,
                                           const void *Ptr, int64_t Size);

/// Prefault [\p Ptr, \p Ptr + \p Size) in the GPU page table of \p Device.
void __ol_tgt_PrepopulatePageTable(ol_device_handle_t Device, void *Ptr,
                                   int64_t Size);

} // namespace llvm::offload::tmp

#endif // OMPTARGET_OPENMP_AMDGPUSUPPORT_H
