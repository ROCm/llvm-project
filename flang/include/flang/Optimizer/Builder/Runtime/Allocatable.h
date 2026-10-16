//===-- Allocatable.h - generate Allocatable runtime API calls---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_BUILDER_RUNTIME_ALLOCATABLE_H
#define FORTRAN_OPTIMIZER_BUILDER_RUNTIME_ALLOCATABLE_H

#include "mlir/IR/Value.h"

namespace mlir {
class Location;
} // namespace mlir

namespace fir {
class FirOpBuilder;
}

namespace fir::runtime {

/// Generate runtime call to assign \p sourceBox to \p destBox.
/// \p destBox must be a fir.ref<fir.box<T>> and \p sourceBox a fir.box<T>.
/// \p destBox Fortran descriptor may be modified if destBox is an allocatable
/// according to Fortran allocatable assignment rules, otherwise it is not
/// modified.
mlir::Value genMoveAlloc(fir::FirOpBuilder &builder, mlir::Location loc,
                         mlir::Value to, mlir::Value from, mlir::Value hasStat,
                         mlir::Value errMsg);

/// Generate runtime call to apply bounds, cobounds, length type
/// parameters and derived type information from \p mold descriptor
/// to \p desc descriptor. The resulting rank of \p desc descriptor
/// is set to \p rank. The resulting descriptor must be initialized
/// and deallocated before the call.
void genAllocatableApplyMold(fir::FirOpBuilder &builder, mlir::Location loc,
                             mlir::Value desc, mlir::Value mold, int rank);

/// Generate runtime call to set the bounds (\p lowerBound and \p upperBound)
/// for the specified dimension \p dimIndex (zero-based) in the given
/// \p desc descriptor.
void genAllocatableSetBounds(fir::FirOpBuilder &builder, mlir::Location loc,
                             mlir::Value desc, mlir::Value dimIndex,
                             mlir::Value lowerBound, mlir::Value upperBound);

/// Generate runtime call to allocate an allocatable entity
/// as described by the given \p desc descriptor.
void genAllocatableAllocate(fir::FirOpBuilder &builder, mlir::Location loc,
                            mlir::Value desc, mlir::Value hasStat = {},
                            mlir::Value errMsg = {});

/// Generate a runtime call that stamps \p desc so the immediately following
/// AllocatableAllocate will dispatch through the OpenMP runtime using the
/// OpenMP allocator identified by \p handle.  When \p align is non-null and
/// non-zero the allocation goes through __kmpc_aligned_alloc with that
/// alignment; otherwise __kmpc_alloc is used.  A null \p align is treated as
/// 0 (default alignment).  The matching AllocatableDeallocate dispatches
/// through __kmpc_free with the same handle.
///
/// \p desc must be a fir::ref<fir::box<none>>; \p handle and \p align are
/// scalar integers of any width (they will be converted to the runtime's
/// `uintptr_t` / `size_t` types as needed).
void genOmpAllocatorStamp(fir::FirOpBuilder &builder, mlir::Location loc,
                          mlir::Value desc, mlir::Value handle,
                          mlir::Value align = {});

} // namespace fir::runtime
#endif // FORTRAN_OPTIMIZER_BUILDER_RUNTIME_ALLOCATABLE_H
