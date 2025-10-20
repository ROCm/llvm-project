//===- StackToShared.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements transforms to swap stack allocations on the target
// device with device shared memory where applicable.
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/OpenMP/Passes.h"
#include "flang/Utils/OpenMP.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPInterfaces.h"

namespace flangomp {
#define GEN_PASS_DEF_STACKTOSHAREDPASS
#include "flang/Optimizer/OpenMP/Passes.h.inc"
} // namespace flangomp

using namespace mlir;

namespace {
class StackToSharedPass
    : public flangomp::impl::StackToSharedPassBase<StackToSharedPass> {
public:
  StackToSharedPass() = default;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    OpBuilder builder(context);

    func::FuncOp funcOp = getOperation();
    auto offloadIface = funcOp->getParentOfType<omp::OffloadModuleInterface>();
    if (!offloadIface || !offloadIface.getIsTargetDevice())
      return;

    funcOp->walk([&](fir::AllocaOp allocaOp) {
      if (!Fortran::utils::openmp::shouldReplaceAllocaWithDeviceSharedMem(
              *allocaOp))
        return;

      // Replace fir.alloca with omp.alloc_shared_mem.
      builder.setInsertionPoint(allocaOp);
      auto sharedAllocOp = omp::AllocSharedMemOp::create(
          builder, allocaOp->getLoc(), allocaOp.getResult().getType(),
          allocaOp.getInType(), allocaOp.getUniqNameAttr(),
          allocaOp.getBindcNameAttr(), allocaOp.getTypeparams(),
          allocaOp.getShape());
      allocaOp.replaceAllUsesWith(sharedAllocOp.getOperation());
      allocaOp.erase();

      // Create a new omp.free_shared_mem for the allocated buffer prior to
      // exiting the region.
      Fortran::utils::openmp::insertDeviceSharedMemDeallocation(
          builder, sharedAllocOp.getResult());
    });
  }
};
} // namespace
