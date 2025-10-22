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

#include "flang/Optimizer/OpenMP/Passes.h"
#include "flang/Utils/OpenMP.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"

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

    LLVM::LLVMFuncOp funcOp = getOperation();
    auto offloadIface = funcOp->getParentOfType<omp::OffloadModuleInterface>();
    if (!offloadIface || !offloadIface.getIsTargetDevice())
      return;

    llvm::SmallVector<Operation *> toBeDeleted;
    funcOp->walk([&](LLVM::AllocaOp allocaOp) {
      if (!Fortran::utils::openmp::shouldReplaceAllocaWithDeviceSharedMem(
              *allocaOp))
        return;
      // Replace llvm.alloca with omp.alloc_shared_mem.
      Type resultType = allocaOp.getResult().getType();

      // TODO: The handling of non-default address spaces might need to be
      // improved. This currently only handles the case where an alloca to
      // non-default address space must only be used by a single addrspacecast
      // to default address space.
      bool nonDefaultAddrSpace = false;
      if (auto llvmPtrType = dyn_cast<LLVM::LLVMPointerType>(resultType))
        nonDefaultAddrSpace = llvmPtrType.getAddressSpace() != 0;

      builder.setInsertionPoint(allocaOp);
      auto sharedAllocOp = omp::AllocSharedMemOp::create(
          builder, allocaOp->getLoc(), LLVM::LLVMPointerType::get(context),
          allocaOp.getElemType(),
          /*uniq_name=*/nullptr,
          /*bindc_name=*/nullptr, /*typeparams=*/{allocaOp.getArraySize()},
          /*shape=*/{});
      if (nonDefaultAddrSpace) {
        assert(allocaOp->hasOneUse() && "alloca must have only one use");
        auto asCastOp =
            cast<LLVM::AddrSpaceCastOp>(*allocaOp->getUsers().begin());
        asCastOp.replaceAllUsesWith(sharedAllocOp.getOperation());
        // Delete later because we can't delete the cast op before the top-level
        // iteration visits it. Also, the alloca can't be deleted before because
        // it's used by it.
        toBeDeleted.push_back(asCastOp);
        toBeDeleted.push_back(allocaOp);
      } else {
        allocaOp.replaceAllUsesWith(sharedAllocOp.getOperation());
        allocaOp.erase();
      }

      // Create a new omp.free_shared_mem for the allocated buffer prior to
      // exiting the region.
      Fortran::utils::openmp::insertDeviceSharedMemDeallocation(
          builder, sharedAllocOp.getResult());
    });
    for (Operation *op : toBeDeleted)
      op->erase();
  }
};
} // namespace
