//===-- lib/Utisl/OpenMP.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Utils/OpenMP.h"

#include "flang/Lower/ConvertExprToHLFIR.h"
#include "flang/Optimizer/Builder/DirectivesCommon.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Transforms/RegionUtils.h"

mlir::omp::MapInfoOp Fortran::utils::openmp::createMapInfoOp(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value baseAddr,
    mlir::Value varPtrPtr, llvm::StringRef name,
    llvm::ArrayRef<mlir::Value> bounds, llvm::ArrayRef<mlir::Value> members,
    mlir::ArrayAttr membersIndex, mlir::omp::ClauseMapFlags mapType,
    mlir::omp::VariableCaptureKind mapCaptureType, mlir::Type retTy,
    bool partialMap, mlir::FlatSymbolRefAttr mapperId) {

  if (auto boxTy = llvm::dyn_cast<fir::BaseBoxType>(baseAddr.getType())) {
    baseAddr = fir::BoxAddrOp::create(builder, loc, baseAddr);
    retTy = baseAddr.getType();
  }

  mlir::TypeAttr varType = mlir::TypeAttr::get(
      llvm::cast<mlir::omp::PointerLikeType>(retTy).getElementType());

  // For types with unknown extents such as <2x?xi32> we discard the incomplete
  // type info and only retain the base type. The correct dimensions are later
  // recovered through the bounds info.
  if (auto seqType = llvm::dyn_cast<fir::SequenceType>(varType.getValue()))
    if (seqType.hasDynamicExtents())
      varType = mlir::TypeAttr::get(seqType.getEleTy());

  mlir::omp::MapInfoOp op =
      mlir::omp::MapInfoOp::create(builder, loc, retTy, baseAddr, varType,
          builder.getAttr<mlir::omp::ClauseMapFlagsAttr>(mapType),
          builder.getAttr<mlir::omp::VariableCaptureKindAttr>(mapCaptureType),
          varPtrPtr, members, membersIndex, bounds, mapperId,
          builder.getStringAttr(name), builder.getBoolAttr(partialMap));
  return op;
}

mlir::Value Fortran::utils::openmp::mapTemporaryValue(
    fir::FirOpBuilder &firOpBuilder, mlir::omp::TargetOp targetOp,
    mlir::Value val, llvm::StringRef name) {
  mlir::OpBuilder::InsertionGuard guard(firOpBuilder);
  mlir::Operation *valOp = val.getDefiningOp();

  if (valOp)
    firOpBuilder.setInsertionPointAfter(valOp);
  else
    // This means val is a block argument
    firOpBuilder.setInsertionPoint(targetOp);

  auto copyVal = firOpBuilder.createTemporary(val.getLoc(), val.getType());
  firOpBuilder.createStoreWithConvert(copyVal.getLoc(), val, copyVal);

  fir::factory::AddrAndBoundsInfo info = fir::factory::getDataOperandBaseAddr(
      firOpBuilder, val, /*isOptional=*/false, val.getLoc());
  llvm::SmallVector<mlir::Value> bounds =
      fir::factory::genImplicitBoundsOps<mlir::omp::MapBoundsOp,
          mlir::omp::MapBoundsType>(firOpBuilder, info,
          hlfir::translateToExtendedValue(
              val.getLoc(), firOpBuilder, hlfir::Entity{val})
              .first,
          /*dataExvIsAssumedSize=*/false, val.getLoc());

  firOpBuilder.setInsertionPoint(targetOp);

  mlir::omp::ClauseMapFlags mapFlag = mlir::omp::ClauseMapFlags::implicit;
  mlir::omp::VariableCaptureKind captureKind =
      mlir::omp::VariableCaptureKind::ByRef;

  mlir::Type eleType = copyVal.getType();
  if (auto refType = mlir::dyn_cast<fir::ReferenceType>(copyVal.getType())) {
    eleType = refType.getElementType();
  }

  if (fir::isa_trivial(eleType) || fir::isa_char(eleType)) {
    captureKind = mlir::omp::VariableCaptureKind::ByCopy;
  } else if (!fir::isa_builtin_cptr_type(eleType)) {
    mapFlag |= mlir::omp::ClauseMapFlags::to;
  }

  mlir::Value mapOp = createMapInfoOp(firOpBuilder, copyVal.getLoc(), copyVal,
      /*varPtrPtr=*/mlir::Value{}, name.str(), bounds,
      /*members=*/llvm::SmallVector<mlir::Value>{},
      /*membersIndex=*/mlir::ArrayAttr{}, mapFlag, captureKind,
      copyVal.getType());

  auto argIface = llvm::cast<mlir::omp::BlockArgOpenMPOpInterface>(*targetOp);
  mlir::Region &region = targetOp.getRegion();

  // Get the index of the first non-map argument before modifying mapVars,
  // then append an element to mapVars and an associated entry block
  // argument at that index.
  unsigned insertIndex =
      argIface.getMapBlockArgsStart() + argIface.numMapBlockArgs();
  targetOp.getMapVarsMutable().append(mapOp);
  mlir::Value clonedValArg =
      region.insertArgument(insertIndex, copyVal.getType(), copyVal.getLoc());

  mlir::Block *entryBlock = &region.getBlocks().front();
  firOpBuilder.setInsertionPointToStart(entryBlock);
  auto loadOp =
      fir::LoadOp::create(firOpBuilder, clonedValArg.getLoc(), clonedValArg);
  return loadOp.getResult();
}

void Fortran::utils::openmp::cloneOrMapRegionOutsiders(
    fir::FirOpBuilder &firOpBuilder, mlir::omp::TargetOp targetOp) {
  mlir::Region &region = targetOp.getRegion();
  mlir::Block *entryBlock = &region.getBlocks().front();

  llvm::SetVector<mlir::Value> valuesDefinedAbove;
  mlir::getUsedValuesDefinedAbove(region, valuesDefinedAbove);
  while (!valuesDefinedAbove.empty()) {
    for (mlir::Value val : valuesDefinedAbove) {
      mlir::Operation *valOp = val.getDefiningOp();

      // NOTE: We skip BoxDimsOp's as the lesser of two evils is to map the
      // indices separately, as the alternative is to eventually map the Box,
      // which comes with a fairly large overhead comparatively. We could be
      // more robust about this and check using a BackwardsSlice to see if we
      // run the risk of mapping a box.
      if (valOp && mlir::isMemoryEffectFree(valOp) &&
          !mlir::isa<fir::BoxDimsOp>(valOp)) {
        mlir::Operation *clonedOp = valOp->clone();
        entryBlock->push_front(clonedOp);

        auto replace = [entryBlock](mlir::OpOperand &use) {
          return use.getOwner()->getBlock() == entryBlock;
        };

        valOp->getResults().replaceUsesWithIf(clonedOp->getResults(), replace);
        valOp->replaceUsesWithIf(clonedOp, replace);
      } else {
        mlir::Value mappedTemp = mapTemporaryValue(firOpBuilder, targetOp, val,
            /*name=*/{});
        val.replaceUsesWithIf(mappedTemp, [entryBlock](mlir::OpOperand &use) {
          return use.getOwner()->getBlock() == entryBlock;
        });
      }
    }
    valuesDefinedAbove.clear();
    mlir::getUsedValuesDefinedAbove(region, valuesDefinedAbove);
  }
}

/// When a use takes place inside an omp.parallel region and it's not as a
/// private clause argument, or when it is a reduction argument passed to
/// omp.parallel or a function call argument, then the defining allocation is
/// eligible for replacement with shared memory.
static bool allocaUseRequiresDeviceSharedMem(const mlir::OpOperand &use) {
  mlir::Operation *owner = use.getOwner();
  if (auto parallelOp = llvm::dyn_cast<mlir::omp::ParallelOp>(owner)) {
    if (llvm::is_contained(parallelOp.getReductionVars(), use.get()))
      return true;
  } else if (auto callOp = llvm::dyn_cast<mlir::CallOpInterface>(owner)) {
    if (llvm::is_contained(callOp.getArgOperands(), use.get()))
      return true;
  }

  // If it is used directly inside of a parallel region, it has to be replaced
  // unless the use is a private clause.
  if (owner->getParentOfType<mlir::omp::ParallelOp>()) {
    if (auto argIface =
            llvm::dyn_cast<mlir::omp::BlockArgOpenMPOpInterface>(owner)) {
      if (auto privateSyms = llvm::cast_or_null<mlir::ArrayAttr>(
              owner->getAttr("private_syms"))) {
        for (auto [var, sym] :
            llvm::zip_equal(argIface.getPrivateVars(), privateSyms)) {
          if (var != use.get())
            continue;

          auto moduleOp = owner->getParentOfType<mlir::ModuleOp>();
          auto privateOp = llvm::cast<mlir::omp::PrivateClauseOp>(
              moduleOp.lookupSymbol(llvm::cast<mlir::SymbolRefAttr>(sym)));
          return privateOp.getDataSharingType() !=
              mlir::omp::DataSharingClauseType::Private;
        }
      }
    }
    return true;
  }
  return false;
}

static bool shouldReplaceAllocaWithUses(
    const mlir::Operation::use_range &uses) {
  // Check direct uses and also follow hlfir.declare/fir.convert uses.
  for (const mlir::OpOperand &use : uses) {
    mlir::Operation *owner = use.getOwner();
    if (llvm::isa<mlir::LLVM::AddrSpaceCastOp, mlir::LLVM::GEPOp>(owner)) {
      if (shouldReplaceAllocaWithUses(owner->getUses()))
        return true;
    } else if (allocaUseRequiresDeviceSharedMem(use)) {
      return true;
    }
  }

  return false;
}

// TODO: Refactor the logic in `shouldReplaceAllocaWithDeviceSharedMem`,
// `shouldReplaceAllocaWithUses` and `allocaUseRequiresDeviceSharedMem` to
// be reusable by the MLIR to LLVM IR translation stage, as something very
// similar is also implemented there to choose between allocas and device
// shared memory allocations when processing OpenMP reductions, mapping and
// privatization.
bool Fortran::utils::openmp::shouldReplaceAllocaWithDeviceSharedMem(
    mlir::Operation &op) {
  auto offloadIface = op.getParentOfType<mlir::omp::OffloadModuleInterface>();
  if (!offloadIface || !offloadIface.getIsTargetDevice())
    return false;

  auto targetOp = op.getParentOfType<mlir::omp::TargetOp>();

  // It must be inside of a generic omp.target or in a target device function,
  // and not inside of omp.parallel.
  if (auto parallelOp = op.getParentOfType<mlir::omp::ParallelOp>()) {
    if (!targetOp || !targetOp->isProperAncestor(parallelOp))
      return false;
  }

  if (targetOp) {
    if (targetOp.getKernelExecFlags(targetOp.getInnermostCapturedOmpOp()) !=
        mlir::omp::TargetExecMode::generic)
      return false;
  } else {
    auto declTargetIface =
        op.getParentOfType<mlir::omp::DeclareTargetInterface>();
    if (!declTargetIface || !declTargetIface.isDeclareTarget() ||
        declTargetIface.getDeclareTargetDeviceType() ==
            mlir::omp::DeclareTargetDeviceType::host)
      return false;
  }

  return shouldReplaceAllocaWithUses(op.getUses());
}

void Fortran::utils::openmp::insertDeviceSharedMemDeallocation(
    mlir::OpBuilder &builder, mlir::Value allocVal) {
  mlir::Block *allocaBlock = allocVal.getParentBlock();
  mlir::DominanceInfo domInfo;
  for (mlir::Block &block : allocVal.getParentRegion()->getBlocks()) {
    mlir::Operation *terminator = block.getTerminator();
    if (!terminator->hasSuccessors() &&
        domInfo.dominates(allocaBlock, &block)) {
      builder.setInsertionPoint(terminator);
      mlir::omp::FreeSharedMemOp::create(builder, allocVal.getLoc(), allocVal);
    }
  }
}
