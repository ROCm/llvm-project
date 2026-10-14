//===- MyOpenMPPass.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Describe the transformation here.
///
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/OpenMP/Passes.h"

#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "flatten-multi-dimensional_arrays-pass"

namespace flangomp {
#define GEN_PASS_DEF_FLATTENMULTIDIMENSIONALARRAYSPASS
#include "flang/Optimizer/OpenMP/Passes.h.inc"
} // namespace flangomp

namespace {

struct BoundDimDesc {
  mlir::Value LowerBound;
  mlir::Value Extent;
  BoundDimDesc(mlir::Value extent) {
    LowerBound = nullptr;
    Extent = extent;
  }
  BoundDimDesc(mlir::Value lowerBound, mlir::Value extent) {
    LowerBound = lowerBound;
    Extent = extent;
  }
};

bool checkBounds(mlir::Operation::operand_range bounds,
                 llvm::SmallVector<BoundDimDesc> &upperBounds) {
  if (bounds.size() == 1)
    return false;
  for (mlir::Value it : bounds) {
    auto mapBoundsOp =
        llvm::dyn_cast<mlir::omp::MapBoundsOp>(it.getDefiningOp());
    if (!mapBoundsOp)
      return false;
    std::optional<int64_t> lowerBound =
        mapBoundsOp.getLowerBound()
            ? mlir::getConstantIntValue(mapBoundsOp.getLowerBound())
            : std::nullopt;
    if (!lowerBound.has_value())
      return false;
    if (*lowerBound)
      return false;
    std::optional<int64_t> stride =
        mapBoundsOp.getStride()
            ? mlir::getConstantIntValue(mapBoundsOp.getStride())
            : std::nullopt;
    if (!stride.has_value())
      return false;
    if (*stride != 1)
      return false;

    upperBounds.push_back(BoundDimDesc(mapBoundsOp.getUpperBound()));
  }

  return true;
}

bool haveTheSameValue(mlir::Value v1, mlir::Value v2) {
  if (v1 == v2)
    return true;
  if (!v1 || !v2)
    return false;
  mlir::Operation *op1 = v1.getDefiningOp();
  mlir::Operation *op2 = v2.getDefiningOp();
  if (!op1 || !op2)
    return false;
  if (auto const1 = mlir::dyn_cast<mlir::arith::ConstantOp>(op1))
    if (auto const2 = mlir::dyn_cast<mlir::arith::ConstantOp>(op2)) {
      std::optional<int64_t> constVal1 =
          mlir::getConstantIntValue(const1.getValue());
      std::optional<int64_t> constVal2 =
          mlir::getConstantIntValue(const2.getValue());
      if (constVal1.has_value() && constVal2.has_value())
        return *constVal1 == *constVal2;
    }
  // Special-case fir.load whose memref is a Fortran variable defined outside
  // the tree root (loads cannot be matched structurally because they have
  // memory effects). Could be extended to recurse on hlfir.designate operands
  // to cover patterns like x(:, vec(j)) = x(:, vec(j)).
  if (auto load1 = mlir::dyn_cast<fir::LoadOp>(op1))
    if (auto load2 = mlir::dyn_cast<fir::LoadOp>(op2)) {
      return load1.getMemref() == load2.getMemref();
    }
  //      if (areLoadsKnownEquivalent(load1, load2, root))
  //        return true;

  if (!mlir::isMemoryEffectFree(op1)) {
    return false;
  }
  if (!mlir::isMemoryEffectFree(op2)) {
    return false;
  }
  // Otherwise, structural equivalence of pure ops, recursing through operands.
  return mlir::OperationEquivalence::isEquivalentTo(
      op1, op2,
      [&](mlir::Value a, mlir::Value b) {
        return mlir::success(haveTheSameValue(a, b));
      },
      /*markEquivalent=*/nullptr,
      mlir::OperationEquivalence::Flags::IgnoreLocations);
}

class EqualArrays {
  llvm::SmallVector<BoundDimDesc> DeviceBounds;
  llvm::SmallVector<hlfir::DeclareOp> Arrays;

public:
  EqualArrays(llvm::SmallVector<BoundDimDesc> deviceBounds,
              hlfir::DeclareOp item) {
    DeviceBounds = deviceBounds;
    Arrays.push_back({item});
  }
  void attachArray(hlfir::DeclareOp array) { Arrays.push_back(array); }
  llvm::SmallVector<hlfir::DeclareOp> getArrays() { return Arrays; }
  llvm::SmallVector<BoundDimDesc> getBounds() { return DeviceBounds; }
};

bool isTheSameArraySize(const llvm::SmallVector<BoundDimDesc> &upperBounds1,
                        const llvm::SmallVector<BoundDimDesc> &upperBounds2) {
  if (upperBounds1.size() != upperBounds2.size())
    return false;
  for (auto [bound1, bound2] : llvm::zip_equal(upperBounds1, upperBounds2)) {
    if (!haveTheSameValue(bound1.LowerBound, bound2.LowerBound))
      return false;
  }
  return true;
}

hlfir::DeclareOp findOptimizableDeclareOp(mlir::Value &arg) {
  auto type = fir::unwrapSequenceType(fir::unwrapRefType(arg.getType()));
  if (!arg.hasNUses(1))
    return nullptr;
  auto declareOp = llvm::dyn_cast<hlfir::DeclareOp>(*arg.user_begin());
  if (!declareOp)
    return nullptr;

  for (auto it : (*arg.user_begin())->getUsers()) {
    if (!llvm::isa<hlfir::DesignateOp>(it))
      return nullptr;
    if (fir::unwrapRefType(it->getResult(0).getType()) != type)
      return nullptr;
  }
  return declareOp;
}

void groupTheSameSizeArrays(llvm::SmallVector<EqualArrays> &arrayEqualBounds,
                            hlfir::DeclareOp &declareOp,
                            llvm::SmallVector<BoundDimDesc> &upperBounds,
                            llvm::SmallVector<BoundDimDesc> &deviceBounds) {
  for (auto &item : arrayEqualBounds) {
    if (isTheSameArraySize(item.getBounds(), upperBounds)) {
      item.attachArray(declareOp);
      return;
    }
  }
  arrayEqualBounds.push_back(EqualArrays(deviceBounds, declareOp));
}

bool checkHostArray(mlir::Value mapVarPtr,
                    llvm::SmallVector<BoundDimDesc> &bounds) {
  if (!mapVarPtr)
    return false;
  if (!mapVarPtr.getDefiningOp())
    return false;
  auto declareOp = llvm::dyn_cast<hlfir::DeclareOp>(mapVarPtr.getDefiningOp());
  if (!declareOp)
    return false;
  auto shape = declareOp.getShape();
  if (!shape)
    return false;
  auto shapeOp = mlir::dyn_cast<fir::ShapeOp>(shape.getDefiningOp());
  if (shapeOp) {
    for (auto it : shapeOp.getExtents())
      bounds.push_back(BoundDimDesc(it));
  }
  return true;
}

llvm::SmallVector<BoundDimDesc> getDeviceBounds(hlfir::DeclareOp declareOp) {
  llvm::SmallVector<BoundDimDesc> deviceBounds;
  auto shape = declareOp.getShape();
  if (!shape)
    return {};
  auto shapeOp = mlir::dyn_cast<fir::ShapeOp>(shape.getDefiningOp());
  if (shapeOp) {
    for (auto it : shapeOp.getExtents())
      deviceBounds.push_back(BoundDimDesc(it));
  }

  return deviceBounds;
}

mlir::Value
calculateTotalSize(fir::FirOpBuilder &builder, mlir::Location loc,
                   const llvm::SmallVector<BoundDimDesc> &deviceBounds) {
  mlir::Value totalSize = nullptr;
  for (auto bound : deviceBounds) {
    auto int64Extent = fir::ConvertOp::create(
        builder, loc, builder.getI64Type(), bound.Extent);
    if (!totalSize)
      totalSize = int64Extent;
    else
      totalSize =
          mlir::arith::MulIOp::create(builder, loc, totalSize, int64Extent);
  }
  totalSize = fir::ShapeOp::create(builder, loc, totalSize);
  return totalSize;
}

hlfir::DeclareOp genFlattenedArrayDeclareOp(fir::FirOpBuilder &builder,
                                            hlfir::DeclareOp declareOp,
                                            mlir::Value shape) {

  std::string name = (declareOp.getUniqName().str() + ".flat");
  auto ref = declareOp.getMemref();
  auto refTy = mlir::cast<fir::ReferenceType>(ref.getType());
  auto seqTy = mlir::cast<fir::SequenceType>(refTy.getEleTy());

  auto flatSeqTy = fir::SequenceType::get(
      {fir::SequenceType::getUnknownExtent()}, seqTy.getEleTy());
  auto flattenedMemRef = fir::ConvertOp::create(
      builder, declareOp.getLoc(), fir::ReferenceType::get(flatSeqTy), ref);
  auto newDecl = hlfir::DeclareOp::create(
      builder, declareOp.getLoc(), flattenedMemRef, name, shape,
      declareOp.getTypeparams(), declareOp.getDummyScope(),
      declareOp.getStorage(), declareOp.getStorageOffset(),
      declareOp.getFortranAttrsAttr(), declareOp.getDataAttrAttr());

  return newDecl;
}

void updateArrayAccess(fir::FirOpBuilder &builder,
                       hlfir::DeclareOp oldDeclareOp,
                       hlfir::DeclareOp newDeclareOp,
                       const llvm::SmallVector<BoundDimDesc> &bounds) {
  mlir::OpBuilder::InsertionGuard guard(builder);
  llvm::SmallVector<hlfir::DesignateOp> oldDesignateOps;
  for (auto it : (oldDeclareOp->getUsers())) {
    auto designateOp = llvm::dyn_cast<hlfir::DesignateOp>(*it);
    assert(designateOp && "Expected designate op");
    builder.setInsertionPointAfter(designateOp);
    auto loc = designateOp.getLoc();
    mlir::Value linearIndex = fir::ConvertOp::create(
        builder, loc, builder.getI64Type(), designateOp.getIndices()[0]);
    auto one = mlir::arith::ConstantOp::create(builder, loc,
                                               builder.getI64IntegerAttr(-1));
    for (size_t i = 1; i < designateOp.getIndices().size(); ++i) {
      auto int64IdxConv = fir::ConvertOp::create(
          builder, loc, builder.getI64Type(), designateOp.getIndices()[i]);
      mlir::Value int64IdxConvMinOne =
          mlir::arith::AddIOp::create(builder, loc, int64IdxConv, one);
      mlir::Value dimOffset = fir::ConvertOp::create(
          builder, loc, builder.getI64Type(), bounds[0].Extent);
      for (size_t dim = 1; dim < i ; ++dim) {
        auto dimOffsetNext = fir::ConvertOp::create(
            builder, loc, builder.getI64Type(), bounds[dim].Extent);
        dimOffset =
            mlir::arith::MulIOp::create(builder, loc, dimOffset, dimOffsetNext);
      }
      auto dimOffsetProduct = mlir::arith::MulIOp::create(
          builder, loc, int64IdxConvMinOne, dimOffset);
      linearIndex = mlir::arith::AddIOp::create(builder, loc, dimOffsetProduct,
                                                linearIndex);
    }
    auto newDesignateOp = hlfir::DesignateOp::create(
        builder, loc, designateOp.getResult().getType(),
        newDeclareOp.getResult(0), {linearIndex});
    designateOp.replaceAllUsesWith(&*newDesignateOp);
    oldDesignateOps.push_back(designateOp);
  }
  for (auto it : oldDesignateOps)
    (*it).erase();
  oldDeclareOp.erase();
}

class FlattenMultiDimensionalArraysPass
    : public flangomp::impl::FlattenMultiDimensionalArraysPassBase<
          FlattenMultiDimensionalArraysPass> {
public:
  using flangomp::impl::FlattenMultiDimensionalArraysPassBase<
      FlattenMultiDimensionalArraysPass>::FlattenMultiDimensionalArraysPassBase;

  void runOnOperation() override {
    mlir::ModuleOp module = mlir::cast<mlir::ModuleOp>(getOperation());
    fir::KindMapping kindMap = fir::getKindMapping(module);
    fir::FirOpBuilder builder{module, std::move(kindMap)};

    getOperation()->walk([&](mlir::Operation *func) {
      if (!mlir::isa<mlir::func::FuncOp, mlir::omp::DeclareMapperOp>(func))
        return;

      func->walk([&](mlir::omp::TargetOp targetOp) {
        llvm::SmallVector<llvm::SmallVector<BoundDimDesc>> mapBounds;
        llvm::SmallVector<llvm::SmallVector<BoundDimDesc>> deviceBounds;
        llvm::SmallVector<llvm::SmallVector<BoundDimDesc>> hostBounds;
        llvm::SmallVector<EqualArrays> arrayEqualBounds;
        llvm::SmallVector<hlfir::DeclareOp> optimizableDeclareOps;
        auto iface = mlir::cast<mlir::omp::BlockArgOpenMPOpInterface>(
            targetOp.getOperation());
        auto blockArgs = iface.getMapBlockArgs();
        for (auto [mapVar, blockArg] :
             llvm::zip_equal(targetOp.getMapVars(), blockArgs)) {
          llvm::SmallVector<BoundDimDesc> hostSingleArrayBounds;
          if (!llvm::isa<fir::SequenceType>(
                  fir::unwrapRefType(mapVar.getType())))
            continue;
          auto mapInfoOp =
              llvm::dyn_cast<mlir::omp::MapInfoOp>(mapVar.getDefiningOp());
          if (!mapInfoOp)
            continue;
          auto declareOp = findOptimizableDeclareOp(blockArg);
          if (!declareOp)
            continue;
          if (!checkHostArray(mapInfoOp.getVarPtr(), hostSingleArrayBounds))
            continue;
          if (hostSingleArrayBounds.size() < 2)
            continue;

          llvm::SmallVector<BoundDimDesc> upperBounds;
          if (!checkBounds(mapInfoOp.getBounds(), upperBounds))
            continue;
          optimizableDeclareOps.push_back(declareOp);
          mapBounds.push_back(upperBounds);
          deviceBounds.push_back(getDeviceBounds(declareOp));
          hostBounds.push_back(hostSingleArrayBounds);
        }

        for (auto [optimizableDeclareOp, mapBound, hostBound, deviceBound] :
             llvm::zip_equal(optimizableDeclareOps, mapBounds, hostBounds,
                             deviceBounds)) {

          groupTheSameSizeArrays(arrayEqualBounds, optimizableDeclareOp,
                                 mapBound, deviceBound);
        }

        mlir::OpBuilder::InsertionGuard guard(builder);
        for (auto optimizableSetOfArrays : arrayEqualBounds) {
          mlir::Value totalSize = nullptr;
          for (auto arr : optimizableSetOfArrays.getArrays()) {
            builder.setInsertionPointAfter(arr);
            if (!totalSize) {
              totalSize = calculateTotalSize(
                  builder, arr.getLoc(), optimizableSetOfArrays.getBounds());
            }
            auto newDecl = genFlattenedArrayDeclareOp(builder, arr, totalSize);
            updateArrayAccess(builder, arr, newDecl,
                              optimizableSetOfArrays.getBounds());
          }
        }
      });
    });
  }
};

} // namespace
