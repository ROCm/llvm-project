//===- OMPMapInfoFinalization.cpp
//---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
/// \file
/// An OpenMP dialect related pass for FIR/HLFIR which performs some
/// pre-processing of MapInfoOp's after the module has been lowered to
/// finalize them.
///
/// For example, it expands MapInfoOp's containing descriptor related
/// types (fir::BoxType's) into multiple MapInfoOp's containing the parent
/// descriptor and pointer member components for individual mapping,
/// treating the descriptor type as a record type for later lowering in the
/// OpenMP dialect.
///
/// The pass also adds MapInfoOp's that are members of a parent object but are
/// not directly used in the body of a target region to it's BlockArgument list
/// to maintain consistency across all MapInfoOp's tied to a region directly or
/// indirectly via an parent object.
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/KindMapping.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include <algorithm>
#include <iterator>
#include <numeric>

namespace fir {
#define GEN_PASS_DEF_OMPMAPINFOFINALIZATIONPASS
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

namespace {
class OMPMapInfoFinalizationPass
    : public fir::impl::OMPMapInfoFinalizationPassBase<
          OMPMapInfoFinalizationPass> {
  struct ParentAndPlacement {
    mlir::omp::MapInfoOp parent;
    size_t index;
  };

  mlir::DenseIntElementsAttr createDenseElementsAttrFromIndices(
      llvm::SmallVector<llvm::SmallVector<int>> &memberPlacementIndices,
      fir::FirOpBuilder &builder) {
    llvm::SmallVector<int64_t> shape;
    shape.push_back(memberPlacementIndices.size());
    shape.push_back(memberPlacementIndices[0].size());

    llvm::SmallVector<int> indicesFlattened = std::accumulate(
        memberPlacementIndices.begin(), memberPlacementIndices.end(),
        llvm::SmallVector<int>(),
        [](llvm::SmallVector<int> &x, llvm::SmallVector<int> &y) {
          x.insert(x.end(), y.begin(), y.end());
          return x;
        });

    return mlir::DenseIntElementsAttr::get(
        mlir::VectorType::get(llvm::ArrayRef<int64_t>(shape),
                              mlir::IntegerType::get(builder.getContext(), 32)),
        llvm::ArrayRef<int32_t>(indicesFlattened));
  }

  void
  getMemberUserList(mlir::omp::MapInfoOp op,
                    llvm::SmallVectorImpl<ParentAndPlacement> &mapMemberUsers) {
    for (auto *users : op->getUsers()) {
      if (auto map = mlir::dyn_cast_if_present<mlir::omp::MapInfoOp>(users)) {
        for (size_t i = 0; i < map.getMembers().size(); ++i) {
          if (map.getMembers()[i].getDefiningOp() == op) {
            mapMemberUsers.push_back({map, i});
          }
        }
      }
    }
  }

  llvm::SmallVector<llvm::SmallVector<int32_t>>
  getMemberIndicesAsVectors(mlir::omp::MapInfoOp mapInfo) {
    llvm::SmallVector<llvm::SmallVector<int32_t>> indices;

    for (int i = 0;
         i < mapInfo.getMembersIndexAttr().getShapedType().getShape()[0]; ++i) {
      llvm::SmallVector<int32_t> vec;
      for (int j = 0;
           j < mapInfo.getMembersIndexAttr().getShapedType().getShape()[1];
           ++j) {
        vec.push_back(
            mapInfo.getMembersIndexAttr()
                .getValues<int32_t>()[i * mapInfo.getMembersIndexAttr()
                                              .getShapedType()
                                              .getShape()[1] +
                                      j]);
      }
      indices.push_back(vec);
    }

    return indices;
  }

  mlir::Value getDescriptorFromBoxMap(mlir::omp::MapInfoOp boxMap,
                                      fir::FirOpBuilder &builder) {
    mlir::Location loc = builder.getUnknownLoc();
    mlir::Value descriptor = boxMap.getVarPtr();

    if (!fir::isTypeWithDescriptor(boxMap.getVarType())) {
      if (auto addrOp = mlir::dyn_cast_if_present<fir::BoxAddrOp>(
              boxMap.getVarPtr().getDefiningOp())) {
        descriptor = addrOp.getVal();
      }
    }

    // The fir::BoxOffsetOp only works with !fir.ref<!fir.box<...>> types, as
    // allowing it to access non-reference box operations can cause some
    // problematic SSA IR. However, in the case of assumed shape's the type
    // is not a !fir.ref, in these cases to retrieve the appropriate
    // !fir.ref<!fir.box<...>> to access the data we need to map we must
    // perform an alloca and then store to it and retrieve the data from the
    // new alloca.
    if (mlir::isa<fir::BaseBoxType>(descriptor.getType())) {
      mlir::OpBuilder::InsertPoint insPt = builder.saveInsertionPoint();
      builder.setInsertionPointToStart(builder.getAllocaBlock());
      auto alloca = builder.create<fir::AllocaOp>(loc, descriptor.getType());
      builder.restoreInsertionPoint(insPt);
      builder.create<fir::StoreOp>(loc, descriptor, alloca);
      descriptor = alloca;
    }

    return descriptor;
  }

  mlir::omp::MapInfoOp getBaseAddrMap(mlir::Value descriptor,
                                      mlir::OperandRange bounds,
                                      int64_t mapType,
                                      fir::FirOpBuilder &builder) {
    mlir::Location loc = builder.getUnknownLoc();

    mlir::Value baseAddrAddr = builder.create<fir::BoxOffsetOp>(
        loc, descriptor, fir::BoxFieldAttr::base_addr);

    // Member of the descriptor pointing at the allocated data
    return builder.create<mlir::omp::MapInfoOp>(
        loc, baseAddrAddr.getType(), descriptor,
        mlir::TypeAttr::get(llvm::cast<mlir::omp::PointerLikeType>(
                                fir::unwrapRefType(baseAddrAddr.getType()))
                                .getElementType()),
        baseAddrAddr, mlir::SmallVector<mlir::Value>{},
        mlir::DenseIntElementsAttr{}, bounds,
        builder.getIntegerAttr(builder.getIntegerType(64, false), mapType),
        builder.getAttr<mlir::omp::VariableCaptureKindAttr>(
            mlir::omp::VariableCaptureKind::ByRef),
        builder.getStringAttr("") /*name*/,
        builder.getBoolAttr(false) /*partial_map*/);
  }

  void genDescriptorMemberMaps(mlir::omp::MapInfoOp op,
                               fir::FirOpBuilder &builder,
                               mlir::Operation *target) {
    llvm::SmallVector<ParentAndPlacement> mapMemberUsers;
    getMemberUserList(op, mapMemberUsers);

    // NOTE/TODO: We currently only support a MapInfoOp being used in one
    // member list at a time, currently the frontend will generate a new
    // MapInfoOp per map clause, so this should not be an issue, but in the
    // future when we seek to cleanup and optimize the IR, this will need to
    // be extended.
    assert(mapMemberUsers.size() <= 1 &&
           "genDescriptorMemberMaps currently only supports descriptor used by "
           "one MapInfoOp member list");

    if (!mapMemberUsers.empty()) {
      auto memberIndices = getMemberIndicesAsVectors(mapMemberUsers[0].parent);
      auto baseAddrIndex = memberIndices[mapMemberUsers[0].index];
      auto *negIdx = std::find(baseAddrIndex.begin(), baseAddrIndex.end(), -1);
      if (negIdx != baseAddrIndex.end()) {
        *negIdx = 0;
      } else {
        baseAddrIndex.push_back(0);
        for (size_t i = 0; i < memberIndices.size(); ++i)
          memberIndices[i].push_back(-1);
      }

      memberIndices.insert(
          std::next(memberIndices.begin(), mapMemberUsers[0].index + 1),
          baseAddrIndex);

      mlir::DenseIntElementsAttr newEleAttr =
          createDenseElementsAttrFromIndices(memberIndices, builder);

      mlir::Value descriptor = getDescriptorFromBoxMap(op, builder);
      auto baseAddr = getBaseAddrMap(descriptor, op.getBounds(),
                                     op.getMapType().value(), builder);

      llvm::SmallVector<mlir::Value> newMemberOps;
      mlir::OperandRange membersArr = mapMemberUsers[0].parent.getMembers();
      for (size_t i = 0; i < membersArr.size(); ++i) {
        newMemberOps.push_back(membersArr[i]);
        if (membersArr[i] == op) {
          newMemberOps.push_back(baseAddr);
        }
      }

      mapMemberUsers[0].parent.getMembersMutable().assign(newMemberOps);
      mapMemberUsers[0].parent.setMembersIndexAttr(newEleAttr);

      if (auto mapClauseOwner =
              llvm::dyn_cast<mlir::omp::MapClauseOwningOpInterface>(target)) {
        llvm::SmallVector<mlir::Value> newMapOps;
        mlir::OperandRange mapOperandsArr = mapClauseOwner.getMapOperands();

        for (size_t i = 0; i < mapOperandsArr.size(); ++i) {
          if (mapOperandsArr[i] == op) {
            // Push new implicit maps generated for the descriptor.
            newMapOps.push_back(baseAddr);

            // for TargetOp's which have IsolatedFromAbove we must align the
            // new additional map operand with an appropriate BlockArgument,
            // as the printing and later processing currently requires a 1:1
            // mapping of BlockArgs to MapInfoOp's at the same placement in
            // each array (BlockArgs and MapOperands).
            if (auto targetOp = llvm::dyn_cast<mlir::omp::TargetOp>(target))
              targetOp.getRegion().insertArgument(i, baseAddr.getType(),
                                                  builder.getUnknownLoc());
          }
          newMapOps.push_back(mapOperandsArr[i]);
        }
        mapClauseOwner.getMapOperandsMutable().assign(newMapOps);
      }

      mlir::Value newDescParentMapOp = builder.create<mlir::omp::MapInfoOp>(
          op->getLoc(), op.getResult().getType(), descriptor,
          mlir::TypeAttr::get(fir::unwrapRefType(descriptor.getType())),
          mlir::Value{}, mlir::SmallVector<mlir::Value>{},
          mlir::DenseIntElementsAttr{} /*members_index*/,
          mlir::SmallVector<mlir::Value>{},
          builder.getIntegerAttr(builder.getIntegerType(64, false),
                                 op.getMapType().value()),
          op.getMapCaptureTypeAttr(), op.getNameAttr(), op.getPartialMapAttr());
      op.replaceAllUsesWith(newDescParentMapOp);
      op->erase();
      
    } else {
      mlir::Value descriptor = getDescriptorFromBoxMap(op, builder);
      auto baseAddr = getBaseAddrMap(descriptor, op.getBounds(),
                                     op.getMapType().value(), builder);

      // TODO: map the addendum segment of the descriptor, similarly to the
      // above base address/data pointer member.

      if (auto mapClauseOwner =
              llvm::dyn_cast<mlir::omp::MapClauseOwningOpInterface>(target)) {
        llvm::SmallVector<mlir::Value> newMapOps;
        mlir::OperandRange mapOperandsArr = mapClauseOwner.getMapOperands();

        for (size_t i = 0; i < mapOperandsArr.size(); ++i) {
          if (mapOperandsArr[i] == op) {
            // Push new implicit maps generated for the descriptor.
            newMapOps.push_back(baseAddr);

            // for TargetOp's which have IsolatedFromAbove we must align the
            // new additional map operand with an appropriate BlockArgument,
            // as the printing and later processing currently requires a 1:1
            // mapping of BlockArgs to MapInfoOp's at the same placement in
            // each array (BlockArgs and MapOperands).
            if (auto targetOp = llvm::dyn_cast<mlir::omp::TargetOp>(target))
              targetOp.getRegion().insertArgument(i, baseAddr.getType(),
                                                  builder.getUnknownLoc());
          }
          newMapOps.push_back(mapOperandsArr[i]);
        }
        mapClauseOwner.getMapOperandsMutable().assign(newMapOps);
      }

      mlir::Value newDescParentMapOp = builder.create<mlir::omp::MapInfoOp>(
          op->getLoc(), op.getResult().getType(), descriptor,
          mlir::TypeAttr::get(fir::unwrapRefType(descriptor.getType())),
          mlir::Value{}, mlir::SmallVector<mlir::Value>{baseAddr},
          mlir::DenseIntElementsAttr::get(
              mlir::VectorType::get(
                  llvm::ArrayRef<int64_t>({1, 1}),
                  mlir::IntegerType::get(builder.getContext(), 32)),
              llvm::ArrayRef<int32_t>({0})) /*members_index*/,
          mlir::SmallVector<mlir::Value>{},
          builder.getIntegerAttr(builder.getIntegerType(64, false),
                                 op.getMapType().value()),
          op.getMapCaptureTypeAttr(), op.getNameAttr(), op.getPartialMapAttr());
      op.replaceAllUsesWith(newDescParentMapOp);
      op->erase();
    }
  }

  // For all mapped record members not directly used in the target region
  // we add them to the block arguments in front of their parent and place
  // them into the map operands list for consistency.
  //
  // These indirect uses (via accesses to their parent) will still be
  // mapped individually in most cases, and a parent mapping doesn't
  // guarantee the parent will be mapped in its totality, partial
  // mapping is common.
  //
  // For example:
  //    map(tofrom: x%y)
  //
  // Will generate a mapping for "x" (the parent) and "y" (the member),
  // the parent "x" will not be mapped, only the member "y" will,
  // however, we must have the parent as a BlockArg and MapOperand in
  // these cases to maintain the correct uses within the region and
  // it helps to track that the member is part of a larger object.
  //
  // In the case of:
  //    map(tofrom: x%y, x%z)
  //
  // The parent member becomes more critical, as we perform a partial
  // structure mapping, where we link the mapping of x and y together
  // via the parent (at a kernel argument level in LLVM IR not just
  // MLIR, important to maintain similarity to Clang and for the runtime
  // to do the correct thing), however, we still do not map the structure
  // in its totality, we do however, generate an un-sized "binding"
  // map entry for it.
  //
  // In the case of:
  //    map(tofrom: x, x%y, x%z)
  //
  // We do actually map the entirety of "x", so the explicit
  // mapping of x%y, x%z becomes unneccesary. It also doesn't
  // quite make sense to write this from a Fortran OpenMP
  // perspective (although it is legal), as even if the members were
  // allocatables or pointers, we are mandated by the specification
  // to map these (and any recursive components) in their entirety,
  // which is different to the C++ equivelant, which requires
  // explicit mapping of these segments.
  void addImplicitMembersToTarget(mlir::omp::MapInfoOp op,
                                  fir::FirOpBuilder &builder,
                                  mlir::Operation *target) {
    auto mapClauseOwner =
        llvm::dyn_cast<mlir::omp::MapClauseOwningOpInterface>(target);
    if (!mapClauseOwner)
      return;

    llvm::SmallVector<mlir::Value> newMapOps;
    mlir::OperandRange mapOperandsArr = mapClauseOwner.getMapOperands();

    for (size_t i = 0; i < mapOperandsArr.size(); ++i) {
      if (mapOperandsArr[i] == op) {
        // Push member maps
        for (size_t j = 0; j < op.getMembers().size(); ++j) {
          newMapOps.push_back(op.getMembers()[j]);
          // for TargetOp's which have IsolatedFromAbove we must align the
          // new additional map operand with an appropriate BlockArgument,
          // as the printing and later processing currently requires a 1:1
          // mapping of BlockArgs to MapInfoOp's at the same placement in
          // each array (BlockArgs and MapOperands).
          if (auto targetOp = llvm::dyn_cast<mlir::omp::TargetOp>(target)) {
            targetOp.getRegion().insertArgument(
                i + j, op.getMembers()[j].getType(), builder.getUnknownLoc());
          }
        }
      }
      newMapOps.push_back(mapOperandsArr[i]);
    }
    mapClauseOwner.getMapOperandsMutable().assign(newMapOps);
  }

  // This pass executes on mlir::ModuleOp's finding omp::MapInfoOp's containing
  // descriptor based types (allocatables, pointers, assumed shape etc.) and
  // expanding them into multiple omp::MapInfoOp's for each pointer member
  // contained within the descriptor.
  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    mlir::ModuleOp module = func->getParentOfType<mlir::ModuleOp>();
    fir::KindMapping kindMap = fir::getKindMapping(module);
    fir::FirOpBuilder builder{module, std::move(kindMap)};

    func->walk([&](mlir::omp::MapInfoOp op) {
      // TODO: Currently only supports a single user for the MapInfoOp, this
      // is fine for the moment as the Fortran frontend will generate a
      // new MapInfoOp per Target operation and clause for the moment.
      // However, when/if we optimise/cleanup the IR, it likely isn't too
      // difficult to extend this function, it would require some
      // modification to create a single new MapInfoOp per new MapInfoOp
      // generated and share it across all users appropriately, making sure
      // to only add a single member link per new generation for the original
      // originating descriptor MapInfoOp.
      assert(llvm::hasSingleElement(op->getUsers()) &&
             "OMPMapInfoFinalization currently only supports single users "
             "of a MapInfoOp");

      if (!op.getMembers().empty()) {
        addImplicitMembersToTarget(op, builder, *op->getUsers().begin());
      } else if (fir::isTypeWithDescriptor(op.getVarType()) ||
                 mlir::isa_and_present<fir::BoxAddrOp>(
                     op.getVarPtr().getDefiningOp())) {
        builder.setInsertionPoint(op);
        genDescriptorMemberMaps(op, builder, *op->getUsers().begin());
      }
    });
  }
};

} // namespace

namespace fir {
std::unique_ptr<mlir::Pass> createOMPMapInfoFinalizationPass() {
  return std::make_unique<OMPMapInfoFinalizationPass>();
}
} // namespace fir
