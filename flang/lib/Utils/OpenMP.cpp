//===-- lib/Utils/OpenMP.cpp ------------------------------------*- C++ -*-===//
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
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIRBoxUtils.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"

#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/RegionUtils.h"

namespace Fortran::utils::openmp {
mlir::omp::MapInfoOp createMapInfoOp(mlir::OpBuilder &builder,
    mlir::Location loc, mlir::Value baseAddr, mlir::Value varPtrPtr,
    llvm::StringRef name, llvm::ArrayRef<mlir::Value> bounds,
    llvm::ArrayRef<mlir::Value> members, mlir::ArrayAttr membersIndex,
    mlir::omp::ClauseMapFlags mapType,
    mlir::omp::VariableCaptureKind mapCaptureType, mlir::Type retTy,
    bool partialMap, mlir::FlatSymbolRefAttr mapperId) {

  auto getPtrVarType = [](mlir::Type ptrType) {
    mlir::TypeAttr varType = mlir::TypeAttr::get(
        llvm::cast<mlir::omp::PointerLikeType>(ptrType).getElementType());

    // For types with unknown extents such as <2x?xi32> we discard the
    // incomplete type info and only retain the base type. The correct
    // dimensions are later recovered through the bounds info.
    if (auto seqType = llvm::dyn_cast<fir::SequenceType>(varType.getValue()))
      if (seqType.hasDynamicExtents())
        varType = mlir::TypeAttr::get(seqType.getEleTy());
    return varType;
  };

  if (auto boxTy = llvm::dyn_cast<fir::BaseBoxType>(baseAddr.getType())) {
    baseAddr = fir::BoxAddrOp::create(builder, loc, baseAddr);
    retTy = baseAddr.getType();
  }

  auto varPtrType = getPtrVarType(retTy);
  auto varPtrPtrTy =
      varPtrPtr ? getPtrVarType(varPtrPtr.getType()) : mlir::TypeAttr{};
  mlir::omp::MapInfoOp op =
      mlir::omp::MapInfoOp::create(builder, loc, retTy, baseAddr, varPtrType,
          builder.getAttr<mlir::omp::ClauseMapFlagsAttr>(mapType),
          builder.getAttr<mlir::omp::VariableCaptureKindAttr>(mapCaptureType),
          varPtrPtr, varPtrPtrTy, members, membersIndex, bounds, mapperId,
          builder.getStringAttr(name), builder.getBoolAttr(partialMap));
  return op;
}

mlir::Value mapTemporaryValue(fir::FirOpBuilder &firOpBuilder,
    mlir::omp::TargetOp targetOp, mlir::Value val, llvm::StringRef name) {
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

void cloneOrMapRegionOutsiders(
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

/// Gets or generates a default declare mapper for a given record type.
///
/// \param firOpBuilder The builder to use for generating the mapper.
/// \param loc The location to use for the generated operations.
/// \param recordType The record type to generate the mapper for.
/// \param mapperNameStr The name of the mapper to generate.
/// \param mangler A function to mangle the mapper name for nested types.
mlir::FlatSymbolRefAttr getOrGenImplicitDefaultDeclareMapper(
    fir::FirOpBuilder &firOpBuilder, mlir::Location loc,
    fir::RecordType recordType, llvm::StringRef mapperNameStr,
    RecordMemberMapperMangler mangler) {
  if (mapperNameStr.empty())
    return {};

  mlir::ModuleOp moduleOp = firOpBuilder.getModule();
  if (moduleOp.lookupSymbol(mapperNameStr))
    return mlir::FlatSymbolRefAttr::get(
        firOpBuilder.getContext(), mapperNameStr);

  mlir::OpBuilder::InsertionGuard guard(firOpBuilder);

  firOpBuilder.setInsertionPointToStart(moduleOp.getBody());
  auto declMapperOp = mlir::omp::DeclareMapperOp::create(
      firOpBuilder, loc, mapperNameStr, recordType);
  auto &region = declMapperOp.getRegion();
  firOpBuilder.createBlock(&region);
  auto mapperArg = region.addArgument(firOpBuilder.getRefType(recordType), loc);

  auto declareOp = hlfir::DeclareOp::create(firOpBuilder, loc, mapperArg,
      /*uniq_name=*/"");

  const auto genBoundsOps = [&](mlir::Value mapVal,
                                llvm::SmallVectorImpl<mlir::Value> &bounds) {
    fir::ExtendedValue extVal = hlfir::translateToExtendedValue(mapVal.getLoc(),
        firOpBuilder, hlfir::Entity{mapVal},
        /*contiguousHint=*/true)
                                    .first;
    fir::factory::AddrAndBoundsInfo info = fir::factory::getDataOperandBaseAddr(
        firOpBuilder, mapVal, /*isOptional=*/false, mapVal.getLoc());
    bounds = fir::factory::genImplicitBoundsOps<mlir::omp::MapBoundsOp,
        mlir::omp::MapBoundsType>(firOpBuilder, info, extVal,
        /*dataExvIsAssumedSize=*/false, mapVal.getLoc());
  };

  const auto getFieldRef = [&](mlir::Value rec, llvm::StringRef fieldName,
                               mlir::Type fieldTy, mlir::Type recType) {
    mlir::Value field = fir::FieldIndexOp::create(firOpBuilder, loc,
        fir::FieldType::get(recType.getContext()), fieldName, recType,
        fir::getTypeParams(rec));
    return fir::CoordinateOp::create(
        firOpBuilder, loc, firOpBuilder.getRefType(fieldTy), rec, field);
  };

  llvm::SmallVector<llvm::SmallVector<int64_t>> memberPlacementIndices;
  llvm::SmallVector<mlir::Value> memberMapOps;

  mlir::omp::ClauseMapFlags mapFlag = mlir::omp::ClauseMapFlags::to |
      mlir::omp::ClauseMapFlags::from | mlir::omp::ClauseMapFlags::implicit;
  mlir::omp::VariableCaptureKind captureKind =
      mlir::omp::VariableCaptureKind::ByRef;

  for (const auto &entry : llvm::enumerate(recordType.getTypeList())) {
    const auto &memberName = entry.value().first;
    const auto &memberType = entry.value().second;

    // OpenMP 5.0, 5.1, 5.2: Default Map Clause
    //
    // "If a component of a derived type list item is a map clause list item
    // that results  from the predefined default mapper for that derived type,
    // and the component is not also an explicit list item or the array base
    // of an explicit list item on the same construct, then: if it has the
    // POINTER attribute, it is attach-INELIGIBLE. If a list item in a map
    // clause is an associated pointer that is attach-ineligible, the effect
    // of the map clause does not apply to its pointer target."
    //
    // What this comes down to is we wish to skip emitting a map inside of
    // the implicit declare mapper generation for pointer components. As well
    // as preventing any nested record types that are pointers from being
    // processed further by the declare mapper infrastructure. The
    // descriptor ("pointer") should be mapped by the containing derived
    // type, this prevents the data ("pointee") from being mapped and
    // processed any further. We should, however, keep an eye on if the
    // record types mapping applying to this descriptor poses issues, or
    // if attach-ineligible still requires an attach map to be emitted
    // alongside the descriptor, even if the pointee has no map emitted.
    // if either case applies, then we will need to emit the maps here and
    // then opt out of base address expansion for these implicit declare
    // mappers in the MapInfoFinalization pass.
    //
    // Notably, this caveat does not apply to allocatables. They get
    // deep-copy semantics.
    if (fir::isPointerType(fir::unwrapRefType(memberType)))
      continue;

    mlir::FlatSymbolRefAttr mapperId;
    if (auto recType = mlir::dyn_cast<fir::RecordType>(
            fir::getFortranElementType(memberType))) {
      std::string mapperIdName =
          recType.getName().str() + llvm::omp::OmpDefaultMapperName;
      mangler(mapperIdName, memberName);
      mapperId = getOrGenImplicitDefaultDeclareMapper(
          firOpBuilder, loc, recType, mapperIdName, mangler);
    }

    auto ref =
        getFieldRef(declareOp.getBase(), memberName, memberType, recordType);
    llvm::SmallVector<mlir::Value> bounds;
    genBoundsOps(ref, bounds);
    mlir::Value mapOp = Fortran::utils::openmp::createMapInfoOp(firOpBuilder,
        loc, ref, /*varPtrPtr=*/mlir::Value{}, /*name=*/"", bounds,
        /*members=*/{},
        /*membersIndex=*/mlir::ArrayAttr{}, mapFlag, captureKind, ref.getType(),
        /*partialMap=*/false, mapperId);
    memberMapOps.emplace_back(mapOp);
    memberPlacementIndices.emplace_back(
        llvm::SmallVector<int64_t>{(int64_t)entry.index()});
  }

  llvm::SmallVector<mlir::Value> bounds;
  genBoundsOps(declareOp.getOriginalBase(), bounds);
  mlir::omp::ClauseMapFlags parentMapFlag = mlir::omp::ClauseMapFlags::implicit;
  mlir::omp::MapInfoOp mapOp = Fortran::utils::openmp::createMapInfoOp(
      firOpBuilder, loc, declareOp.getOriginalBase(),
      /*varPtrPtr=*/mlir::Value(), /*name=*/"", bounds, memberMapOps,
      firOpBuilder.create2DI64ArrayAttr(memberPlacementIndices), parentMapFlag,
      captureKind, declareOp.getType(0),
      /*partialMap=*/true);

  mlir::omp::DeclareMapperInfoOperands clauseOps;
  clauseOps.mapVars.emplace_back(mapOp);
  mlir::omp::DeclareMapperInfoOp::create(firOpBuilder, loc, clauseOps);
  return mlir::FlatSymbolRefAttr::get(firOpBuilder.getContext(), mapperNameStr);
}

LiveInShapeInfo::LiveInShapeInfo(mlir::Value liveIn) {
  mlir::Value shape = nullptr;
  mlir::Operation *liveInDefiningOp = liveIn.getDefiningOp();

  // The live-in may be produced by either declare flavor depending on where
  // in the pipeline the caller runs: DoConcurrentConversion sees
  // `hlfir.declare` (HLFIR still present), LowerWorkdistribute sees
  // `fir.declare` (HLFIR already lowered). Both implement
  // `fir::FortranVariableOpInterface`, which exposes the shape uniformly.
  if (auto varOp = mlir::dyn_cast_if_present<fir::FortranVariableOpInterface>(
          liveInDefiningOp))
    shape = varOp.getShape();

  if (!shape)
    return;

  mlir::Operation *shapeDefiningOp = shape.getDefiningOp();
  auto shapeOp = mlir::dyn_cast_if_present<fir::ShapeOp>(shapeDefiningOp);
  auto shapeShiftOp =
      mlir::dyn_cast_if_present<fir::ShapeShiftOp>(shapeDefiningOp);

  if (!shapeOp && !shapeShiftOp)
    TODO(liveIn.getLoc(),
        "Shapes not defined by `fir.shape` or `fir.shape_shift` op's are"
        "not supported yet.");

  if (shapeShiftOp != nullptr)
    startIndices = shapeShiftOp.getOrigins();

  extents = shapeOp != nullptr
      ? std::vector<mlir::Value>(
            shapeOp.getExtents().begin(), shapeOp.getExtents().end())
      : shapeShiftOp.getExtents();
}

void LiveInShapeInfo::materializeExtents(
    fir::FirOpBuilder &builder, mlir::Value liveIn) {
  // A shaped declare already gave us the extents; nothing to materialize.
  if (isShapedValue())
    return;

  // The live-in is not a shaped declare (e.g. an array-section
  // `hlfir.designate` result), so there is no `fir.shape` to read. Recover the
  // extents from the value's FIR array type so the device-side declare gets a
  // well-formed shape instead of a shapeless one.
  auto seqTy = mlir::dyn_cast_or_null<fir::SequenceType>(
      fir::dyn_cast_ptrOrBoxEleTy(liveIn.getType()));
  if (!seqTy)
    return;

  mlir::Operation *liveInDefiningOp = liveIn.getDefiningOp();
  mlir::OpBuilder::InsertionGuard guard(builder);
  // Materialize extents where the live-in is available so they dominate the
  // target op inserted later at the wrapped loop.
  if (liveInDefiningOp)
    builder.setInsertionPointAfter(liveInDefiningOp);
  else
    builder.setInsertionPointToStart(liveIn.getParentBlock());

  mlir::Location loc = liveIn.getLoc();

  // Boxed arrays carry their (possibly runtime) extents in the descriptor.
  if (mlir::isa<fir::BaseBoxType>(liveIn.getType())) {
    llvm::SmallVector<mlir::Value> boxExtents;
    fir::genDimInfoFromBox(builder, loc, liveIn, /*lbounds=*/nullptr,
        &boxExtents, /*strides=*/nullptr);
    extents.assign(boxExtents.begin(), boxExtents.end());
    return;
  }

  // Raw-address arrays only expose constant extents via the type; dynamic
  // extents are not recoverable from the value alone.
  // TODO: recover dynamic extents (e.g. from the wrapped loop's trip count).
  if (seqTy.hasDynamicExtents() || seqTy.hasUnknownShape())
    return;

  mlir::Type idxTy = builder.getIndexType();
  for (fir::SequenceType::Extent extent : seqTy.getShape())
    extents.push_back(builder.createIntegerConstant(loc, idxTy, extent));
}

namespace {
/// Build implicit `omp.map.bounds` ops describing `liveIn`'s shape.
static void genImplicitMapBounds(fir::FirOpBuilder &builder, mlir::Value liveIn,
    mlir::Value rawAddr, llvm::SmallVectorImpl<mlir::Value> &boundsOps) {
  mlir::Location loc = rawAddr.getLoc();

  // A bare ref/heap/pointer to a dynamically-sized array (e.g. a `dimension(n)`
  // dummy) is not a Fortran entity, so `hlfir::Entity{}` would assert: only a
  // descriptor carries the runtime shape. Recover the extents from the defining
  // declare's shape and describe the value as an `ArrayBoxValue` directly.
  auto seqTy = mlir::dyn_cast_or_null<fir::SequenceType>(
      fir::dyn_cast_ptrOrBoxEleTy(liveIn.getType()));
  bool dynamicRawArray = seqTy && seqTy.hasDynamicExtents() &&
      !mlir::isa<fir::BaseBoxType>(liveIn.getType());

  llvm::SmallVector<mlir::Value> extents;
  if (dynamicRawArray) {
    // The runtime extents of a dynamic raw array come from the defining
    // declare's `fir.shape`, which the pure constructor reads directly; no
    // materialization is needed (or possible) here.
    LiveInShapeInfo shapeInfo(liveIn);
    extents.assign(shapeInfo.extents.begin(), shapeInfo.extents.end());
  }

  fir::ExtendedValue extVal = !extents.empty()
      ? fir::ExtendedValue(fir::ArrayBoxValue(rawAddr, extents))
      : hlfir::translateToExtendedValue(
            loc, builder, hlfir::Entity{liveIn}, /*contiguousHint=*/true)
            .first;
  fir::factory::AddrAndBoundsInfo info = fir::factory::getDataOperandBaseAddr(
      builder, rawAddr, /*isOptional=*/false, loc);
  boundsOps = fir::factory::genImplicitBoundsOps<mlir::omp::MapBoundsOp,
      mlir::omp::MapBoundsType>(builder, info, extVal,
      /*dataExvIsAssumedSize=*/false, loc);
}

/// Reconstruct a declare op inside `targetOp` for the device-side value
/// `liveInArg`, rebuilding the shape via `mapTemporaryValue` so shape extents
/// and origins are themselves mapped into the region. The declare op itself is
/// emitted by `declareBuilder`, letting the caller pick the dialect flavor.
static LiveInDeclareResult genLiveInDeclare(fir::FirOpBuilder &builder,
    mlir::omp::TargetOp targetOp, mlir::Value liveInArg,
    mlir::omp::MapInfoOp liveInMapInfoOp,
    const LiveInShapeInfo &targetShapeCreationInfo,
    LiveInDeclareBuilder declareBuilder) {
  std::string liveInName = liveInMapInfoOp.getName().has_value()
      ? liveInMapInfoOp.getName().value().str()
      : std::string("");

  mlir::Value shape = [&]() -> mlir::Value {
    if (!targetShapeCreationInfo.isShapedValue())
      return {};

    if (targetShapeCreationInfo.isShapeShiftedValue()) {
      llvm::SmallVector<mlir::Value> shapeShiftOperands;

      size_t shapeIdx = 0;
      for (auto [startIndex, extent] :
          llvm::zip_equal(targetShapeCreationInfo.startIndices,
              targetShapeCreationInfo.extents)) {
        shapeShiftOperands.push_back(
            mapTemporaryValue(builder, targetOp, startIndex,
                liveInName + ".start_idx.dim" + std::to_string(shapeIdx)));
        shapeShiftOperands.push_back(mapTemporaryValue(builder, targetOp,
            extent, liveInName + ".extent.dim" + std::to_string(shapeIdx)));
        ++shapeIdx;
      }

      auto shapeShiftType = fir::ShapeShiftType::get(
          builder.getContext(), shapeShiftOperands.size() / 2);
      return fir::ShapeShiftOp::create(
          builder, liveInArg.getLoc(), shapeShiftType, shapeShiftOperands);
    }

    llvm::SmallVector<mlir::Value> shapeOperands;
    size_t shapeIdx = 0;
    for (auto extent : targetShapeCreationInfo.extents) {
      shapeOperands.push_back(mapTemporaryValue(builder, targetOp, extent,
          liveInName + ".extent.dim" + std::to_string(shapeIdx)));
      ++shapeIdx;
    }

    return fir::ShapeOp::create(builder, liveInArg.getLoc(), shapeOperands);
  }();

  return declareBuilder(
      builder, liveInArg.getLoc(), liveInArg, liveInName, shape);
}
} // namespace

mlir::omp::MapInfoOp genMapInfoOpForLiveIn(
    fir::FirOpBuilder &builder, mlir::Value liveIn, bool isReductionVar) {
  mlir::Value rawAddr = liveIn;
  llvm::StringRef name;

  mlir::Operation *liveInDefiningOp = liveIn.getDefiningOp();

  // Recover the raw address and unique name from the defining declare op,
  // supporting both the HLFIR (`hlfir.declare`) and post-lowering FIR
  // (`fir.declare`) flavors. `FortranVariableOpInterface` does not expose the
  // unique name or the un-boxed base, so match the concrete ops.
  if (auto hlfirDeclareOp =
          mlir::dyn_cast_if_present<hlfir::DeclareOp>(liveInDefiningOp)) {
    // Prefer the raw address: it avoids unboxing `fir.box` values when the
    // direct memory reference is available.
    rawAddr = hlfirDeclareOp.getOriginalBase();
    name = hlfirDeclareOp.getUniqName();
  } else if (auto firDeclareOp =
                 mlir::dyn_cast_if_present<fir::DeclareOp>(liveInDefiningOp)) {
    // A `fir.declare` has a single result that already is the memory
    // reference, so it doubles as the raw address.
    rawAddr = firDeclareOp.getResult();
    name = firDeclareOp.getUniqName();
  }

  if (!llvm::isa<mlir::omp::PointerLikeType>(rawAddr.getType())) {
    mlir::OpBuilder::InsertionGuard guard(builder);
    // Block-argument live-ins (e.g. a VALUE scalar dummy) have no defining op;
    // materialize the temp at the top of the owning block instead.
    if (liveInDefiningOp)
      builder.setInsertionPointAfter(liveInDefiningOp);
    else
      builder.setInsertionPointToStart(liveIn.getParentBlock());
    auto copyVal = builder.createTemporary(liveIn.getLoc(), liveIn.getType());
    builder.createStoreWithConvert(copyVal.getLoc(), liveIn, copyVal);
    rawAddr = copyVal;
  }

  mlir::Type liveInType = liveIn.getType();
  mlir::Type eleType = liveInType;
  if (auto refType = mlir::dyn_cast<fir::ReferenceType>(liveInType))
    eleType = refType.getElementType();

  mlir::omp::ClauseMapFlags mapFlag = mlir::omp::ClauseMapFlags::implicit;
  mlir::omp::VariableCaptureKind captureKind =
      mlir::omp::VariableCaptureKind::ByRef;

  if (isReductionVar) {
    mapFlag |= mlir::omp::ClauseMapFlags::to;
    mapFlag |= mlir::omp::ClauseMapFlags::from;
  } else if (fir::isa_trivial(eleType) || fir::isa_char(eleType)) {
    captureKind = mlir::omp::VariableCaptureKind::ByCopy;
  } else if (!fir::isa_builtin_cptr_type(eleType)) {
    mapFlag |= mlir::omp::ClauseMapFlags::to;
    mapFlag |= mlir::omp::ClauseMapFlags::from;
  }

  llvm::SmallVector<mlir::Value> boundsOps;
  genImplicitMapBounds(builder, liveIn, rawAddr, boundsOps);

  auto asRecordType = [&](mlir::Type eleType) {
    return mlir::dyn_cast<fir::RecordType>(
        fir::getDerivedType(fir::unwrapRefType(eleType)));
  };

  fir::RecordType recordType = asRecordType(eleType);

  bool requiresImplicitMapper = [&]() {
    if (!recordType)
      return false;

    for (auto [fieldName, fieldType] : recordType.getTypeList()) {
      if (fir::isAllocatableType(fieldType))
        return true;

      if (asRecordType(fieldType))
        TODO(liveIn.getLoc(), "Nested record types are not supported yet.");
    }

    return false;
  }();

  mlir::FlatSymbolRefAttr mapperId;
  if (requiresImplicitMapper) {
    std::string mapperIdName =
        recordType.getName().str() + llvm::omp::OmpDefaultMapperName;
    // TODO Add a mangler callback once nested record types are supported.
    mapperId = getOrGenImplicitDefaultDeclareMapper(
        builder, liveIn.getLoc(), recordType, mapperIdName);
  }

  return createMapInfoOp(builder, liveIn.getLoc(), rawAddr,
      /*varPtrPtr=*/{}, name.str(), boundsOps,
      /*members=*/{},
      /*membersIndex=*/mlir::ArrayAttr{}, mapFlag, captureKind,
      rawAddr.getType(), /*partialMap=*/false, mapperId);
}

mlir::omp::TargetOp genTargetOpFromLiveIns(mlir::Location loc,
    mlir::RewriterBase &rewriter, mlir::IRMapping &mapper,
    llvm::ArrayRef<mlir::Value> liveIns,
    mlir::omp::TargetExtOperands &clauseOps,
    mlir::omp::LoopNestOperands &loopNestClauseOps,
    const LiveInShapeInfoMap &liveInShapeInfoMap,
    LiveInDeclareBuilder declareBuilder) {
  auto targetOp = mlir::omp::TargetOp::create(rewriter, loc, clauseOps);
  auto argIface = llvm::cast<mlir::omp::BlockArgOpenMPOpInterface>(*targetOp);

  mlir::Region &region = targetOp.getRegion();

  llvm::SmallVector<mlir::Type> regionArgTypes;
  llvm::SmallVector<mlir::Location> regionArgLocs;

  for (auto var : llvm::concat<const mlir::Value>(
           clauseOps.hostEvalVars, clauseOps.mapVars)) {
    regionArgTypes.push_back(var.getType());
    regionArgLocs.push_back(var.getLoc());
  }

  rewriter.createBlock(&region, {}, regionArgTypes, regionArgLocs);
  fir::FirOpBuilder builder(rewriter,
      fir::getKindMapping(targetOp->getParentOfType<mlir::ModuleOp>()));

  // The loop body may surface additional values that need mapping (e.g. shape
  // info for arrays), extending the map block args. Re-query the block args on
  // each iteration so we always see the current vector.
  int idx = 0;
  for (auto [mapInfoOp, mappedVar] :
      llvm::zip_equal(clauseOps.mapVars, liveIns)) {
    auto miOp = mlir::cast<mlir::omp::MapInfoOp>(mapInfoOp.getDefiningOp());
    LiveInDeclareResult liveInDeclare =
        genLiveInDeclare(builder, targetOp, argIface.getMapBlockArgs()[idx],
            miOp, liveInShapeInfoMap.at(mappedVar), declareBuilder);
    ++idx;

    // If `mappedVar.getDefiningOp()` is a `fir::BoxAddrOp`, we likely need to
    // unpack the box by chasing its defining op. Block-arg live-ins have no
    // defining op, so guard against null. Not hit in practice yet.
    if (mlir::isa_and_present<fir::BoxAddrOp>(mappedVar.getDefiningOp()))
      TODO(mappedVar.getLoc(),
          "Mapped variable defined by `BoxAddrOp` are not supported yet");

    auto mapHostValueToDevice = [&](mlir::Value hostValue,
                                    mlir::Value deviceValue) {
      if (!llvm::isa<mlir::omp::PointerLikeType>(hostValue.getType()))
        mapper.map(
            hostValue, builder.loadIfRef(hostValue.getLoc(), deviceValue));
      else
        mapper.map(hostValue, deviceValue);
    };

    mapHostValueToDevice(mappedVar, liveInDeclare.originalBase);

    // A host `hlfir.declare` exposes a distinct "hlfir base" (its second
    // result) beyond the original base; map that to the device-side `base` too.
    // Single-result host declares (e.g. `fir.declare`) have nothing extra here,
    // in which case `originalBase` and `base` alias.
    if (auto origDeclareOp = mlir::dyn_cast_if_present<hlfir::DeclareOp>(
            mappedVar.getDefiningOp()))
      mapHostValueToDevice(origDeclareOp.getBase(), liveInDeclare.base);
  }

  for (auto [arg, hostEval] :
      llvm::zip_equal(argIface.getHostEvalBlockArgs(), clauseOps.hostEvalVars))
    mapper.map(hostEval, arg);

  for (unsigned i = 0; i < loopNestClauseOps.loopLowerBounds.size(); ++i) {
    loopNestClauseOps.loopLowerBounds[i] =
        mapper.lookup(loopNestClauseOps.loopLowerBounds[i]);
    loopNestClauseOps.loopUpperBounds[i] =
        mapper.lookup(loopNestClauseOps.loopUpperBounds[i]);
    loopNestClauseOps.loopSteps[i] =
        mapper.lookup(loopNestClauseOps.loopSteps[i]);
  }

  // Cloning the bounds may have introduced new outer-region dependencies;
  // either clone the MemoryEffectFree ones too, or copy them to a temporary,
  // add them to the map/block-arg lists, and replace their uses with the
  // temporary.
  cloneOrMapRegionOutsiders(builder, targetOp);
  rewriter.setInsertionPoint(
      mlir::omp::TerminatorOp::create(rewriter, targetOp.getLoc()));

  return targetOp;
}
} // namespace Fortran::utils::openmp
