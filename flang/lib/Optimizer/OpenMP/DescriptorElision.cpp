//===- DescriptorElision.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass aims to elide descriptors that are mapped into an `omp.target` region
// as part of an array type (assumed size/rank/shape/allocatables/pointers) when
// the descriptor is proven to be unrequired in the target region or we can
// compartmentalize components of the descriptor to be mapped across in its place.
//
// The intent of the optimization is to gain performance through ideally mapping
// less to the device, but more concretely to simplify the in region accesses to
// the data by bypassing the descriptor.
//
// Not only are the maps rewritten slightly, the in-region accesses are rewritten
// to operate directly on the raw base address, aligning more closely to regular
// array accesses, which means the descriptor never needs to be mapped to the
// device. Except in scenarios where user code implicitly or explicitly demands
// part of the descriptor, e.g. a user invoking ubound or some other Fortran
// intrinsic on the array in a target region. In these cases we do our best effort
// to compartmentalize the descriptor into the necessary constituent components
// to transfer to device, in cases where we cannot do this we try to detect and
// opt out of the optimization.
//
// The pass currently attempts to optimize all descriptor array types, it
// however, will not touch those within derived types. This perhaps could
// be tackled in the future, if it seems like it'd be worth the effort.
//
// It is of note that this optimization ONLY works on contigious arrays, but we
// do not restrict the pass to the CONTIGUOUS keyword in Fortran, so it's up to
// a user to adhere to the contiguity restriction when utilising this pass.
//
// This pass has two options that can be used, one for enabling the pass itself
// and the other for emitting remarks at compile time that attempt to state why
// a particular case could not be optimized when encountered:
//
//  * -fomp-descriptor-elision = enables the pass
//  * -fomp-descriptor-elision-remarks = enables the remark mechanism
//
//
// As far as element addressing rewrites for the array goes we convert
// to a direct fir.coordinate_of on the base pointer using a linear
// element offset:
//
// offset = Σ_d (index_d - lb_d) * stride_d
//
// where, for each dimension d, the descriptor's actual lower bound (lb_d) and
// element stride (stride_d = byte_stride_d / element_size) are computed on the
// host and passed into the region as lightweight scalar (ByCopy) map entries.
//
// In addition to plain element addressing, a limited set of descriptor
// inquiries are supported by passing the relevant descriptor scalars across as
// firstprivate map entries and rewriting the in-region inquiry to use them:
//   * SIZE / LBOUND / UBOUND / SHAPE (fir.box_dims) -> the per-dimension lower
//     bound and extent are computed on the host, mapped ByCopy and substituted
//     for the fir.box_dims results (the stride result must be unused).
//   * rank inquiries (fir.box_rank) -> replaced by the statically known rank.
//   * raw base-address extraction (fir.box_addr) -> replaced by the mapped base
//     address (which is exactly the data we already map), no descriptor needed.
//
// It is of note, that when replacing these intrinsics we have to be careful that
// host_eval does not contain an access to these (and by extent the descriptor we
// are eliding) or any other part of the descriptor, and if it does, we have to
// correct the host_eval to point to the relevant mapped scalar replacement. This
// can occur when a user has utilised ubound or another such intrinsic as a loop
// extent inside of a target parallel teams distribute amongst other cases.
//
// We also avoid touching anything with a user explicit pointer modifier e.g.
// ref_ptr/ptee or attach_always/none. These are cases where a user has intended
// for a specific pointer mapping behaviour to occur. Although, there is still a
// chance this pass causes conflicts with user intent as it doesn't analyze all
// usage of a descriptor (not feasible due to module/function boundaries etc.),
// just the current targets usage.
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/KindMapping.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/OpenMP/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "omp-descriptor-elision"

namespace flangomp {
#define GEN_PASS_DEF_DESCRIPTORELISIONPASS
#include "flang/Optimizer/OpenMP/Passes.h.inc"
} // namespace flangomp

namespace {
class DescriptorElisionPass
    : public flangomp::impl::DescriptorElisionPassBase<DescriptorElisionPass> {
public:
  DescriptorElisionPass() = default;
  DescriptorElisionPass(const flangomp::DescriptorElisionPassOptions &options)
      : DescriptorElisionPassBase(options) {}

  /// Used to store the results of analyzing the in-region uses of an elision
  /// candidate. Effectively recording key usages of the descriptor block
  /// argument inside of the target region so we can perform rewrites.
  struct RegionUses {
    /// In-region hlfir.declare that the descriptor block argument feeds.
    hlfir::DeclareOp declareOp;
    /// In-region loads of the descriptor via hlfir.declare.
    llvm::SmallVector<fir::LoadOp> boxLoads;
    /// Plain element-access designates of the descriptor data.
    llvm::SmallVector<hlfir::DesignateOp> designates;
    /// SIZE/LBOUND/UBOUND/SHAPE inquiries of the descriptor.
    llvm::SmallVector<fir::BoxDimsOp> boxDims;
    /// Rank inquiries of the descriptor.
    llvm::SmallVector<fir::BoxRankOp> boxRanks;
    /// Raw base-address extractions from the descriptor.
    llvm::SmallVector<fir::BoxAddrOp> boxAddrs;
  };

  /// Return true if a map carries user-level pointer-attach / reference
  /// qualifiers that make it unsafe to elide.
  static bool mapHasDisqualifyingFlags(mlir::omp::MapInfoOp mapOp) {
    mlir::omp::ClauseMapFlags flags = mapOp.getMapType();
    return mlir::omp::bitEnumContainsAny(
        flags, mlir::omp::ClauseMapFlags::ref_ptr |
                   mlir::omp::ClauseMapFlags::ref_ptee |
                   mlir::omp::ClauseMapFlags::attach |
                   mlir::omp::ClauseMapFlags::attach_always |
                   mlir::omp::ClauseMapFlags::attach_never |
                   mlir::omp::ClauseMapFlags::attach_auto);
  }

  /// Return the array type described by a box type
  /// TIDYUP: Should be a simpler utility function for this that
  /// FIR already has that we can reuse.
  static fir::SequenceType getElisionSeqType(mlir::Type boxTy) {
    if (mlir::isa<fir::ClassType>(boxTy))
      return {};
    auto baseBoxTy = mlir::dyn_cast<fir::BaseBoxType>(boxTy);
    if (!baseBoxTy)
      return {};
    mlir::Type eleTy = baseBoxTy.getEleTy();
    if (auto heapTy = mlir::dyn_cast<fir::HeapType>(eleTy))
      eleTy = heapTy.getEleTy();
    else if (auto ptrTy = mlir::dyn_cast<fir::PointerType>(eleTy))
      eleTy = ptrTy.getEleTy();
    return mlir::dyn_cast<fir::SequenceType>(eleTy);
  }

  /// Converts an unsupported user operation to a human readable format so
  /// that it can be emitted by the remarks system.
  ///
  /// Some of these we can try to relax over time, some may not be
  /// feasible without altering the users original intent.
  static const char *descriptorObservingReason(mlir::Operation *user) {
    if (mlir::isa<fir::BoxEleSizeOp>(user))
      return "descriptor element-size inquiry";
    if (mlir::isa<fir::BoxIsAllocOp, fir::BoxIsPtrOp, fir::BoxIsArrayOp,
                  fir::BoxTypeDescOp>(user))
      return "descriptor property query (ALLOCATED/ASSOCIATED/rank/type)";
    if (mlir::isa<fir::EmboxOp, fir::ReboxOp, hlfir::ReshapeOp,
                  hlfir::AssociateOp>(user))
      return "descriptor reshape/rebox/embox";
    if (mlir::isa<fir::CallOp, fir::DispatchOp>(user))
      return "descriptor passed to a call";
    if (mlir::isa<fir::StoreOp>(user))
      return "descriptor stored (reallocation/reassociation)";
    return nullptr;
  }

  static void remarkCannotElide(mlir::Operation *user, const char *reason) {
    user->emitRemark("OpenMP descriptor elision: descriptor cannot be "
                     "elided because it is observed by ")
        << (reason ? reason : "an unexpected/unsupported operation");
  }

  /// Verifies the usage of a descriptor fits into a category that we can handle
  /// and (in theory) safely perform elision with. Stores the use in the relevant
  /// field of our RegionUses structure and returns true, or emits a remark and
  /// returns false if we cannot elide because of the usage.
  ///
  /// NOTE: This very likely isn't robust enough to catch all edge cases and
  /// probably elides some cases it shouldn't at the moment.
  static bool classifyBoxUse(mlir::Operation *user, unsigned rank,
                             RegionUses &uses, bool emitRemarks) {
    // Only plain element indexing is allowed for the moment.
    if (auto des = mlir::dyn_cast<hlfir::DesignateOp>(user)) {
      if (des.getComponent())
        return false;
      if (des.getIndices().size() != rank)
        return false;
      if (!des.getSubstring().empty())
        return false;
      uses.designates.push_back(des);
      return true;
    }

    if (auto bd = mlir::dyn_cast<fir::BoxDimsOp>(user)) {
      // The stride result exposes the descriptor's actual layout; if it is
      // consumed we cannot safely assume contiguity, so bail.
      if (!bd.getResult(2).use_empty()) {
        if (emitRemarks)
          remarkCannotElide(user, "descriptor stride inquiry "
                                  "(non-contiguous access)");
        return false;
      }
      // We must know the dimension statically to map the right scalars.
      if (!mlir::getConstantIntValue(bd.getDim())) {
        if (emitRemarks)
          remarkCannotElide(user, "descriptor bounds inquiry with a "
                                  "non-constant dimension");
        return false;
      }
      uses.boxDims.push_back(bd);
      return true;
    }

    if (auto br = mlir::dyn_cast<fir::BoxRankOp>(user)) {
      uses.boxRanks.push_back(br);
      return true;
    }

    if (auto ba = mlir::dyn_cast<fir::BoxAddrOp>(user)) {
      uses.boxAddrs.push_back(ba);
      return true;
    }

    const char *reason = descriptorObservingReason(user);
    if (emitRemarks)
      remarkCannotElide(user, reason);
    return false;
  }

  /// Analyse in-region uses of the descriptor block argument, gathering data for
  /// rewrite by placing it into RegionUses, also verifies if a descriptor can or
  /// cannot be elided through its usage patterns within the target region.
  ///
  /// \p descriptorSlot selects the allocatable/pointer form, where the block
  /// argument is a reference to a descriptor and the region loads the box before
  /// using it, so the supported consumers hang off the fir.load results rather
  /// than the declare results directly.
  static bool analyseRegionUses(mlir::BlockArgument blockArg, unsigned rank,
                                bool descriptorSlot, RegionUses &uses,
                                bool emitRemarks) {
    uses.declareOp = nullptr;
    // Currently we only handle the case where a descriptor block argument is
    // consumed solely by its hlfir.declare.
    /// TIDYUP: There should be a simpler more sane way of doing this, perhaps
    /// just returning false if uses > 1 and that use is not a DeclareOp?
    for (mlir::OpOperand &use : blockArg.getUses()) {
      mlir::Operation *user = use.getOwner();
      auto decl = mlir::dyn_cast<hlfir::DeclareOp>(user);
      if (!decl) {
        const char *reason = descriptorObservingReason(user);
        if (emitRemarks)
          remarkCannotElide(user, reason);
        return false;
      }
      if (uses.declareOp && uses.declareOp != decl)
        return false;
      uses.declareOp = decl;
    }
    if (!uses.declareOp)
      return false;

    // Handles the allocatable/pointer cases where the DeclareOp's results
    // are pre-faced by a load before being utilised by another user. So, for
    // example:
    //
    //  hlfir.declare(blockArg) -> fir.load -> hlfir.designate(element access)
    //
    // Effectively "!fir.ref<!fir.box" cases vs !fir.box.
    /// TIDYUP: Can we get rid of the need for descriptorSlot and collapse
    /// this into one set of loops/case handling for both the allocatable/pointer
    /// and assumed size/shape cases?
    if (descriptorSlot) {
      for (mlir::Value res : uses.declareOp.getResults()) {
        for (mlir::OpOperand &use : res.getUses()) {
          mlir::Operation *user = use.getOwner();
          auto load = mlir::dyn_cast<fir::LoadOp>(user);
          if (!load) {
            const char *reason = descriptorObservingReason(user);
            if (emitRemarks)
              remarkCannotElide(user, reason);
            return false;
          }
          uses.boxLoads.push_back(load);
          for (mlir::OpOperand &boxUse : load.getResult().getUses())
            if (!classifyBoxUse(boxUse.getOwner(), rank, uses, emitRemarks))
              return false;
        }
      }
      return true;
    }

    // Handle assumed-shape/size cases where there's no intermediate load between
    // box users. For example:
    //   hlfir.declare(blockArg) -> hlfir.designate(element access)
    for (mlir::Value res : uses.declareOp.getResults())
      for (mlir::OpOperand &use : res.getUses())
        if (!classifyBoxUse(use.getOwner(), rank, uses, emitRemarks))
          return false;
    return true;
  }

  /// Alter the map to the array base address so that MapInfoFinalization does
  /// not expand it into a descriptor mapping, and so only the array data is
  /// mapped.
  /// TIDYUP: Can perhaps just alter the existing maps relevant fields instead
  /// of creating an entirely new one for the "descriptorSlot" case as well
  void redirectMapToBaseAddress(fir::FirOpBuilder &builder,
                                mlir::omp::MapInfoOp mapOp,
                                mlir::Type baseAddrTy, bool descriptorSlot,
                                mlir::Value box, fir::SequenceType seqTy) {
    if (descriptorSlot) {
      mlir::Location loc = mapOp.getLoc();
      builder.setInsertionPoint(mapOp);
      mlir::Value ba = fir::BoxAddrOp::create(builder, loc, box);
      mlir::Value conv = builder.createConvert(loc, baseAddrTy, ba);
      auto newMap = mlir::omp::MapInfoOp::create(
          builder, loc, /*result=*/baseAddrTy, /*var_ptr=*/conv,
          /*var_type=*/mlir::TypeAttr::get(seqTy),
          /*map_type=*/mapOp.getMapTypeAttr(),
          /*map_capture_type=*/mapOp.getMapCaptureTypeAttr(),
          /*varPtrPtr=*/mlir::Value{}, /*varPtrPtrType=*/mlir::TypeAttr{},
          /*members=*/llvm::SmallVector<mlir::Value>{},
          /*member_index=*/mlir::ArrayAttr{},
          /*bounds=*/mapOp.getBounds(),
          /*mapperId=*/mlir::FlatSymbolRefAttr(),
          /*name=*/mapOp.getNameAttr(),
          /*partialMap=*/builder.getBoolAttr(false));
      mapOp.getResult().replaceAllUsesWith(newMap.getResult());
      mapOp.erase();
      return;
    }
    if (auto boxAddr = mapOp.getVarPtr().getDefiningOp<fir::BoxAddrOp>()) {
      builder.setInsertionPointAfter(boxAddr);
      mlir::Value conv = fir::ConvertOp::create(builder, boxAddr.getLoc(),
                                                baseAddrTy, boxAddr.getResult());
      mapOp.getVarPtrMutable().assign(conv);
    }
  }

  /// Create a scalar ByCopy map and insert it into the target regions block argument
  /// and clause list, effectively creating a firstprivate map for the scalar.
  mlir::BlockArgument addScalarByCopyMap(fir::FirOpBuilder &builder,
                                         mlir::omp::TargetOp target,
                                         mlir::Value varPtr,
                                         llvm::StringRef name) {
    mlir::Location loc = varPtr.getLoc();
    auto refTy = llvm::cast<mlir::omp::PointerLikeType>(varPtr.getType());
    auto mapOp = mlir::omp::MapInfoOp::create(
        builder, loc, varPtr.getType(), varPtr,
        mlir::TypeAttr::get(refTy.getElementType()),
        builder.getAttr<mlir::omp::ClauseMapFlagsAttr>(
            mlir::omp::ClauseMapFlags::to),
        builder.getAttr<mlir::omp::VariableCaptureKindAttr>(
            mlir::omp::VariableCaptureKind::ByCopy),
        /*varPtrPtr=*/mlir::Value{}, /*varPtrPtrType=*/mlir::TypeAttr{},
        /*members=*/llvm::SmallVector<mlir::Value>{},
        /*member_index=*/mlir::ArrayAttr{},
        /*bounds=*/llvm::SmallVector<mlir::Value>{},
        /*mapperId=*/mlir::FlatSymbolRefAttr(),
        /*name=*/builder.getStringAttr(name),
        /*partialMap=*/builder.getBoolAttr(false));

    auto argIface = llvm::cast<mlir::omp::BlockArgOpenMPOpInterface>(*target);
    unsigned insertIndex =
        argIface.getMapBlockArgsStart() + argIface.numMapBlockArgs();
    target.getMapVarsMutable().append(mlir::ValueRange{mapOp});
    return target.getRegion().insertArgument(insertIndex, varPtr.getType(),
                                             loc);
  }

  /// Perform the descriptor elision transform for a single candidate map.
  ///
  /// The descriptor block argument and map is retyped to the array address.
  /// Element accesses are rewritten to direct addressing. Supported descriptor
  /// inquiries are satisfied by mapping the relevant descriptor scalars and
  /// substituting them for the inquiry results. The descriptor mapping
  /// itself is redirected to the base address.
  void transformTarget(fir::FirOpBuilder &builder, mlir::omp::TargetOp target,
                       mlir::omp::MapInfoOp mapOp, mlir::BlockArgument blockArg,
                       mlir::Value box, bool descriptorSlot,
                       fir::SequenceType seqTy, unsigned rank,
                       RegionUses &uses) {
    mlir::Location loc = mapOp.getLoc();
    mlir::Type baseAddrTy = descriptorSlot
                                ? mlir::Type(fir::ReferenceType::get(seqTy))
                                : mapOp.getResult().getType();
    mlir::Type idxTy = builder.getIndexType();
    hlfir::DeclareOp declareOp = uses.declareOp;

    // A dense map recording the scalar values we transfer to device to
    // replace various descriptor components (extent etc.), indexed by
    // dimension. Predominantly used to make sure any host_eval's that
    // were expecting to access the relevant component of the descriptor
    // are capable of being re-aligned to point at the newly materialized
    // scalar field instead, effectively keeping everything in synch.
    //
    /// TIDYUP: Do we actually need to create intermediate allocations for these
    /// scalars, can't we just load and map the relevant elements from the
    /// descriptor directly. It MIGHT be required as it needs to represent
    /// more clearly what the device side data will now be for host_eval
    /// usage. E.g. a scalar access of a scalar, rather than directly going
    /// through the descriptor
    llvm::DenseMap<unsigned, mlir::Value> extentSlot, lbSlot, strideSlot;

    // Recover the box by loading the descriptor just before the map.
    /// TIDYUP: Likely unecessary if we can retrieve the original box
    if (descriptorSlot) {
      builder.setInsertionPoint(mapOp);
      box = fir::LoadOp::create(builder, loc, mapOp.getVarPtr());
    }

    // Host Side Helpers

    /// TIDYUP: Can we refactor these helper functions to be 1-3 functions rather than
    /// what they are at the moment, there's a fair bit of reuse.
    llvm::DenseMap<unsigned, fir::BoxDimsOp> boxDims;
    auto getBoxDims = [&](unsigned d) -> fir::BoxDimsOp {
      auto it = boxDims.find(d);
      if (it != boxDims.end())
        return it->second;
      builder.setInsertionPoint(mapOp);
      mlir::Value dimC = builder.createIntegerConstant(loc, idxTy, d);
      auto dims = fir::BoxDimsOp::create(builder, loc, idxTy, idxTy, idxTy, box,
                                         dimC);
      boxDims[d] = dims;
      return dims;
    };

    llvm::DenseMap<unsigned, mlir::BlockArgument> extentArg;
    auto getExtentArg = [&](unsigned d) -> mlir::BlockArgument {
      auto it = extentArg.find(d);
      if (it != extentArg.end())
        return it->second;
      fir::BoxDimsOp dims = getBoxDims(d);
      builder.setInsertionPoint(mapOp);
      mlir::Value slot = fir::AllocaOp::create(builder, loc, idxTy);
      fir::StoreOp::create(builder, loc, dims.getResult(1), slot);
      mlir::BlockArgument arg =
          addScalarByCopyMap(builder, target, slot, "arr.extent");
      extentArg[d] = arg;
      extentSlot[d] = slot;
      return arg;
    };

    llvm::DenseMap<unsigned, mlir::BlockArgument> lbArg;
    auto getLbArg = [&](unsigned d) -> mlir::BlockArgument {
      auto it = lbArg.find(d);
      if (it != lbArg.end())
        return it->second;
      fir::BoxDimsOp dims = getBoxDims(d);
      builder.setInsertionPoint(mapOp);
      mlir::Value slot = fir::AllocaOp::create(builder, loc, idxTy);
      fir::StoreOp::create(builder, loc, dims.getResult(0), slot);
      mlir::BlockArgument arg =
          addScalarByCopyMap(builder, target, slot, "arr.lb");
      lbArg[d] = arg;
      lbSlot[d] = slot;
      return arg;
    };

    // Element size of the array, materialized once. Used to convert the descriptor's
    // byte strides into element strides.
    mlir::Value hostEleSize;
    auto getEleSize = [&]() -> mlir::Value {
      if (hostEleSize)
        return hostEleSize;
      builder.setInsertionPoint(mapOp);
      hostEleSize = fir::BoxEleSizeOp::create(builder, loc, idxTy, box);
      return hostEleSize;
    };

    // Per-dimension element stride (byte_stride / element_size), mapped to device to
    // aid array indexing.
    llvm::DenseMap<unsigned, mlir::BlockArgument> strideArg;
    auto getStrideArg = [&](unsigned d) -> mlir::BlockArgument {
      auto it = strideArg.find(d);
      if (it != strideArg.end())
        return it->second;
      mlir::Value eleSize = getEleSize();
      fir::BoxDimsOp dims = getBoxDims(d);
      /// TIDYUP: May be better to place this insertion point above getBoxDims/EleSize
      builder.setInsertionPoint(mapOp);
      mlir::Value eltStride = mlir::arith::DivUIOp::create(
          builder, loc, dims.getResult(2), eleSize);
      mlir::Value slot = fir::AllocaOp::create(builder, loc, idxTy);
      fir::StoreOp::create(builder, loc, eltStride, slot);
      mlir::BlockArgument arg =
          addScalarByCopyMap(builder, target, slot, "arr.stride");
      strideArg[d] = arg;
      strideSlot[d] = slot;
      return arg;
    };

    // Target Region Helpers

    llvm::DenseMap<unsigned, mlir::Value> extentVal;
    auto getExtentVal = [&](unsigned d) -> mlir::Value {
      auto it = extentVal.find(d);
      if (it != extentVal.end())
        return it->second;
      mlir::BlockArgument arg = getExtentArg(d);
      builder.setInsertionPoint(declareOp);
      mlir::Value v = fir::LoadOp::create(builder, loc, arg);
      extentVal[d] = v;
      return v;
    };

    llvm::DenseMap<unsigned, mlir::Value> lbVal;
    auto getLbVal = [&](unsigned d) -> mlir::Value {
      auto it = lbVal.find(d);
      if (it != lbVal.end())
        return it->second;
      mlir::BlockArgument arg = getLbArg(d);
      builder.setInsertionPoint(declareOp);
      mlir::Value v = fir::LoadOp::create(builder, loc, arg);
      lbVal[d] = v;
      return v;
    };

    llvm::DenseMap<unsigned, mlir::Value> strideVal;
    auto getStrideVal = [&](unsigned d) -> mlir::Value {
      auto it = strideVal.find(d);
      if (it != strideVal.end())
        return it->second;
      mlir::BlockArgument arg = getStrideArg(d);
      builder.setInsertionPoint(declareOp);
      mlir::Value v = fir::LoadOp::create(builder, loc, arg);
      strideVal[d] = v;
      return v;
    };


    // Step 1: Rewrite various descriptor using operations prior to modifying our DeclareOp
    blockArg.setType(baseAddrTy);
    for (fir::BoxAddrOp ba : uses.boxAddrs) {
      builder.setInsertionPoint(ba);
      mlir::Value repl =
          builder.createConvert(ba.getLoc(), ba.getResult().getType(), blockArg);
      ba.getResult().replaceAllUsesWith(repl);
      ba.erase();
    }

    for (fir::BoxRankOp br : uses.boxRanks) {
      builder.setInsertionPoint(br);
      mlir::Value c = builder.createIntegerConstant(
          br.getLoc(), br.getResult().getType(), rank);
      br.getResult().replaceAllUsesWith(c);
      br.erase();
    }

    for (fir::BoxDimsOp bd : uses.boxDims) {
      std::optional<int64_t> dimC = mlir::getConstantIntValue(bd.getDim());
      assert(dimC && "box_dims dimension must be constant (checked earlier)");
      unsigned d = static_cast<unsigned>(*dimC);
      mlir::Value lb = getLbVal(d);
      mlir::Value ext = getExtentVal(d);
      builder.setInsertionPoint(bd);
      if (!bd.getResult(0).use_empty()) {
        mlir::Value lbConv =
            builder.createConvert(bd.getLoc(), bd.getResult(0).getType(), lb);
        bd.getResult(0).replaceAllUsesWith(lbConv);
      }
      if (!bd.getResult(1).use_empty()) {
        mlir::Value extConv =
            builder.createConvert(bd.getLoc(), bd.getResult(1).getType(), ext);
        bd.getResult(1).replaceAllUsesWith(extConv);
      }
      bd.erase();
    }

    // Step 2: Rewrite the the array accesses to directly index on the array
    // using linear indexing

    mlir::Type eleTy = seqTy.getEleTy();
    mlir::Type refEleTy = fir::ReferenceType::get(eleTy);
    fir::SequenceType flatSeqTy =
        fir::SequenceType::get({fir::SequenceType::getUnknownExtent()}, eleTy);
    mlir::Type flatRefTy = fir::ReferenceType::get(flatSeqTy);
    for (hlfir::DesignateOp des : uses.designates) {
      mlir::Location desLoc = des.getLoc();
      llvm::SmallVector<mlir::Value> lbs, strides;
      for (unsigned d = 0; d < rank; ++d) {
        lbs.push_back(getLbVal(d));
        strides.push_back(getStrideVal(d));
      }
      builder.setInsertionPoint(des);
      mlir::Value flatBase = builder.createConvert(desLoc, flatRefTy, blockArg);
      mlir::Value offset = builder.createIntegerConstant(desLoc, idxTy, 0);
      for (unsigned d = 0; d < rank; ++d) {
        mlir::Value idx =
            builder.createConvert(desLoc, idxTy, des.getIndices()[d]);
        mlir::Value lb = builder.createConvert(desLoc, idxTy, lbs[d]);
        mlir::Value zeroBased =
            mlir::arith::SubIOp::create(builder, desLoc, idx, lb);
        mlir::Value term =
            mlir::arith::MulIOp::create(builder, desLoc, zeroBased, strides[d]);
        offset = mlir::arith::AddIOp::create(builder, desLoc, offset, term);
      }
      auto coord = fir::CoordinateOp::create(
          builder, desLoc, refEleTy, flatBase, mlir::ValueRange{offset});
      des.getResult().replaceAllUsesWith(coord.getResult());
      des.erase();
    }

    // We can in theory delete all loads now that we have rewritten all users, if we haven't
    // then we have let a case we do not handle through, which should either be fixed or
    // protected against by rejecting the transformation.
    for (fir::LoadOp load : uses.boxLoads)
      load.erase();
    declareOp.erase();

    // Step 3: Rewrite relevant host_eval uses to point to the new scalar
    // replacement maps rather than the now defunct descriptor map. This
    // is relevant when loop descriptors such as trip count are tied to the
    // descriptor and now need to be directed to the new source of truth.

    // There can be a rather complicated use chain we need to process to find
    // our desired root descriptor/the source of the descriptor access, this
    // function tries to find the canonical root. We effectively want to
    // utilise this to make sure the Box we are processing is the Box the
    // host_eval is accessing.
    auto rootOf = [&](mlir::Value v) -> mlir::Value {
      llvm::DenseSet<mlir::Value> seen;
      while (v && seen.insert(v).second) {
        mlir::Operation *def = v.getDefiningOp();
        if (auto rb = mlir::dyn_cast_or_null<fir::ReboxOp>(def))
          v = rb.getBox();
        else if (auto cv = mlir::dyn_cast_or_null<fir::ConvertOp>(def))
          v = cv.getValue();
        else if (auto hd = mlir::dyn_cast_or_null<hlfir::DeclareOp>(def))
          v = hd.getMemref();
        else if (auto fd = mlir::dyn_cast_or_null<fir::DeclareOp>(def))
          v = fd.getMemref();
        else
          break;
      }
      return v;
    };
    mlir::Value boxRoot = rootOf(box);
    auto resolvesToBox = [&](mlir::Value v) -> bool {
      return boxRoot && rootOf(v) == boxRoot;
    };

    // Search the values backward slice for a fir.box_dims reading our descriptor
    // and verify which result it uses.
    /// TIDYUP: There's a backwards slice utility in MLIR that we could utilise
    /// instead of defining our own here.
    auto findBoxDimLeaf = [&](mlir::Value v, fir::BoxDimsOp &outBd,
                              unsigned &outIdx) -> bool {
      llvm::SmallVector<mlir::Value> wl{v};
      llvm::DenseSet<mlir::Value> seen;
      while (!wl.empty()) {
        mlir::Value cur = wl.pop_back_val();
        if (!seen.insert(cur).second)
          continue;
        mlir::Operation *def = cur.getDefiningOp();
        if (!def)
          continue;
        if (auto bd = mlir::dyn_cast<fir::BoxDimsOp>(def)) {
          if (resolvesToBox(bd.getVal()))
            for (unsigned i = 0; i < 3; ++i)
              if (bd.getResult(i) == cur) {
                outBd = bd;
                outIdx = i;
                return true;
              }
          continue;
        }
        for (mlir::Value o : def->getOperands())
          wl.push_back(o);
      }
      return false;
    };

    llvm::SmallVector<mlir::Value> hostEval(target.getHostEvalVars().begin(),
                                            target.getHostEvalVars().end());
    for (unsigned i = 0, e = hostEval.size(); i < e; ++i) {
      fir::BoxDimsOp bd;
      unsigned idx = 0;
      if (!findBoxDimLeaf(hostEval[i], bd, idx))
        continue;
      std::optional<int64_t> dC = mlir::getConstantIntValue(bd.getDim());
      if (!dC)
        continue;
      unsigned d = static_cast<unsigned>(*dC);

      mlir::Value slot;
      if (idx == 0) {
        getLbArg(d);
        slot = lbSlot[d];
      } else if (idx == 1) {
        getExtentArg(d);
        slot = extentSlot[d];
      } else {
        getStrideArg(d);
        slot = strideSlot[d];
      }
      if (!slot)
        continue;
      builder.setInsertionPoint(target);
      mlir::Value ld = fir::LoadOp::create(builder, loc, slot);
      mlir::Value conv = builder.createConvert(loc, hostEval[i].getType(), ld);
      target.getHostEvalVarsMutable().slice(i, 1).assign(conv);
    }

    redirectMapToBaseAddress(builder, mapOp, baseAddrTy, descriptorSlot, box,
                             seqTy);
  }

  void processTarget(fir::FirOpBuilder &builder, mlir::omp::TargetOp target) {
    auto mapClauseOwner =
        llvm::dyn_cast<mlir::omp::MapClauseOwningOpInterface>(*target);
    auto argIface =
        llvm::dyn_cast<mlir::omp::BlockArgOpenMPOpInterface>(*target);
    if (!mapClauseOwner || !argIface)
      return;

    llvm::SmallVector<mlir::omp::MapInfoOp> candidates;
    for (mlir::Value mapVar : mapClauseOwner.getMapVars())
      if (auto mapOp = mapVar.getDefiningOp<mlir::omp::MapInfoOp>())
        candidates.push_back(mapOp);

    for (mlir::omp::MapInfoOp mapOp : candidates) {
      if (!mapOp.getMembers().empty()) {
        if (emitDescriptorElisionRemarks)
          remarkCannotElide(mapOp, "a derived-type structure (parent) map");
        continue;
      }

      if (llvm::any_of(mapOp->getUsers(), [](mlir::Operation *user) {
            return mlir::isa<mlir::omp::MapInfoOp>(user);
          })) {
        if (emitDescriptorElisionRemarks)
          remarkCannotElide(mapOp, "a derived-type component (member) map");
        continue;
      }

      if (mapHasDisqualifyingFlags(mapOp)) {
        if (emitDescriptorElisionRemarks)
          remarkCannotElide(mapOp, "a user-level ref_ptr/ref_ptee/attach map "
                                   "qualifier");
        continue;
      }

      // Determine the descriptor form and recover the box type, this will change the
      // passes handling in certain cases.
      //  - assumed-shape/size: var_ptr is fir.box_addr(box); box value is directly
      //     available.
      //  - allocatable/pointer: var_ptr is a !fir.ref<!fir.box<...>>
      bool descriptorSlot = false;
      mlir::Value box;
      mlir::Type boxTy;
      if (auto boxAddr = mapOp.getVarPtr().getDefiningOp<fir::BoxAddrOp>()) {
        box = boxAddr.getVal();
        boxTy = box.getType();
      } else if (auto refTy = mlir::dyn_cast<fir::ReferenceType>(
                     mapOp.getVarPtr().getType())) {
        auto bt = mlir::dyn_cast<fir::BaseBoxType>(refTy.getEleTy());
        if (!bt)
          continue;
        descriptorSlot = true;
        boxTy = bt;
      } else {
        continue;
      }

      fir::SequenceType seqTy = getElisionSeqType(boxTy);
      if (!seqTy)
        continue;
      unsigned rank = seqTy.getDimension();
      if (rank == 0)
        continue;

      // If it's not a map clause then skip it e.g. use_device_ptr etc.
      int64_t mapVarIdx = mapClauseOwner.getOperandIndexForMap(mapOp);
      if (mapVarIdx < 0 ||
          mapVarIdx >= static_cast<int64_t>(argIface.getMapBlockArgs().size()))
        continue;
      mlir::BlockArgument blockArg = argIface.getMapBlockArgs()[mapVarIdx];

      RegionUses uses;
      if (!analyseRegionUses(blockArg, rank, descriptorSlot, uses,
                             emitDescriptorElisionRemarks))
        continue;

      transformTarget(builder, target, mapOp, blockArg, box, descriptorSlot,
                      seqTy, rank, uses);
    }
  }

  void runOnOperation() override {
    // if (!enableDescriptorElision)
    //   return;

    mlir::ModuleOp module = mlir::cast<mlir::ModuleOp>(getOperation());
    fir::KindMapping kindMap = fir::getKindMapping(module);
    fir::FirOpBuilder builder{module, std::move(kindMap)};

    module.walk(
        [&](mlir::omp::TargetOp target) { processTarget(builder, target); });
  }
};
} // namespace

std::unique_ptr<mlir::Pass>
flangomp::createDescriptorElisionPass(bool enableDescriptorElision,
                                      bool emitDescriptorElisionRemarks) {
  DescriptorElisionPassOptions options;
  options.enableDescriptorElision = true;//enableDescriptorElision;
  options.emitDescriptorElisionRemarks = true;//emitDescriptorElisionRemarks;
  return std::make_unique<DescriptorElisionPass>(options);
}
