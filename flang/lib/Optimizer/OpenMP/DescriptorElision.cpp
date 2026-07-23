//===- DescriptorElision.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass elides descriptors for array dummy arguments that are mapped into
// an `omp.target` region when the descriptor is provably not required on the
// device. In that case the in-region accesses are rewritten to operate directly
// on the raw base address (an explicit-shape style access), which means the
// descriptor never needs to be mapped to the device.
//
// The following descriptor-carrying array dummies are handled:
//   * plain assumed-shape arrays (`arr(:)`),
//   * allocatable arrays (`allocatable :: arr(:)`) - always contiguous, so
//     always eligible when otherwise unobserved, and
//   * pointer arrays (`pointer :: arr(:)`) - only when the pointer is declared
//     CONTIGUOUS, since a general pointer target may be strided even for a
//     whole-array map.
//
// Assumed-rank array dummies (`arr(..)`) are handled transparently: an
// assumed-rank entity can only be element-indexed inside a SELECT RANK
// construct, and within a RANK(k) branch the compiler resolves the assumed-rank
// descriptor to a concrete rank-k assumed-shape box (`fir.convert` to
// `!fir.box<!fir.array<?x...>>`). It is that concrete-rank box which is mapped,
// so it flows through the plain assumed-shape path above with no special
// handling required here.
//
// Element addressing is rewritten to a direct fir.coordinate_of on the raw base
// pointer using a linear element offset:
//   offset = Σ_d (index_d - lb_d) * stride_d
// where, for each dimension d, the descriptor's actual lower bound (lb_d) and
// element stride (stride_d = byte_stride_d / element_size) are computed on the
// host and passed into the region as lightweight scalar (ByCopy) map entries.
// Using the descriptor's real lower bounds means non-unit dummy lower bounds
// (e.g. arr(0:)) are handled correctly, and using its real strides means the
// addressing is correct for any layout the mapped storage actually has. In all
// cases the descriptor mapping itself is dropped.
//
// In addition to plain element addressing, a limited set of descriptor
// inquiries are supported by passing the relevant descriptor scalars across as
// ByCopy map entries and rewriting the in-region inquiry to use them:
//   * SIZE / LBOUND / UBOUND / SHAPE (fir.box_dims) -> the per-dimension lower
//     bound and extent are computed on the host, mapped ByCopy and substituted
//     for the fir.box_dims results (the stride result must be unused).
//   * rank inquiries (fir.box_rank) -> replaced by the statically known rank.
//   * raw base-address extraction (fir.box_addr) -> replaced by the mapped base
//     address (which is exactly the data we already map), no descriptor needed.
//
// DESCRIPTOR FORMS: the pre-finalization map for a plain assumed-shape dummy
// has `var_ptr(fir.box_addr(box))` and a block argument that is the box itself;
// element accesses hang directly off the in-region hlfir.declare. For
// allocatable/pointer dummies the map instead has `var_ptr(<descriptor slot>)`
// where the slot is a `!fir.ref<!fir.box<...>>`, the block argument is that
// reference, and the region does `hlfir.declare -> fir.load` to recover the box
// before designating/inquiring. This pass handles both forms (see
// `descriptorSlot`).
//
// CONTIGUITY NOTE: the strides mapped to the device are the *host* descriptor's
// strides, which describe the original array's layout. This is valid only when
// the mapped storage on the device has that same layout, i.e. when the mapped
// region is contiguous (a whole-array map, or a column-major array section whose
// leading dimensions span their full extent). A non-contiguous section (e.g. a
// rectangular sub-block arr(rlo:rhi, clo:chi)) is copied to a packed device
// buffer whose layout differs from the host strides, so such maps are rejected
// (see mapSectionIsContiguous). We also refuse to elide when the descriptor
// stride is directly observed (fir.box_dims stride result in use). Pointer
// dummies are only elided when declared CONTIGUOUS.
//
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

  /// The result of analysing the in-region uses of an elision candidate's
  /// descriptor block argument. Populated by analyseRegionUses and consumed by
  /// transformTarget.
  struct RegionUses {
    /// The in-region hlfir.declare that the descriptor block argument feeds.
    hlfir::DeclareOp declareOp;
    /// For the descriptor-slot form (allocatable/pointer dummies), the block
    /// argument is a *reference* to a descriptor, so the region loads the box
    /// from the declared descriptor slot before designating/inquiring on it.
    /// These loads sit between the declare result and the actual consumers and
    /// are erased once their consumers have been rewritten.
    llvm::SmallVector<fir::LoadOp> boxLoads;
    /// Plain element-access designates (rewritten to direct base addressing).
    llvm::SmallVector<hlfir::DesignateOp> designates;
    /// SIZE/LBOUND/UBOUND/SHAPE inquiries (rewritten to mapped scalars). Each
    /// has a statically-constant dimension index and an unused stride result.
    llvm::SmallVector<fir::BoxDimsOp> boxDims;
    /// Rank inquiries (rewritten to the statically known rank constant).
    llvm::SmallVector<fir::BoxRankOp> boxRanks;
    /// Raw base-address extractions (rewritten to the mapped base address).
    llvm::SmallVector<fir::BoxAddrOp> boxAddrs;
  };

  /// Return true if a map carries user-level pointer-attach / reference
  /// qualifiers that make it unsafe to elide. When the user explicitly requests
  /// pointer/pointee reference semantics (ref_ptr / ref_ptee / ref_ptr_ptee) or
  /// any attach behaviour (attach / attach_always / attach_never / attach_auto),
  /// the descriptor (or the pointer/pointee relationship it encodes) is
  /// semantically significant on the device and must not be replaced by a plain
  /// data map. Detecting ref_ptr or ref_ptee also covers the combined
  /// ref_ptr_ptee case. Such maps are left untouched by this pass.
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

  /// Return true if a map bound spans the full extent of its dimension with a
  /// zero origin, i.e. lower_bound == 0 and upper_bound == extent - 1. This is
  /// the pattern the compiler emits for a whole (non-sectioned) dimension. Both
  /// the fully-constant case and the common symbolic case (upper_bound defined
  /// as `extent - 1`) are recognised; anything else is conservatively treated as
  /// not-full.
  static bool isFullExtentBound(mlir::omp::MapBoundsOp b) {
    mlir::Value lb = b.getLowerBound();
    mlir::Value ub = b.getUpperBound();
    mlir::Value ext = b.getExtent();
    if (!lb || !ub || !ext)
      return false;
    std::optional<int64_t> lbC = mlir::getConstantIntValue(lb);
    if (!lbC || *lbC != 0)
      return false;
    std::optional<int64_t> ubC = mlir::getConstantIntValue(ub);
    std::optional<int64_t> extC = mlir::getConstantIntValue(ext);
    if (ubC && extC)
      return *ubC == *extC - 1;
    // Symbolic: recognise upper_bound == extent - 1.
    if (auto sub = ub.getDefiningOp<mlir::arith::SubIOp>()) {
      std::optional<int64_t> one = mlir::getConstantIntValue(sub.getRhs());
      if (one && *one == 1 && sub.getLhs() == ext)
        return true;
    }
    return false;
  }

  /// Determine whether the storage described by a map's bounds is contiguous, so
  /// that linear (explicit-shape) addressing is valid on device. A whole-array
  /// map (no bounds) is contiguous. For a sectioned map in column-major layout,
  /// the section is contiguous iff every dimension *except the last* (the
  /// slowest-varying) spans its full extent; the last dimension may be any
  /// contiguous sub-range. A rectangular sub-block such as a(rlo:rhi, clo:chi)
  /// is therefore non-contiguous and rejected. Any bounds shape we cannot prove
  /// contiguous is conservatively rejected.
  static bool mapSectionIsContiguous(mlir::omp::MapInfoOp mapOp, unsigned rank) {
    mlir::OperandRange bounds = mapOp.getBounds();
    if (bounds.empty())
      return true; // whole-array map
    if (bounds.size() != rank)
      return false; // unexpected shape - be conservative
    // Every dimension except the last must be full-extent.
    for (unsigned d = 0; d + 1 < rank; ++d) {
      auto b = mlir::dyn_cast_or_null<mlir::omp::MapBoundsOp>(
          bounds[d].getDefiningOp());
      if (!b || !isFullExtentBound(b))
        return false;
    }
    return true;
  }

  /// Return the sequence (array) type described by a box type \p boxTy, if the
  /// box is an array descriptor eligible for elision. Handles plain
  /// assumed-shape descriptors (`!fir.box<!fir.array<...>>`) as well as
  /// allocatable (`!fir.box<!fir.heap<!fir.array<...>>>`) and pointer
  /// (`!fir.box<!fir.ptr<!fir.array<...>>>`) descriptors, unwrapping the
  /// intermediate heap/ptr wrapper for the latter two. Polymorphic entities
  /// (fir.class) are rejected.
  static fir::SequenceType getElisionSeqType(mlir::Type boxTy) {
    // Reject polymorphic entities.
    if (mlir::isa<fir::ClassType>(boxTy))
      return {};
    auto baseBoxTy = mlir::dyn_cast<fir::BaseBoxType>(boxTy);
    if (!baseBoxTy)
      return {};
    mlir::Type eleTy = baseBoxTy.getEleTy();
    // Allocatable/pointer descriptors wrap the array in a heap/ptr type.
    if (auto heapTy = mlir::dyn_cast<fir::HeapType>(eleTy))
      eleTy = heapTy.getEleTy();
    else if (auto ptrTy = mlir::dyn_cast<fir::PointerType>(eleTy))
      eleTy = ptrTy.getEleTy();
    return mlir::dyn_cast<fir::SequenceType>(eleTy);
  }

  /// Classify an operation that consumes a descriptor (box) value but is *not*
  /// a supported consumer. Returns a short human-readable reason describing why
  /// the descriptor is observable (and therefore why the transform is unsafe),
  /// or nullptr if the op is unknown/unclassified.
  ///
  /// These are the concrete op forms that user-level descriptor operations
  /// lower to on device. Some of them are now *supported* by the transform
  /// (box_dims / box_rank / box_addr, handled directly in analyseRegionUses)
  /// and so are intentionally omitted here; the remainder genuinely observe the
  /// descriptor metadata and disqualify elision:
  ///   * storage size inquiries                    -> fir.box_elesize
  ///   * ALLOCATED / ASSOCIATED / rank / type      -> fir.box_isalloc,
  ///                                                  fir.box_isptr,
  ///                                                  fir.box_isarray,
  ///                                                  fir.box_tdesc
  ///   * RESHAPE / reassociation / re-descriptor   -> fir.embox, fir.rebox,
  ///                                                  hlfir.reshape,
  ///                                                  hlfir.associate
  ///   * passing the whole array to a procedure    -> fir.call, fir.dispatch
  ///   * copying/reallocating the descriptor       -> fir.store
  /// For allocatable/pointer dummies, an in-region ALLOCATE / DEALLOCATE /
  /// pointer reassociation lowers to a fir.store into the descriptor slot (or a
  /// runtime call), both of which are classified as observing here, so those
  /// cases are correctly rejected. Unknown consumers are treated conservatively
  /// as observing (see analyseRegionUses).
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

  /// Emit an optimization remark explaining why elision was not possible.
  static void remarkCannotElide(mlir::Operation *user, const char *reason) {
    user->emitRemark("OpenMP descriptor elision: descriptor cannot be "
                     "elided because it is observed by ")
        << (reason ? reason : "an unexpected/unsupported operation");
  }

  /// Classify a single use of the box value (either the declare result directly
  /// for the assumed-shape form, or the loaded box for the descriptor-slot
  /// form). On a supported consumer the relevant entry is appended to \p uses
  /// and true is returned; otherwise a remark may be emitted and false is
  /// returned.
  static bool classifyBoxUse(mlir::Operation *user, unsigned rank,
                             RegionUses &uses, bool emitRemarks) {
    if (auto des = mlir::dyn_cast<hlfir::DesignateOp>(user)) {
      // Only plain element indexing is allowed: no component access, no
      // substrings, no vector subscripts, one scalar index per dimension.
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

  /// Analyse all in-region uses of the descriptor block argument. The accepted
  /// patterns are the implicit compiler-generated element-access pattern:
  ///   hlfir.declare(blockArg) -> hlfir.designate(element access)          (or)
  ///   hlfir.declare(blockArg) -> fir.load -> hlfir.designate(element access)
  /// (the latter for the descriptor-slot form) together with a limited set of
  /// descriptor inquiries that can be satisfied by passing descriptor scalars
  /// across ByCopy: fir.box_dims (SIZE/LBOUND/UBOUND/SHAPE), fir.box_rank
  /// (rank), and fir.box_addr (raw base address).
  ///
  /// Any other user-level descriptor query (is_contiguous, element size,
  /// property queries), reshape/rebox/embox, storing the descriptor
  /// (reallocation/reassociation), or passing it to a call makes the descriptor
  /// observable and disqualifies the transform.
  ///
  /// \p descriptorSlot selects the allocatable/pointer form, where the block
  /// argument is a reference to a descriptor and the region loads the box before
  /// using it, so the supported consumers hang off the fir.load results rather
  /// than the declare results directly.
  ///
  /// On success \p uses is fully populated.
  static bool analyseRegionUses(mlir::BlockArgument blockArg, unsigned rank,
                                bool descriptorSlot, RegionUses &uses,
                                bool emitRemarks) {
    uses.declareOp = nullptr;
    // Structural precondition: the raw descriptor block argument must be
    // consumed solely by its hlfir.declare. In HLFIR a dummy's descriptor is
    // always routed through a declare before any use, so this is the expected
    // shape; we still guard it because we are about to *retype* the block
    // argument to a raw base address, which would break any consumer that reads
    // it as a box directly. The substantive descriptor-observation restrictions
    // are enforced on the box value's uses in the loops below - that is where
    // user-level descriptor uses actually appear.
    for (mlir::OpOperand &use : blockArg.getUses()) {
      mlir::Operation *user = use.getOwner();
      auto decl = mlir::dyn_cast<hlfir::DeclareOp>(user);
      if (!decl) {
        const char *reason = descriptorObservingReason(user);
        LLVM_DEBUG({
          llvm::dbgs() << "omp-descriptor-elision: block arg used by "
                       << (reason ? reason : "unexpected op")
                       << ", skipping: " << *user << "\n";
        });
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

    if (descriptorSlot) {
      // Allocatable/pointer form: the declare results are references to the
      // descriptor slot. Every use must be a fir.load recovering the box; the
      // box's own uses are then the supported consumers. Any non-load use (e.g.
      // a fir.store implementing an in-region ALLOCATE/reassociation) makes the
      // descriptor observable and disqualifies the transform.
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

    // Assumed-shape form: the declare results are the box itself, so the
    // supported consumers hang directly off the declare results.
    for (mlir::Value res : uses.declareOp.getResults())
      for (mlir::OpOperand &use : res.getUses())
        if (!classifyBoxUse(use.getOwner(), rank, uses, emitRemarks))
          return false;
    return true;
  }

  /// Redirect a map to the raw base address so that MapInfoFinalization does
  /// not (re-)expand it into a descriptor mapping, and so the raw base address
  /// is what gets mapped.
  ///
  /// For the assumed-shape form the var_ptr is a fir.box_addr already and the
  /// map result type is already the base-address reference, so we simply insert
  /// an identity conversion so the defining op is no longer a box_addr.
  ///
  /// For the descriptor-slot form the var_ptr is the descriptor slot reference
  /// and the map result type is a descriptor reference (`!fir.ref<!fir.box<>>`),
  /// which is wrong once we map raw data. We therefore build a *new*
  /// omp.map.info whose result type, var_ptr and var_type are all the plain
  /// base-address reference (`!fir.ref<seq>`), replace the old map with it in
  /// the enclosing target's map operands, and erase the old map. The matching
  /// region block argument has already been retyped to \p baseAddrTy by the
  /// caller, so the new map, its result type and the block argument are all
  /// consistent.
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

  /// Create a scalar ByCopy `omp.map.info` for \p varPtr (a reference to a
  /// trivial value) and add it to \p target as a new map entry, returning the
  /// corresponding new region block argument.
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
  /// The descriptor block argument is retyped to the raw base address. Element
  /// accesses are rewritten to direct addressing: a linear element offset
  /// (Σ_d (index_d - lb_d) * stride_d) computed from the mapped per-dimension
  /// lower bounds and strides, applied via fir.coordinate_of on the base pointer
  /// viewed as a flattened rank-1 array. Supported descriptor inquiries are
  /// satisfied by mapping the relevant descriptor scalars ByCopy and
  /// substituting them for the inquiry results. Finally the descriptor mapping
  /// itself is redirected to the base address.
  ///
  /// For the descriptor-slot form (\p descriptorSlot) the host box is recovered
  /// by loading the descriptor slot just before the map, the base-address type
  /// is a reference to the array element type, and the in-region fir.load ops
  /// that recovered the box are erased once their consumers are rewritten.
  void transformTarget(fir::FirOpBuilder &builder, mlir::omp::TargetOp target,
                       mlir::omp::MapInfoOp mapOp, mlir::BlockArgument blockArg,
                       mlir::Value box, bool descriptorSlot,
                       fir::SequenceType seqTy, unsigned rank,
                       RegionUses &uses) {
    mlir::Location loc = mapOp.getLoc();
    // For the descriptor-slot form the map result type is a reference to a
    // descriptor, not the base address, so build the base-address type from the
    // element type instead. For the assumed-shape form the map result type is
    // already the base-address reference.
    mlir::Type baseAddrTy = descriptorSlot
                                ? mlir::Type(fir::ReferenceType::get(seqTy))
                                : mapOp.getResult().getType();
    mlir::Type idxTy = builder.getIndexType();
    hlfir::DeclareOp declareOp = uses.declareOp;

    // Host-side alloca slots backing each per-dimension ByCopy scalar map,
    // keyed by dimension. Recorded so that `omp.target` host_eval operands that
    // were derived from the (now elided) descriptor can be rebound to a host
    // load of the *same* slot, keeping the host_eval trip-count value and the
    // mapped scalar in synch (see rebindHostEvalToScalars below).
    llvm::DenseMap<unsigned, mlir::Value> extentSlot, lbSlot, strideSlot;

    // For the descriptor-slot form, recover the host box by loading the
    // descriptor slot just before the map. This dominates every host-side
    // descriptor inquiry we are about to emit.
    if (descriptorSlot) {
      builder.setInsertionPoint(mapOp);
      box = fir::LoadOp::create(builder, loc, mapOp.getVarPtr());
    }

    // --- Host-side: lazily materialize per-dimension descriptor scalars and
    // map them ByCopy. fir.box_dims is only created once per dimension. ---
    llvm::DenseMap<unsigned, fir::BoxDimsOp> hostDims;
    auto getHostDims = [&](unsigned d) -> fir::BoxDimsOp {
      auto it = hostDims.find(d);
      if (it != hostDims.end())
        return it->second;
      builder.setInsertionPoint(mapOp);
      mlir::Value dimC = builder.createIntegerConstant(loc, idxTy, d);
      auto dims = fir::BoxDimsOp::create(builder, loc, idxTy, idxTy, idxTy, box,
                                         dimC);
      hostDims[d] = dims;
      return dims;
    };

    llvm::DenseMap<unsigned, mlir::BlockArgument> extentArg;
    auto getExtentArg = [&](unsigned d) -> mlir::BlockArgument {
      auto it = extentArg.find(d);
      if (it != extentArg.end())
        return it->second;
      fir::BoxDimsOp dims = getHostDims(d);
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
      fir::BoxDimsOp dims = getHostDims(d);
      builder.setInsertionPoint(mapOp);
      mlir::Value slot = fir::AllocaOp::create(builder, loc, idxTy);
      fir::StoreOp::create(builder, loc, dims.getResult(0), slot);
      mlir::BlockArgument arg =
          addScalarByCopyMap(builder, target, slot, "arr.lb");
      lbArg[d] = arg;
      lbSlot[d] = slot;
      return arg;
    };

    // Host element size (in bytes) of the array, materialized once. Used to
    // convert the descriptor's byte strides into element strides.
    mlir::Value hostEleSize;
    auto getEleSize = [&]() -> mlir::Value {
      if (hostEleSize)
        return hostEleSize;
      builder.setInsertionPoint(mapOp);
      hostEleSize = fir::BoxEleSizeOp::create(builder, loc, idxTy, box);
      return hostEleSize;
    };

    // Per-dimension element stride (byte_stride / element_size), mapped ByCopy.
    // This is what makes addressing contiguity-agnostic: we replicate the
    // descriptor's real strides rather than assuming a packed explicit-shape
    // layout. A stride is only ever materialized for a dimension that is
    // actually indexed.
    llvm::DenseMap<unsigned, mlir::BlockArgument> strideArg;
    auto getStrideArg = [&](unsigned d) -> mlir::BlockArgument {
      auto it = strideArg.find(d);
      if (it != strideArg.end())
        return it->second;
      mlir::Value eleSize = getEleSize();
      fir::BoxDimsOp dims = getHostDims(d);
      // Anchor at mapOp so that both `dims` and `eleSize` (which are emitted
      // just before mapOp, in either order) dominate the division.
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

    // --- Region-side: lazily load the mapped scalars once, before the
    // declare so they dominate every in-region use. ---
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

    // The block argument now carries the raw base address instead of a box.
    blockArg.setType(baseAddrTy);

    // --- Rewrite supported descriptor inquiries (must be done before the
    // declare is erased/replaced, since they consume the declare results). ---

    // fir.box_addr -> the mapped base address.
    for (fir::BoxAddrOp ba : uses.boxAddrs) {
      builder.setInsertionPoint(ba);
      mlir::Value repl =
          builder.createConvert(ba.getLoc(), ba.getResult().getType(), blockArg);
      ba.getResult().replaceAllUsesWith(repl);
      ba.erase();
    }

    // fir.box_rank -> statically known rank.
    for (fir::BoxRankOp br : uses.boxRanks) {
      builder.setInsertionPoint(br);
      mlir::Value c = builder.createIntegerConstant(
          br.getLoc(), br.getResult().getType(), rank);
      br.getResult().replaceAllUsesWith(c);
      br.erase();
    }

    // fir.box_dims -> mapped (lower bound, extent). The stride result is known
    // to be unused (checked in analyseRegionUses).
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

    // --- Rewrite element addressing (stride-based, contiguity-agnostic). ---
    // Each element access becomes a direct fir.coordinate_of on the base
    // pointer using a linear element offset computed from the descriptor's real
    // lower bounds and (element) strides:
    //   offset = Σ_d (index_d - lb_d) * stride_d
    // Because the strides come from the descriptor (mapped ByCopy) rather than
    // being synthesised from extents under a packing assumption, this is correct
    // for both contiguous and non-contiguous layouts, and never needs the
    // extents at all (so no extent map entry is emitted for pure addressing).
    mlir::Type eleTy = seqTy.getEleTy();
    mlir::Type refEleTy = fir::ReferenceType::get(eleTy);
    // A single linear offset must index a 1-D array type, so view the base
    // pointer as a flattened rank-1 array of the element type.
    fir::SequenceType flatSeqTy =
        fir::SequenceType::get({fir::SequenceType::getUnknownExtent()}, eleTy);
    mlir::Type flatRefTy = fir::ReferenceType::get(flatSeqTy);
    for (hlfir::DesignateOp des : uses.designates) {
      mlir::Location desLoc = des.getLoc();
      // Materialize the per-dimension lower bounds and strides first: these
      // helpers create loads before the declare and move the builder insertion
      // point, so we must gather them before emitting the address arithmetic
      // (otherwise the arithmetic ops would not dominate / be misplaced).
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

    // Erase the in-region box loads (descriptor-slot form) now that all of
    // their consumers have been rewritten, then the declare itself.
    for (fir::LoadOp load : uses.boxLoads)
      load.erase();
    declareOp.erase();

    // --- Keep omp.target host_eval trip-count operands in synch with the
    // elided descriptor. ---
    // target teams distribute pre-computes its loop trip counts on the host and
    // threads them into the region via `host_eval`. When a loop bound (e.g. an
    // UBOUND used as the inner loop upper bound) is derived from the descriptor
    // we are eliding, its host_eval operand is computed from a `fir.box_dims`
    // of that descriptor. Once the descriptor mapping is dropped, that value is
    // no longer backed by anything the device can recover, and the host_eval
    // channel and the (reshaped) map list fall out of synch - producing a wrong
    // inner-loop bound for >=2 nested loops. To fix this we rebind each such
    // host_eval operand to a host load of the *same* ByCopy scalar slot
    // (lb/extent/stride) that backs the corresponding in-region descriptor
    // quantity, so the trip-count value and the mapped scalar share one source.
    {
      // A fir.box_dims feeding a host_eval trip count often reads a reshaped
      // view of our descriptor rather than the exact SSA value we elided: the
      // compiler routes the dummy through a fir.declare and/or fir.rebox (and
      // possibly fir.convert) before inquiring its bounds. `box` and the
      // box_dims operand may each sit at a *different* point along that
      // declare->rebox->convert chain, so comparing them directly (in either
      // direction) is unreliable. Instead reduce any descriptor value to a
      // canonical root by stripping rebox/convert/declare, and treat two values
      // as the same descriptor when their roots coincide.
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

      // Search v's backward slice for a fir.box_dims reading our descriptor and
      // report which result (0=lb, 1=extent, 2=stride) it consumes.
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

      // Snapshot the host_eval operands up front: rebinding may add new ByCopy
      // scalar maps (via getLbArg/getExtentArg/getStrideArg ->
      // addScalarByCopyMap), which appends to the target's operand storage and
      // would invalidate a live OperandRange over host_eval. We drive the loop
      // from the snapshot and only touch the mutable range for the by-index
      // assignment, whose host_eval segment length is unchanged by map appends.
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
        // Ensure the matching ByCopy scalar (and its host slot) exists, then
        // rebind the host_eval operand to a host load of that slot.
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
      // Never touch maps that participate in a derived-type parent/member
      // mapping structure. A MapInfoOp is required to have at most a single
      // user (the enclosing target/directive); a map that is also referenced
      // by another MapInfoOp is a *member* of a parent structure map, and a
      // map that itself has members is such a parent. Rewriting either would
      // break the parent/member relationship that MapInfoFinalization relies
      // on (and would leave a map with more than one user), so we leave these
      // structure maps entirely alone.
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

      // Never touch maps that carry user-level pointer-attach / reference
      // qualifiers (ref_ptr / ref_ptee / ref_ptr_ptee / attach*). For these the
      // descriptor and its pointer/pointee relationship are semantically
      // significant on the device and must be mapped as-is.
      if (mapHasDisqualifyingFlags(mapOp)) {
        if (emitDescriptorElisionRemarks)
          remarkCannotElide(mapOp, "a user-level ref_ptr/ref_ptee/attach map "
                                   "qualifier");
        continue;
      }

      // Determine the descriptor form and recover the box type.
      //   * assumed-shape: var_ptr is fir.box_addr(box); box value is directly
      //     available.
      //   * allocatable/pointer: var_ptr is a !fir.ref<!fir.box<...>>
      //     descriptor slot; the box is loaded from it (on host in transform).
      bool descriptorSlot = false;
      mlir::Value box;         // assumed-shape: the box SSA value
      mlir::Type boxTy;        // the box type in all cases
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

      bool isAlloc = fir::isAllocatableType(boxTy);
      bool isPtr = fir::isPointerType(boxTy);
      // The descriptor-slot form only ever arises for allocatable/pointer
      // dummies; a plain box in a reference is not something we expect here.
      if (descriptorSlot && !isAlloc && !isPtr)
        continue;

      // Linear (explicit-shape) addressing on device is only valid if the
      // mapped storage is contiguous. A whole-array map is fine; an array
      // section is only contiguous if every dimension except the last spans its
      // full extent (column-major). Reject e.g. a rectangular sub-block
      // a(rlo:rhi, clo:chi), which would otherwise be miscompiled.
      if (!mapSectionIsContiguous(mapOp, rank)) {
        if (emitDescriptorElisionRemarks)
          remarkCannotElide(mapOp, "a non-contiguous array-section map");
        continue;
      }

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

      // Pointer dummies may point at strided storage even for a whole-array
      // map, so the host descriptor's strides would not describe the mapped
      // (packed) device buffer. Only elide when the pointer is declared
      // CONTIGUOUS, which statically guarantees a packed layout. Allocatables
      // are always contiguous and need no such gate.
      if (isPtr) {
        std::optional<fir::FortranVariableFlagsEnum> attrs =
            uses.declareOp.getFortranAttrs();
        bool contiguous =
            attrs && fir::bitEnumContainsAny(
                         *attrs, fir::FortranVariableFlagsEnum::contiguous);
        if (!contiguous) {
          if (emitDescriptorElisionRemarks)
            remarkCannotElide(mapOp,
                              "a POINTER that is not declared CONTIGUOUS");
          continue;
        }
      }

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
  options.enableDescriptorElision = enableDescriptorElision;
  options.emitDescriptorElisionRemarks = emitDescriptorElisionRemarks;
  return std::make_unique<DescriptorElisionPass>(options);
}
