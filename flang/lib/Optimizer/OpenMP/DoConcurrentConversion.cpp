//===- DoConcurrentConversion.cpp -- map `DO CONCURRENT` to OpenMP loops --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/DirectivesCommon.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/HLFIRTools.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/OpenMP/Passes.h"
#include "flang/Optimizer/OpenMP/Utils.h"
#include "flang/Support/OpenMP-utils.h"
#include "flang/Utils/OpenMP.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"

namespace flangomp {
#define GEN_PASS_DEF_DOCONCURRENTCONVERSIONPASS
#include "flang/Optimizer/OpenMP/Passes.h.inc"
} // namespace flangomp

#define DEBUG_TYPE "do-concurrent-conversion"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

namespace {
namespace looputils {
/// Stores info needed about the induction/iteration variable for each `do
/// concurrent` in a loop nest.
struct InductionVariableInfo {
  InductionVariableInfo(fir::DoConcurrentLoopOp loop,
                        mlir::Value inductionVar) {
    populateInfo(loop, inductionVar);
  }
  /// The operation allocating memory for iteration variable.
  mlir::Operation *iterVarMemDef;
  /// the operation(s) updating the iteration variable with the current
  /// iteration number.
  llvm::SmallVector<mlir::Operation *, 2> indVarUpdateOps;

private:
  /// For the \p doLoop parameter, find the following:
  ///
  /// 1. The operation that declares its iteration variable or allocates memory
  /// for it. For example, give the following loop:
  /// ```
  ///   ...
  ///   %i:2 = hlfir.declare %0 {uniq_name = "_QFEi"} : ...
  ///   ...
  ///   fir.do_concurrent.loop (%ind_var) = (%lb) to (%ub) step (%s) {
  ///     %ind_var_conv = fir.convert %ind_var : (index) -> i32
  ///     fir.store %ind_var_conv to %i#1 : !fir.ref<i32>
  ///     ...
  ///   }
  /// ```
  ///
  /// This function sets the `iterVarMemDef` member to the `hlfir.declare` op
  /// for `%i`.
  ///
  /// 2. The operation(s) that update the loop's iteration variable from its
  /// induction variable. For the above example, the `indVarUpdateOps` is
  /// populated with the first 2 ops in the loop's body.
  ///
  /// Note: The current implementation is dependent on how flang emits loop
  /// bodies; which is sufficient for the current simple test/use cases. If this
  /// proves to be insufficient, this should be made more generic.
  void populateInfo(fir::DoConcurrentLoopOp loop, mlir::Value inductionVar) {
    mlir::Value result = nullptr;

    // Checks if a StoreOp is updating the memref of the loop's iteration
    // variable.
    auto isStoringIV = [&](fir::StoreOp storeOp) {
      // Direct store into the IV memref.
      if (storeOp.getValue() == inductionVar) {
        indVarUpdateOps.push_back(storeOp);
        return true;
      }

      // Indirect store into the IV memref.
      if (auto convertOp = mlir::dyn_cast<fir::ConvertOp>(
              storeOp.getValue().getDefiningOp())) {
        if (convertOp.getOperand() == inductionVar) {
          indVarUpdateOps.push_back(convertOp);
          indVarUpdateOps.push_back(storeOp);
          return true;
        }
      }

      return false;
    };

    for (mlir::Operation &op : loop) {
      if (auto storeOp = mlir::dyn_cast<fir::StoreOp>(op))
        if (isStoringIV(storeOp)) {
          result = storeOp.getMemref();
          break;
        }
    }

    assert(result != nullptr && result.getDefiningOp() != nullptr);
    iterVarMemDef = result.getDefiningOp();
  }
};

using InductionVariableInfos = llvm::SmallVector<InductionVariableInfo>;

/// Collect the list of values used inside the loop but defined outside of it.
void collectLoopLiveIns(fir::DoConcurrentLoopOp loop,
                        llvm::SmallVectorImpl<mlir::Value> &liveIns) {
  llvm::SmallDenseSet<mlir::Value> seenValues;
  llvm::SmallPtrSet<mlir::Operation *, 8> seenOps;

  for (auto [lb, ub, st] : llvm::zip_equal(
           loop.getLowerBound(), loop.getUpperBound(), loop.getStep())) {
    liveIns.push_back(lb);
    liveIns.push_back(ub);
    liveIns.push_back(st);
  }

  mlir::visitUsedValuesDefinedAbove(
      loop.getRegion(), [&](mlir::OpOperand *operand) {
        if (!seenValues.insert(operand->get()).second)
          return;

        mlir::Operation *definingOp = operand->get().getDefiningOp();
        // We want to collect ops corresponding to live-ins only once.
        if (definingOp && !seenOps.insert(definingOp).second)
          return;

        liveIns.push_back(operand->get());
      });

  for (mlir::Value local : loop.getLocalVars())
    liveIns.push_back(local);

  for (mlir::Value reduce : loop.getReduceVars())
    liveIns.push_back(reduce);
}

/// Collects values that are local to a loop: "loop-local values". A loop-local
/// value is one that is used exclusively inside the loop but allocated outside
/// of it. This usually corresponds to temporary values that are used inside the
/// loop body for initialzing other variables for example.
///
/// See `flang/test/Transforms/DoConcurrent/locally_destroyed_temp.f90` for an
/// example of why we need this.
///
/// \param [in] doLoop - the loop within which the function searches for values
/// used exclusively inside.
///
/// \param [out] locals - the list of loop-local values detected for \p doLoop.
void collectLoopLocalValues(fir::DoConcurrentLoopOp loop,
                            llvm::SetVector<mlir::Value> &locals) {
  loop.walk([&](mlir::Operation *op) {
    for (mlir::Value operand : op->getOperands()) {
      if (locals.contains(operand))
        continue;

      bool isLocal = true;

      if (!mlir::isa_and_present<fir::AllocaOp>(operand.getDefiningOp()))
        continue;

      // Values defined inside the loop are not interesting since they do not
      // need to be localized.
      if (loop->isAncestor(operand.getDefiningOp()))
        continue;

      for (auto *user : operand.getUsers()) {
        if (!loop->isAncestor(user)) {
          isLocal = false;
          break;
        }
      }

      if (isLocal)
        locals.insert(operand);
    }
  });
}

/// For a "loop-local" value \p local within a loop's scope, localizes that
/// value within the scope of the parallel region the loop maps to. Towards that
/// end, this function moves the allocation of \p local within \p allocRegion.
///
/// \param local - the value used exclusively within a loop's scope (see
/// collectLoopLocalValues).
///
/// \param allocRegion - the parallel region where \p local's allocation will be
/// privatized.
///
/// \param rewriter - builder used for updating \p allocRegion.
static void localizeLoopLocalValue(mlir::Value local, mlir::Region &allocRegion,
                                   mlir::ConversionPatternRewriter &rewriter) {
  rewriter.moveOpBefore(local.getDefiningOp(), &allocRegion.front().front());
}
} // namespace looputils

class DoConcurrentConversion
    : public mlir::OpConversionPattern<fir::DoConcurrentOp> {
public:
  using mlir::OpConversionPattern<fir::DoConcurrentOp>::OpConversionPattern;

  DoConcurrentConversion(
      mlir::MLIRContext *context, bool mapToDevice,
      llvm::DenseSet<fir::DoConcurrentOp> &concurrentLoopsToSkip,
      mlir::SymbolTable &moduleSymbolTable)
      : OpConversionPattern(context), mapToDevice(mapToDevice),
        concurrentLoopsToSkip(concurrentLoopsToSkip),
        moduleSymbolTable(moduleSymbolTable) {}

  mlir::LogicalResult
  matchAndRewrite(fir::DoConcurrentOp doLoop, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    looputils::InductionVariableInfos ivInfos;
    auto loop = mlir::cast<fir::DoConcurrentLoopOp>(
        doLoop.getRegion().back().getTerminator());

    auto indVars = loop.getLoopInductionVars();
    assert(indVars.has_value());

    for (mlir::Value indVar : *indVars)
      ivInfos.emplace_back(loop, indVar);

    llvm::SmallVector<mlir::Value> loopNestLiveIns;
    looputils::collectLoopLiveIns(loop, loopNestLiveIns);
    assert(!loopNestLiveIns.empty());

    llvm::SetVector<mlir::Value> locals;
    looputils::collectLoopLocalValues(loop, locals);

    // We do not want to map "loop-local" values to the device through
    // `omp.map.info` ops. Therefore, we remove them from the list of live-ins.
    loopNestLiveIns.erase(llvm::remove_if(loopNestLiveIns,
                                          [&](mlir::Value liveIn) {
                                            return locals.contains(liveIn);
                                          }),
                          loopNestLiveIns.end());

    mlir::omp::TargetOp targetOp;
    mlir::omp::LoopNestOperands loopNestClauseOps;

    mlir::IRMapping mapper;

    if (mapToDevice) {
      mlir::ModuleOp module = doLoop->getParentOfType<mlir::ModuleOp>();
      bool isTargetDevice =
          llvm::cast<mlir::omp::OffloadModuleInterface>(*module)
              .getIsTargetDevice();

      mlir::omp::TargetExtOperands targetClauseOps;
      genLoopNestClauseOps(doLoop.getLoc(), rewriter, loop, loopNestClauseOps,
                           isTargetDevice ? nullptr : &targetClauseOps);

      Fortran::utils::openmp::LiveInShapeInfoMap liveInShapeInfoMap;
      fir::FirOpBuilder builder(
          rewriter,
          fir::getKindMapping(doLoop->getParentOfType<mlir::ModuleOp>()));

      for (mlir::Value liveIn : loopNestLiveIns) {
        bool isReductionVar = llvm::find(loop.getReduceVars(), liveIn) !=
                              loop.getReduceVars().end();
        targetClauseOps.mapVars.push_back(
            Fortran::utils::openmp::genMapInfoOpForLiveIn(builder, liveIn,
                                                          isReductionVar));
        Fortran::utils::openmp::LiveInShapeInfo liveInShape(liveIn);
        liveInShape.materializeExtents(builder, liveIn);
        liveInShapeInfoMap.insert({liveIn, liveInShape});
      }

      targetClauseOps.kernelType = mlir::omp::TargetExecModeAttr::get(
          rewriter.getContext(), mlir::omp::TargetExecMode::spmd);
      targetOp = Fortran::utils::openmp::genTargetOpFromLiveIns(
          doLoop.getLoc(), rewriter, mapper, loopNestLiveIns, targetClauseOps,
          loopNestClauseOps, liveInShapeInfoMap,
          [](fir::FirOpBuilder &builder, mlir::Location loc,
             mlir::Value liveInArg, llvm::StringRef name,
             mlir::Value shape) -> Fortran::utils::openmp::LiveInDeclareResult {
            auto declareOp =
                hlfir::DeclareOp::create(builder, loc, liveInArg, name, shape);
            return {declareOp.getOriginalBase(), declareOp.getBase()};
          });
      auto teamsOp = genTeamsOp(rewriter, loop, mapper);
      targetOp.setCombined(true);
      teamsOp.setCombined(true);
    }

    mlir::omp::ParallelOp parallelOp =
        genParallelOp(rewriter, loop, ivInfos, mapper);

    // Only set as composite when part of `distribute parallel do`, and only set
    // as combined when part of `parallel do`.
    parallelOp.setComposite(mapToDevice);
    parallelOp.setCombined(!mapToDevice);

    if (!mapToDevice)
      genLoopNestClauseOps(doLoop.getLoc(), rewriter, loop, loopNestClauseOps);

    for (mlir::Value local : locals)
      looputils::localizeLoopLocalValue(local, parallelOp.getRegion(),
                                        rewriter);

    if (mapToDevice)
      genDistributeOp(doLoop.getLoc(), rewriter).setComposite(/*val=*/true);

    auto [loopNestOp, wsLoopOp] =
        genWsLoopOp(rewriter, loop, mapper, loopNestClauseOps,
                    /*isComposite=*/mapToDevice);

    // `local` region arguments are transferred/cloned from the `do concurrent`
    // loop to the loopnest op when the region is cloned above. Instead, these
    // region arguments should be on the workshare loop's region.
    if (mapToDevice) {
      for (auto [parallelArg, loopNestArg] : llvm::zip_equal(
               parallelOp.getRegion().getArguments(),
               loopNestOp.getRegion().getArguments().slice(
                   loop.getLocalOperandsStart(), loop.getNumLocalOperands())))
        rewriter.replaceAllUsesWith(loopNestArg, parallelArg);

      for (auto [wsloopArg, loopNestArg] : llvm::zip_equal(
               wsLoopOp.getRegion().getArguments(),
               loopNestOp.getRegion().getArguments().slice(
                   loop.getReduceOperandsStart(), loop.getNumReduceOperands())))
        rewriter.replaceAllUsesWith(loopNestArg, wsloopArg);
    } else {
      for (auto [wsloopArg, loopNestArg] :
           llvm::zip_equal(wsLoopOp.getRegion().getArguments(),
                           loopNestOp.getRegion().getArguments().drop_front(
                               loopNestClauseOps.loopLowerBounds.size())))
        rewriter.replaceAllUsesWith(loopNestArg, wsloopArg);
    }

    for (unsigned i = 0;
         i < loop.getLocalVars().size() + loop.getReduceVars().size(); ++i)
      loopNestOp.getRegion().eraseArgument(
          loopNestClauseOps.loopLowerBounds.size());

    rewriter.setInsertionPoint(doLoop);
    fir::FirOpBuilder builder(
        rewriter,
        fir::getKindMapping(doLoop->getParentOfType<mlir::ModuleOp>()));

    // Collect iteration variable(s) allocations so that we can move them
    // outside the `fir.do_concurrent` wrapper (before erasing it).
    llvm::SmallVector<mlir::Operation *> opsToMove;
    for (mlir::Operation &op : llvm::drop_end(doLoop))
      opsToMove.push_back(&op);

    mlir::Block *allocBlock = builder.getAllocaBlock();

    for (mlir::Operation *op : llvm::reverse(opsToMove)) {
      rewriter.moveOpBefore(op, allocBlock, allocBlock->begin());
    }

    // Mark `unordered` loops that are not perfectly nested to be skipped from
    // the legality check of the `ConversionTarget` since we are not interested
    // in mapping them to OpenMP.
    loopNestOp->walk([&](fir::DoConcurrentOp doLoop) {
      concurrentLoopsToSkip.insert(doLoop);
    });

    rewriter.eraseOp(doLoop);

    return mlir::success();
  }

private:
  mlir::omp::ParallelOp
  genParallelOp(mlir::ConversionPatternRewriter &rewriter,
                fir::DoConcurrentLoopOp loop,
                looputils::InductionVariableInfos &ivInfos,
                mlir::IRMapping &mapper) const {
    mlir::omp::ParallelOperands parallelOps;

    if (mapToDevice)
      genPrivatizers(rewriter, mapper, loop, parallelOps);

    mlir::Location loc = loop.getLoc();
    auto parallelOp = mlir::omp::ParallelOp::create(rewriter, loc, parallelOps);
    Fortran::common::openmp::EntryBlockArgs parallelArgs;
    parallelArgs.privVars = parallelOps.privateVars;
    Fortran::common::openmp::genEntryBlock(rewriter, parallelArgs,
                                           parallelOp.getRegion());
    rewriter.setInsertionPoint(mlir::omp::TerminatorOp::create(rewriter, loc));

    genLoopNestIndVarAllocs(rewriter, ivInfos, mapper);
    return parallelOp;
  }

  void genLoopNestIndVarAllocs(mlir::ConversionPatternRewriter &rewriter,
                               looputils::InductionVariableInfos &ivInfos,
                               mlir::IRMapping &mapper) const {

    for (auto &indVarInfo : ivInfos)
      genInductionVariableAlloc(rewriter, indVarInfo.iterVarMemDef, mapper);
  }

  mlir::Operation *
  genInductionVariableAlloc(mlir::ConversionPatternRewriter &rewriter,
                            mlir::Operation *indVarMemDef,
                            mlir::IRMapping &mapper) const {
    assert(
        indVarMemDef != nullptr &&
        "Induction variable memdef is expected to have a defining operation.");

    llvm::SmallSetVector<mlir::Operation *, 2> indVarDeclareAndAlloc;
    for (auto operand : indVarMemDef->getOperands())
      indVarDeclareAndAlloc.insert(operand.getDefiningOp());
    indVarDeclareAndAlloc.insert(indVarMemDef);

    mlir::Operation *result;
    for (mlir::Operation *opToClone : indVarDeclareAndAlloc)
      result = rewriter.clone(*opToClone, mapper);

    return result;
  }

  void genLoopNestClauseOps(
      mlir::Location loc, mlir::ConversionPatternRewriter &rewriter,
      fir::DoConcurrentLoopOp loop,
      mlir::omp::LoopNestOperands &loopNestClauseOps,
      mlir::omp::TargetExtOperands *targetClauseOps = nullptr) const {
    assert(loopNestClauseOps.loopLowerBounds.empty() &&
           "Loop nest bounds were already emitted!");

    auto populateBounds = [](mlir::Value var,
                             llvm::SmallVectorImpl<mlir::Value> &bounds) {
      bounds.push_back(var.getDefiningOp()->getResult(0));
    };

    auto hostEvalCapture = [&](mlir::Value var,
                               llvm::SmallVectorImpl<mlir::Value> &bounds) {
      populateBounds(var, bounds);

      // Ensure that loop-nest bounds are evaluated in the host and forwarded to
      // the nested omp constructs when we map to the device.
      if (targetClauseOps)
        targetClauseOps->hostEvalVars.push_back(var);
    };

    for (auto [lb, ub, st] : llvm::zip_equal(
             loop.getLowerBound(), loop.getUpperBound(), loop.getStep())) {
      hostEvalCapture(lb, loopNestClauseOps.loopLowerBounds);
      hostEvalCapture(ub, loopNestClauseOps.loopUpperBounds);
      hostEvalCapture(st, loopNestClauseOps.loopSteps);
    }

    loopNestClauseOps.loopInclusive = rewriter.getUnitAttr();
    loopNestClauseOps.collapseNumLoops =
        rewriter.getI64IntegerAttr(loopNestClauseOps.loopLowerBounds.size());
  }

  std::pair<mlir::omp::LoopNestOp, mlir::omp::WsloopOp>
  genWsLoopOp(mlir::ConversionPatternRewriter &rewriter,
              fir::DoConcurrentLoopOp loop, mlir::IRMapping &mapper,
              const mlir::omp::LoopNestOperands &clauseOps,
              bool isComposite) const {
    mlir::omp::WsloopOperands wsloopClauseOps;
    if (!mapToDevice)
      genPrivatizers(rewriter, mapper, loop, wsloopClauseOps);

    genReductions(rewriter, mapper, loop, wsloopClauseOps);

    auto wsloopOp =
        mlir::omp::WsloopOp::create(rewriter, loop.getLoc(), wsloopClauseOps);
    wsloopOp.setComposite(isComposite);

    Fortran::common::openmp::EntryBlockArgs wsloopArgs;
    wsloopArgs.privVars = wsloopClauseOps.privateVars;
    wsloopArgs.reductionVars = wsloopClauseOps.reductionVars;
    Fortran::common::openmp::genEntryBlock(rewriter, wsloopArgs,
                                           wsloopOp.getRegion());

    auto loopNestOp =
        mlir::omp::LoopNestOp::create(rewriter, loop.getLoc(), clauseOps);

    // Clone the loop's body inside the loop nest construct using the
    // mapped values.
    rewriter.cloneRegionBefore(loop.getRegion(), loopNestOp.getRegion(),
                               loopNestOp.getRegion().begin(), mapper);

    rewriter.setInsertionPointToEnd(&loopNestOp.getRegion().back());
    mlir::omp::YieldOp::create(rewriter, loop->getLoc());

    return {loopNestOp, wsloopOp};
  }

  mlir::omp::TeamsOp genTeamsOp(mlir::ConversionPatternRewriter &rewriter,
                                fir::DoConcurrentLoopOp loop,
                                mlir::IRMapping &mapper) const {
    mlir::omp::TeamsOperands teamsOps;
    genReductions(rewriter, mapper, loop, teamsOps);

    mlir::Location loc = loop.getLoc();
    auto teamsOp = mlir::omp::TeamsOp::create(rewriter, loc, teamsOps);
    Fortran::common::openmp::EntryBlockArgs teamsArgs;
    teamsArgs.reductionVars = teamsOps.reductionVars;
    Fortran::common::openmp::genEntryBlock(rewriter, teamsArgs,
                                           teamsOp.getRegion());

    rewriter.setInsertionPoint(mlir::omp::TerminatorOp::create(rewriter, loc));

    for (auto [loopVar, teamsArg] : llvm::zip_equal(
             loop.getReduceVars(), teamsOp.getRegion().getArguments())) {
      mapper.map(loopVar, teamsArg);
    }

    return teamsOp;
  }

  mlir::omp::DistributeOp
  genDistributeOp(mlir::Location loc,
                  mlir::ConversionPatternRewriter &rewriter) const {
    auto distOp = mlir::omp::DistributeOp::create(
        rewriter, loc, /*clauses=*/mlir::omp::DistributeOperands{});

    rewriter.createBlock(&distOp.getRegion());
    return distOp;
  }

  void cloneFIRRegionToOMP(mlir::ConversionPatternRewriter &rewriter,
                           mlir::Region &firRegion,
                           mlir::Region &ompRegion) const {
    if (!firRegion.empty()) {
      rewriter.cloneRegionBefore(firRegion, ompRegion, ompRegion.begin());
      auto firYield =
          mlir::cast<fir::YieldOp>(ompRegion.back().getTerminator());
      rewriter.setInsertionPoint(firYield);
      mlir::omp::YieldOp::create(rewriter, firYield.getLoc(),
                                 firYield.getOperands());
      rewriter.eraseOp(firYield);
    }
  }

  /// Generate bodies of OpenMP privatizers by cloning the bodies of FIR
  /// privatizers.
  ///
  /// \param [in] rewriter - used to driver IR generation for privatizers.
  /// \param [in] mapper - value mapping from FIR to OpenMP constructs.
  /// \param [in] loop - FIR loop to convert its localizers.
  ///
  /// \param [out] privateClauseOps - OpenMP privatizers to gen their bodies.
  void genPrivatizers(mlir::ConversionPatternRewriter &rewriter,
                      mlir::IRMapping &mapper, fir::DoConcurrentLoopOp loop,
                      mlir::omp::PrivateClauseOps &privateClauseOps) const {
    // For `local` (and `local_init`) operands, emit corresponding `private`
    // clauses and attach these clauses to the workshare loop.
    if (!loop.getLocalVars().empty())
      for (auto [var, sym, arg] : llvm::zip_equal(
               loop.getLocalVars(),
               loop.getLocalSymsAttr().getAsRange<mlir::SymbolRefAttr>(),
               loop.getRegionLocalArgs())) {
        auto localizer = moduleSymbolTable.lookup<fir::LocalitySpecifierOp>(
            sym.getLeafReference());
        if (localizer.getLocalitySpecifierType() ==
            fir::LocalitySpecifierType::LocalInit)
          TODO(localizer.getLoc(),
               "local_init conversion is not supported yet");

        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointAfter(localizer);

        auto privatizer = mlir::omp::PrivateClauseOp::create(
            rewriter, localizer.getLoc(), sym.getLeafReference().str() + ".omp",
            localizer.getTypeAttr().getValue(),
            mlir::omp::DataSharingClauseType::Private);

        cloneFIRRegionToOMP(rewriter, localizer.getInitRegion(),
                            privatizer.getInitRegion());
        cloneFIRRegionToOMP(rewriter, localizer.getDeallocRegion(),
                            privatizer.getDeallocRegion());

        moduleSymbolTable.insert(privatizer);

        privateClauseOps.privateVars.push_back(mapToDevice ? mapper.lookup(var)
                                                           : var);
        privateClauseOps.privateSyms.push_back(
            mlir::SymbolRefAttr::get(privatizer));
      }
  }

  void genReductions(mlir::ConversionPatternRewriter &rewriter,
                     mlir::IRMapping &mapper, fir::DoConcurrentLoopOp loop,
                     mlir::omp::ReductionClauseOps &reductionClauseOps) const {
    if (!loop.getReduceVars().empty()) {
      for (auto [var, byRef, sym, arg] : llvm::zip_equal(
               loop.getReduceVars(), loop.getReduceByrefAttr().asArrayRef(),
               loop.getReduceSymsAttr().getAsRange<mlir::SymbolRefAttr>(),
               loop.getRegionReduceArgs())) {
        auto firReducer = moduleSymbolTable.lookup<fir::DeclareReductionOp>(
            sym.getLeafReference());

        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointAfter(firReducer);
        std::string ompReducerName = sym.getLeafReference().str() + ".omp";

        auto ompReducer =
            moduleSymbolTable.lookup<mlir::omp::DeclareReductionOp>(
                rewriter.getStringAttr(ompReducerName));

        if (!ompReducer) {
          ompReducer = mlir::omp::DeclareReductionOp::create(
              rewriter, firReducer.getLoc(), ompReducerName,
              firReducer.getTypeAttr().getValue(),
              firReducer.getByrefElementTypeAttr());

          cloneFIRRegionToOMP(rewriter, firReducer.getAllocRegion(),
                              ompReducer.getAllocRegion());
          cloneFIRRegionToOMP(rewriter, firReducer.getInitializerRegion(),
                              ompReducer.getInitializerRegion());
          cloneFIRRegionToOMP(rewriter, firReducer.getReductionRegion(),
                              ompReducer.getReductionRegion());
          cloneFIRRegionToOMP(rewriter, firReducer.getAtomicReductionRegion(),
                              ompReducer.getAtomicReductionRegion());
          cloneFIRRegionToOMP(rewriter, firReducer.getCleanupRegion(),
                              ompReducer.getCleanupRegion());
          moduleSymbolTable.insert(ompReducer);
        }

        reductionClauseOps.reductionVars.push_back(
            mapToDevice ? mapper.lookup(var) : var);
        reductionClauseOps.reductionByref.push_back(byRef);
        reductionClauseOps.reductionSyms.push_back(
            mlir::SymbolRefAttr::get(ompReducer));
      }
    }
  }

  bool mapToDevice;
  llvm::DenseSet<fir::DoConcurrentOp> &concurrentLoopsToSkip;
  mlir::SymbolTable &moduleSymbolTable;
};

/// A listener that forwards notifyOperationErased to the given callback.
struct CallbackListener : public mlir::RewriterBase::Listener {
  CallbackListener(std::function<void(mlir::Operation *op)> onOperationErased)
      : onOperationErased(onOperationErased) {}

  void notifyOperationErased(mlir::Operation *op) override {
    onOperationErased(op);
  }

  std::function<void(mlir::Operation *op)> onOperationErased;
};

class DoConcurrentConversionPass
    : public flangomp::impl::DoConcurrentConversionPassBase<
          DoConcurrentConversionPass> {
public:
  DoConcurrentConversionPass() = default;

  DoConcurrentConversionPass(
      const flangomp::DoConcurrentConversionPassOptions &options)
      : DoConcurrentConversionPassBase(options) {}

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::MLIRContext *context = &getContext();
    mlir::SymbolTable moduleSymbolTable(module);

    if (mapTo != flangomp::DoConcurrentMappingKind::DCMK_Host &&
        mapTo != flangomp::DoConcurrentMappingKind::DCMK_Device) {
      mlir::emitWarning(mlir::UnknownLoc::get(context),
                        "DoConcurrentConversionPass: invalid `map-to` value. "
                        "Valid values are: `host` or `device`");
      return;
    }

    llvm::DenseSet<fir::DoConcurrentOp> concurrentLoopsToSkip;
    CallbackListener callbackListener([&](mlir::Operation *op) {
      if (auto loop = mlir::dyn_cast<fir::DoConcurrentOp>(op))
        concurrentLoopsToSkip.erase(loop);
    });
    mlir::RewritePatternSet patterns(context);
    patterns.insert<DoConcurrentConversion>(
        context, mapTo == flangomp::DoConcurrentMappingKind::DCMK_Device,
        concurrentLoopsToSkip, moduleSymbolTable);
    mlir::ConversionTarget target(*context);
    target.addDynamicallyLegalOp<fir::DoConcurrentOp>(
        [&](fir::DoConcurrentOp op) {
          return concurrentLoopsToSkip.contains(op);
        });
    target.markUnknownOpDynamicallyLegal(
        [](mlir::Operation *) { return true; });

    mlir::ConversionConfig config;
    config.allowPatternRollback = false;
    config.listener = &callbackListener;
    if (mlir::failed(mlir::applyFullConversion(module, target,
                                               std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass>
flangomp::createDoConcurrentConversionPass(bool mapToDevice) {
  DoConcurrentConversionPassOptions options;
  options.mapTo = mapToDevice ? flangomp::DoConcurrentMappingKind::DCMK_Device
                              : flangomp::DoConcurrentMappingKind::DCMK_Host;

  return std::make_unique<DoConcurrentConversionPass>(options);
}
