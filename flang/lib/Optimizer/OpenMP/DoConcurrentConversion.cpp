//===- DoConcurrentConversion.cpp -- map `DO CONCURRENT` to OpenMP loops --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/HLFIR/HLFIRDialect.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/OpenMP/Passes.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>

namespace flangomp {
#define GEN_PASS_DEF_DOCONCURRENTCONVERSIONPASS
#include "flang/Optimizer/OpenMP/Passes.h.inc"
} // namespace flangomp

#define DEBUG_TYPE "do-concurrent-conversion"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

namespace Fortran {
namespace lower {
namespace omp {
namespace internal {
// TODO The following 2 functions are copied from "flang/Lower/OpenMP/Utils.h".
// This duplication is temporary until we find a solution for a shared location
// for these utils that does not introduce circular CMake deps.
mlir::omp::MapInfoOp
createMapInfoOp(mlir::OpBuilder &builder, mlir::Location loc,
                mlir::Value baseAddr, mlir::Value varPtrPtr, std::string name,
                llvm::ArrayRef<mlir::Value> bounds,
                llvm::ArrayRef<mlir::Value> members,
                mlir::DenseIntElementsAttr membersIndex, uint64_t mapType,
                mlir::omp::VariableCaptureKind mapCaptureType, mlir::Type retTy,
                bool partialMap = false) {
  if (auto boxTy = llvm::dyn_cast<fir::BaseBoxType>(baseAddr.getType())) {
    baseAddr = builder.create<fir::BoxAddrOp>(loc, baseAddr);
    retTy = baseAddr.getType();
  }

  mlir::TypeAttr varType = mlir::TypeAttr::get(
      llvm::cast<mlir::omp::PointerLikeType>(retTy).getElementType());

  mlir::omp::MapInfoOp op = builder.create<mlir::omp::MapInfoOp>(
      loc, retTy, baseAddr, varType, varPtrPtr, members, membersIndex, bounds,
      builder.getIntegerAttr(builder.getIntegerType(64, false), mapType),
      builder.getAttr<mlir::omp::VariableCaptureKindAttr>(mapCaptureType),
      builder.getStringAttr(name), builder.getBoolAttr(partialMap));

  return op;
}

mlir::Value calculateTripCount(fir::FirOpBuilder &builder, mlir::Location loc,
                               const mlir::omp::LoopRelatedOps &ops) {
  using namespace mlir::arith;
  assert(ops.loopLowerBounds.size() == ops.loopUpperBounds.size() &&
         ops.loopLowerBounds.size() == ops.loopSteps.size() &&
         !ops.loopLowerBounds.empty() && "Invalid bounds or step");

  // Get the bit width of an integer-like type.
  auto widthOf = [](mlir::Type ty) -> unsigned {
    if (mlir::isa<mlir::IndexType>(ty)) {
      return mlir::IndexType::kInternalStorageBitWidth;
    }
    if (auto tyInt = mlir::dyn_cast<mlir::IntegerType>(ty)) {
      return tyInt.getWidth();
    }
    llvm_unreachable("Unexpected type");
  };

  // For a type that is either IntegerType or IndexType, return the
  // equivalent IntegerType. In the former case this is a no-op.
  auto asIntTy = [&](mlir::Type ty) -> mlir::IntegerType {
    if (ty.isIndex()) {
      return mlir::IntegerType::get(ty.getContext(), widthOf(ty));
    }
    assert(ty.isIntOrIndex() && "Unexpected type");
    return mlir::cast<mlir::IntegerType>(ty);
  };

  // For two given values, establish a common signless IntegerType
  // that can represent any value of type of x and of type of y,
  // and return the pair of x, y converted to the new type.
  auto unifyToSignless =
      [&](fir::FirOpBuilder &b, mlir::Value x,
          mlir::Value y) -> std::pair<mlir::Value, mlir::Value> {
    auto tyX = asIntTy(x.getType()), tyY = asIntTy(y.getType());
    unsigned width = std::max(widthOf(tyX), widthOf(tyY));
    auto wideTy = mlir::IntegerType::get(b.getContext(), width,
                                         mlir::IntegerType::Signless);
    return std::make_pair(b.createConvert(loc, wideTy, x),
                          b.createConvert(loc, wideTy, y));
  };

  // Start with signless i32 by default.
  auto tripCount = builder.createIntegerConstant(loc, builder.getI32Type(), 1);

  for (auto [origLb, origUb, origStep] :
       llvm::zip(ops.loopLowerBounds, ops.loopUpperBounds, ops.loopSteps)) {
    auto tmpS0 = builder.createIntegerConstant(loc, origStep.getType(), 0);
    auto [step, step0] = unifyToSignless(builder, origStep, tmpS0);
    auto reverseCond =
        builder.create<CmpIOp>(loc, CmpIPredicate::slt, step, step0);
    auto negStep = builder.create<SubIOp>(loc, step0, step);
    mlir::Value absStep =
        builder.create<SelectOp>(loc, reverseCond, negStep, step);

    auto [lb, ub] = unifyToSignless(builder, origLb, origUb);
    auto start = builder.create<SelectOp>(loc, reverseCond, ub, lb);
    auto end = builder.create<SelectOp>(loc, reverseCond, lb, ub);

    mlir::Value range = builder.create<SubIOp>(loc, end, start);
    auto rangeCond =
        builder.create<CmpIOp>(loc, CmpIPredicate::slt, end, start);
    std::tie(range, absStep) = unifyToSignless(builder, range, absStep);
    // numSteps = (range /u absStep) + 1
    auto numSteps = builder.create<AddIOp>(
        loc, builder.create<DivUIOp>(loc, range, absStep),
        builder.createIntegerConstant(loc, range.getType(), 1));

    auto trip0 = builder.createIntegerConstant(loc, numSteps.getType(), 0);
    auto loopTripCount =
        builder.create<SelectOp>(loc, rangeCond, trip0, numSteps);
    auto [totalTC, thisTC] = unifyToSignless(builder, tripCount, loopTripCount);
    tripCount = builder.create<MulIOp>(loc, totalTC, thisTC);
  }

  return tripCount;
}

mlir::Value mapTemporaryValue(fir::FirOpBuilder &builder,
                              mlir::omp::TargetOp targetOp, mlir::Value val,
                              std::string name = "") {
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfterValue(val);
  auto copyVal = builder.createTemporary(val.getLoc(), val.getType());
  builder.createStoreWithConvert(copyVal.getLoc(), val, copyVal);

  llvm::SmallVector<mlir::Value> bounds;
  builder.setInsertionPoint(targetOp);
  mlir::Value mapOp = createMapInfoOp(
      builder, copyVal.getLoc(), copyVal,
      /*varPtrPtr=*/mlir::Value{}, name, bounds,
      /*members=*/llvm::SmallVector<mlir::Value>{},
      /*membersIndex=*/mlir::DenseIntElementsAttr{},
      static_cast<std::underlying_type_t<llvm::omp::OpenMPOffloadMappingFlags>>(
          llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_IMPLICIT),
      mlir::omp::VariableCaptureKind::ByCopy, copyVal.getType());
  targetOp.getMapVarsMutable().append(mapOp);

  mlir::Region &targetRegion = targetOp.getRegion();
  mlir::Block *targetEntryBlock = &targetRegion.getBlocks().front();
  mlir::Value clonedValArg =
      targetRegion.addArgument(copyVal.getType(), copyVal.getLoc());
  builder.setInsertionPointToStart(targetEntryBlock);
  auto loadOp =
      builder.create<fir::LoadOp>(clonedValArg.getLoc(), clonedValArg);
  return loadOp.getResult();
}

/// Check if cloning the bounds introduced any dependency on the outer region.
/// If so, then either clone them as well if they are MemoryEffectFree, or else
/// copy them to a new temporary and add them to the map and block_argument
/// lists and replace their uses with the new temporary.
///
/// TODO: similar to the above functions, this is copied from OpenMP lowering
/// (in this case, from `genBodyOfTargetOp`). Once we move to a common lib for
/// these utils this will move as well.
void cloneOrMapRegionOutsiders(fir::FirOpBuilder &builder,
                               mlir::omp::TargetOp targetOp) {
  mlir::Region &targetRegion = targetOp.getRegion();
  mlir::Block *targetEntryBlock = &targetRegion.getBlocks().front();
  llvm::SetVector<mlir::Value> valuesDefinedAbove;
  mlir::getUsedValuesDefinedAbove(targetRegion, valuesDefinedAbove);

  while (!valuesDefinedAbove.empty()) {
    for (mlir::Value val : valuesDefinedAbove) {
      mlir::Operation *valOp = val.getDefiningOp();
      assert(valOp != nullptr);
      if (mlir::isMemoryEffectFree(valOp)) {
        mlir::Operation *clonedOp = valOp->clone();
        targetEntryBlock->push_front(clonedOp);
        assert(clonedOp->getNumResults() == 1);
        val.replaceUsesWithIf(
            clonedOp->getResult(0), [targetEntryBlock](mlir::OpOperand &use) {
              return use.getOwner()->getBlock() == targetEntryBlock;
            });
      } else {
        mlir::Value mappedTemp = mapTemporaryValue(builder, targetOp, val);
        val.replaceUsesWithIf(
            mappedTemp, [targetEntryBlock](mlir::OpOperand &use) {
              return use.getOwner()->getBlock() == targetEntryBlock;
            });
      }
    }
    valuesDefinedAbove.clear();
    mlir::getUsedValuesDefinedAbove(targetRegion, valuesDefinedAbove);
  }
}
} // namespace internal
} // namespace omp
} // namespace lower
} // namespace Fortran

namespace {
namespace looputils {
/// Stores info needed about the induction/iteration variable for each `do
/// concurrent` in a loop nest. This includes:
/// * the operation allocating memory for iteration variable,
/// * the operation(s) updating the iteration variable with the current
///   iteration number.
struct InductionVariableInfo {
  mlir::Operation *iterVarMemDef;
  llvm::SetVector<mlir::Operation *> indVarUpdateOps;
};

using LoopNestToIndVarMap =
    llvm::MapVector<fir::DoLoopOp, InductionVariableInfo>;

/// Given an operation `op`, this returns true if `op`'s operand is ultimately
/// the loop's induction variable. Detecting this helps finding the live-in
/// value corresponding to the induction variable in case the induction variable
/// is indirectly used in the loop (e.g. throught a cast op).
bool isIndVarUltimateOperand(mlir::Operation *op, fir::DoLoopOp doLoop) {
  while (op != nullptr && op->getNumOperands() > 0) {
    auto ivIt = llvm::find_if(op->getOperands(), [&](mlir::Value operand) {
      return operand == doLoop.getInductionVar();
    });

    if (ivIt != op->getOperands().end())
      return true;

    op = op->getOperand(0).getDefiningOp();
  }

  return false;
};

mlir::Value findLoopIndVar(fir::DoLoopOp doLoop) {
  mlir::Value result = nullptr;
  mlir::visitUsedValuesDefinedAbove(
      doLoop.getRegion(), [&](mlir::OpOperand *operand) {
        if (isIndVarUltimateOperand(operand->getOwner(), doLoop))
          result = operand->get();
      });

  assert(result != nullptr);
  return result;
}

/// Collect the list of values used inside the loop but defined outside of it.
/// The first item in the returned list is always the loop's induction
/// variable.
void collectLoopNestLiveIns(
    LoopNestToIndVarMap &loopNest, llvm::SmallVectorImpl<mlir::Value> &liveIns,
    llvm::DenseMap<mlir::Value, std::string> *liveInToName = nullptr) {
  llvm::SmallDenseSet<mlir::Value> seenValues;
  llvm::SmallDenseSet<mlir::Operation *> seenOps;

  auto addValueToLiveIns = [&](mlir::Value liveIn) {
    if (!seenValues.insert(liveIn).second)
      return false;

    mlir::Operation *definingOp = liveIn.getDefiningOp();
    // We want to collect ops corresponding to live-ins only once.
    if (definingOp && !seenOps.insert(definingOp).second)
      return false;

    liveIns.push_back(liveIn);
    return true;
  };

  size_t nestLevel = 0;
  for (auto [loop, _] : loopNest) {
    auto addBoundOrStepToLiveIns = [&](mlir::Value operand, std::string name) {
      (*liveInToName)[operand] = name;
      addValueToLiveIns(operand);
    };

    addBoundOrStepToLiveIns(loop.getLowerBound(),
                            "loop." + std::to_string(nestLevel) + ".lb");
    addBoundOrStepToLiveIns(loop.getUpperBound(),
                            "loop." + std::to_string(nestLevel) + ".ub");
    addBoundOrStepToLiveIns(loop.getStep(),
                            "loop." + std::to_string(nestLevel) + ".step");
    ++nestLevel;
  }

  mlir::visitUsedValuesDefinedAbove(
      loopNest.front().first.getRegion(),
      [&](mlir::OpOperand *operand) { addValueToLiveIns(operand->get()); });
}

/// Collects the op(s) responsible for updating a loop's iteration variable with
/// the current iteration number. For example, for the input IR:
/// ```
/// %i = fir.alloca i32 {bindc_name = "i"}
/// %i_decl:2 = hlfir.declare %i ...
/// ...
/// fir.do_loop %i_iv = %lb to %ub step %step unordered {
///   %1 = fir.convert %i_iv : (index) -> i32
///   fir.store %1 to %i_decl#1 : !fir.ref<i32>
///   ...
/// }
/// ```
/// this function would return the first 2 ops in the `fir.do_loop`'s region.
llvm::SetVector<mlir::Operation *>
extractIndVarUpdateOps(fir::DoLoopOp doLoop) {
  mlir::Value indVar = doLoop.getInductionVar();
  llvm::SetVector<mlir::Operation *> indVarUpdateOps;

  llvm::SmallVector<mlir::Value> toProcess;
  toProcess.push_back(indVar);

  llvm::DenseSet<mlir::Value> done;

  while (!toProcess.empty()) {
    mlir::Value val = toProcess.back();
    toProcess.pop_back();

    if (!done.insert(val).second)
      continue;

    for (mlir::Operation *user : val.getUsers()) {
      indVarUpdateOps.insert(user);

      for (mlir::Value result : user->getResults())
        toProcess.push_back(result);
    }
  }

  return std::move(indVarUpdateOps);
}

/// Starting with a value at the end of a definition/conversion chain, walk the
/// chain backwards and collect all the visited ops along the way. This is the
/// same as the "backward slice" of the use-def chain of \p link.
///
/// If the root of the chain/slice is a constant op  (where convert operations
/// on constant count as constants as well), then populate \p opChain with the
/// extracted chain/slice. If not, then \p opChain will contains a single value:
/// \p link.
///
/// The purpose of this function is that we pull in the chain of
/// constant+conversion ops inside the parallel region if possible; which
/// prevents creating an unnecessary shared/mapped value that crosses the OpenMP
/// region.
///
/// For example, given this IR:
/// ```
/// %c10 = arith.constant 10 : i32
/// %10 = fir.convert %c10 : (i32) -> index
/// ```
/// and giving `%10` as the starting input: `link`, `defChain` would contain
/// both of the above ops.
void collectIndirectConstOpChain(mlir::Operation *link,
                                 llvm::SetVector<mlir::Operation *> &opChain) {
  mlir::BackwardSliceOptions options;
  options.inclusive = true;
  mlir::getBackwardSlice(link, &opChain, options);

  assert(!opChain.empty());

  bool isConstantChain = [&]() {
    if (!mlir::isa_and_present<mlir::arith::ConstantOp>(opChain.front()))
      return false;

    return llvm::all_of(llvm::drop_begin(opChain), [](mlir::Operation *op) {
      return mlir::isa_and_present<fir::ConvertOp>(op);
    });
  }();

  if (isConstantChain)
    return;

  opChain.clear();
  opChain.insert(link);
}

/// Loop \p innerLoop is considered perfectly-nested inside \p outerLoop iff
/// there are no operations in \p outerloop's other than:
///
/// 1. those operations needed to setup \p innerLoop's LB, UB, and step values,
/// 2. the operations needed to assing/update \p outerLoop's induction variable.
/// 3. \p innerLoop itself.
///
/// \p return true if \p innerLoop is perfectly nested inside \p outerLoop
/// according to the above definition.
bool isPerfectlyNested(fir::DoLoopOp outerLoop, fir::DoLoopOp innerLoop) {
  mlir::BackwardSliceOptions backwardSliceOptions;
  backwardSliceOptions.inclusive = true;
  // We will collect the backward slices for innerLoop's LB, UB, and step.
  // However, we want to limit the scope of these slices to the scope of
  // outerLoop's region.
  backwardSliceOptions.filter = [&](mlir::Operation *op) {
    return !mlir::areValuesDefinedAbove(op->getResults(),
                                        outerLoop.getRegion());
  };

  llvm::SetVector<mlir::Operation *> lbSlice;
  mlir::getBackwardSlice(innerLoop.getLowerBound(), &lbSlice,
                         backwardSliceOptions);

  llvm::SetVector<mlir::Operation *> ubSlice;
  mlir::getBackwardSlice(innerLoop.getUpperBound(), &ubSlice,
                         backwardSliceOptions);

  llvm::SetVector<mlir::Operation *> stepSlice;
  mlir::getBackwardSlice(innerLoop.getStep(), &stepSlice, backwardSliceOptions);

  mlir::ForwardSliceOptions forwardSliceOptions;
  forwardSliceOptions.inclusive = true;
  // We don't care of the outer loop's induction variable's uses within the
  // inner loop, so we filter out these uses.
  forwardSliceOptions.filter = [&](mlir::Operation *op) {
    return mlir::areValuesDefinedAbove(op->getResults(), innerLoop.getRegion());
  };

  llvm::SetVector<mlir::Operation *> indVarSlice;
  mlir::getForwardSlice(outerLoop.getInductionVar(), &indVarSlice,
                        forwardSliceOptions);

  llvm::SetVector<mlir::Operation *> innerLoopSetupOpsVec;
  innerLoopSetupOpsVec.set_union(indVarSlice);
  innerLoopSetupOpsVec.set_union(lbSlice);
  innerLoopSetupOpsVec.set_union(ubSlice);
  innerLoopSetupOpsVec.set_union(stepSlice);
  llvm::DenseSet<mlir::Operation *> innerLoopSetupOpsSet;

  for (mlir::Operation *op : innerLoopSetupOpsVec)
    innerLoopSetupOpsSet.insert(op);

  llvm::DenseSet<mlir::Operation *> loopBodySet;
  outerLoop.walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
    if (op == outerLoop)
      return mlir::WalkResult::advance();

    if (op == innerLoop)
      return mlir::WalkResult::skip();

    if (op->hasTrait<mlir::OpTrait::IsTerminator>())
      return mlir::WalkResult::advance();

    loopBodySet.insert(op);
    return mlir::WalkResult::advance();
  });

  bool result = (loopBodySet == innerLoopSetupOpsSet);
  LLVM_DEBUG(DBGS() << "Loop pair starting at location " << outerLoop.getLoc()
                    << " is" << (result ? "" : " not")
                    << " perfectly nested\n");
  return result;
}

/// Starting with `outerLoop` collect a perfectly nested loop nest, if any. This
/// function collects as much as possible loops in the nest; it case it fails to
/// recognize a certain nested loop as part of the nest it just returns the
/// parent loops it discovered before.
mlir::LogicalResult collectLoopNest(fir::DoLoopOp currentLoop,
                                    LoopNestToIndVarMap &loopNest) {
  assert(currentLoop.getUnordered());
  while (true) {
    loopNest.try_emplace(
        currentLoop,
        InductionVariableInfo{
            findLoopIndVar(currentLoop).getDefiningOp(),
            std::move(looputils::extractIndVarUpdateOps(currentLoop))});

    auto directlyNestedLoops = currentLoop.getRegion().getOps<fir::DoLoopOp>();
    llvm::SmallVector<fir::DoLoopOp> unorderedLoops;

    for (auto nestedLoop : directlyNestedLoops)
      if (nestedLoop.getUnordered())
        unorderedLoops.push_back(nestedLoop);

    if (unorderedLoops.empty())
      break;

    if (unorderedLoops.size() > 1)
      return mlir::failure();

    fir::DoLoopOp nestedUnorderedLoop = unorderedLoops.front();

    if ((nestedUnorderedLoop.getLowerBound().getDefiningOp() == nullptr) ||
        (nestedUnorderedLoop.getUpperBound().getDefiningOp() == nullptr) ||
        (nestedUnorderedLoop.getStep().getDefiningOp() == nullptr))
      return mlir::failure();

    if (!isPerfectlyNested(currentLoop, nestedUnorderedLoop))
      return mlir::failure();

    currentLoop = nestedUnorderedLoop;
  }

  return mlir::success();
}

/// Prepares the `fir.do_loop` nest to be easily mapped to OpenMP. In
/// particular, this function would take this input IR:
/// ```
/// fir.do_loop %i_iv = %i_lb to %i_ub step %i_step unordered {
///   fir.store %i_iv to %i#1 : !fir.ref<i32>
///   %j_lb = arith.constant 1 : i32
///   %j_ub = arith.constant 10 : i32
///   %j_step = arith.constant 1 : index
///
///   fir.do_loop %j_iv = %j_lb to %j_ub step %j_step unordered {
///     fir.store %j_iv to %j#1 : !fir.ref<i32>
///     ...
///   }
/// }
/// ```
///
/// into the following form (using generic op form since the result is
/// technically an invalid `fir.do_loop` op:
///
/// ```
/// "fir.do_loop"(%i_lb, %i_ub, %i_step) <{unordered}> ({
/// ^bb0(%i_iv: index):
///   %j_lb = "arith.constant"() <{value = 1 : i32}> : () -> i32
///   %j_ub = "arith.constant"() <{value = 10 : i32}> : () -> i32
///   %j_step = "arith.constant"() <{value = 1 : index}> : () -> index
///
///   "fir.do_loop"(%j_lb, %j_ub, %j_step) <{unordered}> ({
///   ^bb0(%new_i_iv: index, %new_j_iv: index):
///     "fir.store"(%new_i_iv, %i#1) : (i32, !fir.ref<i32>) -> ()
///     "fir.store"(%new_j_iv, %j#1) : (i32, !fir.ref<i32>) -> ()
///     ...
///   })
/// ```
///
/// What happened to the loop nest is the following:
///
/// * the innermost loop's entry block was updated from having one operand to
///   having `n` operands where `n` is the number of loops in the nest,
///
/// * the outer loop(s)' ops that update the IVs were sank inside the innermost
///   loop (see the `"fir.store"(%new_i_iv, %i#1)` op above),
///
/// * the innermost loop's entry block's arguments were mapped in order from the
///   outermost to the innermost IV.
///
/// With this IR change, we can directly inline the innermost loop's region into
/// the newly generated `omp.loop_nest` op.
///
/// Note that this function has a pre-condition that \p loopNest consists of
/// perfectly nested loops; i.e. there are no in-between ops between 2 nested
/// loops except for the ops to setup the inner loop's LB, UB, and step. These
/// ops are handled/cloned by `genLoopNestClauseOps(..)`.
void sinkLoopIVArgs(mlir::ConversionPatternRewriter &rewriter,
                    looputils::LoopNestToIndVarMap &loopNest) {
  if (loopNest.size() <= 1)
    return;

  fir::DoLoopOp innermostLoop = loopNest.back().first;
  mlir::Operation &innermostFirstOp = innermostLoop.getRegion().front().front();

  llvm::SmallVector<mlir::Type> argTypes;
  llvm::SmallVector<mlir::Location> argLocs;

  for (auto &[doLoop, indVarInfo] : llvm::drop_end(loopNest)) {
    // Sink the IV update ops to the innermost loop. We need to do for all loops
    // except for the innermost one, hence the `drop_end` usage above.
    for (mlir::Operation *op : indVarInfo.indVarUpdateOps)
      op->moveBefore(&innermostFirstOp);

    argTypes.push_back(doLoop.getInductionVar().getType());
    argLocs.push_back(doLoop.getInductionVar().getLoc());
  }

  mlir::Region &innermmostRegion = innermostLoop.getRegion();
  // Extend the innermost entry block with arguments to represent the outer IVs.
  innermmostRegion.addArguments(argTypes, argLocs);

  unsigned idx = 1;
  // In reverse, remap the IVs of the loop nest from the old values to the new
  // ones. We do that in reverse since the first argument before this loop is
  // the old IV for the innermost loop. Therefore, we want to replace it first
  // before the old value (1st argument in the block) is remapped to be the IV
  // of the outermost loop in the nest.
  for (auto &[doLoop, _] : llvm::reverse(loopNest)) {
    doLoop.getInductionVar().replaceAllUsesWith(
        innermmostRegion.getArgument(innermmostRegion.getNumArguments() - idx));
    ++idx;
  }
}

/// Collects values that are local to a loop: "loop-local values". A loop-local
/// value is one that is used exclusively inside the loop but allocated outside
/// of it. This usually corresponds to temporary values that are used inside the
/// loop body for initialzing other variables for example.
///
/// \param [in] doLoop - the loop within which the function searches for values
/// used exclusively inside.
///
/// \param [out] locals - the list of loop-local values detected for \p doLoop.
static void collectLoopLocalValues(fir::DoLoopOp doLoop,
                                   llvm::SetVector<mlir::Value> &locals) {
  doLoop.walk([&](mlir::Operation *op) {
    for (mlir::Value operand : op->getOperands()) {
      if (locals.contains(operand))
        continue;

      bool isLocal = true;

      if (!mlir::isa_and_present<fir::AllocaOp>(operand.getDefiningOp()))
        continue;

      // Values defined inside the loop are not interesting since they do not
      // need to be localized.
      if (doLoop->isAncestor(operand.getDefiningOp()))
        continue;

      for (auto *user : operand.getUsers()) {
        if (!doLoop->isAncestor(user)) {
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

class DoConcurrentConversion : public mlir::OpConversionPattern<fir::DoLoopOp> {
public:
  using mlir::OpConversionPattern<fir::DoLoopOp>::OpConversionPattern;

  DoConcurrentConversion(mlir::MLIRContext *context, bool mapToDevice,
                         llvm::DenseSet<fir::DoLoopOp> &concurrentLoopsToSkip)
      : OpConversionPattern(context), mapToDevice(mapToDevice),
        concurrentLoopsToSkip(concurrentLoopsToSkip) {}

  mlir::LogicalResult
  matchAndRewrite(fir::DoLoopOp doLoop, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Operation *lbOp = doLoop.getLowerBound().getDefiningOp();
    mlir::Operation *ubOp = doLoop.getUpperBound().getDefiningOp();
    mlir::Operation *stepOp = doLoop.getStep().getDefiningOp();

    if (lbOp == nullptr || ubOp == nullptr || stepOp == nullptr) {
      return rewriter.notifyMatchFailure(
          doLoop, "At least one of the loop's LB, UB, or step doesn't have a "
                  "defining operation.");
    }

    looputils::LoopNestToIndVarMap loopNest;
    bool hasRemainingNestedLoops =
        failed(looputils::collectLoopNest(doLoop, loopNest));
    if (hasRemainingNestedLoops)
      mlir::emitWarning(doLoop.getLoc(),
                        "Some `do concurent` loops are not perfectly-nested. "
                        "These will be serialzied.");

    llvm::DenseMap<mlir::Value, std::string> liveInToName;
    llvm::SmallVector<mlir::Value> loopNestLiveIns;

    // TODO outline into a separete function. This hoists the ops to compute
    // bounds of all loops in the entire loop nest outside the outermost loop.
    // Without this hoisting, values/variables that are required to compute
    // these bounds will be stuck inside the original `fir.do_loop` ops and
    // therefore their SSA values won't be visible from within the `target`
    // region.
    {
      fir::DoLoopOp outermostLoop = loopNest.front().first;

      mlir::BackwardSliceOptions backwardSliceOptions;
      backwardSliceOptions.inclusive = true;
      // We will collect the backward slices for innerLoop's LB, UB, and step.
      // However, we want to limit the scope of these slices to the scope of
      // outerLoop's region.
      backwardSliceOptions.filter = [&](mlir::Operation *op) {
        return !mlir::areValuesDefinedAbove(op->getResults(),
                                            outermostLoop.getRegion());
      };

      for (auto [loop, _] : loopNest) {
        auto moveBoundOrStepOutOfLoopNest = [&](mlir::Value operand) {
          llvm::SetVector<mlir::Operation *> loopOperandSlice;
          mlir::getBackwardSlice(operand, &loopOperandSlice,
                                 backwardSliceOptions);

          for (mlir::Operation *sliceOp : loopOperandSlice) {
            outermostLoop.moveOutOfLoop(sliceOp);
          }
        };

        moveBoundOrStepOutOfLoopNest(loop.getLowerBound());
        moveBoundOrStepOutOfLoopNest(loop.getUpperBound());
        moveBoundOrStepOutOfLoopNest(loop.getStep());
      }
    }

    looputils::collectLoopNestLiveIns(loopNest, loopNestLiveIns, &liveInToName);
    assert(!loopNestLiveIns.empty());

    llvm::SetVector<mlir::Value> locals;
    looputils::collectLoopLocalValues(loopNest.back().first, locals);
    // We do not want to map "loop-local" values to the device through
    // `omp.map.info` ops. Therefore, we remove them from the list of live-ins.
    loopNestLiveIns.erase(llvm::remove_if(loopNestLiveIns,
                                          [&](mlir::Value liveIn) {
                                            return locals.contains(liveIn);
                                          }),
                          loopNestLiveIns.end());

    looputils::sinkLoopIVArgs(rewriter, loopNest);

    mlir::omp::TargetOp targetOp;
    mlir::omp::LoopNestOperands loopNestClauseOps;

    mlir::IRMapping mapper;

    if (mapToDevice) {
      mlir::omp::TargetOperands targetClauseOps;
      LiveInShapeInfoMap liveInShapeInfoMap;

      // The outermost loop will contain all the live-in values in all nested
      // loops since live-in values are collected recursively for all nested
      // ops.
      for (mlir::Value liveIn : loopNestLiveIns) {
        targetClauseOps.mapVars.push_back(genMapInfoOpForLiveIn(
            rewriter, liveIn, liveInToName, liveInShapeInfoMap[liveIn]));
      }

      targetOp = genTargetOp(doLoop.getLoc(), rewriter, mapper, loopNestLiveIns,
                             targetClauseOps, liveInShapeInfoMap);
      genTeamsOp(doLoop.getLoc(), rewriter);
    }

    mlir::omp::ParallelOp parallelOp = genParallelOp(
        doLoop.getLoc(), rewriter, loopNest, mapper, loopNestClauseOps);
    // Only set as composite when part of `distribute parallel do`.
    parallelOp.setComposite(mapToDevice);

    for (mlir::Value local : locals)
      looputils::localizeLoopLocalValue(local, parallelOp.getRegion(),
                                        rewriter);

    if (mapToDevice)
      genDistributeOp(doLoop.getLoc(), rewriter).setComposite(/*val=*/true);

    mlir::omp::LoopNestOp ompLoopNest =
        genWsLoopOp(rewriter, loopNest.back().first, mapper, loopNestClauseOps,
                    /*isComposite=*/mapToDevice);

    // Now that we created the nested `ws.loop` op, we set can the `target` op's
    // trip count.
    if (mapToDevice) {
      rewriter.setInsertionPoint(targetOp);
      auto parentModule = doLoop->getParentOfType<mlir::ModuleOp>();
      fir::FirOpBuilder firBuilder(rewriter, fir::getKindMapping(parentModule));

      mlir::omp::LoopRelatedOps loopClauseOps;
      loopClauseOps.loopLowerBounds.push_back(lbOp->getResult(0));
      loopClauseOps.loopUpperBounds.push_back(ubOp->getResult(0));
      loopClauseOps.loopSteps.push_back(stepOp->getResult(0));

      mlir::cast<mlir::omp::TargetOp>(targetOp).getTripCountMutable().assign(
          Fortran::lower::omp::internal::calculateTripCount(
              firBuilder, doLoop.getLoc(), loopClauseOps));
    }

    rewriter.eraseOp(doLoop);

    // Mark `unordered` loops that are not perfectly nested to be skipped from
    // the legality check of the `ConversionTarget` since we are not interested
    // in mapping them to OpenMP.
    ompLoopNest->walk([&](fir::DoLoopOp doLoop) {
      if (doLoop.getUnordered()) {
        concurrentLoopsToSkip.insert(doLoop);
      }
    });

    return mlir::success();
  }

private:
  struct TargetDeclareShapeCreationInfo {
    std::vector<mlir::Value> startIndices{};
    std::vector<mlir::Value> extents{};

    bool isShapedValue() const { return !extents.empty(); }
    bool isShapeShiftedValue() const { return !startIndices.empty(); }
  };

  using LiveInShapeInfoMap =
      llvm::DenseMap<mlir::Value, TargetDeclareShapeCreationInfo>;

  void
  genBoundsOps(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
               mlir::Value shape, llvm::SmallVectorImpl<mlir::Value> &boundsOps,
               TargetDeclareShapeCreationInfo &targetShapeCreationInfo) const {
    if (shape == nullptr) {
      return;
    }

    auto shapeOp =
        mlir::dyn_cast_if_present<fir::ShapeOp>(shape.getDefiningOp());
    auto shapeShiftOp =
        mlir::dyn_cast_if_present<fir::ShapeShiftOp>(shape.getDefiningOp());

    if (shapeOp == nullptr && shapeShiftOp == nullptr)
      TODO(loc,
           "Shapes not defined by `fir.shape` or `fir.shape_shift` op's are "
           "not supported yet.");

    auto extents = shapeOp != nullptr
                       ? std::vector<mlir::Value>(shapeOp.getExtents().begin(),
                                                  shapeOp.getExtents().end())
                       : shapeShiftOp.getExtents();

    mlir::Type idxType = extents.front().getType();

    auto one = rewriter.create<mlir::arith::ConstantOp>(
        loc, idxType, rewriter.getIntegerAttr(idxType, 1));
    // For non-shifted values, that starting index is the default Fortran
    // value: 1.
    std::vector<mlir::Value> startIndices =
        shapeOp != nullptr ? std::vector<mlir::Value>(extents.size(), one)
                           : shapeShiftOp.getOrigins();

    auto genBoundsOp = [&](mlir::Value startIndex, mlir::Value extent) {
      // We map the entire range of data by default, therefore, we always map
      // from the start.
      auto normalizedLB = rewriter.create<mlir::arith::ConstantOp>(
          loc, idxType, rewriter.getIntegerAttr(idxType, 0));

      mlir::Value ub = rewriter.create<mlir::arith::SubIOp>(loc, extent, one);

      return rewriter.create<mlir::omp::MapBoundsOp>(
          loc, rewriter.getType<mlir::omp::MapBoundsType>(), normalizedLB, ub,
          extent,
          /*stride=*/mlir::Value{}, /*stride_in_bytes=*/false, startIndex);
    };

    for (auto [startIndex, extent] : llvm::zip_equal(startIndices, extents))
      boundsOps.push_back(genBoundsOp(startIndex, extent));

    if (shapeShiftOp != nullptr)
      targetShapeCreationInfo.startIndices = std::move(startIndices);
    targetShapeCreationInfo.extents = std::move(extents);
  }

  mlir::omp::MapInfoOp genMapInfoOpForLiveIn(
      mlir::ConversionPatternRewriter &rewriter, mlir::Value liveIn,
      const llvm::DenseMap<mlir::Value, std::string> &liveInToName,
      TargetDeclareShapeCreationInfo &targetShapeCreationInfo) const {
    mlir::Value rawAddr = liveIn;
    mlir::Value shape = nullptr;
    std::string name = "";

    mlir::Operation *liveInDefiningOp = liveIn.getDefiningOp();
    auto declareOp =
        mlir::dyn_cast_if_present<hlfir::DeclareOp>(liveInDefiningOp);

    if (declareOp != nullptr) {
      // Use the raw address to avoid unboxing `fir.box` values whenever
      // possible. Put differently, if we have access to the direct value memory
      // reference/address, we use it.
      rawAddr = declareOp.getOriginalBase();
      shape = declareOp.getShape();
      name = declareOp.getUniqName().str();
    } else if (liveInToName.contains(liveIn))
      name = liveInToName.at(liveIn);

    if (!llvm::isa<mlir::omp::PointerLikeType>(rawAddr.getType())) {
      fir::FirOpBuilder builder(
          rewriter, fir::getKindMapping(
                        liveInDefiningOp->getParentOfType<mlir::ModuleOp>()));
      builder.setInsertionPointAfter(liveInDefiningOp);
      auto copyVal = builder.createTemporary(liveIn.getLoc(), liveIn.getType());
      builder.createStoreWithConvert(copyVal.getLoc(), liveIn, copyVal);
      rawAddr = copyVal;
    }

    mlir::Type liveInType = liveIn.getType();
    mlir::Type eleType = liveInType;
    if (auto refType = mlir::dyn_cast<fir::ReferenceType>(liveInType))
      eleType = refType.getElementType();

    llvm::omp::OpenMPOffloadMappingFlags mapFlag =
        llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_IMPLICIT;
    mlir::omp::VariableCaptureKind captureKind =
        mlir::omp::VariableCaptureKind::ByRef;

    if (fir::isa_trivial(eleType) || fir::isa_char(eleType)) {
      captureKind = mlir::omp::VariableCaptureKind::ByCopy;
    } else if (!fir::isa_builtin_cptr_type(eleType)) {
      mapFlag |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_TO;
      mapFlag |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_FROM;
    }

    llvm::SmallVector<mlir::Value> boundsOps;
    genBoundsOps(rewriter, liveIn.getLoc(), shape, boundsOps,
                 targetShapeCreationInfo);

    return Fortran::lower::omp ::internal::createMapInfoOp(
        rewriter, liveIn.getLoc(), rawAddr,
        /*varPtrPtr=*/{}, name, boundsOps,
        /*members=*/{},
        /*membersIndex=*/mlir::DenseIntElementsAttr{},
        static_cast<
            std::underlying_type_t<llvm::omp::OpenMPOffloadMappingFlags>>(
            mapFlag),
        captureKind, rawAddr.getType());
  }

  mlir::omp::TargetOp
  genTargetOp(mlir::Location loc, mlir::ConversionPatternRewriter &rewriter,
              mlir::IRMapping &mapper,
              const llvm::ArrayRef<mlir::Value> liveIns,
              const mlir::omp::TargetOperands &clauseOps,
              const LiveInShapeInfoMap &liveInShapeInfoMap) const {
    auto targetOp = rewriter.create<mlir::omp::TargetOp>(loc, clauseOps);

    mlir::Region &region = targetOp.getRegion();

    llvm::SmallVector<mlir::Type> liveInTypes;
    llvm::SmallVector<mlir::Location> liveInLocs;

    for (mlir::Value mapInfoOp : clauseOps.mapVars) {
      auto miOp = mlir::cast<mlir::omp::MapInfoOp>(mapInfoOp.getDefiningOp());
      liveInTypes.push_back(miOp.getVarPtr().getType());
      liveInLocs.push_back(miOp.getVarPtr().getLoc());
    }

    rewriter.createBlock(&region, {}, liveInTypes, liveInLocs);
    fir::FirOpBuilder builder(
        rewriter,
        fir::getKindMapping(targetOp->getParentOfType<mlir::ModuleOp>()));

    size_t argIdx = 0;
    for (auto [liveIn, mapInfoOp] :
         llvm::zip_equal(liveIns, clauseOps.mapVars)) {
      auto miOp = mlir::cast<mlir::omp::MapInfoOp>(mapInfoOp.getDefiningOp());
      hlfir::DeclareOp liveInDeclare =
          genLiveInDeclare(builder, targetOp, region.getArgument(argIdx), miOp,
                           liveInShapeInfoMap.at(liveIn));

      // TODO If `liveIn.getDefiningOp()` is a `fir::BoxAddrOp`, we probably
      // need to "unpack" the box by getting the defining op of it's value.
      // However, we did not hit this case in reality yet so leaving it as a
      // todo for now.

      if (!llvm::isa<mlir::omp::PointerLikeType>(liveIn.getType()))
        mapper.map(liveIn, builder.loadIfRef(liveIn.getLoc(),
                                             liveInDeclare.getOriginalBase()));
      else
        mapper.map(liveIn, liveInDeclare.getOriginalBase());

      if (auto origDeclareOp = mlir::dyn_cast_if_present<hlfir::DeclareOp>(
              liveIn.getDefiningOp())) {
        mapper.map(origDeclareOp.getBase(), liveInDeclare.getBase());
      }
      ++argIdx;
    }

    Fortran::lower::omp::internal::cloneOrMapRegionOutsiders(builder, targetOp);
    rewriter.setInsertionPoint(
        rewriter.create<mlir::omp::TerminatorOp>(targetOp.getLoc()));

    return targetOp;
  }

  hlfir::DeclareOp genLiveInDeclare(
      fir::FirOpBuilder &builder, mlir::omp::TargetOp targetOp,
      mlir::Value liveInArg, mlir::omp::MapInfoOp liveInMapInfoOp,
      const TargetDeclareShapeCreationInfo &targetShapeCreationInfo) const {
    mlir::Type liveInType = liveInArg.getType();
    std::string liveInName = liveInMapInfoOp.getName().has_value()
                                 ? liveInMapInfoOp.getName().value().str()
                                 : std::string("");

    if (fir::isa_ref_type(liveInType))
      liveInType = fir::unwrapRefType(liveInType);

    mlir::Value shape = [&]() -> mlir::Value {
      if (!targetShapeCreationInfo.isShapedValue())
        return {};

      llvm::SmallVector<mlir::Value> extentOperands;
      llvm::SmallVector<mlir::Value> startIndexOperands;

      if (targetShapeCreationInfo.isShapeShiftedValue()) {
        llvm::SmallVector<mlir::Value> shapeShiftOperands;

        size_t shapeIdx = 0;
        for (auto [startIndex, extent] :
             llvm::zip_equal(targetShapeCreationInfo.startIndices,
                             targetShapeCreationInfo.extents)) {
          shapeShiftOperands.push_back(
              Fortran::lower::omp::internal::mapTemporaryValue(
                  builder, targetOp, startIndex,
                  liveInName + ".start_idx.dim" + std::to_string(shapeIdx)));
          shapeShiftOperands.push_back(
              Fortran::lower::omp::internal::mapTemporaryValue(
                  builder, targetOp, extent,
                  liveInName + ".extent.dim" + std::to_string(shapeIdx)));
          ++shapeIdx;
        }

        auto shapeShiftType = fir::ShapeShiftType::get(
            builder.getContext(), shapeShiftOperands.size() / 2);
        return builder.create<fir::ShapeShiftOp>(
            liveInArg.getLoc(), shapeShiftType, shapeShiftOperands);
      }

      llvm::SmallVector<mlir::Value> shapeOperands;

      size_t shapeIdx = 0;
      for (auto extent : targetShapeCreationInfo.extents) {
        shapeOperands.push_back(
            Fortran::lower::omp::internal::mapTemporaryValue(
                builder, targetOp, extent,
                liveInName + ".extent.dim" + std::to_string(shapeIdx)));
        ++shapeIdx;
      }

      return builder.create<fir::ShapeOp>(liveInArg.getLoc(), shapeOperands);
    }();

    return builder.create<hlfir::DeclareOp>(liveInArg.getLoc(), liveInArg,
                                            liveInName, shape);
  }

  mlir::omp::TeamsOp
  genTeamsOp(mlir::Location loc,
             mlir::ConversionPatternRewriter &rewriter) const {
    auto teamsOp = rewriter.create<mlir::omp::TeamsOp>(
        loc, /*clauses=*/mlir::omp::TeamsOperands{});

    rewriter.createBlock(&teamsOp.getRegion());
    rewriter.setInsertionPoint(rewriter.create<mlir::omp::TerminatorOp>(loc));

    return teamsOp;
  }

  void genLoopNestClauseOps(
      mlir::Location loc, mlir::ConversionPatternRewriter &rewriter,
      looputils::LoopNestToIndVarMap &loopNest, mlir::IRMapping &mapper,
      mlir::omp::LoopNestOperands &loopNestClauseOps) const {
    assert(loopNestClauseOps.loopLowerBounds.empty() &&
           "Loop nest bounds were already emitted!");

    // Clones the chain of ops defining a certain loop bound or its step into
    // the parallel region. For example, if the value of a bound is defined by a
    // `fir.convert`op, this lambda clones the `fir.convert` as well as the
    // value it converts from. We do this since `omp.target` regions are
    // isolated from above.
    auto cloneBoundOrStepOpChain =
        [&](mlir::Operation *operation) -> mlir::Operation * {
      llvm::SetVector<mlir::Operation *> opChain;
      looputils::collectIndirectConstOpChain(operation, opChain);

      mlir::Operation *result;
      for (mlir::Operation *link : opChain) {
        result = rewriter.clone(*link, mapper);
      }

      return result;
    };

    for (auto &[doLoop, _] : loopNest) {
      auto addBoundsOrStep =
          [&](mlir::Value value,
              llvm::SmallVectorImpl<mlir::Value> &boundsOrStepVec) {
            if (mapper.contains(value))
              boundsOrStepVec.push_back(mapper.lookup(value));
            else {
              mlir::Operation *definingOp = value.getDefiningOp();
              boundsOrStepVec.push_back(
                  cloneBoundOrStepOpChain(definingOp)->getResult(0));
            }
          };

      addBoundsOrStep(doLoop.getLowerBound(),
                      loopNestClauseOps.loopLowerBounds);
      addBoundsOrStep(doLoop.getUpperBound(),
                      loopNestClauseOps.loopUpperBounds);
      addBoundsOrStep(doLoop.getStep(), loopNestClauseOps.loopSteps);
    }

    loopNestClauseOps.loopInclusive = rewriter.getUnitAttr();
  }

  mlir::omp::DistributeOp
  genDistributeOp(mlir::Location loc,
                  mlir::ConversionPatternRewriter &rewriter) const {
    auto distOp = rewriter.create<mlir::omp::DistributeOp>(
        loc, /*clauses=*/mlir::omp::DistributeOperands{});

    rewriter.createBlock(&distOp.getRegion());
    rewriter.setInsertionPoint(rewriter.create<mlir::omp::TerminatorOp>(loc));

    return distOp;
  }

  void genLoopNestIndVarAllocs(mlir::ConversionPatternRewriter &rewriter,
                               looputils::LoopNestToIndVarMap &loopNest,
                               mlir::IRMapping &mapper) const {

    for (auto &[_, indVarInfo] : loopNest)
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

  mlir::omp::ParallelOp
  genParallelOp(mlir::Location loc, mlir::ConversionPatternRewriter &rewriter,
                looputils::LoopNestToIndVarMap &loopNest,
                mlir::IRMapping &mapper,
                mlir::omp::LoopNestOperands &loopNestClauseOps) const {
    auto parallelOp = rewriter.create<mlir::omp::ParallelOp>(loc);
    rewriter.createBlock(&parallelOp.getRegion());
    rewriter.setInsertionPoint(rewriter.create<mlir::omp::TerminatorOp>(loc));

    genLoopNestIndVarAllocs(rewriter, loopNest, mapper);
    genLoopNestClauseOps(loc, rewriter, loopNest, mapper, loopNestClauseOps);

    return parallelOp;
  }

  mlir::omp::LoopNestOp
  genWsLoopOp(mlir::ConversionPatternRewriter &rewriter, fir::DoLoopOp doLoop,
              mlir::IRMapping &mapper,
              const mlir::omp::LoopNestOperands &clauseOps,
              bool isComposite) const {

    auto wsloopOp = rewriter.create<mlir::omp::WsloopOp>(doLoop.getLoc());
    wsloopOp.setComposite(isComposite);
    rewriter.createBlock(&wsloopOp.getRegion());
    rewriter.setInsertionPoint(
        rewriter.create<mlir::omp::TerminatorOp>(wsloopOp.getLoc()));

    auto loopNestOp =
        rewriter.create<mlir::omp::LoopNestOp>(doLoop.getLoc(), clauseOps);

    // Clone the loop's body inside the loop nest construct using the
    // mapped values.
    rewriter.cloneRegionBefore(doLoop.getRegion(), loopNestOp.getRegion(),
                               loopNestOp.getRegion().begin(), mapper);

    mlir::Operation *terminator = loopNestOp.getRegion().back().getTerminator();
    rewriter.setInsertionPointToEnd(&loopNestOp.getRegion().back());
    rewriter.create<mlir::omp::YieldOp>(terminator->getLoc());
    rewriter.eraseOp(terminator);

    return loopNestOp;
  }

  bool mapToDevice;
  llvm::DenseSet<fir::DoLoopOp> &concurrentLoopsToSkip;
};

class DoConcurrentConversionPass
    : public flangomp::impl::DoConcurrentConversionPassBase<
          DoConcurrentConversionPass> {
public:
  using flangomp::impl::DoConcurrentConversionPassBase<
      DoConcurrentConversionPass>::DoConcurrentConversionPassBase;

  DoConcurrentConversionPass() = default;

  DoConcurrentConversionPass(
      const flangomp::DoConcurrentConversionPassOptions &options)
      : DoConcurrentConversionPassBase(options) {}

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();

    if (func.isDeclaration()) {
      return;
    }

    auto *context = &getContext();

    if (mapTo != flangomp::DoConcurrentMappingKind::DCMK_Host &&
        mapTo != flangomp::DoConcurrentMappingKind::DCMK_Device) {
      mlir::emitWarning(mlir::UnknownLoc::get(context),
                        "DoConcurrentConversionPass: invalid `map-to` value. "
                        "Valid values are: `host` or `device`");
      return;
    }
    llvm::DenseSet<fir::DoLoopOp> concurrentLoopsToSkip;
    mlir::RewritePatternSet patterns(context);
    patterns.insert<DoConcurrentConversion>(
        context, mapTo == flangomp::DoConcurrentMappingKind::DCMK_Device,
        concurrentLoopsToSkip);
    mlir::ConversionTarget target(*context);
    target.addLegalDialect<
        fir::FIROpsDialect, hlfir::hlfirDialect, mlir::arith::ArithDialect,
        mlir::func::FuncDialect, mlir::omp::OpenMPDialect,
        mlir::cf::ControlFlowDialect, mlir::math::MathDialect>();

    target.addDynamicallyLegalOp<fir::DoLoopOp>([&](fir::DoLoopOp op) {
      return !op.getUnordered() || concurrentLoopsToSkip.contains(op);
    });

    if (mlir::failed(mlir::applyFullConversion(getOperation(), target,
                                               std::move(patterns)))) {
      mlir::emitError(mlir::UnknownLoc::get(context),
                      "error in converting do-concurrent op");
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
