//===-- include/flang/Utils/OpenMP.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_UTILS_OPENMP_H_
#define FORTRAN_UTILS_OPENMP_H_

#include "mlir/Dialect/OpenMP/OpenMPDialect.h"

#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"

#include <vector>

namespace mlir {
class RewriterBase;
} // namespace mlir

namespace fir {
class FirOpBuilder;
class RecordType;
} // namespace fir

namespace Fortran::utils::openmp {
// TODO We can probably move the stuff inside `Support/OpenMP-utils.h/.cpp` here
// as well.

/// Create an `omp.map.info` op. Parameters other than the ones documented below
/// correspond to operation arguments in the OpenMPOps.td file, see op docs for
/// more details.
///
/// \param [in] builder - MLIR operation builder.
/// \param [in] loc     - Source location of the created op.
mlir::omp::MapInfoOp createMapInfoOp(mlir::OpBuilder &builder,
    mlir::Location loc, mlir::Value baseAddr, mlir::Value varPtrPtr,
    llvm::StringRef name, llvm::ArrayRef<mlir::Value> bounds,
    llvm::ArrayRef<mlir::Value> members, mlir::ArrayAttr membersIndex,
    mlir::omp::ClauseMapFlags mapType,
    mlir::omp::VariableCaptureKind mapCaptureType, mlir::Type retTy,
    bool partialMap = false,
    mlir::FlatSymbolRefAttr mapperId = mlir::FlatSymbolRefAttr());

/// For an mlir value that does not have storage, allocate temporary storage
/// (outside the target region), store the value in that storage, and map the
/// storage to the target region.
///
/// \param firOpBuilder - Operation builder.
/// \param targetOp     - Target op to which the temporary value is mapped.
/// \param val          - Temp value that should be mapped to the target region.
/// \param name         - A string used to identify the created `omp.map.info`
/// op.
///
/// \returns The loaded mapped value inside the target region.
mlir::Value mapTemporaryValue(fir::FirOpBuilder &firOpBuilder,
    mlir::omp::TargetOp targetOp, mlir::Value val,
    llvm::StringRef name = "tmp.map");

/// For values used inside a target region but defined outside, either clone
/// these value inside the target region or map them to the region. This
/// function first tries to clone values (if they are defined by
/// memory-effect-free ops, otherwise, the values are mapped.
///
/// \param firOpBuilder - Operation builder.
/// \param targetOp     - The target that needs to be extended by clones and/or
/// maps.
void cloneOrMapRegionOutsiders(
    fir::FirOpBuilder &firOpBuilder, mlir::omp::TargetOp targetOp);

using RecordMemberMapperMangler =
    std::function<void(std::string &mapperId, llvm::StringRef memberName)>;

mlir::FlatSymbolRefAttr getOrGenImplicitDefaultDeclareMapper(
    fir::FirOpBuilder &firOpBuilder, mlir::Location loc,
    fir::RecordType recordType, llvm::StringRef mapperNameStr,
    RecordMemberMapperMangler mangler = {});

struct LiveInShapeInfo {
  // Note: We use `std::vector` (rather than `llvm::SmallVector` as usual) to
  // interface more easily `ShapeShiftOp::getOrigins()` which returns
  // `std::vector`.
  std::vector<mlir::Value> startIndices;
  std::vector<mlir::Value> extents;

  LiveInShapeInfo(mlir::Value liveIn);

  bool isShapedValue() const { return !extents.empty(); }
  bool isShapeShiftedValue() const { return !startIndices.empty(); }
};

using LiveInShapeInfoMap = llvm::DenseMap<mlir::Value, LiveInShapeInfo>;

/// Build an `omp.map.info` op that maps a value defined outside a target
/// region (a "live-in") into the region. Handles trivial scalars, character
/// types, arrays, and Fortran derived types (using an implicit declare
/// mapper for records with allocatable components).
mlir::omp::MapInfoOp genMapInfoOpForLiveIn(fir::FirOpBuilder &builder,
    mlir::Value liveIn, bool isReductionVar = false);

/// The two SSA handles a device-side declare exposes for a mapped live-in. For
/// an `hlfir.declare` these are its two distinct results (`originalBase` and
/// `base`); for a single-result declare such as `fir.declare` both fields alias
/// that one result, so callers can treat either uniformly.
struct LiveInDeclareResult {
  mlir::Value originalBase;
  mlir::Value base;
};

/// Factory that emits the device-side declare op for a mapped live-in inside
/// the target region. `genTargetOpFromLiveIns` rebuilds the live-in's `shape`
/// (with its extents/origins already mapped into the region) and hands it,
/// along with the region block argument (`liveInArg`) and the live-in's `name`,
/// to this callback -- which owns the choice of declare flavor. For example,
/// `DoConcurrentConversion` (before HLFIR-to-FIR) emits `hlfir.declare`, while
/// `LowerWorkdistribute` (after HLFIR-to-FIR) emits `fir.declare`; emitting
/// `hlfir.declare` that late would leave HLFIR ops that fail FIR-to-LLVM
/// legalization.
using LiveInDeclareBuilder = llvm::function_ref<LiveInDeclareResult(
    fir::FirOpBuilder &builder, mlir::Location loc, mlir::Value liveInArg,
    llvm::StringRef name, mlir::Value shape)>;

/// Create an `omp.target` op whose region maps the given `liveIns` (matched
/// 1:1 with `clauseOps.mapVars`) by emitting a device-side declare for each via
/// `declareBuilder`, populating `mapper` so users of the live-ins inside the
/// region see the device-side declares, and cloning/mapping any remaining
/// outsiders.
///
/// `loopNestClauseOps`'s loop bounds/steps are also remapped through `mapper`.
mlir::omp::TargetOp genTargetOpFromLiveIns(mlir::Location loc,
    mlir::RewriterBase &rewriter, mlir::IRMapping &mapper,
    llvm::ArrayRef<mlir::Value> liveIns,
    mlir::omp::TargetExtOperands &clauseOps,
    mlir::omp::LoopNestOperands &loopNestClauseOps,
    const LiveInShapeInfoMap &liveInShapeInfoMap,
    LiveInDeclareBuilder declareBuilder);

} // namespace Fortran::utils::openmp

#endif // FORTRAN_UTILS_OPENMP_H_
