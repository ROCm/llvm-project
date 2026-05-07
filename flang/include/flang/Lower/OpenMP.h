//===-- Lower/OpenMP.h -- lower Open MP directives --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_OPENMP_H
#define FORTRAN_LOWER_OPENMP_H

#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cinttypes>
#include <optional>
#include <utility>

namespace mlir {
class Operation;
class Location;
class Type;
namespace omp {
enum class DeclareTargetDeviceType : uint32_t;
enum class DeclareTargetCaptureClause : uint32_t;
} // namespace omp
} // namespace mlir

namespace fir {
class FirOpBuilder;
} // namespace fir

namespace Fortran {
namespace parser {
struct OpenMPConstruct;
struct OpenMPDeclarativeConstruct;
struct OmpClauseList;
} // namespace parser

namespace semantics {
class Symbol;
class Scope;
class SemanticsContext;
} // namespace semantics

namespace lower {

class AbstractConverter;
class SymMap;

namespace pft {
struct Evaluation;
struct Variable;
} // namespace pft

struct OMPDeferredDeclareTargetInfo {
  mlir::omp::DeclareTargetCaptureClause declareTargetCaptureClause;
  mlir::omp::DeclareTargetDeviceType declareTargetDeviceType;
  bool automap = false;
  const Fortran::semantics::Symbol &sym;
};

// Generate the OpenMP terminator for Operation at Location.
mlir::Operation *genOpenMPTerminator(fir::FirOpBuilder &, mlir::Operation *,
                                     mlir::Location);

void genOpenMPConstruct(AbstractConverter &, Fortran::lower::SymMap &,
                        semantics::SemanticsContext &, pft::Evaluation &,
                        const parser::OpenMPConstruct &);
void genOpenMPDeclarativeConstruct(AbstractConverter &,
                                   Fortran::lower::SymMap &,
                                   semantics::SemanticsContext &,
                                   pft::Evaluation &,
                                   const parser::OpenMPDeclarativeConstruct &);
/// Symbols in OpenMP code can have flags (e.g. threadprivate directive)
/// that require additional handling when lowering the corresponding
/// variable. Perform such handling according to the flags on the symbol.
/// The variable \p var is required to have a `Symbol`.
void genOpenMPSymbolProperties(AbstractConverter &converter,
                               const pft::Variable &var);

void genGroupprivateOp(AbstractConverter &, const pft::Variable &);
void genThreadprivateOp(AbstractConverter &, const pft::Variable &);
void genDeclareTargetIntGlobal(AbstractConverter &, const pft::Variable &);
bool isOpenMPTargetConstruct(const parser::OpenMPConstruct &);
bool isOpenMPDeviceDeclareTarget(Fortran::lower::AbstractConverter &,
                                 Fortran::semantics::SemanticsContext &,
                                 Fortran::lower::pft::Evaluation &,
                                 const parser::OpenMPDeclarativeConstruct &);
void gatherOpenMPDeferredDeclareTargets(
    Fortran::lower::AbstractConverter &, Fortran::semantics::SemanticsContext &,
    Fortran::lower::pft::Evaluation &,
    const parser::OpenMPDeclarativeConstruct &,
    llvm::SmallVectorImpl<OMPDeferredDeclareTargetInfo> &);
bool markOpenMPDeferredDeclareTargetFunctions(
    mlir::Operation *, llvm::SmallVectorImpl<OMPDeferredDeclareTargetInfo> &,
    AbstractConverter &);
void genOpenMPRequires(mlir::Operation *, const Fortran::semantics::Symbol *);

/// Information registered by the Flang lowering of the OpenMP 5.0+
/// `!$omp allocators` construct for each allocatable listed in an ALLOCATE
/// clause.  The lowering of the matching ALLOCATE statement (in
/// lib/Lower/Allocatable.cpp) queries this side table via
/// lookupOmpAllocatorInfo(); on a hit it routes the allocation through the
/// Fortran runtime (rather than the fir.allocmem inline fast path) and
/// emits a _FortranAOmpAllocatorStamp(descriptor, handle, align) call so
/// that Descriptor::Allocate dispatches through __kmpc_alloc /
/// __kmpc_aligned_alloc with the recorded handle and alignment.
struct OmpAllocatorInfo {
  /// MLIR value holding the OpenMP allocator handle (uintptr_t at the
  /// runtime ABI boundary).  Non-null when the entry is live.
  mlir::Value handle;
  /// MLIR value holding the requested alignment in bytes (size_t at the
  /// runtime ABI boundary).  Null means "default alignment" which the
  /// runtime translates into a plain __kmpc_alloc call.
  mlir::Value align;
};

/// Register \p info as the active OpenMP allocator binding for \p sym.  A
/// subsequent lookupOmpAllocatorInfo(sym) returns the previously registered
/// info (or nullopt, if there is none); callers are responsible for calling
/// unregisterOmpAllocatorInfo(sym, prev) on scope exit so that nested
/// `!$omp allocators` constructs restore the outer binding correctly.
///
/// Returns the previously registered info so the caller can save/restore.
std::optional<OmpAllocatorInfo>
registerOmpAllocatorInfo(const Fortran::semantics::Symbol &sym,
                         OmpAllocatorInfo info);

/// Restore a previously registered info (or remove the entry if \p prev is
/// nullopt).  Should be called at the matching scope exit.
void unregisterOmpAllocatorInfo(const Fortran::semantics::Symbol &sym,
                                std::optional<OmpAllocatorInfo> prev);

/// Look up the active OpenMP allocator binding for \p sym, if any.
/// Returns nullopt when the symbol is not currently inside an
/// `!$omp allocators` ALLOCATE clause.
std::optional<OmpAllocatorInfo>
lookupOmpAllocatorInfo(const Fortran::semantics::Symbol &sym);

/// Returns true if \p sym has ever appeared in an `!$omp allocators`
/// ALLOCATE clause in the current compilation unit.  This marker is
/// "sticky": once set it stays set for the rest of the lowering, even
/// after the construct's scope ends.  It is used by Allocatable.cpp to
/// force both ALLOCATE and DEALLOCATE statements for that symbol through
/// the Fortran runtime path (rather than the inline fir.allocmem /
/// fir.freemem fast paths), so that the descriptor's allocator-index
/// dispatch in Descriptor::Allocate / Descriptor::Deallocate properly
/// routes the call through __kmpc_alloc / __kmpc_aligned_alloc /
/// __kmpc_free.  Inline allocate/free would call malloc/free directly
/// and corrupt libomp's chunk metadata, leading to "double free or
/// corruption" errors at runtime.
bool isOmpAllocatorTouchedSymbol(const Fortran::semantics::Symbol &sym);

// Materialize omp.declare_mapper ops for mapper declarations found in
// imported modules. If \p scope is null, materialize for the whole
// semantics global scope; otherwise, operate recursively starting at \p scope.
void materializeOpenMPDeclareMappers(
    Fortran::lower::AbstractConverter &, Fortran::semantics::SemanticsContext &,
    const Fortran::semantics::Scope *scope = nullptr);

// Mark declare target globals and functions that were imported from a
// module file.
void markOpenMPImportedDeclareTargets(
    Fortran::lower::AbstractConverter &converter,
    semantics::SemanticsContext &semaCtx);

namespace omp {
/// If \p base carries OpenMP DECLARE VARIANT entries, return the variant symbol
/// that best matches the enclosing OpenMP context, or nullptr if none matches.
/// \p base is expected to have variant entries.
const Fortran::semantics::Symbol *
resolveDeclareVariantCallee(const Fortran::semantics::Symbol &base,
                            AbstractConverter &converter);
} // namespace omp

// Materialize (idempotently) the omp.declare_reduction op for one already-
// resolved imported user reduction and one requested element type, at module
// scope, for separate compilation. Only the per-type instance whose scoped op
// name equals \p requestedOpName is emitted, so a multi-type declaration's
// other listed types (and any unsupported sibling type) are left alone. No-op
// if the requested type's shape is not lowerable (a combiner-in-clause form or
// an unsupported element type) or the op already exists. Safe to call
// mid-function (uses createDeclareReductionHelper's InsertionGuard). \p
// isByRef selects the by-ref/by-value naming variant used by the clause.
void materializeUserReduction(
    Fortran::lower::AbstractConverter &, Fortran::semantics::SemanticsContext &,
    const Fortran::semantics::Symbol &resolvedReduction,
    llvm::StringRef requestedOpName, mlir::Type requestedType, bool isByRef);

} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_OPENMP_H
