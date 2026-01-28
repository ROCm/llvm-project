//===-- Bridge.cpp -- bridge to lower to MLIR -----------------------------===//
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

#include "flang/Lower/Bridge.h"

#include "flang/Lower/Allocatable.h"
#include "flang/Lower/CUDA.h"
#include "flang/Lower/CallInterface.h"
#include "flang/Lower/ConvertCall.h"
#include "flang/Lower/ConvertExpr.h"
#include "flang/Lower/ConvertExprToHLFIR.h"
#include "flang/Lower/ConvertType.h"
#include "flang/Lower/ConvertVariable.h"
#include "flang/Lower/DirectivesCommon.h"
#include "flang/Lower/HostAssociations.h"
#include "flang/Lower/IO.h"
#include "flang/Lower/IterationSpace.h"
#include "flang/Lower/Mangler.h"
#include "flang/Lower/MultiImageFortran.h"
#include "flang/Lower/OpenACC.h"
#include "flang/Lower/OpenMP.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Lower/Runtime.h"
#include "flang/Lower/StatementContext.h"
#include "flang/Lower/Support/ReductionProcessor.h"
#include "flang/Lower/Support/Utils.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/CUFCommon.h"
#include "flang/Optimizer/Builder/Character.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Runtime/Assign.h"
#include "flang/Optimizer/Builder/Runtime/Character.h"
#include "flang/Optimizer/Builder/Runtime/Derived.h"
#include "flang/Optimizer/Builder/Runtime/EnvironmentDefaults.h"
#include "flang/Optimizer/Builder/Runtime/Exceptions.h"
#include "flang/Optimizer/Builder/Runtime/Main.h"
#include "flang/Optimizer/Builder/Runtime/Ragged.h"
#include "flang/Optimizer/Builder/Runtime/Stop.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/CUF/Attributes/CUFAttr.h"
#include "flang/Optimizer/Dialect/CUF/CUFOps.h"
#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/Support/DataLayout.h"
#include "flang/Optimizer/Support/FatalError.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Parser/tools.h"
#include "flang/Runtime/iostat-consts.h"
#include "flang/Semantics/openmp-dsa.h"
#include "flang/Semantics/runtime-type-info.h"
#include "flang/Semantics/symbol.h"
#include "flang/Semantics/tools.h"
#include "flang/Support/Flags.h"
#include "flang/Support/Version.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/StateStack.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Target/TargetMachine.h"
#include <optional>

#define DEBUG_TYPE "flang-lower-bridge"

static llvm::cl::opt<bool> forceLoopToExecuteOnce(
    "always-execute-loop-body", llvm::cl::init(false),
    llvm::cl::desc("force the body of a loop to execute at least once"));

namespace {
/// Information for generating a structured or unstructured increment loop.
struct IncrementLoopInfo {
  template <typename T>
  explicit IncrementLoopInfo(Fortran::semantics::Symbol &sym, const T &lower,
                             const T &upper, const std::optional<T> &step,
                             bool isConcurrent = false)
      : loopVariableSym{&sym}, lowerExpr{Fortran::semantics::GetExpr(lower)},
        upperExpr{Fortran::semantics::GetExpr(upper)},
        stepExpr{Fortran::semantics::GetExpr(step)},
        isConcurrent{isConcurrent} {}

  IncrementLoopInfo(IncrementLoopInfo &&) = default;
  IncrementLoopInfo &operator=(IncrementLoopInfo &&x) = default;

  bool isStructured() const { return !headerBlock; }

  mlir::Type getLoopVariableType() const {
    assert(loopVariable && "must be set");
    return fir::unwrapRefType(loopVariable.getType());
  }

  bool hasLocalitySpecs() const {
    return !localSymList.empty() || !localInitSymList.empty() ||
           !reduceSymList.empty() || !sharedSymList.empty();
  }

  // Data members common to both structured and unstructured loops.
  const Fortran::semantics::Symbol *loopVariableSym;
  const Fortran::lower::SomeExpr *lowerExpr;
  const Fortran::lower::SomeExpr *upperExpr;
  const Fortran::lower::SomeExpr *stepExpr;
  const Fortran::lower::SomeExpr *maskExpr = nullptr;
  bool isConcurrent;
  llvm::SmallVector<const Fortran::semantics::Symbol *> localSymList;
  llvm::SmallVector<const Fortran::semantics::Symbol *> localInitSymList;
  llvm::SmallVector<const Fortran::semantics::Symbol *> reduceSymList;
  llvm::SmallVector<fir::ReduceOperationEnum> reduceOperatorList;
  llvm::SmallVector<const Fortran::semantics::Symbol *> sharedSymList;
  mlir::Value loopVariable = nullptr;

  // Data members for structured loops.
  mlir::Operation *loopOp = nullptr;

  // Data members for unstructured loops.
  bool hasRealControl = false;
  mlir::Value tripVariable = nullptr;
  mlir::Value stepVariable = nullptr;
  mlir::Block *headerBlock = nullptr; // loop entry and test block
  mlir::Block *maskBlock = nullptr;   // concurrent loop mask block
  mlir::Block *bodyBlock = nullptr;   // first loop body block
  mlir::Block *exitBlock = nullptr;   // loop exit target block
};

/// Information to support stack management, object deallocation, and
/// object finalization at early and normal construct exits.
struct ConstructContext {
  explicit ConstructContext(Fortran::lower::pft::Evaluation &eval,
                            Fortran::lower::StatementContext &stmtCtx)
      : eval{eval}, stmtCtx{stmtCtx} {}

  Fortran::lower::pft::Evaluation &eval;     // construct eval
  Fortran::lower::StatementContext &stmtCtx; // construct exit code
  std::optional<hlfir::Entity> selector;     // construct selector, if any.
  bool pushedScope = false; // was a scoped pushed for this construct?
};

/// Helper to gather the lower bounds of array components with non deferred
/// shape when they are not all ones. Return an empty array attribute otherwise.
static mlir::DenseI64ArrayAttr
gatherComponentNonDefaultLowerBounds(mlir::Location loc,
                                     mlir::MLIRContext *mlirContext,
                                     const Fortran::semantics::Symbol &sym) {
  if (Fortran::semantics::IsAllocatableOrObjectPointer(&sym))
    return {};
  mlir::DenseI64ArrayAttr lbs_attr;
  if (const auto *objDetails =
          sym.detailsIf<Fortran::semantics::ObjectEntityDetails>()) {
    llvm::SmallVector<std::int64_t> lbs;
    bool hasNonDefaultLbs = false;
    for (const Fortran::semantics::ShapeSpec &bounds : objDetails->shape())
      if (auto lb = bounds.lbound().GetExplicit()) {
        if (auto constant = Fortran::evaluate::ToInt64(*lb)) {
          hasNonDefaultLbs |= (*constant != 1);
          lbs.push_back(*constant);
        } else {
          TODO(loc, "generate fir.dt_component for length parametrized derived "
                    "types");
        }
      }
    if (hasNonDefaultLbs) {
      assert(static_cast<int>(lbs.size()) == sym.Rank() &&
             "expected component bounds to be constant or deferred");
      lbs_attr = mlir::DenseI64ArrayAttr::get(mlirContext, lbs);
    }
  }
  return lbs_attr;
}

// Helper class to generate name of fir.global containing component explicit
// default value for objects, and initial procedure target for procedure pointer
// components.
static mlir::FlatSymbolRefAttr gatherComponentInit(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::semantics::Symbol &sym, fir::RecordType derivedType) {
  mlir::MLIRContext *mlirContext = &converter.getMLIRContext();
  // Return procedure target mangled name for procedure pointer components.
  if (const auto *procPtr =
          sym.detailsIf<Fortran::semantics::ProcEntityDetails>()) {
    if (std::optional<const Fortran::semantics::Symbol *> maybeInitSym =
            procPtr->init()) {
      // So far, do not make distinction between p => NULL() and p without init,
      // f18 always initialize pointers to NULL anyway.
      if (!*maybeInitSym)
        return {};
      return mlir::FlatSymbolRefAttr::get(mlirContext,
                                          converter.mangleName(**maybeInitSym));
    }
  }

  const auto *objDetails =
      sym.detailsIf<Fortran::semantics::ObjectEntityDetails>();
  if (!objDetails || !objDetails->init().has_value())
    return {};
  // Object component initial value. Semantic package component object default
  // value into compiler generated symbols that are lowered as read-only
  // fir.global. Get the name of this global.
  std::string name = fir::NameUniquer::getComponentInitName(
      derivedType.getName(), toStringRef(sym.name()));
  return mlir::FlatSymbolRefAttr::get(mlirContext, name);
}

/// Emit fir.use_stmt operations for USE statements in the given function unit
static void
emitUseStatementsFromFunit(Fortran::lower::AbstractConverter &converter,
                           mlir::OpBuilder &builder, mlir::Location loc,
                           const Fortran::lower::pft::FunctionLikeUnit &funit) {
  mlir::MLIRContext *context = builder.getContext();
  const Fortran::semantics::Scope &scope = funit.getScope();

  for (const auto &preservedStmt : funit.preservedUseStmts) {

    auto getMangledName = [&](const std::string &localName) -> std::string {
      Fortran::parser::CharBlock charBlock{localName.data(), localName.size()};
      const auto *sym = scope.FindSymbol(charBlock);
      if (!sym)
        return "";

      const auto &ultimateSym = sym->GetUltimate();

      // Skip cases which can cause mangleName to fail.
      if (ultimateSym.has<Fortran::semantics::DerivedTypeDetails>())
        return "";

      if (ultimateSym.has<Fortran::semantics::UseErrorDetails>())
        return "";

      if (const auto *generic =
              ultimateSym.detailsIf<Fortran::semantics::GenericDetails>()) {
        if (!generic->specific())
          return "";
      }

      return converter.mangleName(ultimateSym);
    };

    mlir::StringAttr moduleNameAttr =
        mlir::StringAttr::get(context, preservedStmt.moduleName);

    llvm::SmallVector<mlir::Attribute> onlySymbolAttrs;
    llvm::SmallVector<mlir::Attribute> renameAttrs;

    // Handle only
    for (const auto &name : preservedStmt.onlyNames) {
      std::string mangledName = getMangledName(name);
      if (!mangledName.empty())
        onlySymbolAttrs.push_back(
            mlir::FlatSymbolRefAttr::get(context, mangledName));
    }

    // Handle renames
    for (const auto &local : preservedStmt.renames) {
      std::string mangledName = getMangledName(local);
      if (!mangledName.empty()) {
        auto localAttr = mlir::StringAttr::get(context, local);
        auto symbolRef = mlir::FlatSymbolRefAttr::get(context, mangledName);
        renameAttrs.push_back(
            fir::UseRenameAttr::get(context, localAttr, symbolRef));
      }
    }

    // Create optional array attributes
    mlir::ArrayAttr onlySymbolsAttr =
        onlySymbolAttrs.empty()
            ? mlir::ArrayAttr()
            : mlir::ArrayAttr::get(context, onlySymbolAttrs);
    mlir::ArrayAttr renamesAttr =
        renameAttrs.empty() ? mlir::ArrayAttr()
                            : mlir::ArrayAttr::get(context, renameAttrs);

    fir::UseStmtOp::create(builder, loc, moduleNameAttr, onlySymbolsAttr,
                           renamesAttr);
  }
}

/// Helper class to generate the runtime type info global data and the
/// fir.type_info operations that contain the dipatch tables (if any).
/// The type info global data is required to describe the derived type to the
/// runtime so that it can operate over it.
/// It must be ensured these operations will be generated for every derived type
/// lowered in the current translated unit. However, these operations
/// cannot be generated before FuncOp have been created for functions since the
/// initializers may take their address (e.g for type bound procedures). This
/// class allows registering all the required type info while it is not
/// possible to create GlobalOp/TypeInfoOp, and to generate this data afte
/// function lowering.
class TypeInfoConverter {
  /// Store the location and symbols of derived type info to be generated.
  /// The location of the derived type instantiation is also stored because
  /// runtime type descriptor symbols are compiler generated and cannot be
  /// mapped to user code on their own.
  struct TypeInfo {
    Fortran::semantics::SymbolRef symbol;
    const Fortran::semantics::DerivedTypeSpec &typeSpec;
    fir::RecordType type;
    mlir::Location loc;
  };

public:
  void registerTypeInfo(Fortran::lower::AbstractConverter &converter,
                        mlir::Location loc,
                        Fortran::semantics::SymbolRef typeInfoSym,
                        const Fortran::semantics::DerivedTypeSpec &typeSpec,
                        fir::RecordType type) {
    if (seen.contains(typeInfoSym))
      return;
    seen.insert(typeInfoSym);
    currentTypeInfoStack->emplace_back(
        TypeInfo{typeInfoSym, typeSpec, type, loc});
    return;
  }

  void createTypeInfo(Fortran::lower::AbstractConverter &converter) {
    createTypeInfoForTypeDescriptorBuiltinType(converter);
    while (!registeredTypeInfoA.empty()) {
      currentTypeInfoStack = &registeredTypeInfoB;
      for (const TypeInfo &info : registeredTypeInfoA)
        createTypeInfoOpAndGlobal(converter, info);
      registeredTypeInfoA.clear();
      currentTypeInfoStack = &registeredTypeInfoA;
      for (const TypeInfo &info : registeredTypeInfoB)
        createTypeInfoOpAndGlobal(converter, info);
      registeredTypeInfoB.clear();
    }
  }

private:
  void createTypeInfoOpAndGlobal(Fortran::lower::AbstractConverter &converter,
                                 const TypeInfo &info) {
    if (!converter.getLoweringOptions().getSkipExternalRttiDefinition())
      Fortran::lower::createRuntimeTypeInfoGlobal(converter, info.symbol.get());
    createTypeInfoOp(converter, info);
  }

  void createTypeInfoForTypeDescriptorBuiltinType(
      Fortran::lower::AbstractConverter &converter) {
    if (registeredTypeInfoA.empty())
      return;
    auto builtinTypeInfoType = llvm::cast<fir::RecordType>(
        converter.genType(registeredTypeInfoA[0].symbol.get()));
    converter.getFirOpBuilder().createTypeInfoOp(
        registeredTypeInfoA[0].loc, builtinTypeInfoType,
        /*parentType=*/fir::RecordType{});
  }

  void createTypeInfoOp(Fortran::lower::AbstractConverter &converter,
                        const TypeInfo &info) {
    fir::RecordType parentType{};
    if (const Fortran::semantics::DerivedTypeSpec *parent =
            Fortran::evaluate::GetParentTypeSpec(info.typeSpec))
      parentType = mlir::cast<fir::RecordType>(converter.genType(*parent));

    fir::FirOpBuilder &builder = converter.getFirOpBuilder();
    fir::TypeInfoOp dt;
    mlir::OpBuilder::InsertPoint insertPointIfCreated;
    std::tie(dt, insertPointIfCreated) =
        builder.createTypeInfoOp(info.loc, info.type, parentType);
    if (!insertPointIfCreated.isSet())
      return; // fir.type_info was already built in a previous call.

    // Set abstract, init, destroy, and nofinal attributes.
    const Fortran::semantics::Symbol &dtSymbol = info.typeSpec.typeSymbol();
    if (dtSymbol.attrs().test(Fortran::semantics::Attr::ABSTRACT))
      dt->setAttr(dt.getAbstractAttrName(), builder.getUnitAttr());

    if (!info.typeSpec.HasDefaultInitialization(/*ignoreAllocatable=*/false,
                                                /*ignorePointer=*/false))
      dt->setAttr(dt.getNoInitAttrName(), builder.getUnitAttr());
    if (!info.typeSpec.HasDestruction())
      dt->setAttr(dt.getNoDestroyAttrName(), builder.getUnitAttr());
    if (!Fortran::semantics::MayRequireFinalization(info.typeSpec))
      dt->setAttr(dt.getNoFinalAttrName(), builder.getUnitAttr());

    const Fortran::semantics::Scope &derivedScope =
        DEREF(info.typeSpec.GetScope());

    // Fill binding table region if the derived type has bindings.
    Fortran::semantics::SymbolVector bindings =
        Fortran::semantics::CollectBindings(derivedScope);
    if (!bindings.empty()) {
      builder.createBlock(&dt.getDispatchTable());
      for (const Fortran::semantics::SymbolRef &binding : bindings) {
        const auto &details =
            binding.get().get<Fortran::semantics::ProcBindingDetails>();
        std::string tbpName = binding.get().name().ToString();
        if (details.numPrivatesNotOverridden() > 0)
          tbpName += "."s + std::to_string(details.numPrivatesNotOverridden());
        std::string bindingName = converter.mangleName(details.symbol());
        auto dtEntry = fir::DTEntryOp::create(
            builder, info.loc,
            mlir::StringAttr::get(builder.getContext(), tbpName),
            mlir::SymbolRefAttr::get(builder.getContext(), bindingName));
        // Propagate DEFERRED attribute on the binding to fir.dt_entry.
        if (binding.get().attrs().test(Fortran::semantics::Attr::DEFERRED))
          dtEntry->setAttr(fir::DTEntryOp::getDeferredAttrNameStr(),
                           builder.getUnitAttr());
      }
      fir::FirEndOp::create(builder, info.loc);
    }
    // Gather info about components that is not reflected in fir.type and may be
    // needed later: component initial values and array component non default
    // lower bounds.
    mlir::Block *componentInfo = nullptr;
    for (const auto &componentName :
         info.typeSpec.typeSymbol()
             .get<Fortran::semantics::DerivedTypeDetails>()
             .componentNames()) {
      auto scopeIter = derivedScope.find(componentName);
      assert(scopeIter != derivedScope.cend() &&
             "failed to find derived type component symbol");
      const Fortran::semantics::Symbol &component = scopeIter->second.get();
      mlir::FlatSymbolRefAttr init_val =
          gatherComponentInit(info.loc, converter, component, info.type);
      mlir::DenseI64ArrayAttr lbs = gatherComponentNonDefaultLowerBounds(
          info.loc, builder.getContext(), component);
      if (init_val || lbs) {
        if (!componentInfo)
          componentInfo = builder.createBlock(&dt.getComponentInfo());
        auto compName = mlir::StringAttr::get(builder.getContext(),
                                              toStringRef(component.name()));
        fir::DTComponentOp::create(builder, info.loc, compName, lbs, init_val);
      }
    }
    if (componentInfo)
      fir::FirEndOp::create(builder, info.loc);
    builder.restoreInsertionPoint(insertPointIfCreated);
  }

  /// Store the front-end data that will be required to generate the type info
  /// for the derived types that have been converted to fir.type<>. There are
  /// two stacks since the type info may visit new types, so the new types must
  /// be added to a new stack.
  llvm::SmallVector<TypeInfo> registeredTypeInfoA;
  llvm::SmallVector<TypeInfo> registeredTypeInfoB;
  llvm::SmallVector<TypeInfo> *currentTypeInfoStack = &registeredTypeInfoA;
  /// Track symbols symbols processed during and after the registration
  /// to avoid infinite loops between type conversions and global variable
  /// creation.
  llvm::SmallSetVector<Fortran::semantics::SymbolRef, 32> seen;
};

using IncrementLoopNestInfo = llvm::SmallVector<IncrementLoopInfo, 8>;
} // namespace

//===----------------------------------------------------------------------===//
// FirConverter
//===----------------------------------------------------------------------===//

namespace {

/// Traverse the pre-FIR tree (PFT) to generate the FIR dialect of MLIR.
class FirConverter : public Fortran::lower::AbstractConverter {
public:
  explicit FirConverter(Fortran::lower::LoweringBridge &bridge)
      : Fortran::lower::AbstractConverter(bridge.getLoweringOptions()),
        bridge{bridge}, foldingContext{bridge.createFoldingContext()},
        mlirSymbolTable{bridge.getModule()} {}
  virtual ~FirConverter() = default;

  /// Convert the PFT to FIR.
  void run(Fortran::lower::pft::Program &pft) {
    // Preliminary translation pass.

    // Lower common blocks, taking into account initialization and the largest
    // size of all instances of each common block. This is done before lowering
    // since the global definition may differ from any one local definition.
    lowerCommonBlocks(pft.getCommonBlocks());

    // - Declare all functions that have definitions so that definition
    //   signatures prevail over call site signatures.
    // - Define module variables and OpenMP/OpenACC declarative constructs so
    //   they are available before lowering any function that may use them.
    bool hasMainProgram = false;
    const Fortran::semantics::Symbol *globalOmpRequiresSymbol = nullptr;
    createBuilderOutsideOfFuncOpAndDo([&]() {
      for (Fortran::lower::pft::Program::Units &u : pft.getUnits()) {
        Fortran::common::visit(
            Fortran::common::visitors{
                [&](Fortran::lower::pft::FunctionLikeUnit &f) {
                  if (f.isMainProgram())
                    hasMainProgram = true;
                  declareFunction(f);
                  if (!globalOmpRequiresSymbol)
                    globalOmpRequiresSymbol = f.getScope().symbol();
                },
                [&](Fortran::lower::pft::ModuleLikeUnit &m) {
                  lowerModuleDeclScope(m);
                  for (Fortran::lower::pft::ContainedUnit &unit :
                       m.containedUnitList)
                    if (auto *f =
                            std::get_if<Fortran::lower::pft::FunctionLikeUnit>(
                                &unit))
                      declareFunction(*f);
                },
                [&](Fortran::lower::pft::BlockDataUnit &b) {
                  if (!globalOmpRequiresSymbol)
                    globalOmpRequiresSymbol = b.symTab.symbol();
                },
                [&](Fortran::lower::pft::CompilerDirectiveUnit &d) {},
                [&](Fortran::lower::pft::OpenACCDirectiveUnit &d) {},
            },
            u);
      }
    });

    // Ensure imported OpenMP declare mappers are materialized at module
    // scope before lowering any constructs that may reference them.
    createBuilderOutsideOfFuncOpAndDo([&]() {
      Fortran::lower::materializeOpenMPDeclareMappers(
          *this, bridge.getSemanticsContext());
    });

    // Create definitions of intrinsic module constants.
    createBuilderOutsideOfFuncOpAndDo(
        [&]() { createIntrinsicModuleDefinitions(pft); });

    // Primary translation pass.
    for (Fortran::lower::pft::Program::Units &u : pft.getUnits()) {
      Fortran::common::visit(
          Fortran::common::visitors{
              [&](Fortran::lower::pft::FunctionLikeUnit &f) { lowerFunc(f); },
              [&](Fortran::lower::pft::ModuleLikeUnit &m) { lowerMod(m); },
              [&](Fortran::lower::pft::BlockDataUnit &b) {},
              [&](Fortran::lower::pft::CompilerDirectiveUnit &d) {},
              [&](Fortran::lower::pft::OpenACCDirectiveUnit &d) {},
          },
          u);
    }

    // Once all the code has been translated, create global runtime type info
    // data structures for the derived types that have been processed, as well
    // as fir.type_info operations for the dispatch tables.
    createBuilderOutsideOfFuncOpAndDo(
        [&]() { typeInfoConverter.createTypeInfo(*this); });

    // Generate the `main` entry point if necessary
    if (hasMainProgram)
      createBuilderOutsideOfFuncOpAndDo([&]() {
        fir::runtime::genMain(*builder, toLocation(),
                              bridge.getEnvironmentDefaults(),
                              getFoldingContext().languageFeatures().IsEnabled(
                                  Fortran::common::LanguageFeature::CUDA),
                              getFoldingContext().languageFeatures().IsEnabled(
                                  Fortran::common::LanguageFeature::Coarray));
      });

    finalizeOpenMPLowering(globalOmpRequiresSymbol);
  }

  /// Declare a function.
  void declareFunction(Fortran::lower::pft::FunctionLikeUnit &funit) {
    CHECK(builder && "declareFunction called with uninitialized builder");
    setCurrentPosition(funit.getStartingSourceLoc());
    for (int entryIndex = 0, last = funit.entryPointList.size();
         entryIndex < last; ++entryIndex) {
      funit.setActiveEntry(entryIndex);
      // Calling CalleeInterface ctor will build a declaration
      // mlir::func::FuncOp with no other side effects.
      // TODO: when doing some compiler profiling on real apps, it may be worth
      // to check it's better to save the CalleeInterface instead of recomputing
      // it later when lowering the body. CalleeInterface ctor should be linear
      // with the number of arguments, so it is not awful to do it that way for
      // now, but the linear coefficient might be non negligible. Until
      // measured, stick to the solution that impacts the code less.
      Fortran::lower::CalleeInterface{funit, *this};
    }
    funit.setActiveEntry(0);

    // Compute the set of host associated entities from the nested functions.
    llvm::SetVector<const Fortran::semantics::Symbol *> escapeHost;
    for (Fortran::lower::pft::ContainedUnit &unit : funit.containedUnitList)
      if (auto *f = std::get_if<Fortran::lower::pft::FunctionLikeUnit>(&unit))
        collectHostAssociatedVariables(*f, escapeHost);
    funit.setHostAssociatedSymbols(escapeHost);

    // Declare internal procedures
    for (Fortran::lower::pft::ContainedUnit &unit : funit.containedUnitList)
      if (auto *f = std::get_if<Fortran::lower::pft::FunctionLikeUnit>(&unit))
        declareFunction(*f);
  }

  /// Get the scope that is defining or using \p sym. The returned scope is not
  /// the ultimate scope, since this helper does not traverse use association.
  /// This allows capturing module variables that are referenced in an internal
  /// procedure but whose use statement is inside the host program.
  const Fortran::semantics::Scope &
  getSymbolHostScope(const Fortran::semantics::Symbol &sym) {
    const Fortran::semantics::Symbol *hostSymbol = &sym;
    while (const auto *details =
               hostSymbol->detailsIf<Fortran::semantics::HostAssocDetails>())
      hostSymbol = &details->symbol();
    return hostSymbol->owner();
  }

  /// Collects the canonical list of all host associated symbols. These bindings
  /// must be aggregated into a tuple which can then be added to each of the
  /// internal procedure declarations and passed at each call site.
  void collectHostAssociatedVariables(
      Fortran::lower::pft::FunctionLikeUnit &funit,
      llvm::SetVector<const Fortran::semantics::Symbol *> &escapees) {
    const Fortran::semantics::Scope *internalScope =
        funit.getSubprogramSymbol().scope();
    assert(internalScope && "internal procedures symbol must create a scope");
    auto addToListIfEscapee = [&](const Fortran::semantics::Symbol &sym) {
      const Fortran::semantics::Symbol &ultimate = sym.GetUltimate();
      const auto *namelistDetails =
          ultimate.detailsIf<Fortran::semantics::NamelistDetails>();
      if (ultimate.has<Fortran::semantics::ObjectEntityDetails>() ||
          Fortran::semantics::IsProcedurePointer(ultimate) ||
          Fortran::semantics::IsDummy(sym) || namelistDetails) {
        const Fortran::semantics::Scope &symbolScope = getSymbolHostScope(sym);
        if (symbolScope.kind() ==
                Fortran::semantics::Scope::Kind::MainProgram ||
            symbolScope.kind() == Fortran::semantics::Scope::Kind::Subprogram)
          if (symbolScope != *internalScope &&
              symbolScope.Contains(*internalScope)) {
            if (namelistDetails) {
              // So far, namelist symbols are processed on the fly in IO and
              // the related namelist data structure is not added to the symbol
              // map, so it cannot be passed to the internal procedures.
              // Instead, all the symbols of the host namelist used in the
              // internal procedure must be considered as host associated so
              // that IO lowering can find them when needed.
              for (const auto &namelistObject : namelistDetails->objects())
                escapees.insert(&*namelistObject);
            } else {
              escapees.insert(&ultimate);
            }
          }
      }
    };
    Fortran::lower::pft::visitAllSymbols(funit, addToListIfEscapee);
  }

  //===--------------------------------------------------------------------===//
  // AbstractConverter overrides
  //===--------------------------------------------------------------------===//

  mlir::Value getSymbolAddress(Fortran::lower::SymbolRef sym) override final {
    return lookupSymbol(sym).getAddr();
  }

  fir::ExtendedValue symBoxToExtendedValue(
      const Fortran::lower::SymbolBox &symBox) override final {
    return symBox.match(
        [](const Fortran::lower::SymbolBox::Intrinsic &box)
            -> fir::ExtendedValue { return box.getAddr(); },
        [](const Fortran::lower::SymbolBox::None &) -> fir::ExtendedValue {
          llvm::report_fatal_error("symbol not mapped");
        },
        [&](const fir::FortranVariableOpInterface &x) -> fir::ExtendedValue {
          return hlfir::translateToExtendedValue(getCurrentLocation(),
                                                 getFirOpBuilder(), x);
        },
        [](const auto &box) -> fir::ExtendedValue { return box; });
  }

  fir::ExtendedValue
  getSymbolExtendedValue(const Fortran::semantics::Symbol &sym,
                         Fortran::lower::SymMap *symMap) override final {
    Fortran::lower::SymbolBox sb = lookupSymbol(sym, symMap);
    if (!sb) {
      LLVM_DEBUG(llvm::dbgs() << "unknown symbol: " << sym << "\nmap: "
                              << (symMap ? *symMap : localSymbols) << '\n');
      fir::emitFatalError(getCurrentLocation(),
                          "symbol is not mapped to any IR value");
    }
    return symBoxToExtendedValue(sb);
  }

  mlir::Value impliedDoBinding(llvm::StringRef name) override final {
    mlir::Value val = localSymbols.lookupImpliedDo(name);
    if (!val)
      fir::emitFatalError(toLocation(), "ac-do-variable has no binding");
    return val;
  }

  void copySymbolBinding(Fortran::lower::SymbolRef src,
                         Fortran::lower::SymbolRef target) override final {
    localSymbols.copySymbolBinding(src, target);
  }

  /// Add the symbol binding to the inner-most level of the symbol map and
  /// return true if it is not already present. Otherwise, return false.
  bool bindIfNewSymbol(Fortran::lower::SymbolRef sym,
                       const fir::ExtendedValue &exval) {
    if (shallowLookupSymbol(sym))
      return false;
    bindSymbol(sym, exval);
    return true;
  }

  void bindSymbol(Fortran::lower::SymbolRef sym,
                  const fir::ExtendedValue &exval) override final {
    addSymbol(sym, exval, /*forced=*/true);
  }

  void bindSymbolStorage(
      Fortran::lower::SymbolRef sym,
      Fortran::lower::SymMap::StorageDesc storage) override final {
    localSymbols.registerStorage(sym, std::move(storage));
  }

  Fortran::lower::SymMap::StorageDesc
  getSymbolStorage(Fortran::lower::SymbolRef sym) override final {
    return localSymbols.lookupStorage(sym);
  }

  Fortran::lower::SymMap &getSymbolMap() override final { return localSymbols; }

  void
  overrideExprValues(const Fortran::lower::ExprToValueMap *map) override final {
    exprValueOverrides = map;
  }

  const Fortran::lower::ExprToValueMap *getExprOverrides() override final {
    return exprValueOverrides;
  }

  bool lookupLabelSet(Fortran::lower::SymbolRef sym,
                      Fortran::lower::pft::LabelSet &labelSet) override final {
    Fortran::lower::pft::FunctionLikeUnit &owningProc =
        *getEval().getOwningProcedure();
    auto iter = owningProc.assignSymbolLabelMap.find(sym);
    if (iter == owningProc.assignSymbolLabelMap.end())
      return false;
    labelSet = iter->second;
    return true;
  }

  Fortran::lower::pft::Evaluation *
  lookupLabel(Fortran::lower::pft::Label label) override final {
    Fortran::lower::pft::FunctionLikeUnit &owningProc =
        *getEval().getOwningProcedure();
    return owningProc.labelEvaluationMap.lookup(label);
  }

  fir::ExtendedValue
  genExprAddr(const Fortran::lower::SomeExpr &expr,
              Fortran::lower::StatementContext &context,
              mlir::Location *locPtr = nullptr) override final {
    mlir::Location loc = locPtr ? *locPtr : toLocation();
    if (lowerToHighLevelFIR())
      return Fortran::lower::convertExprToAddress(loc, *this, expr,
                                                  localSymbols, context);
    return Fortran::lower::createSomeExtendedAddress(loc, *this, expr,
                                                     localSymbols, context);
  }

  fir::ExtendedValue
  genExprValue(const Fortran::lower::SomeExpr &expr,
               Fortran::lower::StatementContext &context,
               mlir::Location *locPtr = nullptr) override final {
    mlir::Location loc = locPtr ? *locPtr : toLocation();
    if (lowerToHighLevelFIR())
      return Fortran::lower::convertExprToValue(loc, *this, expr, localSymbols,
                                                context);
    return Fortran::lower::createSomeExtendedExpression(loc, *this, expr,
                                                        localSymbols, context);
  }

  fir::ExtendedValue
  genExprBox(mlir::Location loc, const Fortran::lower::SomeExpr &expr,
             Fortran::lower::StatementContext &stmtCtx) override final {
    if (lowerToHighLevelFIR())
      return Fortran::lower::convertExprToBox(loc, *this, expr, localSymbols,
                                              stmtCtx);
    return Fortran::lower::createBoxValue(loc, *this, expr, localSymbols,
                                          stmtCtx);
  }

  Fortran::evaluate::FoldingContext &getFoldingContext() override final {
    return foldingContext;
  }

  mlir::Type genType(const Fortran::lower::SomeExpr &expr) override final {
    return Fortran::lower::translateSomeExprToFIRType(*this, expr);
  }
  mlir::Type genType(const Fortran::lower::pft::Variable &var) override final {
    return Fortran::lower::translateVariableToFIRType(*this, var);
  }
  mlir::Type genType(Fortran::lower::SymbolRef sym) override final {
    return Fortran::lower::translateSymbolToFIRType(*this, sym);
  }
  mlir::Type
  genType(Fortran::common::TypeCategory tc, int kind,
          llvm::ArrayRef<std::int64_t> lenParameters) override final {
    return Fortran::lower::getFIRType(&getMLIRContext(), tc, kind,
                                      lenParameters);
  }
  mlir::Type
  genType(const Fortran::semantics::DerivedTypeSpec &tySpec) override final {
    return Fortran::lower::translateDerivedTypeToFIRType(*this, tySpec);
  }
  mlir::Type genType(Fortran::common::TypeCategory tc) override final {
    return Fortran::lower::getFIRType(
        &getMLIRContext(), tc, bridge.getDefaultKinds().GetDefaultKind(tc), {});
  }

  Fortran::lower::TypeConstructionStack &
  getTypeConstructionStack() override final {
    return typeConstructionStack;
  }

  bool
  isPresentShallowLookup(const Fortran::semantics::Symbol &sym) override final {
    return bool(shallowLookupSymbol(sym));
  }

  bool createHostAssociateVarClone(const Fortran::semantics::Symbol &sym,
                                   bool skipDefaultInit) override final {
    mlir::Location loc = genLocation(sym.name());
    mlir::Type symType = genType(sym);
    const auto *details = sym.detailsIf<Fortran::semantics::HostAssocDetails>();
    assert(details && "No host-association found");
    const Fortran::semantics::Symbol &hsym = details->symbol();
    mlir::Type hSymType = genType(hsym.GetUltimate());
    Fortran::lower::SymbolBox hsb =
        lookupSymbol(hsym, /*symMap=*/nullptr, /*forceHlfirBase=*/true);

    auto allocate = [&](llvm::ArrayRef<mlir::Value> shape,
                        llvm::ArrayRef<mlir::Value> typeParams) -> mlir::Value {
      mlir::Value allocVal = builder->allocateLocal(
          loc,
          Fortran::semantics::IsAllocatableOrObjectPointer(&hsym.GetUltimate())
              ? hSymType
              : symType,
          mangleName(sym), toStringRef(sym.GetUltimate().name()),
          /*pinned=*/true, shape, typeParams,
          sym.GetUltimate().attrs().test(Fortran::semantics::Attr::TARGET));
      return allocVal;
    };

    fir::ExtendedValue hexv = symBoxToExtendedValue(hsb);
    fir::ExtendedValue exv = hexv.match(
        [&](const fir::BoxValue &box) -> fir::ExtendedValue {
          const Fortran::semantics::DeclTypeSpec *type = sym.GetType();
          if (type && type->IsPolymorphic())
            TODO(loc, "create polymorphic host associated copy");
          // Create a contiguous temp with the same shape and length as
          // the original variable described by a fir.box.
          llvm::SmallVector<mlir::Value> extents =
              fir::factory::getExtents(loc, *builder, hexv);
          if (box.isDerivedWithLenParameters())
            TODO(loc, "get length parameters from derived type BoxValue");
          if (box.isCharacter()) {
            mlir::Value len = fir::factory::readCharLen(*builder, loc, box);
            mlir::Value temp = allocate(extents, {len});
            return fir::CharArrayBoxValue{temp, len, extents};
          }
          return fir::ArrayBoxValue{allocate(extents, {}), extents};
        },
        [&](const fir::MutableBoxValue &box) -> fir::ExtendedValue {
          // Allocate storage for a pointer/allocatble descriptor.
          // No shape/lengths to be passed to the alloca.
          return fir::MutableBoxValue(allocate({}, {}), {}, {});
        },
        [&](const auto &) -> fir::ExtendedValue {
          mlir::Value temp =
              allocate(fir::factory::getExtents(loc, *builder, hexv),
                       fir::factory::getTypeParams(loc, *builder, hexv));
          return fir::substBase(hexv, temp);
        });

    // Initialise cloned allocatable
    hexv.match(
        [&](const fir::MutableBoxValue &box) -> void {
          const auto new_box = exv.getBoxOf<fir::MutableBoxValue>();
          if (Fortran::semantics::IsPointer(sym.GetUltimate())) {
            // Establish the pointer descriptors. The rank and type code/size
            // at least must be set properly for later inquiry of the pointer
            // to work, and new pointers are always given disassociated status
            // by flang for safety, even if this is not required by the
            // language.
            auto empty = fir::factory::createUnallocatedBox(
                *builder, loc, new_box->getBoxTy(), box.nonDeferredLenParams(),
                {});
            fir::StoreOp::create(*builder, loc, empty, new_box->getAddr());
            return;
          }
          // Copy allocation status of Allocatables, creating new storage if
          // needed.

          // allocate if allocated
          mlir::Value isAllocated =
              fir::factory::genIsAllocatedOrAssociatedTest(*builder, loc, box);
          auto if_builder = builder->genIfThenElse(loc, isAllocated);
          if_builder.genThen([&]() {
            std::string name = mangleName(sym) + ".alloc";
            fir::ExtendedValue read = fir::factory::genMutableBoxRead(
                *builder, loc, box, /*mayBePolymorphic=*/false);
            if (auto read_arr_box = read.getBoxOf<fir::ArrayBoxValue>()) {
              fir::factory::genInlinedAllocation(*builder, loc, *new_box,
                                                 read_arr_box->getLBounds(),
                                                 read_arr_box->getExtents(),
                                                 /*lenParams=*/{}, name,
                                                 /*mustBeHeap=*/true);
            } else if (auto read_char_arr_box =
                           read.getBoxOf<fir::CharArrayBoxValue>()) {
              fir::factory::genInlinedAllocation(
                  *builder, loc, *new_box, read_char_arr_box->getLBounds(),
                  read_char_arr_box->getExtents(), read_char_arr_box->getLen(),
                  name,
                  /*mustBeHeap=*/true);
            } else if (auto read_char_box =
                           read.getBoxOf<fir::CharBoxValue>()) {
              fir::factory::genInlinedAllocation(*builder, loc, *new_box,
                                                 /*lbounds=*/{},
                                                 /*extents=*/{},
                                                 read_char_box->getLen(), name,
                                                 /*mustBeHeap=*/true);
            } else {
              fir::factory::genInlinedAllocation(
                  *builder, loc, *new_box, box.getMutableProperties().lbounds,
                  box.getMutableProperties().extents,
                  box.nonDeferredLenParams(), name,
                  /*mustBeHeap=*/true);
            }
          });
          if_builder.genElse([&]() {
            // nullify box
            auto empty = fir::factory::createUnallocatedBox(
                *builder, loc, new_box->getBoxTy(),
                new_box->nonDeferredLenParams(), {});
            fir::StoreOp::create(*builder, loc, empty, new_box->getAddr());
          });
          if_builder.end();
        },
        [&](const auto &) -> void {
          // Always initialize allocatable component descriptor, even when the
          // value is later copied from the host (e.g. firstprivate) because the
          // assignment from the host to the copy will fail if the component
          // descriptors are not initialized.
          if (skipDefaultInit && !hlfir::mayHaveAllocatableComponent(hSymType))
            return;
          // Initialize local/private derived types with default
          // initialization (Fortran 2023 section 11.1.7.5 and OpenMP 5.2
          // section 5.3). Pointer and allocatable components, when allowed,
          // also need to be established so that flang runtime can later work
          // with them.
          if (const Fortran::semantics::DeclTypeSpec *declTypeSpec =
                  sym.GetType())
            if (const Fortran::semantics::DerivedTypeSpec *derivedTypeSpec =
                    declTypeSpec->AsDerived())
              if (derivedTypeSpec->HasDefaultInitialization(
                      /*ignoreAllocatable=*/false, /*ignorePointer=*/false)) {
                mlir::Value box = builder->createBox(loc, exv);
                fir::runtime::genDerivedTypeInitialize(*builder, loc, box);
              }
        });

    return bindIfNewSymbol(sym, exv);
  }

  void createHostAssociateVarCloneDealloc(
      const Fortran::semantics::Symbol &sym) override final {
    mlir::Location loc = genLocation(sym.name());
    Fortran::lower::SymbolBox hsb =
        lookupSymbol(sym, /*symMap=*/nullptr, /*forceHlfirBase=*/true);

    fir::ExtendedValue hexv = symBoxToExtendedValue(hsb);
    hexv.match(
        [&](const fir::MutableBoxValue &new_box) -> void {
          // Do not process pointers
          if (Fortran::semantics::IsPointer(sym.GetUltimate())) {
            return;
          }
          // deallocate allocated in createHostAssociateVarClone value
          Fortran::lower::genDeallocateIfAllocated(*this, new_box, loc);
        },
        [&](const auto &) -> void {
          // Do nothing
        });
  }

  void copyVar(mlir::Location loc, mlir::Value dst, mlir::Value src,
               fir::FortranVariableFlagsEnum attrs) override final {
    bool isAllocatable =
        bitEnumContainsAny(attrs, fir::FortranVariableFlagsEnum::allocatable);
    bool isPointer =
        bitEnumContainsAny(attrs, fir::FortranVariableFlagsEnum::pointer);

    copyVarHLFIR(loc, Fortran::lower::SymbolBox::Intrinsic{dst},
                 Fortran::lower::SymbolBox::Intrinsic{src}, isAllocatable,
                 isPointer, Fortran::semantics::Symbol::Flags());
  }

  void
  copyHostAssociateVar(const Fortran::semantics::Symbol &sym,
                       mlir::OpBuilder::InsertPoint *copyAssignIP = nullptr,
                       bool hostIsSource = true) override final {
    // 1) Fetch the original copy of the variable.
    assert(sym.has<Fortran::semantics::HostAssocDetails>() &&
           "No host-association found");
    const Fortran::semantics::Symbol &hsym = sym.GetUltimate();
    Fortran::lower::SymbolBox hsb = lookupOneLevelUpSymbol(hsym);
    assert(hsb && "Host symbol box not found");

    // 2) Fetch the copied one that will mask the original.
    Fortran::lower::SymbolBox sb = shallowLookupSymbol(sym);
    assert(sb && "Host-associated symbol box not found");
    assert(hsb.getAddr() != sb.getAddr() &&
           "Host and associated symbol boxes are the same");

    // 3) Perform the assignment.
    mlir::OpBuilder::InsertionGuard guard(*builder);
    if (copyAssignIP && copyAssignIP->isSet())
      builder->restoreInsertionPoint(*copyAssignIP);
    else
      builder->setInsertionPointAfter(sb.getAddr().getDefiningOp());

    Fortran::lower::SymbolBox *lhs_sb, *rhs_sb;
    if (!hostIsSource) {
      lhs_sb = &hsb;
      rhs_sb = &sb;
    } else {
      lhs_sb = &sb;
      rhs_sb = &hsb;
    }

    copyVar(sym, *lhs_sb, *rhs_sb, sym.flags());
  }

  void genEval(Fortran::lower::pft::Evaluation &eval,
               bool unstructuredContext) override final {
    genFIR(eval, unstructuredContext);
  }

  //===--------------------------------------------------------------------===//
  // Utility methods
  //===--------------------------------------------------------------------===//

  void collectSymbolSet(
      Fortran::lower::pft::Evaluation &eval,
      llvm::SetVector<const Fortran::semantics::Symbol *> &symbolSet,
      Fortran::semantics::Symbol::Flag flag, bool collectSymbols,
      bool checkHostAssociatedSymbols) override final {
    auto addToList = [&](const Fortran::semantics::Symbol &sym) {
      std::function<void(const Fortran::semantics::Symbol &, bool)>
          insertSymbols = [&](const Fortran::semantics::Symbol &oriSymbol,
                              bool collectSymbol) {
            if (collectSymbol && oriSymbol.test(flag)) {
              symbolSet.insert(&oriSymbol);
            } else if (const auto *commonDetails =
                           oriSymbol.detailsIf<
                               Fortran::semantics::CommonBlockDetails>()) {
              for (const auto &mem : commonDetails->objects())
                if (collectSymbol && mem->test(flag))
                  symbolSet.insert(&(*mem).GetUltimate());
            } else if (checkHostAssociatedSymbols) {
              if (const auto *details{
                      oriSymbol
                          .detailsIf<Fortran::semantics::HostAssocDetails>()})
                insertSymbols(details->symbol(), true);
            }
          };
      insertSymbols(sym, collectSymbols);
    };
    Fortran::lower::pft::visitAllSymbols(eval, addToList);
  }

  mlir::Location getCurrentLocation() override final { return toLocation(); }

  /// Generate a dummy location.
  mlir::Location genUnknownLocation() override final {
    // Note: builder may not be instantiated yet
    return mlir::UnknownLoc::get(&getMLIRContext());
  }

  static mlir::Location genLocation(Fortran::parser::SourcePosition pos,
                                    mlir::MLIRContext &ctx) {
    llvm::SmallString<256> path(*pos.path);
    llvm::sys::fs::make_absolute(path);
    llvm::sys::path::remove_dots(path);
    return mlir::FileLineColLoc::get(&ctx, path.str(), pos.line, pos.column);
  }

  /// Generate a `Location` from the `CharBlock`.
  mlir::Location
  genLocation(const Fortran::parser::CharBlock &block) override final {
    mlir::Location mainLocation = genUnknownLocation();
    if (const Fortran::parser::AllCookedSources *cooked =
            bridge.getCookedSource()) {
      if (std::optional<Fortran::parser::ProvenanceRange> provenance =
              cooked->GetProvenanceRange(block)) {
        if (std::optional<Fortran::parser::SourcePosition> filePos =
                cooked->allSources().GetSourcePosition(provenance->start()))
          mainLocation = genLocation(*filePos, getMLIRContext());

        llvm::SmallVector<mlir::Location> locs;
        locs.push_back(mainLocation);

        llvm::SmallVector<fir::LocationKindAttr> locAttrs;
        locAttrs.push_back(fir::LocationKindAttr::get(&getMLIRContext(),
                                                      fir::LocationKind::Base));

        // Gather include location information if any.
        std::optional<Fortran::parser::ProvenanceRange> prov = provenance;
        while (prov) {
          if (std::optional<Fortran::parser::ProvenanceRange> include =
                  cooked->allSources().GetInclusionInfo(*prov)) {
            if (std::optional<Fortran::parser::SourcePosition> incPos =
                    cooked->allSources().GetSourcePosition(include->start())) {
              locs.push_back(genLocation(*incPos, getMLIRContext()));
              locAttrs.push_back(fir::LocationKindAttr::get(
                  &getMLIRContext(), fir::LocationKind::Inclusion));
            }
            prov = include;
          } else {
            prov.reset();
          }
        }
        if (locs.size() > 1) {
          assert(locs.size() == locAttrs.size() &&
                 "expect as many attributes as locations");
          return mlir::FusedLocWith<fir::LocationKindArrayAttr>::get(
              &getMLIRContext(), locs,
              fir::LocationKindArrayAttr::get(&getMLIRContext(), locAttrs));
        }
      }
    }
    return mainLocation;
  }

  const Fortran::semantics::Scope &getCurrentScope() override final {
    return bridge.getSemanticsContext().FindScope(currentPosition);
  }

  fir::FirOpBuilder &getFirOpBuilder() override final {
    CHECK(builder && "builder is not set before calling getFirOpBuilder");
    return *builder;
  }

  mlir::ModuleOp getModuleOp() override final { return bridge.getModule(); }

  mlir::MLIRContext &getMLIRContext() override final {
    return bridge.getMLIRContext();
  }
  std::string
  mangleName(const Fortran::semantics::Symbol &symbol) override final {
    return Fortran::lower::mangle::mangleName(
        symbol, scopeBlockIdMap, /*keepExternalInScope=*/false,
        getLoweringOptions().getUnderscoring());
  }
  std::string mangleName(
      const Fortran::semantics::DerivedTypeSpec &derivedType) override final {
    return Fortran::lower::mangle::mangleName(derivedType, scopeBlockIdMap);
  }
  std::string mangleName(std::string &name) override final {
    return Fortran::lower::mangle::mangleName(name, getCurrentScope(),
                                              scopeBlockIdMap);
  }
  std::string
  mangleName(std::string &name,
             const Fortran::semantics::Scope &myScope) override final {
    return Fortran::lower::mangle::mangleName(name, myScope, scopeBlockIdMap);
  }
  std::string getRecordTypeFieldName(
      const Fortran::semantics::Symbol &component) override final {
    return Fortran::lower::mangle::getRecordTypeFieldName(component,
                                                          scopeBlockIdMap);
  }
  const fir::KindMapping &getKindMap() override final {
    return bridge.getKindMap();
  }

  /// Return the current function context, which may be a nested BLOCK context
  /// or a full subprogram context.
  Fortran::lower::StatementContext &getFctCtx() override final {
    if (!activeConstructStack.empty() &&
        activeConstructStack.back().eval.isA<Fortran::parser::BlockConstruct>())
      return activeConstructStack.back().stmtCtx;
    return bridge.fctCtx();
  }

  /// Initializes values for STAT and ERRMSG
  std::pair<mlir::Value, mlir::Value>
  genStatAndErrmsg(mlir::Location loc,
                   const std::list<Fortran::parser::StatOrErrmsg>
                       &statOrErrList) override final {
    Fortran::lower::StatementContext stmtCtx;

    mlir::Value errMsgExpr, statExpr;
    for (const Fortran::parser::StatOrErrmsg &statOrErr : statOrErrList) {
      std::visit(Fortran::common::visitors{
                     [&](const Fortran::parser::StatVariable &statVar) {
                       const Fortran::semantics::SomeExpr *expr =
                           Fortran::semantics::GetExpr(statVar);
                       statExpr =
                           fir::getBase(genExprAddr(*expr, stmtCtx, &loc));
                     },
                     [&](const Fortran::parser::MsgVariable &errMsgVar) {
                       const Fortran::semantics::SomeExpr *expr =
                           Fortran::semantics::GetExpr(errMsgVar);
                       errMsgExpr =
                           fir::getBase(genExprBox(loc, *expr, stmtCtx));
                     }},
                 statOrErr.u);
    }

    return {statExpr, errMsgExpr};
  }

  mlir::Value hostAssocTupleValue() override final { return hostAssocTuple; }

  /// Record a binding for the ssa-value of the tuple for this function.
  void bindHostAssocTuple(mlir::Value val) override final {
    assert(!hostAssocTuple && val);
    hostAssocTuple = val;
  }

  mlir::Value dummyArgsScopeValue() const override final {
    return dummyArgsScope;
  }

  bool isRegisteredDummySymbol(
      Fortran::semantics::SymbolRef symRef) const override final {
    auto *sym = &*symRef;
    return registeredDummySymbols.contains(sym);
  }

  unsigned getDummyArgPosition(
      const Fortran::semantics::Symbol &sym) const override final {
    auto it = dummyArgPositions.find(&sym);
    return (it != dummyArgPositions.end()) ? it->second : 0;
  }

  const Fortran::lower::pft::FunctionLikeUnit *
  getCurrentFunctionUnit() const override final {
    return currentFunctionUnit;
  }

  void checkCoarrayEnabled() override final {
    if (!getFoldingContext().languageFeatures().IsEnabled(
            Fortran::common::LanguageFeature::Coarray))
      fir::emitFatalError(
          getCurrentLocation(),
          "Not yet implemented: Multi-image features are experimental and are "
          "disabled by default, use '-fcoarray' to enable.",
          false);
  }

  void registerTypeInfo(mlir::Location loc,
                        Fortran::lower::SymbolRef typeInfoSym,
                        const Fortran::semantics::DerivedTypeSpec &typeSpec,
                        fir::RecordType type) override final {
    typeInfoConverter.registerTypeInfo(*this, loc, typeInfoSym, typeSpec, type);
  }

  llvm::StringRef
  getUniqueLitName(mlir::Location loc,
                   std::unique_ptr<Fortran::lower::SomeExpr> expr,
                   mlir::Type eleTy) override final {
    std::string namePrefix =
        getConstantExprManglePrefix(loc, *expr.get(), eleTy);
    auto [it, inserted] = literalNamesMap.try_emplace(
        expr.get(), namePrefix + std::to_string(uniqueLitId));
    const auto &name = it->second;
    if (inserted) {
      // Keep ownership of the expr key.
      literalExprsStorage.push_back(std::move(expr));

      // If we've just added a new name, we have to make sure
      // there is no global object with the same name in the module.
      fir::GlobalOp global = builder->getNamedGlobal(name);
      if (global)
        fir::emitFatalError(loc, llvm::Twine("global object with name '") +
                                     llvm::Twine(name) +
                                     llvm::Twine("' already exists"));
      ++uniqueLitId;
      return name;
    }

    // The name already exists. Verify that the prefix is the same.
    if (!llvm::StringRef(name).starts_with(namePrefix))
      fir::emitFatalError(loc, llvm::Twine("conflicting prefixes: '") +
                                   llvm::Twine(name) +
                                   llvm::Twine("' does not start with '") +
                                   llvm::Twine(namePrefix) + llvm::Twine("'"));

    return name;
  }

  /// Find the symbol in the inner-most level of the local map or return null.
  Fortran::lower::SymbolBox
  shallowLookupSymbol(const Fortran::semantics::Symbol &sym) override {
    if (Fortran::lower::SymbolBox v = localSymbols.shallowLookupSymbol(sym))
      return v;
    return {};
  }

private:
  FirConverter() = delete;
  FirConverter(const FirConverter &) = delete;
  FirConverter &operator=(const FirConverter &) = delete;

  //===--------------------------------------------------------------------===//
  // Helper member functions
  //===--------------------------------------------------------------------===//

  mlir::Value createFIRExpr(mlir::Location loc,
                            const Fortran::lower::SomeExpr *expr,
                            Fortran::lower::StatementContext &stmtCtx) {
    return fir::getBase(genExprValue(*expr, stmtCtx, &loc));
  }

  /// Find the symbol in the local map or return null.
  Fortran::lower::SymbolBox
  lookupSymbol(const Fortran::semantics::Symbol &sym,
               Fortran::lower::SymMap *symMap = nullptr,
               bool forceHlfirBase = false) {
    symMap = symMap ? symMap : &localSymbols;
    if (lowerToHighLevelFIR()) {
      if (std::optional<fir::FortranVariableOpInterface> var =
              symMap->lookupVariableDefinition(sym)) {
        auto exv = hlfir::translateToExtendedValue(toLocation(), *builder, *var,
                                                   forceHlfirBase);
        return exv.match(
            [](mlir::Value x) -> Fortran::lower::SymbolBox {
              return Fortran::lower::SymbolBox::Intrinsic{x};
            },
            [](auto x) -> Fortran::lower::SymbolBox { return x; });
      }

      // Entry character result represented as an argument pair
      // needs to be represented in the symbol table even before
      // we can create DeclareOp for it. The temporary mapping
      // is EmboxCharOp that conveys the address and length information.
      // After mapSymbolAttributes is done, the mapping is replaced
      // with the new DeclareOp, and the following table lookups
      // do not reach here.
      if (sym.IsFuncResult())
        if (const Fortran::semantics::DeclTypeSpec *declTy = sym.GetType())
          if (declTy->category() ==
              Fortran::semantics::DeclTypeSpec::Category::Character)
            return symMap->lookupSymbol(sym);

      // Procedure dummies are not mapped with an hlfir.declare because
      // they are not "variable" (cannot be assigned to), and it would
      // make hlfir.declare more complex than it needs to to allow this.
      // Do a regular lookup.
      if (Fortran::semantics::IsProcedure(sym))
        return symMap->lookupSymbol(sym);

      // Commonblock names are not variables, but in some lowerings (like
      // OpenMP) it is useful to maintain the address of the commonblock in an
      // MLIR value and query it. hlfir.declare need not be created for these.
      if (sym.detailsIf<Fortran::semantics::CommonBlockDetails>())
        return symMap->lookupSymbol(sym);

      // For symbols to be privatized in OMP, the symbol is mapped to an
      // instance of `SymbolBox::Intrinsic` (i.e. a direct mapping to an MLIR
      // SSA value). This MLIR SSA value is the block argument to the
      // `omp.private`'s `alloc` block. If this is the case, we return this
      // `SymbolBox::Intrinsic` value.
      if (Fortran::lower::SymbolBox v = symMap->lookupSymbol(sym))
        return v;

      return {};
    }
    if (Fortran::lower::SymbolBox v = symMap->lookupSymbol(sym))
      return v;
    return {};
  }

  /// Find the symbol in one level up of symbol map such as for host-association
  /// in OpenMP code or return null.
  Fortran::lower::SymbolBox
  lookupOneLevelUpSymbol(const Fortran::semantics::Symbol &sym) override {
    if (Fortran::lower::SymbolBox v = localSymbols.lookupOneLevelUpSymbol(sym))
      return v;
    return {};
  }

  mlir::SymbolTable *getMLIRSymbolTable() override { return &mlirSymbolTable; }

  mlir::StateStack &getStateStack() override { return stateStack; }

  /// Add the symbol to the local map and return `true`. If the symbol is
  /// already in the map and \p forced is `false`, the map is not updated.
  /// Instead the value `false` is returned.
  bool addSymbol(const Fortran::semantics::SymbolRef sym,
                 fir::ExtendedValue val, bool forced = false) {
    if (!forced && lookupSymbol(sym))
      return false;
    if (lowerToHighLevelFIR()) {
      Fortran::lower::genDeclareSymbol(*this, localSymbols, sym, val,
                                       fir::FortranVariableFlagsEnum::None,
                                       forced);
    } else {
      localSymbols.addSymbol(sym, val, forced);
    }
    return true;
  }

  void copyVar(const Fortran::semantics::Symbol &sym,
               const Fortran::lower::SymbolBox &lhs_sb,
               const Fortran::lower::SymbolBox &rhs_sb,
               Fortran::semantics::Symbol::Flags flags) {
    mlir::Location loc = genLocation(sym.name());
    if (lowerToHighLevelFIR())
      copyVarHLFIR(loc, lhs_sb, rhs_sb, flags);
    else
      copyVarFIR(loc, sym, lhs_sb, rhs_sb);
  }

  void copyVarHLFIR(mlir::Location loc, Fortran::lower::SymbolBox dst,
                    Fortran::lower::SymbolBox src,
                    Fortran::semantics::Symbol::Flags flags) {
    assert(lowerToHighLevelFIR());

    bool isBoxAllocatable = dst.match(
        [](const fir::MutableBoxValue &box) { return box.isAllocatable(); },
        [](const fir::FortranVariableOpInterface &box) {
          return fir::FortranVariableOpInterface(box).isAllocatable();
        },
        [](const auto &box) { return false; });

    bool isBoxPointer = dst.match(
        [](const fir::MutableBoxValue &box) { return box.isPointer(); },
        [](const fir::FortranVariableOpInterface &box) {
          return fir::FortranVariableOpInterface(box).isPointer();
        },
        [](const fir::AbstractBox &box) {
          return fir::isBoxProcAddressType(box.getAddr().getType());
        },
        [](const auto &box) { return false; });

    copyVarHLFIR(loc, dst, src, isBoxAllocatable, isBoxPointer, flags);
  }

  void copyVarHLFIR(mlir::Location loc, Fortran::lower::SymbolBox dst,
                    Fortran::lower::SymbolBox src, bool isAllocatable,
                    bool isPointer, Fortran::semantics::Symbol::Flags flags) {
    assert(lowerToHighLevelFIR());
    hlfir::Entity lhs{dst.getAddr()};
    hlfir::Entity rhs{src.getAddr()};

    auto copyData = [&](hlfir::Entity l, hlfir::Entity r) {
      // Dereference RHS and load it if trivial scalar.
      r = hlfir::loadTrivialScalar(loc, *builder, r);
      hlfir::AssignOp::create(*builder, loc, r, l, isAllocatable);
    };

    if (isPointer) {
      // Set LHS target to the target of RHS (do not copy the RHS
      // target data into the LHS target storage).
      auto loadVal = fir::LoadOp::create(*builder, loc, rhs);
      fir::StoreOp::create(*builder, loc, loadVal, lhs);
    } else if (isAllocatable &&
               flags.test(Fortran::semantics::Symbol::Flag::OmpCopyIn)) {
      // For copyin allocatable variables, RHS must be copied to lhs
      // only when rhs is allocated.
      hlfir::Entity temp =
          hlfir::derefPointersAndAllocatables(loc, *builder, rhs);
      mlir::Value addr = hlfir::genVariableRawAddress(loc, *builder, temp);
      mlir::Value isAllocated = builder->genIsNotNullAddr(loc, addr);
      builder->genIfThenElse(loc, isAllocated)
          .genThen([&]() { copyData(lhs, rhs); })
          .genElse([&]() {
            fir::ExtendedValue hexv = symBoxToExtendedValue(dst);
            hexv.match(
                [&](const fir::MutableBoxValue &new_box) -> void {
                  // if the allocation status of original list item is
                  // unallocated, unallocate the copy if it is allocated, else
                  // do nothing.
                  Fortran::lower::genDeallocateIfAllocated(*this, new_box, loc);
                },
                [&](const auto &) -> void {});
          })
          .end();
    } else if (isAllocatable &&
               flags.test(Fortran::semantics::Symbol::Flag::OmpFirstPrivate)) {
      // For firstprivate allocatable variables, RHS must be copied
      // only when LHS is allocated.
      hlfir::Entity temp =
          hlfir::derefPointersAndAllocatables(loc, *builder, lhs);
      mlir::Value addr = hlfir::genVariableRawAddress(loc, *builder, temp);
      mlir::Value isAllocated = builder->genIsNotNullAddr(loc, addr);
      builder->genIfThen(loc, isAllocated)
          .genThen([&]() { copyData(lhs, rhs); })
          .end();
    } else {
      copyData(lhs, rhs);
    }
  }

  void copyVarFIR(mlir::Location loc, const Fortran::semantics::Symbol &sym,
                  const Fortran::lower::SymbolBox &lhs_sb,
                  const Fortran::lower::SymbolBox &rhs_sb) {
    assert(!lowerToHighLevelFIR());
    fir::ExtendedValue lhs = symBoxToExtendedValue(lhs_sb);
    fir::ExtendedValue rhs = symBoxToExtendedValue(rhs_sb);
    mlir::Type symType = genType(sym);
    if (auto seqTy = mlir::dyn_cast<fir::SequenceType>(symType)) {
      Fortran::lower::StatementContext stmtCtx;
      Fortran::lower::createSomeArrayAssignment(*this, lhs, rhs, localSymbols,
                                                stmtCtx);
      stmtCtx.finalizeAndReset();
    } else if (lhs.getBoxOf<fir::CharBoxValue>()) {
      fir::factory::CharacterExprHelper{*builder, loc}.createAssign(lhs, rhs);
    } else {
      auto loadVal = fir::LoadOp::create(*builder, loc, fir::getBase(rhs));
      fir::StoreOp::create(*builder, loc, loadVal, fir::getBase(lhs));
    }
  }

  /// Map a block argument to a result or dummy symbol. This is not the
  /// definitive mapping. The specification expression have not been lowered
  /// yet. The final mapping will be done using this pre-mapping in
  /// Fortran::lower::mapSymbolAttributes.
  /// \param argNo The 1-based source position of this argument (0 if
  /// unknown/result)
  bool mapBlockArgToDummyOrResult(const Fortran::semantics::SymbolRef sym,
                                  mlir::Value val, bool isResult,
                                  unsigned argNo = 0) {
    localSymbols.addSymbol(sym, val);
    if (!isResult)
      registerDummySymbol(sym, argNo);

    return true;
  }

  /// Generate the address of loop variable \p sym.
  /// If \p sym is not mapped yet, allocate local storage for it.
  mlir::Value genLoopVariableAddress(mlir::Location loc,
                                     const Fortran::semantics::Symbol &sym,
                                     bool isUnordered) {
    if (!shallowLookupSymbol(sym) &&
        (isUnordered ||
         GetSymbolDSA(sym).test(Fortran::semantics::Symbol::Flag::OmpPrivate) ||
         GetSymbolDSA(sym).test(
             Fortran::semantics::Symbol::Flag::OmpFirstPrivate) ||
         GetSymbolDSA(sym).test(
             Fortran::semantics::Symbol::Flag::OmpLastPrivate) ||
         GetSymbolDSA(sym).test(Fortran::semantics::Symbol::Flag::OmpLinear))) {
      // Do concurrent loop variables are not mapped yet since they are
      // local to the Do concurrent scope (same for OpenMP loops).
      mlir::OpBuilder::InsertPoint insPt = builder->saveInsertionPoint();
      builder->setInsertionPointToStart(builder->getAllocaBlock());
      mlir::Type tempTy = genType(sym);
      mlir::Value temp =
          builder->createTemporaryAlloc(loc, tempTy, toStringRef(sym.name()));
      bindIfNewSymbol(sym, temp);
      builder->restoreInsertionPoint(insPt);
    }
    auto entry = lookupSymbol(sym);
    (void)entry;
    assert(entry && "loop control variable must already be in map");
    Fortran::lower::StatementContext stmtCtx;
    return fir::getBase(
        genExprAddr(Fortran::evaluate::AsGenericExpr(sym).value(), stmtCtx));
  }

  static bool isNumericScalarCategory(Fortran::common::TypeCategory cat) {
    return cat == Fortran::common::TypeCategory::Integer ||
           cat == Fortran::common::TypeCategory::Real ||
           cat == Fortran::common::TypeCategory::Complex ||
           cat == Fortran::common::TypeCategory::Logical;
  }
  static bool isLogicalCategory(Fortran::common::TypeCategory cat) {
    return cat == Fortran::common::TypeCategory::Logical;
  }
  static bool isCharacterCategory(Fortran::common::TypeCategory cat) {
    return cat == Fortran::common::TypeCategory::Character;
  }
  static bool isDerivedCategory(Fortran::common::TypeCategory cat) {
    return cat == Fortran::common::TypeCategory::Derived;
  }

  /// Insert a new block before \p block. Leave the insertion point unchanged.
  mlir::Block *insertBlock(mlir::Block *block) {
    mlir::OpBuilder::InsertPoint insertPt = builder->saveInsertionPoint();
    mlir::Block *newBlock = builder->createBlock(block);
    builder->restoreInsertionPoint(insertPt);
    return newBlock;
  }

  Fortran::lower::pft::Evaluation &evalOfLabel(Fortran::parser::Label label) {
    const Fortran::lower::pft::LabelEvalMap &labelEvaluationMap =
        getEval().getOwningProcedure()->labelEvaluationMap;
    const auto iter = labelEvaluationMap.find(label);
    assert(iter != labelEvaluationMap.end() && "label missing from map");
    return *iter->second;
  }

  void genBranch(mlir::Block *targetBlock) {
    assert(targetBlock && "missing unconditional target block");
    mlir::cf::BranchOp::create(*builder, toLocation(), targetBlock);
  }

  void genConditionalBranch(mlir::Value cond, mlir::Block *trueTarget,
                            mlir::Block *falseTarget) {
    assert(trueTarget && "missing conditional branch true block");
    assert(falseTarget && "missing conditional branch false block");
    mlir::Location loc = toLocation();
    mlir::Value bcc = builder->createConvert(loc, builder->getI1Type(), cond);
    mlir::cf::CondBranchOp::create(*builder, loc, bcc, trueTarget,
                                   mlir::ValueRange{}, falseTarget,
                                   mlir::ValueRange{});
  }
  void genConditionalBranch(mlir::Value cond,
                            Fortran::lower::pft::Evaluation *trueTarget,
                            Fortran::lower::pft::Evaluation *falseTarget) {
    genConditionalBranch(cond, trueTarget->block, falseTarget->block);
  }
  void genConditionalBranch(const Fortran::parser::ScalarLogicalExpr &expr,
                            mlir::Block *trueTarget, mlir::Block *falseTarget) {
    Fortran::lower::StatementContext stmtCtx;
    mlir::Value cond =
        createFIRExpr(toLocation(), Fortran::semantics::GetExpr(expr), stmtCtx);
    stmtCtx.finalizeAndReset();
    genConditionalBranch(cond, trueTarget, falseTarget);
  }
  void genConditionalBranch(const Fortran::parser::ScalarLogicalExpr &expr,
                            Fortran::lower::pft::Evaluation *trueTarget,
                            Fortran::lower::pft::Evaluation *falseTarget) {
    Fortran::lower::StatementContext stmtCtx;
    mlir::Value cond =
        createFIRExpr(toLocation(), Fortran::semantics::GetExpr(expr), stmtCtx);
    stmtCtx.finalizeAndReset();
    genConditionalBranch(cond, trueTarget->block, falseTarget->block);
  }

  void
  genDoWhileAsSCFWhile(const Fortran::parser::ScalarLogicalExpr &whileCondition,
                       Fortran::lower::pft::Evaluation &doConstructEval,
                       Fortran::lower::pft::Evaluation &doStmtEval) {
    mlir::Location loc = toLocation();

    auto scfWhile =
        mlir::scf::WhileOp::create(*builder, loc,
                                   /*resultTypes=*/mlir::TypeRange{},
                                   /*inits=*/mlir::ValueRange{});

    // Fill the "before" region: compute condition.
    mlir::Block *beforeBlock =
        builder->createBlock(&scfWhile.getBefore(), scfWhile.getBefore().end());
    builder->setInsertionPointToStart(beforeBlock);
    Fortran::lower::StatementContext stmtCtx;
    mlir::Value cond = createFIRExpr(
        loc, Fortran::semantics::GetExpr(whileCondition), stmtCtx);
    stmtCtx.finalizeAndReset();
    cond = builder->createConvert(loc, builder->getI1Type(), cond);
    mlir::scf::ConditionOp::create(*builder, loc, cond, mlir::ValueRange{});

    // Fill the "after" region: loop body.
    mlir::Block *afterBlock =
        builder->createBlock(&scfWhile.getAfter(), scfWhile.getAfter().end());
    builder->setInsertionPointToStart(afterBlock);

    // Lower nested evaluations excluding the loop control statement (the
    // NonLabelDoStmt) and the EndDoStmt.
    auto iter = doConstructEval.getNestedEvaluations().begin();
    auto end = doConstructEval.getNestedEvaluations().end();
    assert(iter != end && "malformed DoConstruct evaluation list");
    ++iter; // skip the NonLabelDoStmt
    assert(iter != end && "malformed DoConstruct evaluation list");
    auto endDoIter = std::prev(end);
    for (; iter != endDoIter; ++iter)
      genFIR(*iter, /*unstructuredContext=*/false);

    mlir::scf::YieldOp::create(*builder, loc);
    builder->setInsertionPointAfter(scfWhile);
  }

  /// Return the nearest active ancestor construct of \p eval, or nullptr.
  Fortran::lower::pft::Evaluation *
  getActiveAncestor(const Fortran::lower::pft::Evaluation &eval) {
    Fortran::lower::pft::Evaluation *ancestor = eval.parentConstruct;
    for (; ancestor; ancestor = ancestor->parentConstruct)
      if (ancestor->activeConstruct)
        break;
    return ancestor;
  }

  /// Return the predicate: "a branch to \p targetEval has exit code".
  bool hasExitCode(const Fortran::lower::pft::Evaluation &targetEval) {
    Fortran::lower::pft::Evaluation *activeAncestor =
        getActiveAncestor(targetEval);
    for (auto it = activeConstructStack.rbegin(),
              rend = activeConstructStack.rend();
         it != rend; ++it) {
      if (&it->eval == activeAncestor)
        break;
      if (it->stmtCtx.hasCode())
        return true;
    }
    return false;
  }

  /// Generate a branch to \p targetEval after generating on-exit code for
  /// any enclosing construct scopes that are exited by taking the branch.
  void
  genConstructExitBranch(const Fortran::lower::pft::Evaluation &targetEval) {
    Fortran::lower::pft::Evaluation *activeAncestor =
        getActiveAncestor(targetEval);
    for (auto it = activeConstructStack.rbegin(),
              rend = activeConstructStack.rend();
         it != rend; ++it) {
      if (&it->eval == activeAncestor)
        break;
      it->stmtCtx.finalizeAndKeep();
    }
    genBranch(targetEval.block);
  }

  /// A construct contains nested evaluations. Some of these evaluations
  /// may start a new basic block, others will add code to an existing
  /// block.
  /// Collect the list of nested evaluations that are last in their block,
  /// organize them into two sets:
  /// 1. Exiting evaluations: they may need a branch exiting from their
  ///    parent construct,
  /// 2. Fall-through evaluations: they will continue to the following
  ///    evaluation. They may still need a branch, but they do not exit
  ///    the construct. They appear in cases where the following evaluation
  ///    is a target of some branch.
  void collectFinalEvaluations(
      Fortran::lower::pft::Evaluation &construct,
      llvm::SmallVector<Fortran::lower::pft::Evaluation *> &exits,
      llvm::SmallVector<Fortran::lower::pft::Evaluation *> &fallThroughs) {
    Fortran::lower::pft::EvaluationList &nested =
        construct.getNestedEvaluations();
    if (nested.empty())
      return;

    Fortran::lower::pft::Evaluation *exit = construct.constructExit;
    Fortran::lower::pft::Evaluation *previous = &nested.front();

    for (auto it = ++nested.begin(), end = nested.end(); it != end;
         previous = &*it++) {
      if (it->block == nullptr)
        continue;
      // "*it" starts a new block, check what to do with "previous"
      if (it->isIntermediateConstructStmt() && previous != exit)
        exits.push_back(previous);
      else if (previous->lexicalSuccessor && previous->lexicalSuccessor->block)
        fallThroughs.push_back(previous);
    }
    if (previous != exit)
      exits.push_back(previous);
  }

  /// Generate a SelectOp or branch sequence that compares \p selector against
  /// values in \p valueList and targets corresponding labels in \p labelList.
  /// If no value matches the selector, branch to \p defaultEval.
  ///
  /// Three cases require special processing.
  ///
  /// An empty \p valueList indicates an ArithmeticIfStmt context that requires
  /// two comparisons against 0 or 0.0. The selector may have either INTEGER
  /// or REAL type.
  ///
  /// A nonpositive \p valuelist value indicates an IO statement context
  /// (0 for ERR, -1 for END, -2 for EOR). An ERR branch must be taken for
  /// any positive (IOSTAT) value. A missing (zero) label requires a branch
  /// to \p defaultEval for that value.
  ///
  /// A non-null \p errorBlock indicates an AssignedGotoStmt context that
  /// must always branch to an explicit target. There is no valid defaultEval
  /// in this case. Generate a branch to \p errorBlock for an AssignedGotoStmt
  /// that violates this program requirement.
  ///
  /// If this is not an ArithmeticIfStmt and no targets have exit code,
  /// generate a SelectOp. Otherwise, for each target, if it has exit code,
  /// branch to a new block, insert exit code, and then branch to the target.
  /// Otherwise, branch directly to the target.
  void genMultiwayBranch(mlir::Value selector,
                         llvm::SmallVector<int64_t> valueList,
                         llvm::SmallVector<Fortran::parser::Label> labelList,
                         const Fortran::lower::pft::Evaluation &defaultEval,
                         mlir::Block *errorBlock = nullptr) {
    bool inArithmeticIfContext = valueList.empty();
    assert(((inArithmeticIfContext && labelList.size() == 2) ||
            (valueList.size() && labelList.size() == valueList.size())) &&
           "mismatched multiway branch targets");
    mlir::Block *defaultBlock = errorBlock ? errorBlock : defaultEval.block;
    bool defaultHasExitCode = !errorBlock && hasExitCode(defaultEval);
    bool hasAnyExitCode = defaultHasExitCode;
    if (!hasAnyExitCode)
      for (auto label : labelList)
        if (label && hasExitCode(evalOfLabel(label))) {
          hasAnyExitCode = true;
          break;
        }
    mlir::Location loc = toLocation();
    size_t branchCount = labelList.size();
    if (!inArithmeticIfContext && !hasAnyExitCode &&
        !getEval().forceAsUnstructured()) { // from -no-structured-fir option
      // Generate a SelectOp.
      llvm::SmallVector<mlir::Block *> blockList;
      for (auto label : labelList) {
        mlir::Block *block =
            label ? evalOfLabel(label).block : defaultEval.block;
        assert(block && "missing multiway branch block");
        blockList.push_back(block);
      }
      blockList.push_back(defaultBlock);
      if (valueList[branchCount - 1] == 0) // Swap IO ERR and default blocks.
        std::swap(blockList[branchCount - 1], blockList[branchCount]);
      fir::SelectOp::create(*builder, loc, selector, valueList, blockList);
      return;
    }
    mlir::Type selectorType = selector.getType();
    bool realSelector = mlir::isa<mlir::FloatType>(selectorType);
    assert((inArithmeticIfContext || !realSelector) && "invalid selector type");
    mlir::Value zero;
    if (inArithmeticIfContext)
      zero = realSelector
                 ? mlir::arith::ConstantOp::create(
                       *builder, loc, selectorType,
                       builder->getFloatAttr(selectorType, 0.0))
                 : builder->createIntegerConstant(loc, selectorType, 0);
    for (auto label : llvm::enumerate(labelList)) {
      mlir::Value cond;
      if (realSelector) // inArithmeticIfContext
        cond = mlir::arith::CmpFOp::create(
            *builder, loc,
            label.index() == 0 ? mlir::arith::CmpFPredicate::OLT
                               : mlir::arith::CmpFPredicate::OGT,
            selector, zero);
      else if (inArithmeticIfContext) // INTEGER selector
        cond = mlir::arith::CmpIOp::create(
            *builder, loc,
            label.index() == 0 ? mlir::arith::CmpIPredicate::slt
                               : mlir::arith::CmpIPredicate::sgt,
            selector, zero);
      else // A value of 0 is an IO ERR branch: invert comparison.
        cond = mlir::arith::CmpIOp::create(
            *builder, loc,
            valueList[label.index()] == 0 ? mlir::arith::CmpIPredicate::ne
                                          : mlir::arith::CmpIPredicate::eq,
            selector,
            builder->createIntegerConstant(loc, selectorType,
                                           valueList[label.index()]));
      // Branch to a new block with exit code and then to the target, or branch
      // directly to the target. defaultBlock is the "else" target.
      bool lastBranch = label.index() == branchCount - 1;
      mlir::Block *nextBlock =
          lastBranch && !defaultHasExitCode
              ? defaultBlock
              : builder->getBlock()->splitBlock(builder->getInsertionPoint());
      const Fortran::lower::pft::Evaluation &targetEval =
          label.value() ? evalOfLabel(label.value()) : defaultEval;
      if (hasExitCode(targetEval)) {
        mlir::Block *jumpBlock =
            builder->getBlock()->splitBlock(builder->getInsertionPoint());
        genConditionalBranch(cond, jumpBlock, nextBlock);
        startBlock(jumpBlock);
        genConstructExitBranch(targetEval);
      } else {
        genConditionalBranch(cond, targetEval.block, nextBlock);
      }
      if (!lastBranch) {
        startBlock(nextBlock);
      } else if (defaultHasExitCode) {
        startBlock(nextBlock);
        genConstructExitBranch(defaultEval);
      }
    }
  }

  void pushActiveConstruct(Fortran::lower::pft::Evaluation &eval,
                           Fortran::lower::StatementContext &stmtCtx) {
    activeConstructStack.push_back(ConstructContext{eval, stmtCtx});
    eval.activeConstruct = true;
  }
  void popActiveConstruct() {
    assert(!activeConstructStack.empty() && "invalid active construct stack");
    activeConstructStack.back().eval.activeConstruct = false;
    if (activeConstructStack.back().pushedScope)
      localSymbols.popScope();
    activeConstructStack.pop_back();
  }

  //===--------------------------------------------------------------------===//
  // Termination of symbolically referenced execution units
  //===--------------------------------------------------------------------===//

  /// Exit of a routine
  ///
  /// Generate the cleanup block before the routine exits
  void genExitRoutine(bool earlyReturn, mlir::ValueRange retval = {}) {
    if (blockIsUnterminated()) {
      bridge.openAccCtx().finalizeAndKeep();
      bridge.fctCtx().finalizeAndKeep();
      mlir::func::ReturnOp::create(*builder, toLocation(), retval);
    }
    if (!earlyReturn) {
      bridge.openAccCtx().pop();
      bridge.fctCtx().pop();
    }
  }

  /// END of procedure-like constructs
  ///
  /// Generate the cleanup block before the procedure exits
  void genReturnSymbol(const Fortran::semantics::Symbol &functionSymbol) {
    const Fortran::semantics::Symbol &resultSym =
        functionSymbol.get<Fortran::semantics::SubprogramDetails>().result();
    Fortran::lower::SymbolBox resultSymBox = lookupSymbol(resultSym);
    mlir::Location loc = toLocation();
    if (!resultSymBox) {
      // Create a dummy undefined value of the expected return type.
      // This prevents improper cleanup of StatementContext, which would lead
      // to a crash due to a block with no terminator. See issue #126452.
      mlir::FunctionType funcType = builder->getFunction().getFunctionType();
      mlir::Type resultType = funcType.getResult(0);
      mlir::Value undefResult = fir::UndefOp::create(*builder, loc, resultType);
      genExitRoutine(false, undefResult);
      return;
    }
    mlir::Value resultVal = resultSymBox.match(
        [&](const fir::CharBoxValue &x) -> mlir::Value {
          if (Fortran::semantics::IsBindCProcedure(functionSymbol))
            return fir::LoadOp::create(*builder, loc, x.getBuffer());
          return fir::factory::CharacterExprHelper{*builder, loc}
              .createEmboxChar(x.getBuffer(), x.getLen());
        },
        [&](const fir::MutableBoxValue &x) -> mlir::Value {
          mlir::Value resultRef = resultSymBox.getAddr();
          mlir::Value load = fir::LoadOp::create(*builder, loc, resultRef);
          unsigned rank = x.rank();
          if (x.isAllocatable() && rank > 0) {
            // ALLOCATABLE array result must have default lower bounds.
            // At the call site the result box of a function reference
            // might be considered having default lower bounds, but
            // the runtime box should probably comply with this assumption
            // as well. If the result box has proper lbounds in runtime,
            // this may improve the debugging experience of Fortran apps.
            // We may consider removing this, if the overhead of setting
            // default lower bounds is too big.
            mlir::Value one =
                builder->createIntegerConstant(loc, builder->getIndexType(), 1);
            llvm::SmallVector<mlir::Value> lbounds{rank, one};
            auto shiftTy = fir::ShiftType::get(builder->getContext(), rank);
            mlir::Value shiftOp =
                fir::ShiftOp::create(*builder, loc, shiftTy, lbounds);
            load = fir::ReboxOp::create(*builder, loc, load.getType(), load,
                                        shiftOp, /*slice=*/mlir::Value{});
          }
          return load;
        },
        [&](const auto &) -> mlir::Value {
          mlir::Value resultRef = resultSymBox.getAddr();
          mlir::Type resultType = genType(resultSym);
          mlir::Type resultRefType = builder->getRefType(resultType);
          // A function with multiple entry points returning different types
          // tags all result variables with one of the largest types to allow
          // them to share the same storage. Convert this to the actual type.
          if (resultRef.getType() != resultRefType)
            resultRef = builder->createConvertWithVolatileCast(
                loc, resultRefType, resultRef);
          return fir::LoadOp::create(*builder, loc, resultRef);
        });
    genExitRoutine(false, resultVal);
  }

  /// Get the return value of a call to \p symbol, which is a subroutine entry
  /// point that has alternative return specifiers.
  const mlir::Value
  getAltReturnResult(const Fortran::semantics::Symbol &symbol) {
    assert(Fortran::semantics::HasAlternateReturns(symbol) &&
           "subroutine does not have alternate returns");
    return getSymbolAddress(symbol);
  }

  void genFIRProcedureExit(Fortran::lower::pft::FunctionLikeUnit &funit,
                           const Fortran::semantics::Symbol &symbol) {
    if (mlir::Block *finalBlock = funit.finalBlock) {
      // The current block must end with a terminator.
      if (blockIsUnterminated())
        mlir::cf::BranchOp::create(*builder, toLocation(), finalBlock);
      // Set insertion point to final block.
      builder->setInsertionPoint(finalBlock, finalBlock->end());
    }
    if (Fortran::semantics::IsFunction(symbol)) {
      genReturnSymbol(symbol);
    } else if (Fortran::semantics::HasAlternateReturns(symbol)) {
      mlir::Value retval = fir::LoadOp::create(*builder, toLocation(),
                                               getAltReturnResult(symbol));
      genExitRoutine(false, retval);
    } else {
      genExitRoutine(false);
    }
  }

  //
  // Statements that have control-flow semantics
  //

  /// Generate an If[Then]Stmt condition or its negation.
  template <typename A>
  mlir::Value genIfCondition(const A *stmt, bool negate = false) {
    mlir::Location loc = toLocation();
    Fortran::lower::StatementContext stmtCtx;
    mlir::Value condExpr = createFIRExpr(
        loc,
        Fortran::semantics::GetExpr(
            std::get<Fortran::parser::ScalarLogicalExpr>(stmt->t)),
        stmtCtx);
    stmtCtx.finalizeAndReset();
    mlir::Value cond =
        builder->createConvert(loc, builder->getI1Type(), condExpr);
    if (negate)
      cond = mlir::arith::XOrIOp::create(
          *builder, loc, cond,
          builder->createIntegerConstant(loc, cond.getType(), 1));
    return cond;
  }

  mlir::func::FuncOp getFunc(llvm::StringRef name, mlir::FunctionType ty) {
    if (mlir::func::FuncOp func = builder->getNamedFunction(name)) {
      assert(func.getFunctionType() == ty);
      return func;
    }
    return builder->createFunction(toLocation(), name, ty);
  }

  /// Lowering of CALL statement
  void genFIR(const Fortran::parser::CallStmt &stmt) {
    Fortran::lower::StatementContext stmtCtx;
    Fortran::lower::pft::Evaluation &eval = getEval();
    setCurrentPosition(stmt.source);
    assert(stmt.typedCall && "Call was not analyzed");
    mlir::Value res{};

    // Set 'no_inline', 'inline_hint' or 'always_inline' to true on the
    // ProcedureRef. The NoInline and AlwaysInline attribute will be set in
    // genProcedureRef later.
    for (const auto *dir : eval.dirs) {
      Fortran::common::visit(
          Fortran::common::visitors{
              [&](const Fortran::parser::CompilerDirective::ForceInline &) {
                stmt.typedCall->setAlwaysInline(true);
              },
              [&](const Fortran::parser::CompilerDirective::Inline &) {
                stmt.typedCall->setInlineHint(true);
              },
              [&](const Fortran::parser::CompilerDirective::NoInline &) {
                stmt.typedCall->setNoInline(true);
              },
              [&](const auto &) {}},
          dir->u);
    }

    if (lowerToHighLevelFIR()) {
      std::optional<mlir::Type> resultType;
      if (stmt.typedCall->hasAlternateReturns())
        resultType = builder->getIndexType();
      auto hlfirRes = Fortran::lower::convertCallToHLFIR(
          toLocation(), *this, *stmt.typedCall, resultType, localSymbols,
          stmtCtx);
      if (hlfirRes)
        res = *hlfirRes;
    } else {
      // Call statement lowering shares code with function call lowering.
      res = Fortran::lower::createSubroutineCall(
          *this, *stmt.typedCall, explicitIterSpace, implicitIterSpace,
          localSymbols, stmtCtx, /*isUserDefAssignment=*/false);
    }
    stmtCtx.finalizeAndReset();
    if (!res)
      return; // "Normal" subroutine call.
    // Call with alternate return specifiers.
    // The call returns an index that selects an alternate return branch target.
    llvm::SmallVector<int64_t> indexList;
    llvm::SmallVector<Fortran::parser::Label> labelList;
    int64_t index = 0;
    const auto &call{std::get<Fortran::parser::Call>(stmt.t)};
    for (const Fortran::parser::ActualArgSpec &arg :
         std::get<std::list<Fortran::parser::ActualArgSpec>>(call.t)) {
      const auto &actual = std::get<Fortran::parser::ActualArg>(arg.t);
      if (const auto *altReturn =
              std::get_if<Fortran::parser::AltReturnSpec>(&actual.u)) {
        indexList.push_back(++index);
        labelList.push_back(altReturn->v);
      }
    }
    genMultiwayBranch(res, indexList, labelList, eval.nonNopSuccessor());
  }

  void genFIR(const Fortran::parser::ComputedGotoStmt &stmt) {
    Fortran::lower::StatementContext stmtCtx;
    Fortran::lower::pft::Evaluation &eval = getEval();
    mlir::Value selectExpr =
        createFIRExpr(toLocation(),
                      Fortran::semantics::GetExpr(
                          std::get<Fortran::parser::ScalarIntExpr>(stmt.t)),
                      stmtCtx);
    stmtCtx.finalizeAndReset();
    llvm::SmallVector<int64_t> indexList;
    llvm::SmallVector<Fortran::parser::Label> labelList;
    int64_t index = 0;
    for (Fortran::parser::Label label :
         std::get<std::list<Fortran::parser::Label>>(stmt.t)) {
      indexList.push_back(++index);
      labelList.push_back(label);
    }
    genMultiwayBranch(selectExpr, indexList, labelList, eval.nonNopSuccessor());
  }

  void genFIR(const Fortran::parser::ArithmeticIfStmt &stmt) {
    Fortran::lower::StatementContext stmtCtx;
    mlir::Value expr = createFIRExpr(
        toLocation(),
        Fortran::semantics::GetExpr(std::get<Fortran::parser::Expr>(stmt.t)),
        stmtCtx);
    stmtCtx.finalizeAndReset();
    // Raise an exception if REAL expr is a NaN.
    if (mlir::isa<mlir::FloatType>(expr.getType()))
      expr = mlir::arith::AddFOp::create(*builder, toLocation(), expr, expr);
    // An empty valueList indicates to genMultiwayBranch that the branch is
    // an ArithmeticIfStmt that has two branches on value 0 or 0.0.
    llvm::SmallVector<int64_t> valueList;
    llvm::SmallVector<Fortran::parser::Label> labelList;
    labelList.push_back(std::get<1>(stmt.t));
    labelList.push_back(std::get<3>(stmt.t));
    const Fortran::lower::pft::LabelEvalMap &labelEvaluationMap =
        getEval().getOwningProcedure()->labelEvaluationMap;
    const auto iter = labelEvaluationMap.find(std::get<2>(stmt.t));
    assert(iter != labelEvaluationMap.end() && "label missing from map");
    genMultiwayBranch(expr, valueList, labelList, *iter->second);
  }

  void genFIR(const Fortran::parser::AssignedGotoStmt &stmt) {
    // See Fortran 90 Clause 8.2.4.
    // Relax the requirement that the GOTO variable must have a value in the
    // label list when a list is present, and allow a branch to any non-format
    // target that has an ASSIGN statement for the variable.
    mlir::Location loc = toLocation();
    Fortran::lower::pft::Evaluation &eval = getEval();
    Fortran::lower::pft::FunctionLikeUnit &owningProc =
        *eval.getOwningProcedure();
    const Fortran::lower::pft::SymbolLabelMap &symbolLabelMap =
        owningProc.assignSymbolLabelMap;
    const Fortran::lower::pft::LabelEvalMap &labelEvalMap =
        owningProc.labelEvaluationMap;
    const Fortran::semantics::Symbol &symbol =
        *std::get<Fortran::parser::Name>(stmt.t).symbol;
    auto labelSetIter = symbolLabelMap.find(symbol);
    llvm::SmallVector<int64_t> valueList;
    llvm::SmallVector<Fortran::parser::Label> labelList;
    if (labelSetIter != symbolLabelMap.end()) {
      for (auto &label : labelSetIter->second) {
        const auto evalIter = labelEvalMap.find(label);
        assert(evalIter != labelEvalMap.end() && "assigned goto label missing");
        if (evalIter->second->block) { // non-format statement
          valueList.push_back(label);  // label as an integer
          labelList.push_back(label);
        }
      }
    }
    if (!labelList.empty()) {
      mlir::Value selectExpr = hlfir::loadTrivialScalar(
          loc, *builder, hlfir::Entity{getSymbolAddress(symbol)});
      // Add a default error target in case the goto is nonconforming.
      mlir::Block *errorBlock =
          builder->getBlock()->splitBlock(builder->getInsertionPoint());
      genMultiwayBranch(selectExpr, valueList, labelList,
                        eval.nonNopSuccessor(), errorBlock);
      startBlock(errorBlock);
    }
    fir::runtime::genReportFatalUserError(
        *builder, loc,
        "Assigned GOTO variable '" + symbol.name().ToString() +
            "' does not have a valid target label value");
    fir::UnreachableOp::create(*builder, loc);
  }

  fir::ReduceOperationEnum
  getReduceOperationEnum(const Fortran::parser::ReductionOperator &rOpr) {
    switch (rOpr.v) {
    case Fortran::parser::ReductionOperator::Operator::Plus:
      return fir::ReduceOperationEnum::Add;
    case Fortran::parser::ReductionOperator::Operator::Multiply:
      return fir::ReduceOperationEnum::Multiply;
    case Fortran::parser::ReductionOperator::Operator::And:
      return fir::ReduceOperationEnum::AND;
    case Fortran::parser::ReductionOperator::Operator::Or:
      return fir::ReduceOperationEnum::OR;
    case Fortran::parser::ReductionOperator::Operator::Eqv:
      return fir::ReduceOperationEnum::EQV;
    case Fortran::parser::ReductionOperator::Operator::Neqv:
      return fir::ReduceOperationEnum::NEQV;
    case Fortran::parser::ReductionOperator::Operator::Max:
      return fir::ReduceOperationEnum::MAX;
    case Fortran::parser::ReductionOperator::Operator::Min:
      return fir::ReduceOperationEnum::MIN;
    case Fortran::parser::ReductionOperator::Operator::Iand:
      return fir::ReduceOperationEnum::IAND;
    case Fortran::parser::ReductionOperator::Operator::Ior:
      return fir::ReduceOperationEnum::IOR;
    case Fortran::parser::ReductionOperator::Operator::Ieor:
      return fir::ReduceOperationEnum::IEOR;
    }
    llvm_unreachable("illegal reduction operator");
  }

  /// Collect DO CONCURRENT loop control information.
  IncrementLoopNestInfo getConcurrentControl(
      const Fortran::parser::ConcurrentHeader &header,
      const std::list<Fortran::parser::LocalitySpec> &localityList = {}) {
    IncrementLoopNestInfo incrementLoopNestInfo;
    for (const Fortran::parser::ConcurrentControl &control :
         std::get<std::list<Fortran::parser::ConcurrentControl>>(header.t))
      incrementLoopNestInfo.emplace_back(
          *std::get<0>(control.t).symbol, std::get<1>(control.t),
          std::get<2>(control.t), std::get<3>(control.t), /*isUnordered=*/true);
    IncrementLoopInfo &info = incrementLoopNestInfo.back();
    info.maskExpr = Fortran::semantics::GetExpr(
        std::get<std::optional<Fortran::parser::ScalarLogicalExpr>>(header.t));
    for (const Fortran::parser::LocalitySpec &x : localityList) {
      if (const auto *localList =
              std::get_if<Fortran::parser::LocalitySpec::Local>(&x.u))
        for (const Fortran::parser::Name &x : localList->v)
          info.localSymList.push_back(x.symbol);
      if (const auto *localInitList =
              std::get_if<Fortran::parser::LocalitySpec::LocalInit>(&x.u))
        for (const Fortran::parser::Name &x : localInitList->v)
          info.localInitSymList.push_back(x.symbol);
      for (IncrementLoopInfo &info : incrementLoopNestInfo) {
        if (const auto *reduceList =
                std::get_if<Fortran::parser::LocalitySpec::Reduce>(&x.u)) {
          fir::ReduceOperationEnum reduce_operation = getReduceOperationEnum(
              std::get<Fortran::parser::ReductionOperator>(reduceList->t));
          for (const Fortran::parser::Name &x :
               std::get<std::list<Fortran::parser::Name>>(reduceList->t)) {
            info.reduceSymList.push_back(x.symbol);
            info.reduceOperatorList.push_back(reduce_operation);
          }
        }
      }
      if (const auto *sharedList =
              std::get_if<Fortran::parser::LocalitySpec::Shared>(&x.u))
        for (const Fortran::parser::Name &x : sharedList->v)
          info.sharedSymList.push_back(x.symbol);
    }
    return incrementLoopNestInfo;
  }

  /// Create DO CONCURRENT construct symbol bindings and generate LOCAL_INIT
  /// assignments.
  void handleLocalitySpecs(const IncrementLoopInfo &info) {
    Fortran::semantics::SemanticsContext &semanticsContext =
        bridge.getSemanticsContext();
    fir::LocalitySpecifierOperands privateClauseOps;
    auto doConcurrentLoopOp =
        mlir::dyn_cast_if_present<fir::DoConcurrentLoopOp>(info.loopOp);
    // TODO Promote to using `enableDelayedPrivatization` (which is enabled by
    // default unlike the staging flag) once the implementation of this is more
    // complete.
    bool useDelayedPriv = enableDelayedPrivatization && doConcurrentLoopOp;
    llvm::SetVector<const Fortran::semantics::Symbol *> allPrivatizedSymbols;
    llvm::SmallPtrSet<const Fortran::semantics::Symbol *, 16>
        mightHaveReadHostSym;

    for (const Fortran::semantics::Symbol *symToPrivatize : info.localSymList) {
      if (useDelayedPriv) {
        Fortran::lower::privatizeSymbol<fir::LocalitySpecifierOp>(
            *this, this->getFirOpBuilder(), localSymbols, allPrivatizedSymbols,
            mightHaveReadHostSym, symToPrivatize, &privateClauseOps);
        continue;
      }

      createHostAssociateVarClone(*symToPrivatize, /*skipDefaultInit=*/false);
    }

    for (const Fortran::semantics::Symbol *symToPrivatize :
         info.localInitSymList) {
      if (useDelayedPriv) {
        Fortran::lower::privatizeSymbol<fir::LocalitySpecifierOp>(
            *this, this->getFirOpBuilder(), localSymbols, allPrivatizedSymbols,
            mightHaveReadHostSym, symToPrivatize, &privateClauseOps);
        continue;
      }

      createHostAssociateVarClone(*symToPrivatize, /*skipDefaultInit=*/true);
      const auto *hostDetails =
          symToPrivatize->detailsIf<Fortran::semantics::HostAssocDetails>();
      assert(hostDetails && "missing locality spec host symbol");
      const Fortran::semantics::Symbol *hostSym = &hostDetails->symbol();
      Fortran::evaluate::ExpressionAnalyzer ea{semanticsContext};
      Fortran::evaluate::Assignment assign{
          ea.Designate(Fortran::evaluate::DataRef{*symToPrivatize}).value(),
          ea.Designate(Fortran::evaluate::DataRef{*hostSym}).value()};
      if (Fortran::semantics::IsPointer(*symToPrivatize))
        assign.u = Fortran::evaluate::Assignment::BoundsSpec{};
      genAssignment(assign);
    }

    for (const Fortran::semantics::Symbol *sym : info.sharedSymList) {
      const auto *hostDetails =
          sym->detailsIf<Fortran::semantics::HostAssocDetails>();
      copySymbolBinding(hostDetails->symbol(), *sym);
    }

    if (useDelayedPriv) {
      doConcurrentLoopOp.getLocalVarsMutable().assign(
          privateClauseOps.privateVars);
      doConcurrentLoopOp.setLocalSymsAttr(
          builder->getArrayAttr(privateClauseOps.privateSyms));

      for (auto [sym, privateVar] : llvm::zip_equal(
               allPrivatizedSymbols, privateClauseOps.privateVars)) {
        auto arg = doConcurrentLoopOp.getRegion().begin()->addArgument(
            privateVar.getType(), doConcurrentLoopOp.getLoc());
        bindSymbol(*sym, hlfir::translateToExtendedValue(
                             privateVar.getLoc(), *builder, hlfir::Entity{arg},
                             /*contiguousHint=*/true)
                             .first);
      }
    }

    if (!doConcurrentLoopOp)
      return;

    llvm::SmallVector<bool> reduceVarByRef;
    llvm::SmallVector<mlir::Attribute> reductionDeclSymbols;
    llvm::SmallVector<mlir::Attribute> nestReduceAttrs;

    for (const auto &reduceOp : info.reduceOperatorList)
      nestReduceAttrs.push_back(
          fir::ReduceAttr::get(builder->getContext(), reduceOp));

    llvm::SmallVector<mlir::Value> reduceVars;
    Fortran::lower::omp::ReductionProcessor rp;
    bool result = rp.processReductionArguments<fir::DeclareReductionOp>(
        toLocation(), *this, info.reduceOperatorList, reduceVars,
        reduceVarByRef, reductionDeclSymbols, info.reduceSymList);
    if (!result)
      TODO(toLocation(), "Lowering unrecognised reduction type");

    doConcurrentLoopOp.getReduceVarsMutable().assign(reduceVars);
    doConcurrentLoopOp.setReduceSymsAttr(
        reductionDeclSymbols.empty()
            ? nullptr
            : mlir::ArrayAttr::get(builder->getContext(),
                                   reductionDeclSymbols));
    doConcurrentLoopOp.setReduceAttrsAttr(
        nestReduceAttrs.empty()
            ? nullptr
            : mlir::ArrayAttr::get(builder->getContext(), nestReduceAttrs));
    doConcurrentLoopOp.setReduceByrefAttr(
        reduceVarByRef.empty() ? nullptr
                               : mlir::DenseBoolArrayAttr::get(
                                     builder->getContext(), reduceVarByRef));

    for (auto [sym, reduceVar] :
         llvm::zip_equal(info.reduceSymList, reduceVars)) {
      auto arg = doConcurrentLoopOp.getRegion().begin()->addArgument(
          reduceVar.getType(), doConcurrentLoopOp.getLoc());
      bindSymbol(*sym, hlfir::translateToExtendedValue(
                           reduceVar.getLoc(), *builder, hlfir::Entity{arg},
                           /*contiguousHint=*/true)
                           .first);
    }

    // Note that allocatable, types with ultimate components, and type
    // requiring finalization are forbidden in LOCAL/LOCAL_INIT (F2023 C1130),
    // so no clean-up needs to be generated for these entities.
  }

  void
  genPermutatedLoops(llvm::ArrayRef<Fortran::lower::pft::Evaluation *> doStmts,
                     Fortran::lower::pft::Evaluation *innermostDo) override {
    // Fortran::lower::pft::Evaluation &eval = getEval();
    // bool unstructuredContext = eval.lowerAsUnstructured();

    llvm::SmallVector<mlir::Block *> headerBlocks;
    llvm::SmallVector<IncrementLoopNestInfo, 1> loopInfos;

    auto enterLoop = [&](Fortran::lower::pft::Evaluation &eval) {
      bool unstructuredContext = eval.lowerAsUnstructured();

      // Collect loop nest information.
      // Generate begin loop code directly for infinite and while loops.
      Fortran::lower::pft::Evaluation &doStmtEval =
          eval.getFirstNestedEvaluation();
      auto *doStmt = doStmtEval.getIf<Fortran::parser::NonLabelDoStmt>();
      const auto &loopControl =
          std::get<std::optional<Fortran::parser::LoopControl>>(doStmt->t);
      mlir::Block *preheaderBlock = doStmtEval.block;
      mlir::Block *beginBlock =
          preheaderBlock ? preheaderBlock : builder->getBlock();
      auto createNextBeginBlock = [&]() {
        // Step beginBlock through unstructured preheader, header, and mask
        // blocks, created in outermost to innermost order.
        return beginBlock = beginBlock->splitBlock(beginBlock->end());
      };
      mlir::Block *headerBlock =
          unstructuredContext ? createNextBeginBlock() : nullptr;
      headerBlocks.push_back(headerBlock);
      mlir::Block *bodyBlock = doStmtEval.lexicalSuccessor->block;
      mlir::Block *exitBlock = doStmtEval.parentConstruct->constructExit->block;
      IncrementLoopNestInfo &incrementLoopNestInfo = loopInfos.emplace_back();
      const Fortran::parser::ScalarLogicalExpr *whileCondition = nullptr;
      bool infiniteLoop = !loopControl.has_value();
      if (infiniteLoop) {
        assert(unstructuredContext && "infinite loop must be unstructured");
        startBlock(headerBlock);
      } else if ((whileCondition =
                      std::get_if<Fortran::parser::ScalarLogicalExpr>(
                          &loopControl->u))) {
        assert(unstructuredContext && "while loop must be unstructured");
        maybeStartBlock(preheaderBlock); // no block or empty block
        startBlock(headerBlock);
        genConditionalBranch(*whileCondition, bodyBlock, exitBlock);
      } else if (const auto *bounds =
                     std::get_if<Fortran::parser::LoopControl::Bounds>(
                         &loopControl->u)) {
        // Non-concurrent increment loop.
        IncrementLoopInfo &info = incrementLoopNestInfo.emplace_back(
            *bounds->Name().thing.symbol, bounds->Lower(), bounds->Upper(),
            bounds->Step());
        if (unstructuredContext) {
          maybeStartBlock(preheaderBlock);
          info.hasRealControl = info.loopVariableSym->GetType()->IsNumeric(
              Fortran::common::TypeCategory::Real);
          info.headerBlock = headerBlock;
          info.bodyBlock = bodyBlock;
          info.exitBlock = exitBlock;
        }
      } else {
        llvm_unreachable("Cannot permute DO CONCURRENT");
      }

      // Increment loop begin code. (Infinite/while code was already generated.)
      if (!infiniteLoop && !whileCondition)
        genFIRIncrementLoopBegin(incrementLoopNestInfo, doStmtEval.dirs);
    };

    auto leaveLoop = [&](Fortran::lower::pft::Evaluation &eval,
                         mlir::Block *headerBlock,
                         IncrementLoopNestInfo &incrementLoopNestInfo) {
      bool unstructuredContext = eval.lowerAsUnstructured();

      Fortran::lower::pft::Evaluation &doStmtEval =
          eval.getFirstNestedEvaluation();
      auto *doStmt = doStmtEval.getIf<Fortran::parser::NonLabelDoStmt>();

      const auto &loopControl =
          std::get<std::optional<Fortran::parser::LoopControl>>(doStmt->t);
      bool infiniteLoop = !loopControl.has_value();
      const Fortran::parser::ScalarLogicalExpr *whileCondition =
          std::get_if<Fortran::parser::ScalarLogicalExpr>(&loopControl->u);

      auto iter = std::prev(eval.getNestedEvaluations().end());

      // An EndDoStmt in unstructured code may start a new block.
      Fortran::lower::pft::Evaluation &endDoEval = *iter;
      assert(endDoEval.getIf<Fortran::parser::EndDoStmt>() && "no enddo stmt");
      if (unstructuredContext)
        maybeStartBlock(endDoEval.block);

      // Loop end code.
      if (infiniteLoop || whileCondition)
        genBranch(headerBlock);
      else
        genFIRIncrementLoopEnd(incrementLoopNestInfo);

      // This call may generate a branch in some contexts.
      genFIR(endDoEval, unstructuredContext);
    };

    for (auto l : doStmts)
      enterLoop(*l);

    // Loop body code.
    bool innermostUnstructuredContext = innermostDo->lowerAsUnstructured();

    auto iter = innermostDo->getNestedEvaluations().begin();
    for (auto end = --innermostDo->getNestedEvaluations().end(); iter != end;
         ++iter)
      genFIR(*iter, innermostUnstructuredContext);

    for (auto &&[l, headerBlock, li] :
         llvm::zip_equal(doStmts, headerBlocks, loopInfos))
      leaveLoop(*l, headerBlock, li);
  }

  void attachInlineAttributes(
      mlir::Operation &op,
      const llvm::ArrayRef<const Fortran::parser::CompilerDirective *> &dirs) {
    if (dirs.empty())
      return;

    for (mlir::Value operand : op.getOperands()) {
      if (operand.getDefiningOp())
        attachInlineAttributes(*operand.getDefiningOp(), dirs);
    }

    if (fir::CallOp callOp = mlir::dyn_cast<fir::CallOp>(op)) {
      for (const auto *dir : dirs) {
        Fortran::common::visit(
            Fortran::common::visitors{
                [&](const Fortran::parser::CompilerDirective::NoInline &) {
                  callOp.setInlineAttr(fir::FortranInlineEnum::no_inline);
                },
                [&](const Fortran::parser::CompilerDirective::Inline &) {
                  callOp.setInlineAttr(fir::FortranInlineEnum::inline_hint);
                },
                [&](const Fortran::parser::CompilerDirective::ForceInline &) {
                  callOp.setInlineAttr(fir::FortranInlineEnum::always_inline);
                },
                [&](const auto &) {}},
            dir->u);
      }
    }
  }

  void attachAttributesToDoLoopOperations(
      fir::DoLoopOp &doLoop,
      llvm::SmallVectorImpl<const Fortran::parser::CompilerDirective *> &dirs) {
    if (!doLoop.getOperation() || dirs.empty())
      return;

    for (mlir::Block &block : doLoop.getRegion()) {
      for (mlir::Operation &op : block.getOperations()) {
        if (!dirs.empty())
          attachInlineAttributes(op, dirs);
      }
    }
  }

  // Add AccessGroups attribute on operations in fir::DoLoopOp if this
  // operation has the parallelAccesses attribute.
  void attachAccessGroupAttrToDoLoopOperations(fir::DoLoopOp &doLoop) {
    if (auto loopAnnotAttr = doLoop.getLoopAnnotationAttr()) {
      if (loopAnnotAttr.getParallelAccesses().size()) {
        llvm::SmallVector<mlir::Attribute> accessGroupAttrs(
            loopAnnotAttr.getParallelAccesses().begin(),
            loopAnnotAttr.getParallelAccesses().end());
        mlir::ArrayAttr attrs =
            mlir::ArrayAttr::get(builder->getContext(), accessGroupAttrs);
        doLoop.walk([&](mlir::Operation *op) {
          if (fir::StoreOp storeOp = mlir::dyn_cast<fir::StoreOp>(op)) {
            storeOp.setAccessGroupsAttr(attrs);
          } else if (fir::LoadOp loadOp = mlir::dyn_cast<fir::LoadOp>(op)) {
            loadOp.setAccessGroupsAttr(attrs);
          } else if (hlfir::AssignOp assignOp =
                         mlir::dyn_cast<hlfir::AssignOp>(op)) {
            // In some loops, the HLFIR AssignOp operation can be translated
            // into FIR operation(s) containing StoreOp. It is therefore
            // necessary to forward the AccessGroups attribute.
            assignOp.getOperation()->setAttr(fir::getAccessGroupsAttrName(),
                                             attrs);
          } else if (hlfir::RegionAssignOp regionAssignOp =
                         mlir::dyn_cast<hlfir::RegionAssignOp>(op)) {
            // User defined assignment, WHERE and FORALL assignments are
            // abstracted via hlfir.region_assign at that stage. Set the
            // access group on it so that it can later be propagated to
            // hlfir.assign/fir.store/fir.loads created to implement it.
            regionAssignOp.getOperation()->setAttr(
                fir::getAccessGroupsAttrName(), attrs);
          } else if (fir::CallOp callOp = mlir::dyn_cast<fir::CallOp>(op)) {
            callOp.setAccessGroupsAttr(attrs);
          }
        });
      }
    }
  }

  /// Generate FIR for a DO construct. There are six variants:
  ///  - unstructured infinite and while loops
  ///  - structured and unstructured increment loops
  ///  - structured and unstructured concurrent loops
  void genFIR(const Fortran::parser::DoConstruct &doConstruct) {
    setCurrentPositionAt(doConstruct);
    Fortran::lower::pft::Evaluation &eval = getEval();
    bool unstructuredContext = eval.lowerAsUnstructured();

    // Loops with induction variables inside OpenACC compute constructs
    // need special handling to ensure that the IVs are privatized.
    if (Fortran::lower::isInsideOpenACCComputeConstruct(*builder)) {
      // Open up a new scope for the loop variables.
      localSymbols.pushScope();
      llvm::scope_exit scopeGuard([&]() { localSymbols.popScope(); });

      mlir::Operation *loopOp = Fortran::lower::genOpenACCLoopFromDoConstruct(
          *this, bridge.getSemanticsContext(), localSymbols, doConstruct, eval);
      bool success = loopOp != nullptr;
      if (success) {
        // Sanity check that the builder insertion point is inside the newly
        // generated loop.
        assert(
            loopOp->getRegion(0).isAncestor(
                builder->getInsertionPoint()->getBlock()->getParent()) &&
            "builder insertion point is not inside the newly generated loop");

        // Loop body code.
        auto iter = eval.getNestedEvaluations().begin();
        for (auto end = --eval.getNestedEvaluations().end(); iter != end;
             ++iter)
          genFIR(*iter, unstructuredContext);

        builder->setInsertionPointAfter(loopOp);
        return;
      }
      // Fall back to normal loop handling.
    }

    // Collect loop nest information.
    // Generate begin loop code directly for infinite and while loops.
    Fortran::lower::pft::Evaluation &doStmtEval =
        eval.getFirstNestedEvaluation();
    auto *doStmt = doStmtEval.getIf<Fortran::parser::NonLabelDoStmt>();
    const auto &loopControl =
        std::get<std::optional<Fortran::parser::LoopControl>>(doStmt->t);
    mlir::Block *preheaderBlock = doStmtEval.block;
    mlir::Block *beginBlock =
        preheaderBlock ? preheaderBlock : builder->getBlock();
    auto createNextBeginBlock = [&]() {
      // Step beginBlock through unstructured preheader, header, and mask
      // blocks, created in outermost to innermost order.
      return beginBlock = beginBlock->splitBlock(beginBlock->end());
    };
    mlir::Block *headerBlock =
        unstructuredContext ? createNextBeginBlock() : nullptr;
    mlir::Block *bodyBlock = doStmtEval.lexicalSuccessor->block;
    mlir::Block *exitBlock = doStmtEval.parentConstruct->constructExit->block;
    IncrementLoopNestInfo incrementLoopNestInfo;
    const Fortran::parser::ScalarLogicalExpr *whileCondition = nullptr;
    bool infiniteLoop = !loopControl.has_value();
    if (infiniteLoop) {
      assert(unstructuredContext && "infinite loop must be unstructured");
      startBlock(headerBlock);
    } else if ((whileCondition =
                    std::get_if<Fortran::parser::ScalarLogicalExpr>(
                        &loopControl->u))) {
      // Optionally lower a restricted subset of DO WHILE loops directly to
      // scf.while. This subset excludes early-exit constructs (EXIT/CYCLE/GOTO,
      // etc.) by requiring that the loop body is structured (as decided by the
      // PFT branch analysis), allowing the loop to exit only when the condition
      // becomes false.
      if (!unstructuredContext) {
        maybeStartBlock(preheaderBlock); // no block or empty block
        genDoWhileAsSCFWhile(*whileCondition, eval, doStmtEval);
        return;
      }

      assert(unstructuredContext && "while loop must be unstructured");
      maybeStartBlock(preheaderBlock); // no block or empty block
      startBlock(headerBlock);
      genConditionalBranch(*whileCondition, bodyBlock, exitBlock);
    } else if (const auto *bounds =
                   std::get_if<Fortran::parser::LoopControl::Bounds>(
                       &loopControl->u)) {
      // Non-concurrent increment loop.
      IncrementLoopInfo &info = incrementLoopNestInfo.emplace_back(
          *bounds->Name().thing.symbol, bounds->Lower(), bounds->Upper(),
          bounds->Step());
      if (unstructuredContext) {
        maybeStartBlock(preheaderBlock);
        info.hasRealControl = info.loopVariableSym->GetType()->IsNumeric(
            Fortran::common::TypeCategory::Real);
        info.headerBlock = headerBlock;
        info.bodyBlock = bodyBlock;
        info.exitBlock = exitBlock;
      }
    } else {
      const auto *concurrent =
          std::get_if<Fortran::parser::LoopControl::Concurrent>(
              &loopControl->u);
      assert(concurrent && "invalid DO loop variant");
      incrementLoopNestInfo = getConcurrentControl(
          std::get<Fortran::parser::ConcurrentHeader>(concurrent->t),
          std::get<std::list<Fortran::parser::LocalitySpec>>(concurrent->t));
      if (unstructuredContext) {
        maybeStartBlock(preheaderBlock);
        for (IncrementLoopInfo &info : incrementLoopNestInfo) {
          // The original loop body provides the body and latch blocks of the
          // innermost dimension. The (first) body block of a non-innermost
          // dimension is the preheader block of the immediately enclosed
          // dimension. The latch block of a non-innermost dimension is the
          // exit block of the immediately enclosed dimension.
          auto createNextExitBlock = [&]() {
            // Create unstructured loop exit blocks, outermost to innermost.
            return exitBlock = insertBlock(exitBlock);
          };
          bool isInnermost = &info == &incrementLoopNestInfo.back();
          bool isOutermost = &info == &incrementLoopNestInfo.front();
          info.headerBlock = isOutermost ? headerBlock : createNextBeginBlock();
          info.bodyBlock = isInnermost ? bodyBlock : createNextBeginBlock();
          info.exitBlock = isOutermost ? exitBlock : createNextExitBlock();
          if (info.maskExpr)
            info.maskBlock = createNextBeginBlock();
        }
      }
    }

    // Introduce a `do concurrent` scope to bind symbols corresponding to local,
    // local_init, and reduce region arguments.
    if (!incrementLoopNestInfo.empty() &&
        incrementLoopNestInfo.back().isConcurrent)
      localSymbols.pushScope();

    // Increment loop begin code. (Infinite/while code was already generated.)
    if (!infiniteLoop && !whileCondition)
      genFIRIncrementLoopBegin(incrementLoopNestInfo, doStmtEval.dirs);

    // Loop body code.
    auto iter = eval.getNestedEvaluations().begin();
    for (auto end = --eval.getNestedEvaluations().end(); iter != end; ++iter)
      genFIR(*iter, unstructuredContext);

    // An EndDoStmt in unstructured code may start a new block.
    Fortran::lower::pft::Evaluation &endDoEval = *iter;
    assert(endDoEval.getIf<Fortran::parser::EndDoStmt>() && "no enddo stmt");
    if (unstructuredContext)
      maybeStartBlock(endDoEval.block);

    // Loop end code.
    if (infiniteLoop || whileCondition)
      genBranch(headerBlock);
    else
      genFIRIncrementLoopEnd(incrementLoopNestInfo);

    // This call may generate a branch in some contexts.
    genFIR(endDoEval, unstructuredContext);

    // Add AccessGroups attribute on operations in fir::DoLoopOp if necessary
    for (IncrementLoopInfo &info : incrementLoopNestInfo)
      if (auto loopOp = mlir::dyn_cast_if_present<fir::DoLoopOp>(info.loopOp))
        attachAccessGroupAttrToDoLoopOperations(loopOp);

    if (!incrementLoopNestInfo.empty() &&
        incrementLoopNestInfo.back().isConcurrent)
      localSymbols.popScope();

    // Add attribute(s) on operations in fir::DoLoopOp if necessary
    for (IncrementLoopInfo &info : incrementLoopNestInfo)
      if (auto loopOp = mlir::dyn_cast_if_present<fir::DoLoopOp>(info.loopOp))
        attachAttributesToDoLoopOperations(loopOp, doStmtEval.dirs);
  }

  /// Generate FIR to evaluate loop control values (lower, upper and step).
  mlir::Value genControlValue(const Fortran::lower::SomeExpr *expr,
                              const IncrementLoopInfo &info,
                              bool *isConst = nullptr) {
    mlir::Location loc = toLocation();
    mlir::Type controlType = info.isStructured() ? builder->getIndexType()
                                                 : info.getLoopVariableType();
    Fortran::lower::StatementContext stmtCtx;
    if (expr) {
      if (isConst)
        *isConst = Fortran::evaluate::IsConstantExpr(*expr);
      return builder->createConvert(loc, controlType,
                                    createFIRExpr(loc, expr, stmtCtx));
    }

    if (isConst)
      *isConst = true;
    if (info.hasRealControl)
      return builder->createRealConstant(loc, controlType, 1u);
    return builder->createIntegerConstant(loc, controlType, 1); // step
  }

  // For unroll directives without a value, force full unrolling.
  // For unroll directives with a value, if the value is greater than 1,
  // force unrolling with the given factor. Otherwise, disable unrolling.
  mlir::LLVM::LoopUnrollAttr
  genLoopUnrollAttr(std::optional<std::uint64_t> directiveArg) {
    mlir::BoolAttr falseAttr =
        mlir::BoolAttr::get(builder->getContext(), false);
    mlir::BoolAttr trueAttr = mlir::BoolAttr::get(builder->getContext(), true);
    mlir::IntegerAttr countAttr;
    mlir::BoolAttr fullUnrollAttr;
    bool shouldUnroll = true;
    if (directiveArg.has_value()) {
      auto unrollingFactor = directiveArg.value();
      if (unrollingFactor == 0 || unrollingFactor == 1) {
        shouldUnroll = false;
      } else {
        countAttr =
            builder->getIntegerAttr(builder->getI64Type(), unrollingFactor);
      }
    } else {
      fullUnrollAttr = trueAttr;
    }

    mlir::BoolAttr disableAttr = shouldUnroll ? falseAttr : trueAttr;
    return mlir::LLVM::LoopUnrollAttr::get(
        builder->getContext(), /*disable=*/disableAttr, /*count=*/countAttr, {},
        /*full=*/fullUnrollAttr, {}, {}, {});
  }

  // Enabling unroll and jamming directive without a value.
  // For directives with a value, if the value is greater than 1,
  // force unrolling with the given factor. Otherwise, disable unrolling and
  // jamming.
  mlir::LLVM::LoopUnrollAndJamAttr
  genLoopUnrollAndJamAttr(std::optional<std::uint64_t> count) {
    mlir::BoolAttr falseAttr =
        mlir::BoolAttr::get(builder->getContext(), false);
    mlir::BoolAttr trueAttr = mlir::BoolAttr::get(builder->getContext(), true);
    mlir::IntegerAttr countAttr;
    bool shouldUnroll = true;
    if (count.has_value()) {
      auto unrollingFactor = count.value();
      if (unrollingFactor == 0 || unrollingFactor == 1) {
        shouldUnroll = false;
      } else {
        countAttr =
            builder->getIntegerAttr(builder->getI64Type(), unrollingFactor);
      }
    }

    mlir::BoolAttr disableAttr = shouldUnroll ? falseAttr : trueAttr;
    return mlir::LLVM::LoopUnrollAndJamAttr::get(
        builder->getContext(), /*disable=*/disableAttr, /*count*/ countAttr, {},
        {}, {}, {}, {});
  }

  // Enabling loop vectorization attribute.
  mlir::LLVM::LoopVectorizeAttr
  genLoopVectorizeAttr(mlir::BoolAttr disableAttr,
                       mlir::BoolAttr scalableEnable,
                       mlir::IntegerAttr vectorWidth) {
    mlir::LLVM::LoopVectorizeAttr va;
    if (disableAttr)
      va = mlir::LLVM::LoopVectorizeAttr::get(
          builder->getContext(),
          /*disable=*/disableAttr, /*predicate=*/{},
          /*scalableEnable=*/scalableEnable,
          /*vectorWidth=*/vectorWidth, {}, {}, {});
    return va;
  }

  void addLoopAnnotationAttr(
      IncrementLoopInfo &info,
      llvm::SmallVectorImpl<const Fortran::parser::CompilerDirective *> &dirs) {
    mlir::BoolAttr disableVecAttr;
    mlir::BoolAttr scalableEnable;
    mlir::IntegerAttr vectorWidth;
    mlir::LLVM::LoopUnrollAttr ua;
    mlir::LLVM::LoopUnrollAndJamAttr uja;
    llvm::SmallVector<mlir::LLVM::AccessGroupAttr> aga;
    bool has_attrs = false;
    for (const auto *dir : dirs) {
      Fortran::common::visit(
          Fortran::common::visitors{
              [&](const Fortran::parser::CompilerDirective::VectorAlways &) {
                disableVecAttr =
                    mlir::BoolAttr::get(builder->getContext(), false);
                has_attrs = true;
              },
              [&](const Fortran::parser::CompilerDirective::VectorLength &vl) {
                using Kind =
                    Fortran::parser::CompilerDirective::VectorLength::Kind;
                Kind kind = std::get<Kind>(vl.t);
                uint64_t length = std::get<uint64_t>(vl.t);
                disableVecAttr =
                    mlir::BoolAttr::get(builder->getContext(), false);
                if (length != 0)
                  vectorWidth =
                      builder->getIntegerAttr(builder->getI64Type(), length);
                switch (kind) {
                case Kind::Scalable:
                  scalableEnable =
                      mlir::BoolAttr::get(builder->getContext(), true);
                  break;
                case Kind::Fixed:
                  scalableEnable =
                      mlir::BoolAttr::get(builder->getContext(), false);
                  break;
                case Kind::Auto:
                  break;
                }
                has_attrs = true;
              },
              [&](const Fortran::parser::CompilerDirective::Unroll &u) {
                ua = genLoopUnrollAttr(u.v);
                has_attrs = true;
              },
              [&](const Fortran::parser::CompilerDirective::UnrollAndJam &u) {
                uja = genLoopUnrollAndJamAttr(u.v);
                has_attrs = true;
              },
              [&](const Fortran::parser::CompilerDirective::NoVector &u) {
                disableVecAttr =
                    mlir::BoolAttr::get(builder->getContext(), true);
                has_attrs = true;
              },
              [&](const Fortran::parser::CompilerDirective::NoUnroll &u) {
                ua = genLoopUnrollAttr(/*unrollingFactor=*/0);
                has_attrs = true;
              },
              [&](const Fortran::parser::CompilerDirective::NoUnrollAndJam &u) {
                uja = genLoopUnrollAndJamAttr(/*unrollingFactor=*/0);
                has_attrs = true;
              },
              [&](const Fortran::parser::CompilerDirective::IVDep &iv) {
                disableVecAttr =
                    mlir::BoolAttr::get(builder->getContext(), false);
                aga.push_back(
                    mlir::LLVM::AccessGroupAttr::get(builder->getContext()));
                has_attrs = true;
              },
              [&](const auto &) {}},
          dir->u);
    }
    mlir::LLVM::LoopVectorizeAttr va =
        genLoopVectorizeAttr(disableVecAttr, scalableEnable, vectorWidth);
    mlir::LLVM::LoopAnnotationAttr la = mlir::LLVM::LoopAnnotationAttr::get(
        builder->getContext(), {}, /*vectorize=*/va, {}, /*unroll*/ ua,
        /*unroll_and_jam*/ uja, {}, {}, {}, {}, {}, {}, {}, {}, {},
        /*parallelAccesses*/ aga);
    if (has_attrs) {
      if (auto loopOp = mlir::dyn_cast<fir::DoLoopOp>(info.loopOp))
        loopOp.setLoopAnnotationAttr(la);

      if (auto doConcurrentOp =
              mlir::dyn_cast<fir::DoConcurrentLoopOp>(info.loopOp))
        doConcurrentOp.setLoopAnnotationAttr(la);
    }
  }

  /// Generate FIR to begin a structured or unstructured increment loop nest.
  void genFIRIncrementLoopBegin(
      IncrementLoopNestInfo &incrementLoopNestInfo,
      llvm::SmallVectorImpl<const Fortran::parser::CompilerDirective *> &dirs) {
    assert(!incrementLoopNestInfo.empty() && "empty loop nest");
    mlir::Location loc = toLocation();
    mlir::arith::IntegerOverflowFlags iofBackup{};

    llvm::SmallVector<mlir::Value> nestLBs;
    llvm::SmallVector<mlir::Value> nestUBs;
    llvm::SmallVector<mlir::Value> nestSts;
    llvm::SmallVector<mlir::Value> nestReduceOperands;
    llvm::SmallVector<mlir::Attribute> nestReduceAttrs;
    bool genDoConcurrent = false;

    for (IncrementLoopInfo &info : incrementLoopNestInfo) {
      genDoConcurrent = info.isStructured() && info.isConcurrent;

      if (!genDoConcurrent)
        info.loopVariable = genLoopVariableAddress(loc, *info.loopVariableSym,
                                                   info.isConcurrent);

      if (!getLoweringOptions().getIntegerWrapAround()) {
        iofBackup = builder->getIntegerOverflowFlags();
        builder->setIntegerOverflowFlags(
            mlir::arith::IntegerOverflowFlags::nsw);
      }

      nestLBs.push_back(genControlValue(info.lowerExpr, info));
      nestUBs.push_back(genControlValue(info.upperExpr, info));
      bool isConst = true;
      nestSts.push_back(genControlValue(
          info.stepExpr, info, info.isStructured() ? nullptr : &isConst));

      if (!getLoweringOptions().getIntegerWrapAround())
        builder->setIntegerOverflowFlags(iofBackup);

      // Use a temp variable for unstructured loops with non-const step.
      if (!isConst) {
        mlir::Value stepValue = nestSts.back();
        info.stepVariable = builder->createTemporary(loc, stepValue.getType());
        fir::StoreOp::create(*builder, loc, stepValue, info.stepVariable);
      }
    }

    for (auto [info, lowerValue, upperValue, stepValue] :
         llvm::zip_equal(incrementLoopNestInfo, nestLBs, nestUBs, nestSts)) {
      // Structured loop - generate fir.do_loop.
      if (info.isStructured()) {
        if (genDoConcurrent)
          continue;

        // The loop variable is a doLoop op argument.
        mlir::Type loopVarType = info.getLoopVariableType();
        auto loopOp = fir::DoLoopOp::create(
            *builder, loc, lowerValue, upperValue, stepValue,
            /*unordered=*/false,
            /*finalCountValue=*/false,
            builder->createConvert(loc, loopVarType, lowerValue));
        info.loopOp = loopOp;
        builder->setInsertionPointToStart(loopOp.getBody());
        mlir::Value loopValue = loopOp.getRegionIterArgs()[0];

        // Update the loop variable value in case it has non-index references.
        fir::StoreOp::create(*builder, loc, loopValue, info.loopVariable);
        addLoopAnnotationAttr(info, dirs);
        continue;
      }

      // Unstructured loop preheader - initialize tripVariable and loopVariable.
      mlir::Value tripCount;
      if (info.hasRealControl) {
        auto diff1 =
            mlir::arith::SubFOp::create(*builder, loc, upperValue, lowerValue);
        auto diff2 =
            mlir::arith::AddFOp::create(*builder, loc, diff1, stepValue);
        tripCount =
            mlir::arith::DivFOp::create(*builder, loc, diff2, stepValue);
        tripCount =
            builder->createConvert(loc, builder->getIndexType(), tripCount);
      } else {
        auto diff1 =
            mlir::arith::SubIOp::create(*builder, loc, upperValue, lowerValue);
        auto diff2 =
            mlir::arith::AddIOp::create(*builder, loc, diff1, stepValue);
        tripCount =
            mlir::arith::DivSIOp::create(*builder, loc, diff2, stepValue);
      }
      if (forceLoopToExecuteOnce) { // minimum tripCount is 1
        mlir::Value one =
            builder->createIntegerConstant(loc, tripCount.getType(), 1);
        auto cond = mlir::arith::CmpIOp::create(
            *builder, loc, mlir::arith::CmpIPredicate::slt, tripCount, one);
        tripCount =
            mlir::arith::SelectOp::create(*builder, loc, cond, one, tripCount);
      }
      info.tripVariable = builder->createTemporary(loc, tripCount.getType());
      fir::StoreOp::create(*builder, loc, tripCount, info.tripVariable);
      fir::StoreOp::create(*builder, loc, lowerValue, info.loopVariable);

      // Unstructured loop header - generate loop condition and mask.
      // Note - Currently there is no way to tag a loop as a concurrent loop.
      startBlock(info.headerBlock);
      tripCount = fir::LoadOp::create(*builder, loc, info.tripVariable);
      mlir::Value zero =
          builder->createIntegerConstant(loc, tripCount.getType(), 0);
      auto cond = mlir::arith::CmpIOp::create(
          *builder, loc, mlir::arith::CmpIPredicate::sgt, tripCount, zero);
      if (info.maskExpr) {
        genConditionalBranch(cond, info.maskBlock, info.exitBlock);
        startBlock(info.maskBlock);
        mlir::Block *latchBlock = getEval().getLastNestedEvaluation().block;
        assert(latchBlock && "missing masked concurrent loop latch block");
        Fortran::lower::StatementContext stmtCtx;
        mlir::Value maskCond = createFIRExpr(loc, info.maskExpr, stmtCtx);
        stmtCtx.finalizeAndReset();
        genConditionalBranch(maskCond, info.bodyBlock, latchBlock);
      } else {
        genConditionalBranch(cond, info.bodyBlock, info.exitBlock);
        if (&info != &incrementLoopNestInfo.back()) // not innermost
          startBlock(info.bodyBlock); // preheader block of enclosed dimension
      }
      if (info.hasLocalitySpecs()) {
        mlir::OpBuilder::InsertPoint insertPt = builder->saveInsertionPoint();
        builder->setInsertionPointToStart(info.bodyBlock);
        handleLocalitySpecs(info);
        builder->restoreInsertionPoint(insertPt);
      }
    }

    if (genDoConcurrent) {
      auto loopWrapperOp = fir::DoConcurrentOp::create(*builder, loc);
      builder->setInsertionPointToStart(
          builder->createBlock(&loopWrapperOp.getRegion()));

      for (IncrementLoopInfo &info : llvm::reverse(incrementLoopNestInfo)) {
        info.loopVariable = genLoopVariableAddress(loc, *info.loopVariableSym,
                                                   info.isConcurrent);
      }

      builder->setInsertionPointToEnd(loopWrapperOp.getBody());
      auto loopOp = fir::DoConcurrentLoopOp::create(
          *builder, loc, nestLBs, nestUBs, nestSts, /*loopAnnotation=*/nullptr,
          /*local_vars=*/mlir::ValueRange{},
          /*local_syms=*/nullptr, /*reduce_vars=*/mlir::ValueRange{},
          /*reduce_byref=*/nullptr, /*reduce_syms=*/nullptr,
          /*reduce_attrs=*/nullptr);

      llvm::SmallVector<mlir::Type> loopBlockArgTypes(
          incrementLoopNestInfo.size(), builder->getIndexType());
      llvm::SmallVector<mlir::Location> loopBlockArgLocs(
          incrementLoopNestInfo.size(), loc);
      mlir::Region &loopRegion = loopOp.getRegion();
      mlir::Block *loopBlock = builder->createBlock(
          &loopRegion, loopRegion.begin(), loopBlockArgTypes, loopBlockArgLocs);
      builder->setInsertionPointToStart(loopBlock);

      for (auto [info, blockArg] :
           llvm::zip_equal(incrementLoopNestInfo, loopBlock->getArguments())) {
        info.loopOp = loopOp;
        mlir::Value loopValue =
            builder->createConvert(loc, info.getLoopVariableType(), blockArg);
        fir::StoreOp::create(*builder, loc, loopValue, info.loopVariable);

        if (info.maskExpr) {
          Fortran::lower::StatementContext stmtCtx;
          mlir::Value maskCond = createFIRExpr(loc, info.maskExpr, stmtCtx);
          stmtCtx.finalizeAndReset();
          mlir::Value maskCondCast =
              builder->createConvert(loc, builder->getI1Type(), maskCond);
          auto ifOp = fir::IfOp::create(*builder, loc, maskCondCast,
                                        /*withElseRegion=*/false);
          builder->setInsertionPointToStart(&ifOp.getThenRegion().front());
        }
      }

      IncrementLoopInfo &innermostInfo = incrementLoopNestInfo.back();

      if (innermostInfo.hasLocalitySpecs())
        handleLocalitySpecs(innermostInfo);

      addLoopAnnotationAttr(innermostInfo, dirs);
    }
  }

  /// Generate FIR to end a structured or unstructured increment loop nest.
  void genFIRIncrementLoopEnd(IncrementLoopNestInfo &incrementLoopNestInfo) {
    assert(!incrementLoopNestInfo.empty() && "empty loop nest");
    mlir::Location loc = toLocation();
    mlir::arith::IntegerOverflowFlags flags{};
    if (!getLoweringOptions().getIntegerWrapAround())
      flags = bitEnumSet(flags, mlir::arith::IntegerOverflowFlags::nsw);
    auto iofAttr = mlir::arith::IntegerOverflowFlagsAttr::get(
        builder->getContext(), flags);
    for (auto it = incrementLoopNestInfo.rbegin(),
              rend = incrementLoopNestInfo.rend();
         it != rend; ++it) {
      IncrementLoopInfo &info = *it;
      if (info.isStructured()) {
        // End fir.do_concurent.loop.
        if (info.isConcurrent) {
          builder->setInsertionPointAfter(info.loopOp->getParentOp());
          continue;
        }

        // End fir.do_loop.
        // Decrement tripVariable.
        auto doLoopOp = mlir::cast<fir::DoLoopOp>(info.loopOp);
        builder->setInsertionPointToEnd(doLoopOp.getBody());
        // Step loopVariable to help optimizations such as vectorization.
        // Induction variable elimination will clean up as necessary.
        mlir::Value step = builder->createConvert(
            loc, info.getLoopVariableType(), doLoopOp.getStep());
        mlir::Value loopVar =
            fir::LoadOp::create(*builder, loc, info.loopVariable);
        mlir::Value loopVarInc =
            mlir::arith::AddIOp::create(*builder, loc, loopVar, step, iofAttr);
        fir::ResultOp::create(*builder, loc, loopVarInc);
        builder->setInsertionPointAfter(doLoopOp);
        // The loop control variable may be used after the loop.
        fir::StoreOp::create(*builder, loc, doLoopOp.getResult(0),
                             info.loopVariable);
        continue;
      }

      // Unstructured loop - decrement tripVariable and step loopVariable.
      mlir::Value tripCount =
          fir::LoadOp::create(*builder, loc, info.tripVariable);
      mlir::Value one =
          builder->createIntegerConstant(loc, tripCount.getType(), 1);
      tripCount = mlir::arith::SubIOp::create(*builder, loc, tripCount, one);
      fir::StoreOp::create(*builder, loc, tripCount, info.tripVariable);
      mlir::Value value = fir::LoadOp::create(*builder, loc, info.loopVariable);
      mlir::Value step;
      if (info.stepVariable)
        step = fir::LoadOp::create(*builder, loc, info.stepVariable);
      else
        step = genControlValue(info.stepExpr, info);
      if (info.hasRealControl)
        value = mlir::arith::AddFOp::create(*builder, loc, value, step);
      else
        value =
            mlir::arith::AddIOp::create(*builder, loc, value, step, iofAttr);
      fir::StoreOp::create(*builder, loc, value, info.loopVariable);

      genBranch(info.headerBlock);
      if (&info != &incrementLoopNestInfo.front()) // not outermost
        startBlock(info.exitBlock); // latch block of enclosing dimension
    }
  }

  /// Generate structured or unstructured FIR for an IF construct.
  /// The initial statement may be either an IfStmt or an IfThenStmt.
  void genFIR(const Fortran::parser::IfConstruct &) {
    Fortran::lower::pft::Evaluation &eval = getEval();

    // Structured fir.if nest.
    if (eval.lowerAsStructured()) {
      fir::IfOp topIfOp, currentIfOp;
      for (Fortran::lower::pft::Evaluation &e : eval.getNestedEvaluations()) {
        auto genIfOp = [&](mlir::Value cond) {
          Fortran::lower::pft::Evaluation &succ = *e.controlSuccessor;
          bool hasElse = succ.isA<Fortran::parser::ElseIfStmt>() ||
                         succ.isA<Fortran::parser::ElseStmt>();
          auto ifOp = fir::IfOp::create(*builder, toLocation(), cond,
                                        /*withElseRegion=*/hasElse);
          builder->setInsertionPointToStart(&ifOp.getThenRegion().front());
          return ifOp;
        };
        setCurrentPosition(e.position);
        if (auto *s = e.getIf<Fortran::parser::IfThenStmt>()) {
          topIfOp = currentIfOp = genIfOp(genIfCondition(s, e.negateCondition));
        } else if (auto *s = e.getIf<Fortran::parser::IfStmt>()) {
          topIfOp = currentIfOp = genIfOp(genIfCondition(s, e.negateCondition));
        } else if (auto *s = e.getIf<Fortran::parser::ElseIfStmt>()) {
          builder->setInsertionPointToStart(
              &currentIfOp.getElseRegion().front());
          currentIfOp = genIfOp(genIfCondition(s));
        } else if (e.isA<Fortran::parser::ElseStmt>()) {
          builder->setInsertionPointToStart(
              &currentIfOp.getElseRegion().front());
        } else if (e.isA<Fortran::parser::EndIfStmt>()) {
          builder->setInsertionPointAfter(topIfOp);
          genFIR(e, /*unstructuredContext=*/false); // may generate branch
        } else {
          genFIR(e, /*unstructuredContext=*/false);
        }
      }
      return;
    }

    // Unstructured branch sequence.
    llvm::SmallVector<Fortran::lower::pft::Evaluation *> exits, fallThroughs;
    collectFinalEvaluations(eval, exits, fallThroughs);

    for (Fortran::lower::pft::Evaluation &e : eval.getNestedEvaluations()) {
      auto genIfBranch = [&](mlir::Value cond) {
        if (e.lexicalSuccessor == e.controlSuccessor) // empty block -> exit
          genConditionalBranch(cond, e.parentConstruct->constructExit,
                               e.controlSuccessor);
        else // non-empty block
          genConditionalBranch(cond, e.lexicalSuccessor, e.controlSuccessor);
      };
      setCurrentPosition(e.position);
      if (auto *s = e.getIf<Fortran::parser::IfThenStmt>()) {
        maybeStartBlock(e.block);
        genIfBranch(genIfCondition(s, e.negateCondition));
      } else if (auto *s = e.getIf<Fortran::parser::IfStmt>()) {
        maybeStartBlock(e.block);
        genIfBranch(genIfCondition(s, e.negateCondition));
      } else if (auto *s = e.getIf<Fortran::parser::ElseIfStmt>()) {
        startBlock(e.block);
        genIfBranch(genIfCondition(s));
      } else {
        genFIR(e);
        if (blockIsUnterminated()) {
          if (llvm::is_contained(exits, &e))
            genConstructExitBranch(*eval.constructExit);
          else if (llvm::is_contained(fallThroughs, &e))
            genBranch(e.lexicalSuccessor->block);
        }
      }
    }
  }

  void genCaseOrRankConstruct() {
    Fortran::lower::pft::Evaluation &eval = getEval();
    Fortran::lower::StatementContext stmtCtx;
    pushActiveConstruct(eval, stmtCtx);

    llvm::SmallVector<Fortran::lower::pft::Evaluation *> exits, fallThroughs;
    collectFinalEvaluations(eval, exits, fallThroughs);

    for (Fortran::lower::pft::Evaluation &e : eval.getNestedEvaluations()) {
      if (e.getIf<Fortran::parser::EndSelectStmt>())
        maybeStartBlock(e.block);
      else
        genFIR(e);
      if (blockIsUnterminated()) {
        if (llvm::is_contained(exits, &e))
          genConstructExitBranch(*eval.constructExit);
        else if (llvm::is_contained(fallThroughs, &e))
          genBranch(e.lexicalSuccessor->block);
      }
    }
    popActiveConstruct();
  }
  void genFIR(const Fortran::parser::CaseConstruct &) {
    genCaseOrRankConstruct();
  }

  template <typename A>
  void genNestedStatement(const Fortran::parser::Statement<A> &stmt) {
    setCurrentPosition(stmt.source);
    genFIR(stmt.statement);
  }

  /// Force the binding of an explicit symbol. This is used to bind and re-bind
  /// a concurrent control symbol to its value.
  void forceControlVariableBinding(const Fortran::semantics::Symbol *sym,
                                   mlir::Value inducVar) {
    mlir::Location loc = toLocation();
    assert(sym && "There must be a symbol to bind");
    mlir::Type toTy = genType(*sym);
    // FIXME: this should be a "per iteration" temporary.
    mlir::Value tmp =
        builder->createTemporary(loc, toTy, toStringRef(sym->name()),
                                 llvm::ArrayRef<mlir::NamedAttribute>{
                                     fir::getAdaptToByRefAttr(*builder)});
    mlir::Value cast = builder->createConvert(loc, toTy, inducVar);
    fir::StoreOp::create(*builder, loc, cast, tmp);
    addSymbol(*sym, tmp, /*force=*/true);
  }

  /// Process a concurrent header for a FORALL. (Concurrent headers for DO
  /// CONCURRENT loops are lowered elsewhere.)
  void genFIR(const Fortran::parser::ConcurrentHeader &header) {
    llvm::SmallVector<mlir::Value> lows;
    llvm::SmallVector<mlir::Value> highs;
    llvm::SmallVector<mlir::Value> steps;
    if (explicitIterSpace.isOutermostForall()) {
      // For the outermost forall, we evaluate the bounds expressions once.
      // Contrastingly, if this forall is nested, the bounds expressions are
      // assumed to be pure, possibly dependent on outer concurrent control
      // variables, possibly variant with respect to arguments, and will be
      // re-evaluated.
      mlir::Location loc = toLocation();
      mlir::Type idxTy = builder->getIndexType();
      Fortran::lower::StatementContext &stmtCtx =
          explicitIterSpace.stmtContext();
      auto lowerExpr = [&](auto &e) {
        return fir::getBase(genExprValue(e, stmtCtx));
      };
      for (const Fortran::parser::ConcurrentControl &ctrl :
           std::get<std::list<Fortran::parser::ConcurrentControl>>(header.t)) {
        const Fortran::lower::SomeExpr *lo =
            Fortran::semantics::GetExpr(std::get<1>(ctrl.t));
        const Fortran::lower::SomeExpr *hi =
            Fortran::semantics::GetExpr(std::get<2>(ctrl.t));
        auto &optStep =
            std::get<std::optional<Fortran::parser::ScalarIntExpr>>(ctrl.t);
        lows.push_back(builder->createConvert(loc, idxTy, lowerExpr(*lo)));
        highs.push_back(builder->createConvert(loc, idxTy, lowerExpr(*hi)));
        steps.push_back(
            optStep.has_value()
                ? builder->createConvert(
                      loc, idxTy,
                      lowerExpr(*Fortran::semantics::GetExpr(*optStep)))
                : builder->createIntegerConstant(loc, idxTy, 1));
      }
    }
    auto lambda = [&, lows, highs, steps]() {
      // Create our iteration space from the header spec.
      mlir::Location loc = toLocation();
      mlir::Type idxTy = builder->getIndexType();
      llvm::SmallVector<fir::DoLoopOp> loops;
      Fortran::lower::StatementContext &stmtCtx =
          explicitIterSpace.stmtContext();
      auto lowerExpr = [&](auto &e) {
        return fir::getBase(genExprValue(e, stmtCtx));
      };
      const bool outermost = !lows.empty();
      std::size_t headerIndex = 0;
      for (const Fortran::parser::ConcurrentControl &ctrl :
           std::get<std::list<Fortran::parser::ConcurrentControl>>(header.t)) {
        const Fortran::semantics::Symbol *ctrlVar =
            std::get<Fortran::parser::Name>(ctrl.t).symbol;
        mlir::Value lb;
        mlir::Value ub;
        mlir::Value by;
        if (outermost) {
          assert(headerIndex < lows.size());
          if (headerIndex == 0)
            explicitIterSpace.resetInnerArgs();
          lb = lows[headerIndex];
          ub = highs[headerIndex];
          by = steps[headerIndex++];
        } else {
          const Fortran::lower::SomeExpr *lo =
              Fortran::semantics::GetExpr(std::get<1>(ctrl.t));
          const Fortran::lower::SomeExpr *hi =
              Fortran::semantics::GetExpr(std::get<2>(ctrl.t));
          auto &optStep =
              std::get<std::optional<Fortran::parser::ScalarIntExpr>>(ctrl.t);
          lb = builder->createConvert(loc, idxTy, lowerExpr(*lo));
          ub = builder->createConvert(loc, idxTy, lowerExpr(*hi));
          by = optStep.has_value()
                   ? builder->createConvert(
                         loc, idxTy,
                         lowerExpr(*Fortran::semantics::GetExpr(*optStep)))
                   : builder->createIntegerConstant(loc, idxTy, 1);
        }
        auto lp = fir::DoLoopOp::create(
            *builder, loc, lb, ub, by, /*unordered=*/true,
            /*finalCount=*/false, explicitIterSpace.getInnerArgs());
        if ((!loops.empty() || !outermost) && !lp.getRegionIterArgs().empty())
          fir::ResultOp::create(*builder, loc, lp.getResults());
        explicitIterSpace.setInnerArgs(lp.getRegionIterArgs());
        builder->setInsertionPointToStart(lp.getBody());
        forceControlVariableBinding(ctrlVar, lp.getInductionVar());
        loops.push_back(lp);
      }
      if (outermost)
        explicitIterSpace.setOuterLoop(loops[0]);
      explicitIterSpace.appendLoops(loops);
      if (const auto &mask =
              std::get<std::optional<Fortran::parser::ScalarLogicalExpr>>(
                  header.t);
          mask.has_value()) {
        mlir::Type i1Ty = builder->getI1Type();
        fir::ExtendedValue maskExv =
            genExprValue(*Fortran::semantics::GetExpr(mask.value()), stmtCtx);
        mlir::Value cond =
            builder->createConvert(loc, i1Ty, fir::getBase(maskExv));
        auto ifOp = fir::IfOp::create(*builder, loc,
                                      explicitIterSpace.innerArgTypes(), cond,
                                      /*withElseRegion=*/true);
        fir::ResultOp::create(*builder, loc, ifOp.getResults());
        builder->setInsertionPointToStart(&ifOp.getElseRegion().front());
        fir::ResultOp::create(*builder, loc, explicitIterSpace.getInnerArgs());
        builder->setInsertionPointToStart(&ifOp.getThenRegion().front());
      }
    };
    // Push the lambda to gen the loop nest context.
    explicitIterSpace.pushLoopNest(lambda);
  }

  void genFIR(const Fortran::parser::ForallAssignmentStmt &stmt) {
    Fortran::common::visit([&](const auto &x) { genFIR(x); }, stmt.u);
  }

  void genFIR(const Fortran::parser::EndForallStmt &) {
    if (!lowerToHighLevelFIR())
      cleanupExplicitSpace();
  }

  template <typename A>
  void prepareExplicitSpace(const A &forall) {
    if (!explicitIterSpace.isActive())
      analyzeExplicitSpace(forall);
    localSymbols.pushScope();
    explicitIterSpace.enter();
  }

  /// Cleanup all the FORALL context information when we exit.
  void cleanupExplicitSpace() {
    explicitIterSpace.leave();
    localSymbols.popScope();
  }

  /// Generate FIR for a FORALL statement.
  void genFIR(const Fortran::parser::ForallStmt &stmt) {
    const auto &concurrentHeader =
        std::get<
            Fortran::common::Indirection<Fortran::parser::ConcurrentHeader>>(
            stmt.t)
            .value();
    if (lowerToHighLevelFIR()) {
      mlir::OpBuilder::InsertionGuard guard(*builder);
      Fortran::lower::SymMapScope scope(localSymbols);
      genForallNest(concurrentHeader);
      genFIR(std::get<Fortran::parser::UnlabeledStatement<
                 Fortran::parser::ForallAssignmentStmt>>(stmt.t)
                 .statement);
      return;
    }
    prepareExplicitSpace(stmt);
    genFIR(concurrentHeader);
    genFIR(std::get<Fortran::parser::UnlabeledStatement<
               Fortran::parser::ForallAssignmentStmt>>(stmt.t)
               .statement);
    cleanupExplicitSpace();
  }

  /// Generate FIR for a FORALL construct.
  void genFIR(const Fortran::parser::ForallConstruct &forall) {
    mlir::OpBuilder::InsertPoint insertPt = builder->saveInsertionPoint();
    if (lowerToHighLevelFIR())
      localSymbols.pushScope();
    else
      prepareExplicitSpace(forall);
    genNestedStatement(
        std::get<
            Fortran::parser::Statement<Fortran::parser::ForallConstructStmt>>(
            forall.t));
    for (const Fortran::parser::ForallBodyConstruct &s :
         std::get<std::list<Fortran::parser::ForallBodyConstruct>>(forall.t)) {
      Fortran::common::visit(
          Fortran::common::visitors{
              [&](const Fortran::parser::WhereConstruct &b) { genFIR(b); },
              [&](const Fortran::common::Indirection<
                  Fortran::parser::ForallConstruct> &b) { genFIR(b.value()); },
              [&](const auto &b) { genNestedStatement(b); }},
          s.u);
    }
    genNestedStatement(
        std::get<Fortran::parser::Statement<Fortran::parser::EndForallStmt>>(
            forall.t));
    if (lowerToHighLevelFIR()) {
      localSymbols.popScope();
      builder->restoreInsertionPoint(insertPt);
    }
  }

  /// Lower the concurrent header specification.
  void genFIR(const Fortran::parser::ForallConstructStmt &stmt) {
    const auto &concurrentHeader =
        std::get<
            Fortran::common::Indirection<Fortran::parser::ConcurrentHeader>>(
            stmt.t)
            .value();
    if (lowerToHighLevelFIR())
      genForallNest(concurrentHeader);
    else
      genFIR(concurrentHeader);
  }

  /// Generate hlfir.forall and hlfir.forall_mask nest given a Forall
  /// concurrent header
  void genForallNest(const Fortran::parser::ConcurrentHeader &header) {
    mlir::Location loc = getCurrentLocation();
    const bool isOutterForall = !isInsideHlfirForallOrWhere();
    hlfir::ForallOp outerForall;
    auto evaluateControl = [&](const auto &parserExpr, mlir::Region &region,
                               bool isMask = false) {
      if (region.empty())
        builder->createBlock(&region);
      Fortran::lower::StatementContext localStmtCtx;
      const Fortran::semantics::SomeExpr *anlalyzedExpr =
          Fortran::semantics::GetExpr(parserExpr);
      assert(anlalyzedExpr && "expression semantics failed");
      // Generate the controls of outer forall outside of the hlfir.forall
      // region. They do not depend on any previous forall indices (C1123) and
      // no assignment has been made yet that could modify their value. This
      // will simplify hlfir.forall analysis because the SSA integer value
      // yielded will obviously not depend on any variable modified by the
      // forall when produced outside of it.
      // This is not done for the mask because it may (and in usual code, does)
      // depend on the forall indices that have just been defined as
      // hlfir.forall block arguments.
      mlir::OpBuilder::InsertPoint innerInsertionPoint;
      if (outerForall && !isMask) {
        innerInsertionPoint = builder->saveInsertionPoint();
        builder->setInsertionPoint(outerForall);
      }
      mlir::Value exprVal =
          fir::getBase(genExprValue(*anlalyzedExpr, localStmtCtx, &loc));
      localStmtCtx.finalizeAndPop();
      if (isMask)
        exprVal = builder->createConvert(loc, builder->getI1Type(), exprVal);
      if (innerInsertionPoint.isSet())
        builder->restoreInsertionPoint(innerInsertionPoint);
      hlfir::YieldOp::create(*builder, loc, exprVal);
    };
    for (const Fortran::parser::ConcurrentControl &control :
         std::get<std::list<Fortran::parser::ConcurrentControl>>(header.t)) {
      auto forallOp = hlfir::ForallOp::create(*builder, loc);
      if (isOutterForall && !outerForall)
        outerForall = forallOp;
      evaluateControl(std::get<1>(control.t), forallOp.getLbRegion());
      evaluateControl(std::get<2>(control.t), forallOp.getUbRegion());
      if (const auto &optionalStep =
              std::get<std::optional<Fortran::parser::ScalarIntExpr>>(
                  control.t))
        evaluateControl(*optionalStep, forallOp.getStepRegion());
      // Create block argument and map it to a symbol via an hlfir.forall_index
      // op (symbols must be mapped to in memory values).
      const Fortran::semantics::Symbol *controlVar =
          std::get<Fortran::parser::Name>(control.t).symbol;
      assert(controlVar && "symbol analysis failed");
      mlir::Type controlVarType = genType(*controlVar);
      mlir::Block *forallBody = builder->createBlock(&forallOp.getBody(), {},
                                                     {controlVarType}, {loc});
      auto forallIndex = hlfir::ForallIndexOp::create(
          *builder, loc, fir::ReferenceType::get(controlVarType),
          forallBody->getArguments()[0],
          builder->getStringAttr(controlVar->name().ToString()));
      localSymbols.addVariableDefinition(*controlVar, forallIndex,
                                         /*force=*/true);
      auto end = fir::FirEndOp::create(*builder, loc);
      builder->setInsertionPoint(end);
    }

    if (const auto &maskExpr =
            std::get<std::optional<Fortran::parser::ScalarLogicalExpr>>(
                header.t)) {
      // Create hlfir.forall_mask and set insertion point in its body.
      auto forallMaskOp = hlfir::ForallMaskOp::create(*builder, loc);
      evaluateControl(*maskExpr, forallMaskOp.getMaskRegion(), /*isMask=*/true);
      builder->createBlock(&forallMaskOp.getBody());
      auto end = fir::FirEndOp::create(*builder, loc);
      builder->setInsertionPoint(end);
    }
  }

  void attachDirectiveToLoop(const Fortran::parser::CompilerDirective &dir,
                             Fortran::lower::pft::Evaluation *e) {
    while (e->isDirective())
      e = e->lexicalSuccessor;

    if (e->isA<Fortran::parser::NonLabelDoStmt>())
      e->dirs.push_back(&dir);
  }

  void
  attachInliningDirectiveToStmt(const Fortran::parser::CompilerDirective &dir,
                                Fortran::lower::pft::Evaluation *e) {
    while (e->isDirective())
      e = e->lexicalSuccessor;

    // If the successor is a statement or a do loop, the compiler
    // will perform inlining.
    if (e->isA<Fortran::parser::CallStmt>() ||
        e->isA<Fortran::parser::NonLabelDoStmt>() ||
        e->isA<Fortran::parser::AssignmentStmt>()) {
      e->dirs.push_back(&dir);
    } else {
      mlir::Location loc = toLocation();
      mlir::emitWarning(loc,
                        "Inlining directive not in front of loops, function"
                        "call or assignment.\n");
    }
  }

  void genFIR(const Fortran::parser::CompilerDirective &dir) {
    Fortran::lower::pft::Evaluation &eval = getEval();

    Fortran::common::visit(
        Fortran::common::visitors{
            [&](const Fortran::parser::CompilerDirective::VectorAlways &) {
              attachDirectiveToLoop(dir, &eval);
            },
            [&](const Fortran::parser::CompilerDirective::VectorLength &) {
              attachDirectiveToLoop(dir, &eval);
            },
            [&](const Fortran::parser::CompilerDirective::Unroll &) {
              attachDirectiveToLoop(dir, &eval);
            },
            [&](const Fortran::parser::CompilerDirective::UnrollAndJam &) {
              attachDirectiveToLoop(dir, &eval);
            },
            [&](const Fortran::parser::CompilerDirective::NoVector &) {
              attachDirectiveToLoop(dir, &eval);
            },
            [&](const Fortran::parser::CompilerDirective::NoUnroll &) {
              attachDirectiveToLoop(dir, &eval);
            },
            [&](const Fortran::parser::CompilerDirective::NoUnrollAndJam &) {
              attachDirectiveToLoop(dir, &eval);
            },
            [&](const Fortran::parser::CompilerDirective::ForceInline &) {
              attachInliningDirectiveToStmt(dir, &eval);
            },
            [&](const Fortran::parser::CompilerDirective::Inline &) {
              attachInliningDirectiveToStmt(dir, &eval);
            },
            [&](const Fortran::parser::CompilerDirective::NoInline &) {
              attachInliningDirectiveToStmt(dir, &eval);
            },
            [&](const Fortran::parser::CompilerDirective::Prefetch &prefetch) {
              for (const auto &p : prefetch.v) {
                Fortran::evaluate::ExpressionAnalyzer ea{
                    bridge.getSemanticsContext()};
                Fortran::lower::SomeExpr expr{*ea.Analyze(
                    std::get<Fortran::parser::DataRef>(p.value().u))};
                Fortran::lower::StatementContext stmtCtx;
                mlir::Location loc = genLocation(dir.source);
                hlfir::Entity var = Fortran::lower::convertExprToHLFIR(
                    loc, *this, expr, localSymbols, stmtCtx);
                mlir::Value memRef =
                    hlfir::genVariableRawAddress(loc, *builder, var);

                // TODO: Don't use default value, instead get the following
                //       info from the directive
                uint32_t isRead{0}, localityHint{3}, isData{1};
                fir::PrefetchOp::create(*builder, loc, memRef, isRead,
                                        localityHint, isData);
              }
            },
            [&](const Fortran::parser::CompilerDirective::IVDep &) {
              attachDirectiveToLoop(dir, &eval);
            },
            [&](const auto &) {}},
        dir.u);
  }

  void genFIR(const Fortran::parser::OpenACCConstruct &acc) {
    mlir::OpBuilder::InsertPoint insertPt = builder->saveInsertionPoint();

    // Cache constructs should not push/pop a scope because they need to update
    // the symbol map for subsequent statements in the same loop body.
    bool isCacheConstruct =
        std::holds_alternative<Fortran::parser::OpenACCCacheConstruct>(acc.u);

    if (!isCacheConstruct)
      localSymbols.pushScope();
    mlir::Value exitCond = genOpenACCConstruct(
        *this, bridge.getSemanticsContext(), getEval(), acc, localSymbols);

    const Fortran::parser::OpenACCLoopConstruct *accLoop =
        std::get_if<Fortran::parser::OpenACCLoopConstruct>(&acc.u);
    const Fortran::parser::OpenACCCombinedConstruct *accCombined =
        std::get_if<Fortran::parser::OpenACCCombinedConstruct>(&acc.u);

    Fortran::lower::pft::Evaluation *curEval = &getEval();
    // Determine collapse depth/force and loopCount
    bool collapseForce = false;
    ui