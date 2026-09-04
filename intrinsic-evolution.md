# Design Document: Intrinsic Signature Evolution with Optional Arguments

## 1. Overview
LLVM IR requires calls to overloaded intrinsics to use the type suffix required
by their mangling rules, for example:

```llvm
%r = call float @llvm.fmuladd.f32(float %a, float %b, float %c)
```

This design permits the equivalent unmangled spelling:

```llvm
%r = call float @llvm.fmuladd(float %a, float %b, float %c)
```

The broader feature is intrinsic signature evolution. An intrinsic may gain
new signatures by appending trailing arguments while every previously declared
signature remains complete and independently valid. Intermediate arities exist
only when declared explicitly. An absent argument has no LLVM IR value: the
parser and construction APIs must not synthesize an operand for it.

The target lowering defines the behavior of each signature variant. For
example, a target may lower a zero-argument asyncmark as applying to all stages
and lower a one-argument asyncmark using the explicit stage operand. That
relationship is inherent in target lowering and may be implemented with
TableGen patterns, custom lowering, or both; generic IR does not materialize an
`ALL` value.

The parser already has the information required to resolve an unmangled
overloaded spelling: the call instruction provides its return type and all
explicit arguments, and the intrinsic signature definitions describe the
valid variants. Direct C++ construction must use the same signature-selection
logic.

The intended result is normal LLVM IR. The temporary base-name reference is
removed, each call uses the canonical declaration for the selected signature,
and its operand list remains unchanged.

### 1.1 Introductory Example

Suppose asyncmark supports both a historical zero-argument signature and a new
signature with an optional trailing `i32` stage. Input may use the base name for
both calls:

```llvm
call void @llvm.amdgcn.asyncmark()
call void @llvm.amdgcn.asyncmark(i32 %stage)
```

After parsing, the zero-argument signature retains the base name and the
present optional argument contributes its resolved type to the mangled name:

```llvm
call void @llvm.amdgcn.asyncmark()
call void @llvm.amdgcn.asyncmark.i32(i32 %stage)
```

The first call is not rewritten to pass `i32 ALL`. Absence is represented by
an empty sequence of optional type-mangling components, while each present
optional parameter contributes one component. This generalizes to multiple
trailing optional parameters:

```text
foo()              -> @llvm.foo
foo(i32)           -> @llvm.foo.i32
foo(i32, ptr)      -> @llvm.foo.i32.p0
```

Optional parameters contribute mangling components even when their types are
concrete rather than ordinary intrinsic overload types. Existing overload
components remain first, followed by the types of present optional parameters.

### 1.2 Goals

- Accept an unmangled base name for an intrinsic when it is used directly as a
  callee.
- Infer all overload types from the complete callsite signature, including the
  return type and explicitly supplied arguments.
- Allow any existing intrinsic, overloaded or non-overloaded, to gain trailing
  optional arguments of any type supported by intrinsic signatures.
- Keep every explicitly declared historical signature valid. New variants must
  never make an older accepted call spelling invalid.
- Select the matching canonical signature during textual parsing,
  intrinsic-specific `IRBuilder` construction, generic `IRBuilder::CreateCall`,
  `CreateInvoke`, `CreateCallBr`, and their direct instruction factories without
  changing operands or control-flow destinations.
- Permit one base name to resolve to multiple declarations when callsites use
  different signature variants or overloads in the same module.
- Preserve existing handling of explicitly mangled intrinsic names.
- Reuse intrinsic descriptor matching and existing intrinsic remangling and
  auto-upgrade infrastructure.
- Leave the implementation and semantics of each variant to target lowering.
- Produce parser diagnostics for unknown names, invalid signatures, and invalid
  uses.

### 1.3 Non-goals

- Allowing an intrinsic base name as a first-class value, function pointer, or
  operand of another call.
- Inferring overload types from uses other than the direct callsite signature.
- Supporting non-trailing signature changes. Every variant extends the owner by
  appending parameters.
- Changing or removing parameters that existed before optional-signature
  evolution began. Arbitrary signature migration remains an explicit
  auto-upgrade task.
- Prescribing whether a target implements variant lowering with TableGen or
  custom lowering.
- Replacing explicit declarations or changing the behavior of ordinary
  non-intrinsic forward references.

## 2. Resolution Flow

```text
  Parse call/invoke/callbr
          |
          | resolveFunctionType builds the callsite FunctionType
          v
  getGlobalVal("llvm.fmuladd", pointer-to-FunctionType)
          |
          | createGlobalFwdRef; record in ForwardRefVals
          | all uses share the placeholder
          v
  LLParser::validate
          |
          | lookupIntrinsicID(base name)
          | for each placeholder use:
          |   require direct CallBase callee
          |   match FunctionType to one signature variant
          |   collect that variant's OverloadTys
          |   insert its canonical declaration
          |   replace only this callee use
          v
  erase placeholder and ForwardRefVals entry
```

The important property is that resolution is per use, rather than per base
name. A module may use one base name with multiple optional-argument variants
and multiple overload types. Each use independently selects one complete
signature and canonical declaration.

Signature selection never completes a shorter call. If the callsite has no
optional operand, it selects the shorter variant; if it has that operand, it
selects a longer variant. In both cases the parsed `FunctionType` and operand
list remain unchanged.

## 3. Detailed Component Design

### A. Call Parsing (`llvm/lib/AsmParser/LLParser.cpp`)

No intrinsic-specific parsing is needed in `parseCall`, `parseInvoke`, or
`parseCallBr`.

1. `resolveFunctionType` handles the short call syntax. If the type before the
   callee is not already a function type, it constructs
   `FunctionType::get(ReturnType, ArgumentTypes, false)` from the return type
   and parsed argument values.
2. The resulting type is stored in `CalleeID.FTy` before
   `convertValIDToValue` resolves the callee.
3. For an unresolved named global, `getGlobalVal` creates a temporary global
   forward reference and records `{placeholder, first-reference-location}` in
   `ForwardRefVals`.
4. Repeated references to the same base name reuse that placeholder. The
   placeholder's use-list consequently records every callsite, regardless of
   its concrete overload type.

The normal parser checks argument values against the callsite `FunctionType`
while constructing the instruction. Intrinsic-specific constraints are
validated later against the intrinsic descriptor table.

### B. Intrinsic Lookup and Signature Matching (`llvm/lib/IR/Intrinsics.cpp`)

`Intrinsic::lookupIntrinsicID` performs the name lookup. For an overloaded
intrinsic, the lookup accepts the base name as well as an exact or prefixed
mangled name. An unknown name returns `Intrinsic::not_intrinsic`.

An intrinsic with optional arguments has multiple generated signature
variants under one intrinsic ID. Each variant is one complete trailing
extension of the owner signature. A variant-aware signature resolver obtains the
generated IIT descriptors and checks, in order:

1. The fixed parameter count and vararg property.
2. The return type.
3. Every parameter present in that variant.
4. Deferred relationships such as `IIT_ARG` constraints.
5. That exactly one variant matches the callsite `FunctionType`.

When an overloaded descriptor matches, the concrete type is appended to
`OverloadTys` in the existing mangling order. The resolver separately records
the types of present optional parameters as `OptionalTys`.

The canonical mangling sequence is:

```text
existing overload type components, followed by present optional types
```

The shortest variant has an empty `OptionalTys` sequence and therefore retains
its existing canonical name. Each longer variant appends one mangling component
per present optional parameter, even when that parameter has a concrete rather
than ordinarily overloaded type.

The parser must not construct suffix strings itself. It passes the selected
variant, `OverloadTys`, and `OptionalTys` to signature-aware
`Intrinsic::getName` and `Intrinsic::getOrInsertDeclaration` APIs.

### C. Finalization (`LLParser::validate`)

The intrinsic pass runs after each direct call's `FunctionType` is known. It
processes unresolved intrinsic forward references and resolved uses of a base
name that may denote a longer optional signature.

Relying only on `ForwardRefVals` is insufficient. The shortest historical
declaration may already own the base name, while another call using that name
selects a longer signature. Such direct calls must still be queued for
signature resolution rather than being permanently bound by name alone.

For each candidate direct call:

1. Compute `IID = Intrinsic::lookupIntrinsicID(Name)`.
2. Require the named value to be the direct callee use.
3. Match `CB->getFunctionType()` to exactly one signature variant.
4. Collect fresh `OverloadTys` and `OptionalTys` for that callsite.
5. Obtain the canonical declaration for those results.
6. Replace only the callee use. Do not replace the `CallBase`, change its
   `FunctionType`, or change any argument operand.
7. Remove an unresolved placeholder after all of its uses have been handled.

The existing auto-upgrade path remains for unrelated historical migrations.
Optional historical signatures do not take an upgrade path because they remain
valid signatures in their original form.

### D. Interaction With Existing Declarations

The shortest historical declaration retains the base name and follows the
ordinary declaration and verification path. It is not remangled merely because
longer optional variants exist. Explicitly mangled longer declarations are
validated against the optional types encoded in their names.

The inserted declaration is reused when another callsite selects the same
`IID`, overload types, and present optional types. `getOrInsertDeclaration`
also handles the existing LLVM case where a declaration with the desired name
has an incompatible function type by renaming the invalid declaration and
inserting the correct one.

## 4. Invariants and Edge Cases

### 4.1 Placeholder invariants

- A `ForwardRefVals` entry represents one unresolved symbol name and one
  placeholder global value.
- Every placeholder use must be rewritable before the placeholder is erased.
- The placeholder must not be replaced globally with one declaration, because
  one base name can represent multiple optional signatures and overloads.
- `ForwardRefVals` must be mutated only after the use-list has been processed.

### 4.2 Supported call forms

Signature-variant resolution applies uniformly to `call`, `invoke`, and
`callbr`, because all derive from `CallBase` and expose the same function type
and callee use. Resolution does not reconstruct the instruction or alter normal,
unwind, default, or indirect destinations.

Existing verifier restrictions remain authoritative. In particular, `callbr`
continues to support only inline asm-goto and selected intrinsic IDs, and
intrinsic `callbr` continues to reject operand bundles. Since all variants share
their owner's intrinsic ID, adding a variant neither grants nor removes
`callbr` eligibility.

### 4.3 Type and naming cases

- The shortest signature contributes no optional suffix and retains the
  intrinsic's existing canonical name.
- Each present optional parameter contributes its resolved type to the mangled
  name, even if that type is concrete rather than ordinarily overloaded.
- Overloaded scalar, vector, integer, floating-point, pointer, token, and
  aggregate cases are determined by the IIT descriptors, not by ad hoc parser
  rules.
- Opaque pointers require the full `FunctionType` and descriptor constraints;
  the parser must not attempt to recover pointee types from pointer syntax.
- A vararg mismatch is a signature error even if all fixed parameters match.
- A malformed or unknown `llvm.` name is not treated as an ordinary undefined
  function. It is diagnosed as an unknown intrinsic.

## 5. Error Handling and Diagnostics

Diagnostics should preserve the existing parser location and intrinsic
validator text. In the current implementation, the location stored in
`ForwardRefVals` is the first reference location for that name, so multiple
uses of one invalid base name may report the first location.

| Case | Expected diagnostic |
| :--- | :--- |
| Non-callee use | `intrinsic can only be used as callee` |
| Unknown `llvm.` name | `unknown intrinsic 'llvm.foobar'` |
| Wrong fixed-argument count | Report the accepted variant counts and the actual count. |
| No matching variant | Report the callsite function type and candidate variant diagnostics. |
| Multiple matching variants | Internal/generated-table error; TableGen must reject the overlap. |
| Mangled optional types disagree with the function type | Report an incorrect intrinsic mangling diagnostic. |
| Type constraint violation | Descriptor-specific text, for example `intrinsic argument 1 type (matching overload type 0) expected i8, but got i16` |
| Vararg mismatch | `intrinsic was not defined with variable arguments!` or `intrinsic was defined with variable arguments!` |
| Legacy upgrade failure | The original intrinsic validation error, unless the name is unknown |

Errors should be returned before erasing the unresolved placeholder. A failed
parse does not need to leave a usable partial module, but avoiding premature
mutation makes the failure path deterministic and prevents use-after-erase
problems while diagnostics are being formed.

## 6. Testing Strategy

Tests belong under `llvm/test/Assembler` and should exercise both assembly
acceptance and canonical output.

### 6.1 Positive tests

- A scalar integer overload such as `llvm.umax` resolves to `.i8`, `.i16`, and
  `.i32` in one module.
- A floating-point overload resolves to the expected scalar suffix.
- The same unmangled base name is used by both `call` and `invoke`.
- A `callbr` use resolves through the same path.
- Multiple uses selecting the same overload share one declaration.
- Existing explicitly mangled names remain unchanged.
- A non-overloaded intrinsic resolves without an added suffix.
- A legacy spelling still takes the auto-upgrade path.

### 6.2 Negative tests

- `llvm/test/Assembler/implicit-intrinsic-declaration-invalid.ll`: related
  argument types disagree.
- `implicit-intrinsic-declaration-invalid2.ll`: the base name is used as a
  non-callee operand.
- `implicit-intrinsic-declaration-invalid3.ll`: the `llvm.` name is unknown.
- `implicit-intrinsic-declaration-invalid4.ll`: argument count is invalid.
- Add coverage for return-type mismatch, vararg mismatch, pointer address-space
  constraints, and an invalid `invoke` or `callbr` signature.

### 6.3 Regression commands

The minimum focused checks are:

```bash
llvm-lit llvm/test/Assembler/implicit-intrinsic-declaration.ll \
  llvm/test/Assembler/implicit-intrinsic-declaration-invalid*.ll
```

For a configured build, also run `llvm-as` followed by `opt -S` on the positive
test and verify that the output contains canonical mangled declarations and no
unresolved base-name calls. Existing intrinsic and auto-upgrade assembler
tests should be run to detect regressions in the shared finalization path.

## 7. Existing Unmangled-Name Implementation Checklist

- [x] Build the callsite `FunctionType` in the existing call parsers.
- [x] Record unresolved `llvm.` names through `ForwardRefVals`.
- [x] Look up the intrinsic ID during parser finalization.
- [x] Validate each callsite and collect independent overload types.
- [x] Insert/reuse canonical intrinsic declarations.
- [x] Rewire individual callee uses and erase the placeholder.
- [x] Preserve legacy intrinsic auto-upgrade behavior.
- [x] Diagnose non-callee, unknown-name, and signature errors.
- [ ] Add any missing `invoke`, `callbr`, vararg, and return-type regression tests.

## 8. Decisions and Future Work

- Whether diagnostics should report the specific failing callsite rather than
  the first reference location stored for the shared placeholder.
- The shortest historical declaration retains its existing name and signature.
- Optional signature variants are a permanent language and construction
  feature, not a parser compatibility mode. No strict-mode option disables it.
- Bitcode and textual IR round-tripping preserve each selected signature and
  operand list while canonicalizing longer variant names.
- Signature selection applies uniformly to `call`, `invoke`, and `callbr` while
  retaining their existing verifier restrictions.

## 9. Generalized Feature: Trailing Optional Arguments

The feature is not specific to asyncmark and should be exposed as a general
intrinsic facility. Any intrinsic may opt into it by declaring trailing
optional parameters; no parser or builder code should identify a target or
intrinsic by name. The compatibility contract is:

```text
Historical signature:  foo(T0, T1)
New signature:         foo(T0, T1, T2?)
```

Both input forms are accepted:

```llvm
call ... @llvm.foo(T0 %a, T1 %b)
call ... @llvm.foo(T0 %a, T1 %b, T2 %c)
```

Both remain complete calls in memory, with distinct canonical declarations:

```llvm
call ... @llvm.foo(T0 %a, T1 %b)
call ... @llvm.foo.<T2-mangle>(T0 %a, T1 %b, T2 %c)
```

The shorter form does not imply a value for `T2`. Target lowering handles that
variant according to its own semantics. The longer form adds the resolved type
of `T2` to the intrinsic's mangling sequence.

### 9.1 TableGen Interface

The existing intrinsic definition remains unchanged. Each additional signature
is a separate, named record that explicitly references its owner and lists only
the appended parameters:

```tablegen
def int_foo : Intrinsic<[], [], []>;

def int_foo_with_mode :
  IntrinsicSignatureVariant<
      int_foo,
      [llvm_anyint_ty]>;

def int_foo_with_mode_and_scope :
  IntrinsicSignatureVariant<
      int_foo,
      [llvm_anyint_ty, llvm_i32_ty]>;

def int_foo_with_i32_mode_and_wide_scope :
  IntrinsicSignatureVariant<
      int_foo,
      [llvm_i32_ty, llvm_i64_ty]>;
```

This combines the two forms of variation. Different mode types are ordinary
type overloads within one signature, while the zero-, one-, and two-argument
forms are distinct signature variants:

```llvm
declare void @llvm.foo()

declare void @llvm.foo.i8(i8)
declare void @llvm.foo.i16(i16)

declare void @llvm.foo.i8.i32(i8, i32)
declare void @llvm.foo.i16.i32(i16, i32)
declare void @llvm.foo.i32.i32(i32, i32)
declare void @llvm.foo.i32.i64(i32, i64)
```

The first mangling component in the longer forms is the resolved overloaded
mode type. The final `.i32` in the two-argument forms records the presence and
type of the appended scope parameter. Each arity is explicit: removing
`int_foo_with_mode` would remove the one-argument signatures without affecting
the zero- or two-argument signatures.

The concrete `int_foo_with_i32_mode_and_wide_scope` variant is disjoint from
the generic `(anyint, i32)` form because its scope is `i64`. The complete
function type therefore selects exactly one variant without a specificity
ordering.

The division of responsibility is:

* Forms with the same parameter count and IIT structure use ordinary LLVM
  overload types within one signature record.
* Forms with different fixed parameter counts use separate
  `IntrinsicSignatureVariant` records.
* Each signature variant may itself contain ordinary overload types, as the
  mode parameters above demonstrate.

This produces a hierarchy of one intrinsic ID, several explicit arity
signatures, and potentially many type instantiations of each signature.

Conceptually, the new class is:

```tablegen
class IntrinsicSignatureVariant<
    Intrinsic owner,
    list<LLVMType> appendedParamTypes,
    list<IntrinsicProperty> additionalProperties = []>;
```

The record name, such as `int_foo_with_scope`, identifies the variant for
TableGen diagnostics, generated APIs, and target patterns. No additional string
identifier is needed. The record does not create another public
`Intrinsic::ID`; it contributes a signature to its owner's ID.

Owner return types, parameters, and properties are inherited. Argument indexes
in `additionalProperties` refer to the complete variant signature. Existing
parameter attributes keep their indexes, and properties such as `ImmArg` apply
only when their parameter is present.

TableGen enforces these rules:

1. `owner` refers to an `Intrinsic`, not another variant.
2. At least one parameter is appended.
3. Owner return types, parameters, and existing properties cannot be replaced
   or removed by the variant.
4. The complete variant is valid under the IIT type and dependency rules.
5. Additional argument properties use valid complete-signature indexes.
6. No two variants may accept the same concrete function type.
7. TableGen rejects overlapping variants; runtime resolution requires exactly
   one matching variant and never applies specificity ranking.

Specificity ranking is deliberately excluded because it would make signature
resolution depend on the current set of TableGen records rather than only on
the IR function type. For example, suppose an intrinsic initially has this
generic variant:

```tablegen
def int_foo_with_mode_and_scope :
  IntrinsicSignatureVariant<
      int_foo,
      [llvm_anyint_ty, llvm_i32_ty]>;
```

The generic variant accepts this call:

```llvm
call void @llvm.foo.i32.i32(i32 %mode, i32 %scope)
```

If a later LLVM release added the following conflicting variant:

```tablegen
def int_foo_with_i32_mode_and_scope :
  IntrinsicSignatureVariant<
      int_foo,
      [llvm_i32_ty, llvm_i32_ty]>;
```

a “most-specific wins” rule would select the new record for exactly the same
IR. Variant-specific properties or lowering could then change without any
change to the module, silently reinterpreting existing textual IR and bitcode.

Rejecting overlap preserves the compatibility invariant: a concrete function
type identifies the same signature variant before and after unrelated variants
are added. Special lowering for one instantiation of an existing overloaded
variant should inspect its resolved types in target lowering rather than add an
overlapping signature record.

Variants are explicit. Adding `[B, C]` creates `foo(A, B, C)` but does not
implicitly create `foo(A, B)`; that intermediate signature requires its own
variant record. TableGen emits the IIT descriptor sequence and mangling
information for each listed variant. It emits no default values and no
operand-materialization table.

### 9.2 Permanent Historical Prefixes

Optional signatures are a permanent compatibility commitment. If an intrinsic evolves as
follows:

```text
v1: foo(A)
v2: foo(A, B?)
v3: foo(A, B?, C?)
```

then argument counts one, two, and three remain accepted indefinitely as
distinct signatures. Adding `C` creates a new variant but does not change the
function type, name, or semantics of the `foo(A)` and `foo(A, B)` variants.

Once released, a variant's parameter types, attributes, name, and semantics are
immutable. Later evolution may only append another optional parameter. Code
review and TableGen tests should treat generated variants as part of the stable
intrinsic compatibility surface.

This guarantee applies only to evolution by appending optional parameters.
Changing the type or meaning of an existing parameter, reordering parameters,
removing parameters, or inserting a parameter before the trailing optional block
requires a conventional explicit auto-upgrade and is outside this facility.

The guarantee applies equally to textual IR, bitcode read by current LLVM, and
supported C++ construction entry points. Readers and builders preserve the
selected signature and canonicalize only its declaration name when necessary.

### 9.3 Shared Signature Resolution API

Signature evolution needs a shared operation that selects one complete variant
from a callsite `FunctionType`. Conceptually:

```cpp
struct IntrinsicSignature {
  FunctionType *FunctionTy;
  SmallVector<Type *, 4> OverloadTypes;
  SmallVector<Type *, 4> OptionalTypes;
  unsigned NumPresentOptionalArgs;
};

std::optional<IntrinsicSignature> resolveIntrinsicSignature(
    Intrinsic::ID ID, Type *ReturnType, ArrayRef<Type *> ExplicitArgTypes,
    bool IsVarArg, raw_ostream &ErrorOS = nulls());
```

The exact API shape may differ, but it must be usable by the parser, verifier,
remangler, `IRBuilder`, and low-level `CallBase` construction. Its algorithm is:

1. Read every generated signature variant for the intrinsic ID.
2. Filter candidates by fixed parameter count and vararg property.
3. Match the return and parameter types against each candidate's IIT
   descriptors.
4. Validate deferred IIT relationships within that candidate.
5. Collect ordinary overload types and the types of present optional
   parameters separately.
6. Require exactly one matching variant.
7. Return the unchanged callsite type and the two mangling sequences.

`Intrinsic::isSignatureValid` can delegate to this resolver when only a boolean
answer is needed. A successful result never contains values to append.

### 9.4 Overload Completeness

Every declared variant must be independently valid, including for overloaded
intrinsics. For example:

```text
foo<T>(T value, i32 stage?) -> T
```

produces a short variant overloaded by `T` and a long variant whose mangling
also contains `i32`.

An overload type introduced only by an optional parameter simply does not exist
in a shorter variant:

```text
foo<T>(T value?) -> void
```

The zero-argument variant has no `T` and retains the base name. A present `T`
is inferred from the operand and added to the longer variant's mangled name.
TableGen rejects only variants whose remaining return or parameter descriptors
depend on an overload definition that was removed with an absent suffix.

### 9.5 Orthogonality to Varargs

LLVM intrinsics already support varargs through a trailing `llvm_vararg_ty` in
their TableGen parameter list. Signature variants preserve that existing
mechanism unchanged.

For a vararg owner, `AppendedParamTypes` extends the fixed parameter list before
the existing `llvm_vararg_ty` marker. The marker remains last, and every variant
inherits the owner's vararg property:

```tablegen
def int_foo :
  Intrinsic<[], [llvm_i32_ty, llvm_vararg_ty], []>;

def int_foo_with_mode :
  IntrinsicSignatureVariant<
      int_foo,
      [llvm_i8_ty]>;
```

This describes two signatures:

```llvm
declare void @llvm.foo(i32, ...)
declare void @llvm.foo.i8(i32, i8, ...)
```

Variant selection uses the fixed parameters and vararg bit encoded in the
callsite `FunctionType`. Actual operands passed through `...` do not select a
variant and do not contribute mangling components.

An existing call with type `void (i32, ...)` therefore remains the base variant
even if its first vararg is an `i8`. To select `void (i32, i8, ...)`, textual IR
or a C++ caller must provide that explicit function type or obtain the
corresponding canonical declaration. Generic `CreateCall` must not reconstruct
fixed parameters from the types of vararg operands.

### 9.6 Parser Normalization

`LLParser` should use the shared signature resolver for unresolved base names
and for base-name calls that initially bind to an existing shorter declaration:

```text
lookupIntrinsicID(Name)
for each candidate direct call:
  require a direct CallBase callee
  resolve its complete FunctionType to exactly one variant
  collect OverloadTys and OptionalTys
  get or insert that variant's canonical declaration
  replace only the callee use
erase the placeholder after all uses are handled
```

The individual call parsers must not special-case intrinsic names or inject
target-specific constants. The shared resolver handles `CallInst`,
`InvokeInst`, and `CallBrInst` without replacing the instruction, changing its
`FunctionType`, changing operands, or changing control-flow destinations.

### 9.7 Direct C++ Construction

Parser resolution does not cover direct construction. A caller may start from
the base declaration and supply operands selecting another variant:

```cpp
Function *F = Intrinsic::getOrInsertDeclaration(M, Intrinsic::foo);
IRBuilder<> Builder(...);
Builder.CreateCall(F, {});             // Selects the base variant.
Builder.CreateCall(F, {Mode, Scope});  // Selects a longer variant.
```

Permanent convenience support applies to both intrinsic-specific builders and
generic call construction. The following entry points must share one
signature-selection helper:

* `IRBuilderBase::CreateIntrinsic` and `CreateIntrinsicWithoutFolding`.
* Every `IRBuilderBase::CreateCall` overload after it converges on a
  `FunctionType`, callee, and operand array.
* Every `IRBuilderBase::CreateInvoke` and `CreateCallBr` overload.
* Every public `CallInst::Create`, `InvokeInst::Create`, and
  `CallBrInst::Create` factory after it converges on the same inputs.

The generic path performs selection only when the callee is a direct `Function`
with a valid intrinsic ID. It does not look through arbitrary casts or alter
ordinary calls. It combines the requested return type and actual operand types
into a candidate callsite signature, resolves exactly one variant, and obtains
that variant's canonical declaration. A detached declaration cannot be
canonicalized because there is no module in which to obtain the selected
declaration and is rejected.

The common operation should:

1. Resolve the intrinsic ID from the direct callee.
2. Build or obtain the requested callsite `FunctionType` from the return type,
   operands, and vararg information.
3. Resolve exactly one signature variant and its mangling types.
4. Obtain that variant's canonical declaration from the module.
5. Perform folding with the original operand list.
6. Construct the requested `CallBase` subclass with the selected declaration,
   unchanged operands, and unchanged control-flow destinations.

The implementation belongs in the generic IR intrinsic layer. All `CreateCall`
entry points must select the variant before allocating the `CallBase`; there
must not be a low-level escape hatch that permanently binds a base declaration
without considering the requested callsite signature.

The low-level factories keep their existing failure model. A valid variant
resolves uniquely; malformed or ambiguous calls continue to trigger the same
assertions or verification failures as malformed calls do today. APIs that
already return `Error` or `Expected` should return a structured signature
diagnostic instead.

Intrinsic folding must run after signature selection so it receives the
selected intrinsic declaration and exactly the operands supplied by the caller.
No construction path appends, removes, or substitutes operands.

### 9.8 Interaction With Auto-Upgrade

Optional signatures are not an auto-upgrade. The base signature remains valid,
and a call selecting it keeps the same function type and operands. New variants
are resolved directly through the current signature table.

`AutoUpgrade` remains responsible for unrelated historical changes that remove
or alter a formerly valid signature. It may reuse the shared matcher when it
needs to identify a current variant, but optional-signature support does not add
an operand-rewriting path and does not route a valid shorter signature through
`UpgradeIntrinsicCall`.

Pre-feature base names remain canonical for the base signature. A longer call
written with the base name is resolved to its type-mangled variant by the parser
or construction API, not by auto-upgrade.

### 9.9 Comparison With AutoUpgrade

AutoUpgrade and signature variants express different compatibility contracts.

| Aspect | AutoUpgrade | Signature variants |
| :--- | :--- | :--- |
| Status of old signature | Legacy input accepted for migration | Permanently valid current IR |
| In-memory result | Rewritten to the new signature | Selected signature remains unchanged |
| Operand list | New operands may be synthesized | Never changed |
| Target lowering | Handles the new canonical form | Handles each declared signature |
| Textual round-trip | Old call prints with the new operands | Old call retains its operands and base declaration |
| Bitcode reading | Migrates the old form | Resolves the old form as a current variant |
| Direct C++ construction | Not automatically covered | Supported by signature-aware construction APIs |
| Generated intrinsic data | One current IIT signature | Multiple IIT signatures under one intrinsic ID |

AutoUpgrade is appropriate when the old operation is exactly equivalent at the
IR level to a new operation with a concrete operand:

```text
old_call(args) == new_call(args, CONSTANT)
```

It provides one canonical representation for optimizers and target lowering.
The reader recognizes an old declaration, creates the current declaration, and
rewrites each old call with the required operand.

Signature variants are appropriate when absence is part of the intrinsic
interface itself:

```text
operation()        target-defined implicit behavior
operation(value)   explicit behavior selected by value
```

Target lowering may implement the first form similarly to a particular explicit
value, but generic IR does not assert that equivalence. Optimizers continue to
see the shorter call unless a separate transform is validly defined to replace
it.

The distinction is visible during C++ construction. AutoUpgrade normally runs
while serialized IR or old declarations are read; it is not an implicit stage
of `IRBuilder::CreateCall`. Signature-aware construction instead selects a
currently valid declaration before creating the call, so both shorter and
longer forms are valid immediately.

Varargs reinforce the difference. AutoUpgrade must not reinterpret an existing
vararg operand as a newly added fixed operand. Signature variants select from
the callsite `FunctionType`, so a call declared as `void (i32, ...)` remains the
base vararg signature even if its first vararg happens to match the type of a
new `void (i32, i8, ...)` variant.

For asyncmark, signature variants are the intended contract. The zero-argument
form remains a zero-argument call whose `ALL` behavior belongs to AMDGPU
lowering. Using AutoUpgrade would establish the stronger and unwanted IR-level
claim that the call is canonically equivalent to one with an explicit `ALL`
operand.

### 9.10 Compatibility Matrix

| Input/API form | Result |
| :--- | :--- |
| Base-name call matching the owner signature | Accepted; base name and operands remain unchanged. |
| Base-name call matching an added variant | Accepted; callee is changed to the type-mangled variant name. |
| Already canonical owner or variant call | Accepted unchanged. |
| `IRBuilder::CreateIntrinsic` selecting a variant | Accepted; folding receives the caller's original operands. |
| Generic `IRBuilder::CreateCall` selecting a variant | Accepted for a direct intrinsic callee. |
| Direct `CallInst::Create` selecting a variant | Accepted for a direct intrinsic callee. |
| `IRBuilder` or direct `InvokeInst` construction selecting a variant | Accepted; normal and unwind destinations are unchanged. |
| `IRBuilder` or direct `CallBrInst` construction selecting a variant | Accepted when the owner intrinsic is already legal in `callbr`; destinations are unchanged. |
| Function type matching no listed variant | Rejected as an invalid intrinsic signature. |
| Function type matching overlapping variants | Rejected by TableGen; runtime treats it as malformed generated data. |
| Non-constant explicit `ImmArg` | Rejected by intrinsic validation. |

### 9.11 Required Tests

The generic feature needs tests independent of asyncmark:

* An owner plus one and multiple explicit appended signature variants.
* A same-arity ordinary type overload without a signature variant.
* A signature variant whose appended parameter is ordinarily overloaded.
* A signature variant combining inherited owner overloads with appended types.
* The `foo_with_mode_and_scope` example, including disjoint concrete mode and
  scope types.
* TableGen rejection of overlapping generic and concrete variants such as
  `(anyint, i32)` and `(i32, i32)`.
* TableGen rejection of duplicate complete signatures and mangled names.
* Calls selecting the owner, an intermediate explicit variant, and the longest
  variant from the same base name.
* Positive `call`, `invoke`, and legal `callbr` parser coverage for base-name
  variant selection.
* `IRBuilder::CreateIntrinsic`, every `IRBuilder` call-form creator, and direct
  `CallInst`, `InvokeInst`, and `CallBrInst` factories selecting owner and added
  variants.
* Folding receiving the original operands after variant selection.
* Invalid argument counts, type relationships, vararg ambiguity, non-constant
  `ImmArg` values, and unknown names.
* Mangling coverage for every type supported as an appended intrinsic
  parameter, including target extension types.
* Historical bitcode read with the same function type, operand list, and base
  declaration name.

Asyncmark should be one target-specific regression test of this generic
facility. It should verify that `asyncmark()` and `wait.asyncmark(count)` retain
their operand lists and lower with inherent `ALL` semantics, while the explicit
stage variants receive type-mangled names and lower using their operands.

### 9.12 Implementation Checklist

- [ ] Add separately named `IntrinsicSignatureVariant` TableGen records with an
  explicit owner and appended parameter list.
- [ ] Group variants under one `Intrinsic::ID` and generate IIT descriptors and
  mangling metadata for each complete signature.
- [ ] Reject duplicate and overlapping concrete function types without
  specificity ranking.
- [ ] Add exact variant resolution returning `OverloadTys` and `OptionalTys`.
- [ ] Extend `getName`, `getType`, and `getOrInsertDeclaration` for variants.
- [ ] Resolve both unresolved and already-bound base-name calls in textual IR.
- [ ] Select variants before folding in `CreateIntrinsic`.
- [ ] Select variants in every `IRBuilder` call-form creator and public
  `CallInst`, `InvokeInst`, and `CallBrInst` factory for direct intrinsic
  callees without changing operands or destinations.
- [ ] Add generic TableGen, parser, bitcode, IR, verifier, and IRBuilder tests
  covering every intrinsic parameter type category.
- [ ] Add asyncmark as a consumer and target regression test.

## 10. Existing Infrastructure Map

Only unmangled-name resolution is directly relevant in the current checkout.
The existing integer default-argument auto-upgrade facility is a different
mechanism and is not used by this design. Signature variant generation,
matching, mangling, and C++ construction support are proposed work. The
following map identifies the existing components to extend:

| Responsibility | Location | Relevant behavior |
| :--- | :--- | :--- |
| Resolve an abbreviated callsite type | `llvm/lib/AsmParser/LLParser.cpp`, `resolveFunctionType` | Builds a `FunctionType` from the return type and parsed argument list for the short call syntax. |
| Create an unresolved named value | `llvm/lib/AsmParser/LLParser.cpp`, `getGlobalVal` | Creates a temporary forward-reference value and records it in `ForwardRefVals`. |
| Resolve intrinsic forward references | `llvm/lib/AsmParser/LLParser.cpp`, `LLParser::validate` | Walks each `llvm.` placeholder use, validates its callsite type, and rewrites its callee use. |
| Map a name to an intrinsic ID | `llvm/lib/IR/Intrinsics.cpp`, `lookupIntrinsicID` | Uses the generated name tables and accepts an exact name, or a prefix for an overloaded intrinsic. |
| Match the callsite signature | `llvm/lib/IR/Intrinsics.cpp`, `isSignatureValid` | Interprets the generated IIT descriptors and appends inferred overload types. |
| Construct the canonical declaration | `llvm/lib/IR/Intrinsics.cpp`, `getOrInsertDeclaration` | Calls `Intrinsic::getName` with the inferred types and inserts or reuses the declaration. |
| Upgrade old intrinsic signatures | `llvm/lib/IR/AutoUpgrade.cpp` | Handles names and signatures that are no longer accepted by the current IIT descriptor. |

The current parser implementation is intentionally shared by `call`,
`invoke`, and `callbr`. Each parser creates a `CallBase`, and finalization only
depends on `CallBase::getFunctionType()` and the particular callee `Use`.

## 11. Name Lookup Semantics

`lookupIntrinsicID` does not parse a mangled suffix into types. It performs a
generated-table lookup and then applies one important rule:

* An intrinsic with neither ordinary overloads nor signature variants requires
  an exact name match.
* An intrinsic with ordinary overloads or signature variants may match its base
  name or a name beginning with the base name and a mangling suffix.

This distinction prevents an arbitrary prefix from turning a non-overloaded
intrinsic into a valid name. A signature variant makes the owner mangleable even
when every appended type is concrete, as in `asyncmark.i32`. The parser must not
infer the intrinsic ID by stripping arbitrary suffix components itself. The
lookup API owns that policy, while the signature resolver owns type matching.

The generated intrinsic name table contains the canonical table name for each
intrinsic, not a separate parser-only registry of every possible overload.
Possible overload and variant spellings are therefore resolved from the ID plus
the generated signature records, and not by enumerating names in the parser.

For a name that is not present in the generated table, `lookupIntrinsicID`
returns `Intrinsic::not_intrinsic`. An unresolved `llvm.` name receives an
intrinsic diagnostic instead of becoming an ordinary undefined function.

## 12. Signature-Matching Contract

The variant-aware signature resolver is shared by the parser and verifier:

```text
OverloadTys = []
OptionalTys = []
for each generated signature variant:
  filter by parameter count and vararg bit
  match return descriptor against FTy.return_type
  match each parameter descriptor against FTy.parameters
  run deferred IIT relationships
  collect ordinary OverloadTys
  collect the variant's appended parameter types as OptionalTys
require exactly one match
return the selected variant and both type sequences
```

Each variant uses the same IIT descriptor machinery as the verifier today. An
`Overloaded` descriptor records an overload index; dependent descriptors such
as `Match`, `VecElement`, and vector relationship descriptors constrain later
matches against the type already recorded at that index. This is why callers
pass the complete `FunctionType` and do not duplicate type rules.

The API outputs ordered vectors of `Type *`, not strings. `OverloadTys` contains
the owner's ordinary overload components. `OptionalTys` contains every appended
parameter type present in the selected variant, including concrete types. The
parser never constructs suffixes itself; `Intrinsic::getName` remains the
single owner of textual mangling.

Two details are important for callers:

1. Both type vectors and the selected variant must be computed per use. One
   base name can select multiple signatures and overloads in the same module.
2. The function type must include the return type, all fixed parameters, and
   the vararg bit. Matching only argument operands is insufficient because the
   return type can be an overload type and vararg status is part of the
   intrinsic signature.

## 13. Finalization Algorithm and Mutation Safety

The finalization pass is a callee-use rewrite, not a global symbol replacement.
Its candidate set includes unresolved placeholders and direct base-name calls
already bound to the owner declaration:

```text
for each candidate direct intrinsic call:
  IID = lookupIntrinsicID(Name)
  require Use is the callee operand of a CallBase
  resolve CallBase.function_type to exactly one variant
  collect local OverloadTys and OptionalTys
  Use.set(getOrInsertDeclaration(IID, Variant, OverloadTys, OptionalTys))
erase an unresolved placeholder after all its uses are handled
```

The early-increment iteration is required because a successful `Use::set`
removes that use from the placeholder's use-list. Replacing all uses at once
would be incorrect: one base name can denote several canonical declarations.
The placeholder is erased only after all of its uses have been processed.

Already-resolved owner uses are queued separately because they are not in
`ForwardRefVals`. Their callee use may remain unchanged when the owner signature
matches or move to a longer type-mangled declaration when another variant
matches.

The successful path can insert a declaration before another use is handled.
`getOrInsertDeclaration` naturally deduplicates subsequent uses selecting the
same intrinsic ID, variant, and type vectors. If a global with the canonical
name already exists but has the wrong kind or prototype, the existing intrinsic
helper renames it and creates the correctly typed declaration.

The error path deliberately leaves the placeholder intact until the error is
returned. This avoids erasing a value while a later use may still be needed for
diagnostics and keeps failed parsing from depending on partial use-list
mutation. Parser failure does not need to produce a usable module, but this
ordering avoids use-after-erase hazards.

## 14. Validation and Upgrade Ordering

Validation and upgrading solve different compatibility problems and must remain
ordered as follows:

1. Look up the ID and resolve the current callsite against all current signature
   variants.
2. If exactly one variant matches, use its canonical declaration immediately
   without changing operands.
3. If no current variant matches, allow existing auto-upgrade logic to
   recognize an unrelated superseded signature.
4. If no upgrade exists, report `unknown intrinsic` for an unknown ID or the
   signature resolver's diagnostic for a known ID.

Trying auto-upgrade first would make current valid variants dependent on legacy
compatibility behavior. A historical signature retained as an explicit variant
always succeeds in step 2 and never reaches `UpgradeIntrinsicCall`.

## 15. Explicit Declarations and Scope of the Feature

This feature handles unresolved references and direct base-name calls that bind
to an existing owner declaration. In particular:

* A direct call to an unresolved base name is resolved from its function type.
* A direct call bound to the owner declaration is reconsidered when its
  callsite type may select a longer variant.
* A use of the base name as a function value is rejected, because no callsite
  supplies enough information to select an overload.
* The explicit owner declaration retains the base name and is verified against
  the owner's original signature.
* An explicitly mangled variant declaration is verified against the exact
  variant and type sequence encoded in its name.

This boundary preserves symbol semantics: only a direct callsite supplies the
complete function type needed to select among signatures. The feature does not
turn an intrinsic base name into a generally overloadable function value.

## 16. Expanded Test Matrix

The focused tests should verify both the rewrite and the boundaries of the
feature. Each positive test should run through `llvm-as` or `opt` and use
`FileCheck` on canonical output.

| Area | Required coverage | Observable result |
| :--- | :--- | :--- |
| Integer overloads | `i8`, `i16`, and `i32` uses of one base name | Separate `.i8`, `.i16`, and `.i32` declarations/calls. |
| Return overload | Return type selects the overload while parameters are dependent | Canonical name reflects the return-derived type. |
| Vector overload | Fixed and scalable vector forms where supported | Correct vector mangling; no parser-specific vector logic. |
| Pointer overload | Different address spaces and opaque pointers | IIT constraints decide validity and canonical suffixing. |
| Shared declaration | Repeated calls with the same signature | One declaration is reused. |
| Owner and variants | One base name used with multiple arities | Owner call remains unchanged; each longer call receives its type-mangled declaration. |
| Mixed overloads and variants | `foo`, `foo(anyint)`, and `foo(anyint, i32)` | Existing overload components precede present appended types. |
| Disjoint variants | `(anyint, i32)` and `(i32, i64)` | Each concrete function type selects exactly one variant. |
| Overlapping variants | `(anyint, i32)` and `(i32, i32)` | TableGen rejects the definition. |
| Unmangled-name call forms | Complete `call`, `invoke`, and legal `callbr` forms | All forms select variants through the same `CallBase` resolver. |
| Non-overloaded intrinsic | Valid base name without overload types | Exact canonical intrinsic name is used. |
| Explicit variant mangling | Already-mangled call | Encoded appended types must match the exact variant. |
| Non-callee use | Base name in a function-pointer or other operand position | `intrinsic can only be used as callee`. |
| Unknown name | Unresolved `llvm.` symbol not in the generated table | `unknown intrinsic '...'`. |
| Argument count | Too few and too many arguments | IIT argument-count diagnostic. |
| Type mismatch | Broken dependent relationship or concrete type | IIT diagnostic identifies the failing position. |
| Varargs | Both expected and unexpected vararg signatures | Vararg-specific diagnostic. |
| Legacy upgrade | Signature rejected by current IIT but recognized by upgrader | Existing auto-upgrade behavior is preserved. |
| Operand preservation | Owner and longer variants | Parsing and round-trip never add, remove, or replace operands. |

The existing `implicit-intrinsic-declaration.ll` test already covers mixed
`i8`, `i16`, and explicitly mangled `i32` calls. The existing invalid test
covers a dependent argument mismatch. New tests should add the remaining rows
without weakening checks on canonical output.

## 17. Compatibility and Performance

The feature has no effect on ordinary globals or intrinsics without signature
variants. Calls using an owner with variants require signature selection even
when the base declaration is already resolved. The parser should mark such
owners in generated metadata so ordinary calls do not enter this path.

Signature variants can be indexed first by fixed argument count and vararg bit,
then matched by IIT descriptors. Matching is performed once per candidate use
because one base name can select different variants and overloads. Declaration
insertion remains a module-symbol-table lookup shared by repeated selections.

Textual round-tripping leaves the owner signature at its existing base name and
canonicalizes longer base-name calls to names containing their appended types.
It never changes operand lists. Rejecting overlapping variants ensures that
adding another disjoint variant cannot reinterpret the function type of
existing textual IR or bitcode.
