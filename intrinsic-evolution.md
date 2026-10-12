# Design: Intrinsic Signature Families

## 1. Overview

LLVM allows intrinsics to have overloaded argument types. At a call, the
intrinsic name is mangled to reflect the concrete type of the actual argument
being passed. The parser can also accept the base name and resolves it from the
callsite signature:

```llvm
%r = call float @llvm.fmuladd(float %a, float %b, float %c)
; Canonical form: @llvm.fmuladd.f32
```

This document defines an intrinsic signature family: one owner and explicit
signature variants sharing a base name. Each call matches one declared
signature; no variant supplies values for absent arguments or replaces another
signature. Target lowering defines each signature's behavior. Adding a variant
to an existing intrinsic is one application of this mechanism.

Section 2 describes the behavior of families in textual IR and bitcode, which
does not depend on how a family is represented internally. Section 3 describes
how a family is declared in TableGen and exposed to C++ clients.

### 1.1 Introductory Example

Asyncmark may support both zero- and one-argument signatures:

```llvm
call void @llvm.amdgcn.asyncmark()
call void @llvm.amdgcn.asyncmark(i32 %stage)
```

The parser canonicalizes only the callee names:

```llvm
call void @llvm.amdgcn.asyncmark()
call void @llvm.amdgcn.asyncmark.i32(i32 %stage)
```

In this particular example, the zero-argument call is equivalent to passing
`ALL` as an explicit stage. But this is determined by the AMDGPU lowering, and
not by replacing the call in the IR itself.

Families also support same-arity alternatives. With both `foo(i32)` and
`foo(float)`, base-name calls select a signature from their callsite types,
while canonical output uses signature mangling:

```llvm
declare void @llvm.foo.i32(i32)
declare void @llvm.foo.f32(float)
```

### 1.2 Goals

- Define exact, non-overlapping signatures under one base name.
- Resolve direct base-name `call`, `invoke`, and `callbr` uses from the callsite
  `FunctionType`, and accept overload-mangled owner names as input.
- Support ordinary overload types within every signature.
- Preserve every declared signature's function type, operands, and semantics.
- Use the same resolver in the parser, verifier, remangler, and C++ construction
  APIs.
- Leave variant-specific behavior to target lowering.

### 1.3 Non-goals

- Supplying default values for missing arguments or synthesizing operands.
- Managing compatibility when an intrinsic author changes or removes a
  signature in a later revision; that is the author's responsibility.
- Resolving an intrinsic base name used as a first-class value or indirect
  callee.
- Defining whether a target uses TableGen patterns or custom lowering.

## 2. Signature Model and IR Behavior

An intrinsic family consists of one owner and zero or more signature variants.
Each member has a complete IIT signature and may contain ordinary overload
types.

Variants are explicit. Declaring `foo(A, B, C)` does not implicitly declare
`foo(A, B)`. An absent parameter has no IR value and no descriptor in the
selected signature.

Every concrete `FunctionType` must match exactly one member of the family.
TableGen rejects duplicate or overlapping signatures.

### 2.1 Canonical Mangling

LLVM's existing *overload mangling* names an intrinsic with one signature using
the types of its overloaded positions. Once an intrinsic has multiple
signatures, *signature mangling* names every member, including the owner, using
the non-void result type, if any, followed by every fixed parameter type in IIT
order. The vararg marker and actual vararg operands are excluded.

Examples:

```text
foo()           -> @llvm.foo
foo(i32)        -> @llvm.foo.i32
foo(float)      -> @llvm.foo.f32
foo(i32, ptr)   -> @llvm.foo.i32.p0
```

A void zero-argument signature keeps the base name because its type sequence is
empty.

The bare base name always selects a signature from the callsite `FunctionType`,
even if it was previously the owner's canonical name. *Overload-mangled names*
using the owner signature alone remain valid input for the owner's signature.
Signature-mangled names are derived independently, so adding another disjoint
variant does not rename existing family members.

### 2.2 Round Trips

Input may use either a base name or an overload-mangled owner name. After
selection, the module uses the selected member's signature-mangled canonical
name. Parsing and printing can therefore change a callee's spelling without
changing its function type or operands. For a family formed from the existing
`llvm.fmuladd` owner and the integer-middle variant in Section 3.4:

Input:

```llvm
%old = call float @llvm.fmuladd.f32(float %a, float %b, float %c)
%new = call float @llvm.fmuladd(float %a, i32 %mode, float %c)
```

Printed:

```llvm
%old = call float @llvm.fmuladd.f32.f32.f32.f32(float %a, float %b, float %c)
%new = call float @llvm.fmuladd.f32.f32.i32.f32(float %a, i32 %mode, float %c)
```

For `asyncmark`, the base name is also the signature-mangled name of the
zero-argument owner. That spelling does not force owner selection: the callsite
type alone selects the one-argument variant when an `i32` is passed.

```llvm
call void @llvm.amdgcn.asyncmark()           ; stays @llvm.amdgcn.asyncmark
call void @llvm.amdgcn.asyncmark(i32 %stage) ; prints @llvm.amdgcn.asyncmark.i32
```

### 2.3 Evolving a Family

An intrinsic author can preserve existing calls by retaining each signature
and adding disjoint variants. For example:

```text
v1: foo(i32)
v2: foo(i32), foo(float)
v3: foo(i32), foo(float), foo(i32, i8)
```

Each of these families satisfies the same constraints on signatures within that
revision. The feature does not compare revisions or prevent authors from
changing or removing signatures. Authors are responsible for the compatibility
of textual IR, bitcode, and C++ clients across such changes. When a
single-signature intrinsic first becomes a family, its canonical names may
change; overload-mangled names using its unchanged owner signature remain
accepted input.

### 2.4 Initial Constraint: No Specificity Ranking

The initial implementation does not apply a "most specific" rule. Suppose the
family contains the signature `foo(anyint, i32)`. It accepts:

```llvm
call void @llvm.foo.i32.i32(i32 %mode, i32 %scope)
```

Adding the signature `foo(i32, i32)` would overlap it. A specificity rule would
reinterpret unchanged IR as the new signature, possibly changing properties or
lowering. The initial implementation rejects this overlap. This is a
conservative implementation constraint that may be relaxed as discussed in
[Appendix A](#appendix-a-specialization-considerations).

### 2.5 Signature Rules

Each violation below is a TableGen error that fails the build. TableGen verifies
the family as a whole, including symbolic constraints on overloaded types.
Let N be the set of suffixed names the family accepts: the signature-mangled
name of every member, plus the overload-mangled names of every member that
accepts them (Section 3 states which members do).

- **S1.** The complete function type differs from every sibling.
- **S2.** No two signatures accept the same concrete `FunctionType`.
- **S3.** Every name in N denotes exactly one `FunctionType`. The bare base
  name is exempt and always resolves from the callsite type.

Two manglings of one member may produce the same string. Different function
types may not.

For example, `i32 foo()` and `void foo(i32)` are disjoint but would both
produce `@llvm.foo.i32`, so S3 rejects the family. A proposed `float ()`
variant of `llvm.fmuladd` would claim the owner's overload-mangled
`@llvm.fmuladd.f32`, so S3 rejects it. These errors name the conflicting
signatures and, where possible, a concrete type and name that exhibit the
conflict.

Further rules that depend on how a family is declared are listed with that
representation in Section 3.

### 2.6 Resolving Calls

Resolution applies uniformly to `call`, `invoke`, and `callbr` because it only
replaces the direct callee use. It does not reconstruct the instruction or
alter operands or control-flow destinations.

Existing verifier restrictions remain unchanged. In particular, `callbr`
continues to support only inline asm-goto and selected intrinsics.

A signature that accepts varargs ends with a vararg marker after its fixed
parameters. The callsite `FunctionType`, not the actual vararg operands, selects
the signature. A call typed as `void (i32, ...)` remains `foo(i32, ...)` even if
its first vararg is `i8`; selecting `foo(i32, i8, ...)` requires that explicit
function type. Vararg operands do not contribute mangling components.

### 2.7 Interaction With AutoUpgrade

Signature families describe current IR; AutoUpgrade independently rewrites
superseded declarations and calls. Adding a family does not change any
AutoUpgrade rule, its entry points, or the way it rewrites operands. The
interaction depends on the existing input path:

- For an undeclared direct call in text IR, the parser first tries the current
  intrinsic signature. If it matches, the call is resolved without upgrading
  it; otherwise the parser tries AutoUpgrade.
- For an explicit text-IR declaration, the parser offers the declaration and
  its calls to AutoUpgrade during end-of-module processing.
- For bitcode, the reader offers intrinsic declarations to AutoUpgrade before
  materializing and upgrading calls that use them.
- C++ construction does not invoke AutoUpgrade.

For example, `llvm.objectsize` currently has four arguments, and AutoUpgrade
recognizes older two- and three-argument forms. A hypothetical current
three-argument variant with the same type as an old form could be selected by
an undeclared text-IR call, while an explicit text or bitcode declaration may
still be upgraded. The feature neither checks for nor resolves this conflict;
the intrinsic author is responsible for interactions with existing upgrade
rules. TableGen validates the current signature family, not AutoUpgrade cases.

For `asyncmark`, retaining the zero-argument form as a current signature means
its all-stage behavior remains defined by AMDGPU lowering. No IR operand is
added merely because a one-argument variant exists.

### 2.8 Compatibility and Cost

Base-name calls and overload-mangled owner names are accepted as described in
Section 2.1. Unknown `llvm.` names remain errors unless existing AutoUpgrade
recognizes them. A base name cannot be used as a first-class value because no
callsite type is available to select a signature.

Generated metadata should mark families with variants so ordinary calls avoid
this path. Variants can be indexed by fixed parameter count and vararg bit
before IIT matching. Declaration insertion remains shared through the module
symbol table.

## 3. Design: Shared Intrinsic ID

In this design, all members of a family share one `Intrinsic::ID`. The ID
identifies the family, not an argument layout; the selected signature is
recovered from the callsite `FunctionType`. Only the owner accepts
overload-mangled names as input, so N in Section 2.5 contains the owner's
overload-mangled names and every member's signature-mangled name.

### 3.1 TableGen Interface

The owner remains an ordinary `Intrinsic` definition. Each additional signature
is a named record containing a complete return type list, parameter type list,
and property list:

```tablegen
class IntrinsicSignatureVariant<
    Intrinsic owner,
    list<LLVMType> retTypes,
    list<LLVMType> paramTypes,
    list<IntrinsicProperty> properties = []>;
```

The record name identifies the variant in diagnostics, generated APIs, and
target patterns; it does not create another public intrinsic ID.

The overlap described in Section 2.4 would be spelled:

```tablegen
def int_foo_with_mode_and_scope :
  IntrinsicSignatureVariant<
      int_foo, [], [llvm_anyint_ty, llvm_i32_ty]>;

def int_foo_with_i32_mode_and_scope :
  IntrinsicSignatureVariant<
      int_foo, [], [llvm_i32_ty, llvm_i32_ty]>;  // rejected: overlaps above
```

### 3.2 Mixed Arity and Type Overloads

```tablegen
def int_foo : Intrinsic<[], [], []>;

def int_foo_with_mode :
  IntrinsicSignatureVariant<
      int_foo, [], [llvm_anyint_ty]>;

def int_foo_with_mode_and_scope :
  IntrinsicSignatureVariant<
      int_foo, [], [llvm_anyint_ty, llvm_i32_ty]>;

def int_foo_with_i32_mode_and_wide_scope :
  IntrinsicSignatureVariant<
      int_foo, [], [llvm_i32_ty, llvm_i64_ty]>;
```

This defines:

```llvm
declare void @llvm.foo()
declare void @llvm.foo.i8(i8)
declare void @llvm.foo.i16(i16)
declare void @llvm.foo.i8.i32(i8, i32)
declare void @llvm.foo.i16.i32(i16, i32)
declare void @llvm.foo.i32.i32(i32, i32)
declare void @llvm.foo.i32.i64(i32, i64)
```

`llvm_anyint_ty` creates ordinary type overloads within a signature. The
`(i32, i64)` signature is disjoint from `(anyint, i32)` because the scope types
differ.

### 3.3 Varargs

Each vararg signature includes a trailing `llvm_vararg_ty` after its fixed
parameters:

```tablegen
def int_foo :
  Intrinsic<[], [llvm_i32_ty, llvm_vararg_ty], []>;

def int_foo_with_mode :
  IntrinsicSignatureVariant<
      int_foo, [], [llvm_i32_ty, llvm_i8_ty, llvm_vararg_ty]>;
```

Selection follows Section 2.6.

### 3.4 Same-Arity Evolution

A concrete alternative does not necessarily broaden the owner to
`llvm_any_ty`:

```tablegen
def int_typed_foo : Intrinsic<[], [llvm_i32_ty], []>;

def int_typed_foo_f32 :
  IntrinsicSignatureVariant<
      int_typed_foo, [], [llvm_float_ty]>;
```

The resulting family contains `@llvm.typed.foo.i32` and
`@llvm.typed.foo.f32`.

The existing `llvm.fmuladd` owner has a floating-point result and three
operands matching that result. A variant can allow a previously invalid
same-arity call without broadening the owner's operands:

```tablegen
def int_fmuladd_with_integer_middle :
  IntrinsicSignatureVariant<
      int_fmuladd, [llvm_anyfloat_ty],
      [LLVMMatchType<0>, llvm_i32_ty, LLVMMatchType<0>]>;
```

For a `float` result, `float (float, float, float)` selects the owner and
`float (float, i32, float)` selects the variant. Their signature-mangled names
are `@llvm.fmuladd.f32.f32.f32.f32` and
`@llvm.fmuladd.f32.f32.i32.f32`, respectively. The owner's existing
overload-mangled spelling, `@llvm.fmuladd.f32`, remains valid input for its
signature. Lowering defines the meaning of the new form.

### 3.5 Validation Rules

In addition to S1 to S3, each violation below is a TableGen error:

- **T1.** `owner` refers to an `Intrinsic`, not another variant.
- **T2.** IIT descriptors and dependent overload relationships are valid.
- **T3.** Argument properties refer to valid indexes in the complete signature.
- **T4.** Common semantic properties do not conflict within the family.

### 3.6 Call Eligibility

Variants share the owner's ID and therefore its `callbr` eligibility.

### 3.7 Target Lowering

All family members share one intrinsic ID. Lowering can distinguish them by
resolved signature or `FunctionType`; targets should not parse the mangled name.
TableGen patterns and custom lowering are both supported.

### 3.8 ID-Based Consumers

An intrinsic ID identifies a family, not an argument layout. When adding a
variant, the maintainer of that ID is responsible for consumers that inspect
the ID and then assume operand positions, types, or behavior. In particular,
`getArgOperand()` checks bounds in debug builds but does not check the selected
signature; a same-arity variant may silently reach owner-specific code.

TableGen assigns one scoped `Intrinsic::Signature` enumerator to each owner and
named variant record. `resolveIntrinsicSignature(ID, FunctionType)` returns the
selected enumerator. A converted consumer dispatches on that result and
explicitly handles or rejects every member it encounters:

```cpp
auto Sig = Intrinsic::resolveIntrinsicSignature(Intrinsic::fmuladd,
                                                 Call.getFunctionType());
if (!Sig)
  return Sig.takeError();
switch (Sig->SignatureID) {
case Intrinsic::Signature::int_fmuladd:
  return handleFloatingOperands(Call);
case Intrinsic::Signature::int_fmuladd_with_integer_middle:
  return handleIntegerMiddle(Call);
default:
  return reportUnhandledIntrinsicSignature(Call, Sig->SignatureID);
}
```

The enumerator names come from the existing unique TableGen record names; no
per-family enum or visitor is required. Generated consumers can use this
dispatch directly. Hand-written ID-based consumers must be audited when their
ID gains variants; the framework cannot discover assumptions in untouched
switches. A guard against unhandled members in such consumers is a future
extension, not a TableGen check on signature declarations.

### 3.9 C++ Construction

Existing `Intrinsic::getType`, `getName`, and `getOrInsertDeclaration` calls
that take an ID and overload types remain owner-directed. The existing
`getOrInsertDeclaration(ID, RetTy, ArgTys)` overload also retains its current
single-signature behavior, including ignoring those types for a non-overloaded
owner. Existing `IRBuilder::CreateIntrinsic` overloads keep their current owner
construction and folding behavior; adding a variant does not reinterpret those
calls or add a new failure for previously accepted inputs. In particular, the
overload types may include an overloaded result but do not identify the family's
signature: the zero- and one-argument `asyncmark` forms both use an empty
overload-type list.

The variant-aware declaration API takes an ID and an explicit `FunctionType`:

```cpp
Function *Intrinsic::getOrInsertDeclaration(
    Module *M, Intrinsic::ID ID, FunctionType *FTy);
```

It resolves `FTy` against the family and returns the selected member's
canonical declaration. The result type, fixed parameters, and vararg bit are
all specified by `FTy`; actual vararg operands never select a member. Callers
can pass the declaration to existing `CreateCall`, `CreateInvoke`, or
`CreateCallBr` APIs. A typed `CreateIntrinsic(FTy, ID, Args)` form may use the
same resolver. Generic IRBuilder calls and raw `CallInst`, `InvokeInst`, and
`CallBrInst` factories continue to use the callee and function type supplied
by their callers; they do not silently substitute another declaration.

If the typed `CreateIntrinsic` form folds, it resolves the signature before
invoking a folder. Folding is permitted only when that folder explicitly
supports the selected signature, regardless of whether it is the owner or a
variant; otherwise the builder constructs the call. Existing folders receive
only an ID, operands, and result type, so they cannot establish support for a
new signature merely from the shared ID. Teaching a folder a new signature's
semantics is the intrinsic maintainer's responsibility. The signature framework
only controls eligibility for folding through its typed construction path;
other ID-based folders and optimizations need the consumer audit described
above.

Constructors do not invoke AutoUpgrade. Callers should obtain intrinsic
declarations through intrinsic APIs rather than hard-coded mangled-name lookups
in a module.

The implementation and test plan are described in
`intrinsic-evolution-impl.md`.

## Appendix A: Specialization Considerations

The initial non-overlap rule leaves room for a future, deliberately limited
form of specialization.

### A.1 Reinterpreting Existing IR

Adding a more-specific variant may cause an unchanged concrete `FunctionType`
to select the new record in a later compiler. This is an intended capability of
specialization, not inherently a compatibility failure. Existing IR must remain
accepted, but variant selection and lowering may change between compiler
versions.

Removing a released specialization can similarly cause calls to fall back to a
generic variant. Intrinsic authors must account for both effects when evolving
a family.

### A.2 Validation Compatibility

For families using generated validation, overlapping generic and specific
variants must accept and reject every call in their shared domain identically.
TableGen should normalize their validation properties and reject the overlap
unless it can establish a static match.

For example, a specific variant cannot add `ImmArg` to an argument that is
unconstrained in the matching generic variant. Equivalent property sets with
different source ordering remain compatible.

Intrinsics that explicitly opt out of generated validation are outside this
check. Their authors are responsible for preserving validation compatibility.

### A.3 Limited Specificity Ranking

A future implementation may rank only a concrete LLVM type over a compatible
`any` descriptor:

```text
i32       > anyint
float     > anyfloat
<4 x i32> > anyvector
```

No promotions, conversions, or C++-style overload rules are implied. Generic
descriptors are not ranked against one another. Candidates that are more
specific in different positions remain incomparable and are rejected.

This structural rule keeps ranking deterministic and allows TableGen to report
conflicts without implementing general type-constraint implication.

### A.4 Signature Variant or Target Lowering

An intrinsic implementor may represent a specialization as a signature variant
or keep one generic signature and specialize in target lowering. A variant can
carry declarative properties and participate in generated signature selection;
target lowering can specialize directly from concrete operand types without
adding another IR-level record.

The appropriate choice depends on whether the distinction belongs in the
intrinsic interface or only in its implementation.
