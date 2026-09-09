# Design: Intrinsic Signature Variants and Evolution

## 1. Overview

LLVM allows intrinsics to have overloaded argument types. At a call, the
intrinsic name is mangled to reflect the concrete type of the actual argument
being passed. The parser can also accept the base name and resolves it from the
callsite signature:

```llvm
%r = call float @llvm.fmuladd(float %a, float %b, float %c)
; Canonical form: @llvm.fmuladd.f32
```

This document introduces a new mechanism that allows one intrinsic ID to have
several exact signatures. Adding a signature does not replace an existing one or
supply default values for absent arguments. Each call must match one declared
signature, and target lowering defines that signature's behavior.

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

Signature evolution also covers same-arity alternatives. If `foo(i32)` exists
and `foo(float)` is added, historical base-name input remains valid while
canonical output uses complete-signature mangling:

```llvm
declare void @llvm.foo.i32(i32)
declare void @llvm.foo.f32(float)
```

### 1.2 Goals

- Add exact, non-overlapping signatures to an existing intrinsic ID.
- Accept base and historical names for direct `call`, `invoke`, and `callbr`
  uses, then select a signature from the callsite `FunctionType`.
- Support ordinary overload types within every signature.
- Preserve every declared signature's function type, operands, and semantics.
- Use the same resolver in the parser, verifier, remangler, and C++ construction
  APIs.
- Leave variant-specific behavior to target lowering.

### 1.3 Non-goals

- Supplying default values for missing arguments or synthesizing operands.
- Changing or removing an existing signature. Such migrations remain the
  responsibility of explicit auto-upgrade logic.
- Resolving an intrinsic base name used as a first-class value or indirect
  callee.
- Defining whether a target uses TableGen patterns or custom lowering.

## 2. Signature Model

An intrinsic family consists of one owner and zero or more signature variants.
All members share one `Intrinsic::ID`; each member has a complete IIT signature
and may contain ordinary overload types.

Variants are explicit. Declaring `foo(A, B, C)` does not implicitly declare
`foo(A, B)`. An absent parameter has no IR value and no descriptor in the
selected signature.

Every concrete `FunctionType` must match exactly one member of the family.
TableGen rejects duplicate or overlapping signatures.

### 2.1 Canonical Mangling

Once an intrinsic has multiple signatures, the canonical name of every member,
including the owner, uses complete-signature mangling. The generated mangling
sequence contains the signature's mangleable result and fixed parameter types in
deterministic IIT order. The vararg marker and actual vararg operands are
excluded.

Examples:

```text
foo()           -> @llvm.foo
foo(i32)        -> @llvm.foo.i32
foo(float)      -> @llvm.foo.f32
foo(i32, ptr)   -> @llvm.foo.i32.p0
```

A void zero-argument signature keeps the base name because its type sequence is
empty.

Historical names remain accepted input aliases. A round trip may change a
callee's spelling but not its function type, operands, or semantics. Canonical
names are derived independently from complete signatures, so adding another
disjoint variant does not rename existing family members.

### 2.2 Permanent Compatibility

Every released signature remains valid indefinitely. Later revisions may add a
disjoint signature but may not remove or reinterpret an existing one.

For example:

```text
v1: foo(i32)
v2: foo(i32), foo(float)
v3: foo(i32), foo(float), foo(i32, i8)
```

The guarantee applies to textual IR, bitcode, and supported C++ construction
APIs. Canonical names may evolve when a single-signature intrinsic first becomes
a family; accepted signatures and behavior do not.

### 2.3 Initial Constraint: No Specificity Ranking

The initial implementation does not apply a "most specific" rule. Suppose the
family contains:

```tablegen
def int_foo_with_mode_and_scope :
  IntrinsicSignatureVariant<
      int_foo, [], [llvm_anyint_ty, llvm_i32_ty]>;
```

It accepts:

```llvm
call void @llvm.foo.i32.i32(i32 %mode, i32 %scope)
```

Adding this record would overlap it:

```tablegen
def int_foo_with_i32_mode_and_scope :
  IntrinsicSignatureVariant<
      int_foo, [], [llvm_i32_ty, llvm_i32_ty]>;
```

A specificity rule would reinterpret unchanged IR as the new record, possibly
changing properties or lowering. The initial implementation rejects this
overlap. This is a conservative implementation constraint that may be relaxed as
discussed in [Appendix A](#appendix-a-specialization-considerations).

## 3. TableGen Interface

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

### 3.1 Mixed Arity and Type Overloads

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

### 3.2 Same-Arity Evolution

A concrete alternative does not necessarily broadeni the owner to
`llvm_any_ty`:

```tablegen
def int_typed_foo : Intrinsic<[], [llvm_i32_ty], []>;

def int_typed_foo_f32 :
  IntrinsicSignatureVariant<
      int_typed_foo, [], [llvm_float_ty]>;
```

The resulting family contains `@llvm.typed.foo.i32` and
`@llvm.typed.foo.f32`.

### 3.3 Validation Rules

TableGen verifies that:

1. `owner` refers to an `Intrinsic`, not another variant.
2. The complete function type differs from the owner and every sibling.
3. IIT descriptors and dependent overload relationships are valid.
4. Argument properties refer to valid indexes in the complete signature.
5. Common semantic properties do not conflict within the family.
6. No two records accept the same concrete `FunctionType`.

## 4. Resolving Calls

Resolution applies uniformly to `call`, `invoke`, and `callbr` because it only
replaces the direct callee use. It does not reconstruct the instruction or
alter operands or control-flow destinations.

Existing verifier restrictions remain unchanged. In particular, `callbr`
continues to support only inline asm-goto and selected intrinsic IDs. Variants
share the owner's ID and therefore its `callbr` eligibility.

### 4.1 Varargs

Each vararg signature includes a trailing `llvm_vararg_ty` after its fixed
parameters:

```tablegen
def int_foo :
  Intrinsic<[], [llvm_i32_ty, llvm_vararg_ty], []>;

def int_foo_with_mode :
  IntrinsicSignatureVariant<
      int_foo, [], [llvm_i32_ty, llvm_i8_ty, llvm_vararg_ty]>;
```

The callsite `FunctionType`, not the actual vararg operands, selects the
variant. A call typed as `void (i32, ...)` remains the owner even if its first
vararg is `i8`; selecting `void (i32, i8, ...)` requires that explicit function
type. Vararg operands do not contribute mangling components.

### 4.2 Target Lowering

All family members share one intrinsic ID. Lowering can distinguish them by
variant ID or `FunctionType`; targets should not parse the mangled name.
TableGen patterns and custom lowering are both supported.

## 5. Interaction With AutoUpgrade

Signature variants preserve several current signatures. AutoUpgrade migrates a
legacy signature to a different current signature. They therefore have separate
roles:

| Aspect | AutoUpgrade | Signature variants |
| :--- | :--- | :--- |
| Old signature | Legacy input | Current IR |
| In-memory result | Rewritten | Signature and operands unchanged |
| Target lowering | One canonical form | Each declared signature |
| C++ construction | Not automatic | Resolved before construction |
| Generated data | One current IIT signature | Several IIT signatures under one ID |

AutoUpgrade is appropriate when `old(args)` is defined to equal
`new(args, CONSTANT)`. Signature variants are appropriate when absence is part
of the intrinsic interface and generic IR must not assert that equivalence.

For asyncmark, the zero-argument form remains a current signature whose
all-stage behavior belongs to AMDGPU lowering. It is not upgraded to a call
with an explicit `ALL` operand.

If no current variant matches, existing auto-upgrade logic may still recognize
an unrelated superseded signature.

## 6. Compatibility and Cost

Historical names are accepted aliases and canonicalized from the exact
`FunctionType`. Unknown `llvm.` names remain errors. A base name cannot be used
as a first-class value because no callsite type is available to select a
signature.

Generated metadata should mark IDs with variants so ordinary calls avoid this
path. Variants can be indexed by fixed parameter count and vararg bit before IIT
matching. Declaration insertion remains shared through the module symbol table.

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
