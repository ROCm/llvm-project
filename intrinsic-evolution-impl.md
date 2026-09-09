# Implementation: Intrinsic Signature Variants and Evolution

This document implements the design in `intrinsic-evolution.md`.

## 1. Existing Infrastructure

The current checkout already resolves unmangled overloaded intrinsic calls from
their callsite types. Signature variants extend these components:

| Responsibility | Location | Existing behavior |
| :--- | :--- | :--- |
| Build abbreviated call types | `llvm/lib/AsmParser/LLParser.cpp`, `resolveFunctionType` | Builds a `FunctionType` from the return type and arguments. |
| Track unresolved globals | `llvm/lib/AsmParser/LLParser.cpp`, `getGlobalVal` | Records placeholders in `ForwardRefVals`. |
| Resolve intrinsic references | `llvm/lib/AsmParser/LLParser.cpp`, `LLParser::validate` | Matches unresolved direct calls and rewrites callee uses. |
| Look up intrinsic IDs | `llvm/lib/IR/Intrinsics.cpp`, `lookupIntrinsicID` | Maps base or mangled names to an ID. |
| Match signatures | `llvm/lib/IR/Intrinsics.cpp`, `isSignatureValid` | Interprets one generated IIT signature and infers overload types. |
| Create declarations | `llvm/lib/IR/Intrinsics.cpp`, `getOrInsertDeclaration` | Constructs the canonical name and declaration. |
| Upgrade superseded forms | `llvm/lib/IR/AutoUpgrade.cpp` | Rewrites signatures that are no longer current. |

The existing integer default-argument upgrade is unrelated: signature variants
never add operands.

## 2. Generated Data and Core APIs

### 2.1 TableGen Collection

`CodeGenIntrinsics` should collect every `IntrinsicSignatureVariant` under its
owner before emitting intrinsic tables. Only the owner creates an
`Intrinsic::ID`; named variant records remain available to diagnostics and
target TableGen consumers.

For each family member, emit:

- A stable variant identifier.
- Its IIT descriptor sequence.
- Fixed parameter count and vararg bit.
- Complete-signature mangling metadata.
- Function and argument properties.

TableGen rejects malformed IIT dependencies, conflicting properties, duplicate
function types, and overlapping type constraints. Variants are ordered
deterministically; source order must not affect resolution.

### 2.2 Signature Resolver

Add a shared resolver conceptually equivalent to:

```cpp
struct IntrinsicSignature {
  FunctionType *FunctionTy;
  SmallVector<Type *, 8> MangleTypes;
  unsigned VariantID;
};

std::optional<IntrinsicSignature> resolveIntrinsicSignature(
    Intrinsic::ID ID, FunctionType *FTy,
    raw_ostream &ErrorOS = nulls());
```

The implementation:

1. Retrieves the ID's family members.
2. Filters by fixed parameter count and vararg bit.
3. Matches return and parameter IIT descriptors.
4. Runs deferred type relationships.
5. Requires one match and returns its variant ID and mangling types.

Indexing variants by parameter count and vararg bit avoids scanning unrelated
signatures. `isSignatureValid` can delegate to this API.

### 2.3 Naming and Declaration APIs

Extend `Intrinsic::getType`, `getName`, and `getOrInsertDeclaration` to accept a
variant ID and `MangleTypes`. Existing single-signature intrinsic behavior stays
unchanged.

`lookupIntrinsicID` must treat an ID with variants as mangleable even when all
types are concrete. It accepts historical aliases and current canonical names;
the resolver and verifier confirm that the suffix agrees with the selected
signature.

The parser and builders pass type vectors to `getName`; they never construct
suffix strings.

## 3. Parser Integration

### 3.1 Resolution Flow

```text
parse call/invoke/callbr and obtain FunctionType
        |
        v
lookup intrinsic ID from historical or canonical name
        |
        v
resolve exact signature variant
        |
        v
get or insert canonical declaration
        |
        v
replace only the callee use
```

The instruction's function type, operands, bundles, attributes, and control-flow
destinations remain unchanged.

### 3.2 Candidate Calls

Processing only `ForwardRefVals` is insufficient. A historical name may already
resolve to a declaration while the callsite type selects another family member.
The parser must therefore queue:

- Direct uses of unresolved `llvm.` placeholders.
- Direct calls through historical aliases owned by a family with variants.

Every candidate must be a direct `CallBase` callee use. A base name used as a
function value remains invalid.

No intrinsic-specific grammar is needed. `parseCall`, `parseInvoke`, and
`parseCallBr` already provide the complete callsite `FunctionType` through
`resolveFunctionType` and `CalleeID.FTy`.

### 3.3 Finalization and Mutation Safety

Conceptually:

```text
for each candidate direct intrinsic call:
  IID = lookupIntrinsicID(Name)
  Sig = resolveIntrinsicSignature(IID, CallBase.function_type)
  NewFn = getOrInsertDeclaration(IID, Sig.VariantID, Sig.MangleTypes)
  CalleeUse.set(NewFn)
erase an unresolved placeholder after all its uses are handled
```

Placeholder use lists require early-increment iteration because `Use::set`
removes the current use. Replacement is per use because one historical name can
select several declarations. On error, keep the placeholder alive until the
diagnostic is formed.

### 3.4 Explicit Declarations and AutoUpgrade

An explicit historical declaration remains valid input. Resolve its exact
`FunctionType`, then canonicalize its name when the family has multiple
signatures. Canonical declarations are verified against the variant encoded in
their suffix.

Current variants are resolved before AutoUpgrade. If none matches, existing
upgrade logic may recognize an unrelated superseded signature. Valid family
members never enter `UpgradeIntrinsicCall`.

## 4. C++ Construction

Variant selection must happen before allocating a call and before intrinsic
folding. Use one helper for:

- `IRBuilderBase::CreateIntrinsic` and `CreateIntrinsicWithoutFolding`.
- `IRBuilderBase::CreateCall`, `CreateInvoke`, and `CreateCallBr`.
- Public `CallInst::Create`, `InvokeInst::Create`, and `CallBrInst::Create`
  factories.

For a direct intrinsic callee, the helper:

1. Obtains the requested `FunctionType`.
2. Resolves one family member.
3. Gets its canonical declaration from the module.
4. Performs folding with the original operands.
5. Creates the requested `CallBase` subclass without changing operands or
   destinations.

Ordinary and indirect calls are unaffected. A detached historical declaration
cannot be canonicalized because declaration insertion requires a module.

For varargs, use the supplied `FunctionType`; never infer fixed parameters from
actual vararg operands. A caller selecting another vararg variant must provide
its function type or canonical declaration.

Low-level factories retain their current failure model. APIs returning `Error`
or `Expected` should propagate resolver diagnostics.

## 5. Diagnostics

Diagnostics should use the failing callsite location rather than only the first
location stored for a shared placeholder.

| Case | Diagnostic |
| :--- | :--- |
| Non-callee base-name use | `intrinsic can only be used as callee` |
| Unknown `llvm.` name | `unknown intrinsic '...'` |
| Unsupported argument count | Report accepted counts and the actual count. |
| No matching variant | Report the callsite type and relevant candidate failures. |
| Multiple matches | Generated-table error; TableGen should have rejected the family. |
| Mangled types disagree with signature | Incorrect intrinsic mangling. |
| IIT type relationship fails | Preserve the descriptor-specific validator text. |
| Vararg bit differs | Preserve the existing vararg diagnostic. |

Return errors before erasing placeholders or partially changing unrelated uses.

## 6. Testing

### 6.1 TableGen and IR Coverage

| Area | Required coverage |
| :--- | :--- |
| Same arity | Concrete `foo(i32)` owner plus `foo(float)` variant. |
| Different arity | Zero-, one-, and two-argument family members. |
| Ordinary overloads | `llvm_anyint_ty` within a variant. |
| Mixed forms | `foo`, `foo(anyint)`, and `foo(anyint, i32)`. |
| Disjoint constraints | `(anyint, i32)` and `(i32, i64)` both accepted. |
| Overlap rejection | `(anyint, i32)` and `(i32, i32)` rejected. |
| Type coverage | Scalar, pointer/address space, fixed/scalable vector, aggregate, token, metadata, and target extension types. |
| Mangling | Owner and variants use complete-signature names; historical aliases remain input-compatible. |
| Reuse | Repeated selections reuse one declaration. |

### 6.2 Parser and Construction Coverage

- Select every family member from one historical name in a single module.
- Exercise `call`, `invoke`, and a legally supported intrinsic `callbr`.
- Verify that operands, bundles, attributes, and destinations are unchanged.
- Test explicit canonical names, unknown names, non-callee uses, bad counts,
  type mismatches, and incorrect suffixes.
- Confirm that a historical vararg call is not reinterpreted from its actual
  vararg operand types.
- Cover `CreateIntrinsic`, every IRBuilder call form, direct instruction
  factories, folding, and the C API paths that use those factories.
- Read historical bitcode and emit the current canonical name without changing
  the function type or operands.

Asyncmark is the target integration test: zero-argument forms retain inherent
all-stage lowering, and explicit-stage variants consume their operands.

### 6.3 Focused Commands

```bash
llvm-lit llvm/test/Assembler/implicit-intrinsic-declaration.ll \
  llvm/test/Assembler/implicit-intrinsic-declaration-invalid*.ll
```

Add focused TableGen/unit tests for variant generation and C++ construction,
then run existing intrinsic, assembler, verifier, auto-upgrade, and affected
target tests.

## 7. Implementation Checklist

- [ ] Add `IntrinsicSignatureVariant` records and group them by owner.
- [ ] Validate properties, IIT dependencies, duplicates, and overlap.
- [ ] Emit stable variant IDs, descriptor tables, and mangling metadata.
- [ ] Add exact signature resolution and variant-aware core APIs.
- [ ] Extend name lookup, remangling, and verifier checks.
- [ ] Resolve unresolved and already-bound historical calls in `LLParser`.
- [ ] Integrate selection into all IRBuilder and direct `CallBase` factories.
- [ ] Preserve existing AutoUpgrade behavior for non-variant migrations.
- [ ] Add generic TableGen, parser, bitcode, verifier, and C++ API tests.
- [ ] Convert asyncmark and add target lowering tests.
