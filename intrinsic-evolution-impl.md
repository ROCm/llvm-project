# Implementation: Intrinsic Signature Families

This document implements the design in `intrinsic-evolution.md`.

## 1. Existing Infrastructure

The current checkout already resolves unmangled overloaded intrinsic calls from
their callsite types. Signature variants extend these components:

| Responsibility | Location | Existing behavior |
| :--- | :--- | :--- |
| Build abbreviated call types | `llvm/lib/AsmParser/LLParser.cpp`, `resolveFunctionType` | Builds a `FunctionType` from the return type and arguments. |
| Track unresolved globals | `llvm/lib/AsmParser/LLParser.cpp`, `getGlobalVal` | Records placeholders in `ForwardRefVals`. |
| Resolve intrinsic references | `llvm/lib/AsmParser/LLParser.cpp`, `LLParser::validateEndOfModule` | Matches unresolved direct calls and rewrites callee uses. |
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

- A scoped `Intrinsic::Signature` enumerator named for its unique TableGen
  record, including the owner record.
- Its IIT descriptor sequence.
- Fixed parameter count and vararg bit.
- Signature-mangling metadata for each member and overload-mangling metadata
  derived from the owner signature.
- Function and argument properties.

TableGen diagnoses malformed IIT dependencies, conflicting properties, invalid
argument indexes, and any duplicate or overlapping accepted `FunctionType` as
build failures. It also rejects distinct types with the same signature-mangled
name, or a signature-mangled name that claims a suffixed overload-mangled name
of the owner for a different type. The bare base name is exempt. Comparisons
must account for all types admitted by overloaded descriptors, rather than
only enumerated examples; if the constraints cannot establish that a family
is unambiguous, TableGen rejects it. Diagnostics identify both records and a
conflicting concrete type and name where possible. Variants are ordered
deterministically; source order must not affect resolution.

### 2.2 Signature Resolver

Add a shared resolver conceptually equivalent to:

```cpp
struct ResolvedIntrinsicSignature {
  Intrinsic::Signature SignatureID;
  SmallVector<Type *, 8> MangleTypes;
};

Expected<ResolvedIntrinsicSignature> resolveIntrinsicSignature(
    Intrinsic::ID ID, FunctionType *FTy);
```

The implementation:

1. Retrieves the ID's family members.
2. Filters by fixed parameter count and vararg bit.
3. Matches return and parameter IIT descriptors.
4. Runs deferred type relationships.
5. Requires one match and returns its signature enumerator and mangling types.

Indexing variants by parameter count and vararg bit avoids scanning unrelated
signatures. `isSignatureValid` can delegate to this API.

### 2.3 Naming and Declaration APIs

Keep the existing ID/overload-type forms of `Intrinsic::getType`, `getName`, and
`getOrInsertDeclaration` owner-directed. Add signature-aware internal accessors
for generating names and types. Keep the existing
`getOrInsertDeclaration(ID, RetTy, ArgTys)` semantics, including its behavior
for non-overloaded owners. In particular, do not use a family-wide
`isOverloaded(ID)` bit to change how this legacy overload treats a concrete
owner. Add a public typed declaration API:

```cpp
Function *Intrinsic::getOrInsertDeclaration(
    Module *M, Intrinsic::ID ID, FunctionType *FTy);
```

It resolves `FTy` to one signature and obtains its canonical declaration; C++
callers need not provide a signature enumerator. It does not invoke
AutoUpgrade. Existing single-signature behavior is unchanged.

`lookupIntrinsicID` must treat an ID with variants as mangleable even when all
types are concrete. A bare base name identifies the family and selects a member
solely from the callsite `FunctionType`. A suffixed overload-mangled name
computed from the owner signature is also accepted for that owner; other
suffixes must agree with the selected member's signature-mangled canonical
name. The resolver and verifier check the selected function type and name.

The parser and typed construction pass the resolved signature and mangling
types to naming code; they never construct suffix strings.

## 3. Parser Integration

### 3.1 Resolution Flow

```text
parse call/invoke/callbr and obtain FunctionType
        |
        v
lookup intrinsic ID from base, overload-mangled owner, or canonical name
        |
        v
resolve exact family signature
        |
        v
validate base, owner overload-mangled, or canonical spelling
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

Processing only `ForwardRefVals` is insufficient. A base name may already
resolve to a declaration while the callsite type selects another family member.
The parser must therefore queue:

- Direct uses of unresolved `llvm.` placeholders.
- Direct calls through an already-bound base name or an overload-mangled owner
  name in a family with variants.

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
  validate Name as base or canonical (or owner overload mangling if owner)
  NewFn = getOrInsertResolvedDeclaration(IID, Sig)
  CalleeUse.set(NewFn)
erase an unresolved placeholder after all its uses are handled
```

Placeholder use lists require early-increment iteration because `Use::set`
removes the current use. Replacement is per use because one base name can
select several declarations. On error, keep the placeholder alive until the
diagnostic is formed. The internal declaration helper reuses the resolver's
result; the public typed API calls the same helper after resolving `FTy`.

### 3.4 Explicit Declarations and AutoUpgrade

An explicit declaration using an overload-mangled owner name remains valid
input for its owner signature. Resolve its `FunctionType`, then canonicalize
its name when the family has multiple signatures. A declaration at the bare
base name does not pin calls to its signature: direct callsites still select
their own family members. Canonicalize an explicit base-name declaration from
its declared type before inserting a selected canonical declaration under the
same name. Canonical declarations are verified against the selected signature
and its mangled name.

AutoUpgrade is independent of signature variants. Keep its current entry
points and behavior: unresolved text-IR calls first try current signatures and
fall back to AutoUpgrade, while explicit text and bitcode declarations are
offered to AutoUpgrade. No variant rule inspects AutoUpgrade cases or imposes a
cross-path precedence order. If an upgrade rule shadows a variant, the
intrinsic author is responsible for that interaction.

## 4. C++ Construction and Consumers

### 4.1 Constructing Calls

Leave existing `IRBuilderBase::CreateIntrinsic` and
`CreateIntrinsicWithoutFolding` overloads unchanged: their ID and overload
types construct the owner and retain their current folding and failure
behavior. Do not infer a variant from `Args` or add a pre-fold validation guard
to those overloads. Overload types can include the result type but do not
specify all fixed parameters or the vararg boundary.

For a variant, request a declaration with `(Module *, ID, FunctionType *)`,
then use existing `IRBuilderBase::CreateCall`, `CreateInvoke`, or
`CreateCallBr` with that declaration. These generic builders, the public
`CallInst::Create`, `InvokeInst::Create`, and `CallBrInst::Create` factories, and
the C API paths that call them keep their supplied callee and function type;
none silently substitutes another declaration. In particular, they do not
invoke AutoUpgrade. A detached declaration cannot be canonicalized without a
module. Callers should not construct intrinsic declarations through literal
module name lookups.

The typed declaration API checks the requested `FunctionType` against the
family, including return type, fixed parameters, and vararg bit. Never infer
fixed parameters from actual vararg operands. Its error handling must follow
the non-error-returning declaration APIs' existing failure model; the shared
resolver provides diagnostic detail to APIs that can return `Error` or
`Expected`.

### 4.2 Folding

A future convenience overload `CreateIntrinsic(FunctionType *FTy, ID, Args)`
can resolve the signature before folding or constructing a call. It may call an
existing ID-based folder only if that folder explicitly supports the resolved
signature; otherwise it constructs the call without folding. Eligibility is
about the signature's complete `FunctionType`, not owner-versus-variant status.
The existing folder interface receives only ID, operands, and result type, so
support for an additional signature cannot be inferred from a shared ID.

The framework gates folding through new typed construction. Constant folders,
instruction simplifiers, and other ID-based optimizations used after
construction are maintained by the intrinsic author when adding variants;
their semantics are not supplied by signature resolution.

### 4.3 Dispatching ID-Based Consumers

Generate one scoped `Intrinsic::Signature` enum whose names come from the
owner and variant TableGen records. ID-based consumers with positional operand
assumptions can call `resolveIntrinsicSignature(ID, Call.getFunctionType())`
and switch on the returned signature; their default case reports an unhandled
signature rather than reusing owner behavior. The maintainer of each ID is
responsible for converting its affected folders, wrappers, optimizers, and
lowering paths. Generated consumers can use the resolver directly, but the
framework cannot detect assumptions in untouched hand-written ID switches.
An automatic guard for those switches is a stretch goal, not a TableGen rule.

## 5. Diagnostics

Diagnostics should use the failing callsite location rather than only the first
location stored for a shared placeholder.

| Case | Diagnostic |
| :--- | :--- |
| Non-callee base-name use | `intrinsic can only be used as callee` |
| Unknown `llvm.` name | `unknown intrinsic '...'` |
| Unsupported argument count | Report accepted counts and the actual count. |
| No matching signature | Report the callsite type and relevant candidate failures. |
| Multiple matches | Invalid generated tables: TableGen must fail the build instead. |
| Mangled types disagree with signature | Incorrect intrinsic mangling. |
| IIT type relationship fails | Preserve the descriptor-specific validator text. |
| Vararg bit differs | Preserve the existing vararg diagnostic. |
| Unhandled signature in a converted consumer | Identify the intrinsic ID and signature record. |

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
| Mangling | Owner and variants use signature-mangled names; overload-mangled owner names remain accepted input. |
| Reuse | Repeated selections reuse one declaration. |
| Name collision | Distinct signatures with the same signature-mangled name fail TableGen. |
| Owner name collision | A variant claiming a suffixed overload-mangled owner name fails TableGen. |
| Base-name exception | A `void(i32)` owner plus a `void()` variant: both base-name calls select by type, even though the variant's canonical name is the base name. |

### 6.2 Parser and Construction Coverage

- Select every family member from one base name in a single module, including
  when that name is already bound to an explicit declaration.
- Exercise `call`, `invoke`, and a legally supported intrinsic `callbr`.
- Verify that operands, bundles, attributes, and destinations are unchanged.
- Test explicit canonical names, unknown names, non-callee uses, bad counts,
  type mismatches, and incorrect suffixes.
- Confirm that an overload-mangled owner vararg call is not reinterpreted from
  its actual vararg operand types.
- Verify that existing `CreateIntrinsic` overloads still construct the owner,
  including calls with overload types and varargs, without new failure paths.
- Check that the legacy `(ID, RetTy, ArgTys)` declaration overload still ignores
  those types for a concrete owner, even if a variant contains overload types.
- Select every family member using the new `(Module *, ID, FunctionType *)`
  declaration API; construct `call`, `invoke`, and `callbr` through existing
  generic C++ factories, with explicit return and vararg types.
- Verify that typed construction never folds with an inapplicable ID-only
  folder, and that an explicitly supported signature can fold if a typed
  convenience overload is implemented.
- Test converted ID consumers with owner and variant calls, including a
  same-arity type change that would not trip `getArgOperand()` bounds checks.
- Read bitcode containing an overload-mangled owner name and emit the current
  canonical name without changing the function type or operands, absent an
  independently applicable AutoUpgrade rule.

Use `fmuladd` as a disjoint same-arity example: its existing
`float (float, float, float)` call retains `@llvm.fmuladd.f32` as accepted
input, while a new `float (float, i32, float)` call receives the
signature-mangled name `@llvm.fmuladd.f32.f32.i32.f32`. A proposed `float ()`
variant instead fails TableGen because it would claim `@llvm.fmuladd.f32`.
Keep an `objectsize` legacy two- or three-argument example as an AutoUpgrade
test, not as an assumed variant compatibility guarantee. A hypothetical
three-argument variant matching its legacy type can bypass AutoUpgrade for an
undeclared text-IR call, whereas an explicitly declared text or bitcode form
can still be upgraded. Tests document this existing path-dependent behavior;
the feature does not change the upgrade rule.

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
- [ ] Fail TableGen builds for invalid properties, IIT dependencies,
      overlapping types, canonical-name collisions, and collisions with
      suffixed overload-mangled owner names.
- [ ] Emit scoped signature enumerators for owner and variant records,
      descriptor tables, and mangling metadata.
- [ ] Add exact signature resolution and variant-aware core APIs.
- [ ] Extend name lookup, remangling, and verifier checks.
- [ ] Resolve unresolved and already-bound base-name calls and
      overload-mangled owner names in `LLParser`.
- [ ] Add the typed declaration API without changing existing owner-directed
      builders or generic call factories.
- [ ] Audit ID-based consumers for each intrinsic gaining variants; dispatch
      on the resolved signature or report an unhandled signature.
- [ ] Gate any new typed builder folding on explicit signature eligibility.
- [ ] Leave AutoUpgrade entry points and rules unchanged.
- [ ] Add generic TableGen, parser, bitcode, verifier, and C++ API tests.
- [ ] Convert asyncmark and add target lowering tests.
