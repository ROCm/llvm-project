# DIExpression Rework RFC (WIP DRAFT)

This RFC advocates reworking `DIExpression` as a graph-based abstraction of
typed nodes representing operations in an algebra of locations.

:::{warning}
This RFC is human-authored, but LLMs were used during research and for
identifying inconsistencies. Any verbatim LLM output is identified as such in a
contextually-appropriate way.
:::

:::{admonition} TODO
One thing I didn't identify earlier as a potential issue is that many (most?)
edits to the expression occur around arguments, which typically occur earlier
rather than later. While expressions remain small this likely doesn't pose a
significant issue, and the goal of always canonicalizing/optimizing is in-part
to keep the expressions small at all times, but there is some risk here.

This may also limit sharing, and/or make edits more cumbersome. I think this is
roughly analogous to what we have today, where we have to essentially
reconstruct the whole expression anyway for most of these edits, but currently
it is mitigated by spatial locality (in other words, my proposal may introduce
significantly more page faults).

The more compact, non-Metadata based approach inspired by Stephen's suggestions
in the Monday meeting might help, and should bring the aggregate performance
characteristics a bit closer to the current DIExpression editing scheme.
:::

## Goals

- Support key changes in DWARF6
- Support efficient and declarative canonicalization/optimization of
  expressions during compilation
- Either directly resolve, or provide a clear path to enable resolving, several
  issues/bugs/limitations of LLVM debug-info (see the Motiviating Issues section)

## Motivating Issues

:::{admonition} TODO
Expand on why these are relevant, why we expect to resolve them via the
proposed changes, etc.
:::

- [51841 – compile-time explosion at -O3](https://bugs.llvm.org/show_bug.cgi?id=51841)
  - Avoid the exponential growth of linear-scanned expressions (due to e.g. unrolling loops through) fine-grained, mandatory canonicalization
- [[DebugInfo][SimplifyCFG] Salvaged `ge`/`gt`/`le`/`lt` expressions zero-extend signed integer operands · Issue #219412 · llvm/llvm-project](https://github.com/llvm/llvm-project/issues/219412)
- [Assertion `Ty->isPtrOrPtrVectorTy() && "This should only be called with a pointer or pointer vector type"' failed. · Issue #134523 · llvm/llvm-project](https://github.com/llvm/llvm-project/issues/134523)
- [[llvm][DebugInfo] Emit 0/1 for constant boolean values by laxmansole · Pull Request #151225 · llvm/llvm-project](https://github.com/llvm/llvm-project/pull/151225)
- [[Clang][CodeGen][X86] don't coerce int128 into `{i64,i64}` for SysV-like ABIs by T0b1-iOS · Pull Request #135230 · llvm/llvm-project](https://github.com/llvm/llvm-project/pull/135230)
- [[clang] clang generates incompatible types on DWARF stack · Issue #102294 · llvm/llvm-project](https://github.com/llvm/llvm-project/issues/102294)
- [Errors in generating DWARF debug information : DW_TAG_variable has an invalid location · Issue #102756 · llvm/llvm-project](https://github.com/llvm/llvm-project/issues/102756)
- [[DebugInfo][InstCombine] Canonicalizing a shift-based truncation check omits an intermediate truncation from the debug expression · Issue #218684 · llvm/llvm-project](https://github.com/llvm/llvm-project/issues/218684)
- [InstCombine drops debug value info towards end of function · Issue #61153 · llvm/llvm-project](https://github.com/llvm/llvm-project/issues/61153)
- [Incorrect of debug information of a structure variable compiled with "clang -g -O3" · Issue #91542 · llvm/llvm-project](https://github.com/llvm/llvm-project/issues/91542)
- [Debug location for char parameter emitted with DW_OP_piece 4, causes errors in GDB and Clang · Issue #75206 · llvm/llvm-project](https://github.com/llvm/llvm-project/issues/75206)

## Assumptions

- The contribution of `DIExpression` to RSS is negligible, and even an ~10x
  increase would be in the noise on `compile-time-tracker`.
- Compile time regression will be minimal in the context of the whole
  compilation.
  - A very rough approximation, based on an assumption of a pointer-chasing
    graph representation, was attempted in [this
    commit](https://github.com/llvm/llvm-project/commit/3d52ae13d0f6b351beffd7cf54dfb8ca8e0aa200),
    which does redundant work in that it retains the vector in addition to the
    graph.
  - The
    [compile-time-tracker](https://llvm-compile-time-tracker.com/compare.php?from=03801ff8b967997b74e6631723f128d6c65a9e77&to=3d52ae13d0f6b351beffd7cf54dfb8ca8e0aa200&stat=instructions)
    shows `instructions:u` grew by `+0.15%` in the worst case.
  - This doesn't incorporate page faults (TODO)

## Definitions

### Location

A "location" is a tuple `L = (S, N, T)` of a reference to a "location storage"
`S`, an unsigned zero-indexed bit offset `N`, and a type `T`. It symbolically
locates the runtime bits of a program object and describes that object's
runtime representation.

:::{note}
This is similar to the DWARF6 definition of a location, with the addition of a
type directly as part of the location, and without direct support for "multi"
locations defined through e.g. location lists.
:::

### Location Storage

A "location storage" is a linear stream of bits that can hold values. Each
location storage has a size in bits and can be accessed using a zero-based bit
offset. The ordering of bits within a location storage uses the bit numbering
and direction conventions that are appropriate to the current language on the
target architecture.

:::{note}
This is the definition of location storage from
[AMDGPUDwarfExtensionsForHeterogeneousDebugging](./AMDGPUDwarfExtensionsForHeterogeneousDebugging.md)
:::

There are several kinds of location storage:

- "Memory location storage" exists for each target architectural address space,
  with a size equal to the size of that address space and bits which correspond
  to the bits of its addressable units, ordered first by ascending address, and
  then in the target-specific conventional order for the bits within an
  addressable unit.
- "Register location storage" for each MIR register, with the size equal to the
  corresponding target register and bits which correspond to the bits of the
  register.
- "Undefined location storage" represents unavailable/undefined program
  objects.
- "Implicit location storage" represents literal, fixed values that can only be
  read. There is one such storage for each typed literal value.
- "Composite location storage" represents virtual storage which delegates bits
  to other component location storages.
- "SSA location storage" exists for each SSA value in LLVM IR, with the size
  and bits of the SSA value. It is unspecified whether the storage can be
  written.

## Representation-agnostic Semantics

All debug expressions are represented by nodes of a directed acyclic graph
(DAG) with labelled edges.

Nodes in the DAG encode terms in an algebra of locations. Each term computes a
single location, and has a kind which statically defines its set of input
locations, and any associated literals.

:::{note}
These literals can include, e.g. types which allow for static type-checking of
expressions, as well as untyped constants like indexes.
:::

Edges in the DAG represent dependencies between operations. Edge labels
associate an outgoing edge with the term input it satisfies.

Expressions are evaluated in a context that includes the same elements as
{ref}`amdgpu-dwarf-expression-evaluation-context`, except:

- None of *current result kind*, *current object*, or *initial stack*, as these
  are not meaningful.
- Additionally, zero or more typed *arguments* are included, ordered by a
  zero-based index.

Terms compute their output as a function of only their inputs, literals,
evaluation context, and the contents of referenced location storage.

Within a single evaluation, the context and location storage contents are
fixed, which implies structurally equal terms have equal outputs.

:::{note}
This property of nodes applies only *within a single evaluation*, so a
shared term does not imply shared results across multiple evaluations.
:::

### Nodes (aka Operations, Terms)

:::{admonition} TODO
Do we need to encode integer signedness as another dimension on top of the LLVM
type system? Or do we want to reuse DIType information instead of LLVM types (I
think Adrian brought this up in one discussion)? Haven't thought this aspect
through carefully enough to say just yet.

I think we want to keep everything in the LLVM world as long as possible, which
implies we go with fixed-width 2's complement integer types with operation
variants where signedness is significant. Then, the problem becomes one of
legalizing/lowering to DWARF semantics from that.

An LLM identified some interesting cases to ponder:

```
LLVM: sub i8 0, 0x80  -> 0x80
DWARF: signed 8-bit 0x80; DW_OP_neg -> undefined
```

```
ZExt(i32, Add(i8, 255, 1))
This must result in (i32 0), which it won't if the add is evaluated in a wider
type (e.g. the generic type, typically)
```
:::

Informally, the `<...>` shaped "holes" in the following template are filled
in to define the various kinds of supported operations:

```
DIOp<Kind>(<ArgMetaVar>: <SyntaxKind>(<SyntaxParams>), ...) -> (<OutStorage>, <OutOffset>, <OutType>)
```

The `SyntaxKind` gives the "shape" of the syntax for the argument,
parameterized over `SyntaxParams`, and is one of:

- `loc(<Storage>, <Offset>, <Type>)` for a location (i.e. a reference to another node)
- `type(<Type>)` for a type literal identifying `Type`, or a subclass of `Type`
- `value(<Type>)` for a value literal of type `Type`, or a subclass of `Type`

All operations have a fixed, ordered, comma-separated list of arguments.

Reading from left-to-right, meta-syntactic variables (identifiers with only
capital letters and the `'` character) are bound on their first appearance.
Instances of already-bound variables act as constraints, requiring equality.
The placeholder `_` is used to indicate no binding and no constraint on an
element (e.g. "don't care" parameters).

The available types place static and dynamic constraints on the argument, and
are one of:

- `any` for any supported LLVM type, which are:
  - `ptr addrspace(N)` for the {ref}`t_pointer`
  - `int` for any {ref}`t_integer`
- Any previously bound type literal

:::{note}
For example, after `T: type(int)` is bound the arg `X: loc(T)` would be
constrained to refer to nodes with exactly the dynamic type bound to `T`, which
is statically constrained to be an integer type.
:::

The following definitions are also used:

- `typeof(L: loc(_, _, T)) = T`: computes the type of location `L`.
- `bitsizeof(X)`: computes the size in bits of bound type `X`.
- `sizeof(X) = bitsizeof(X) * 8`
- `ptr2offset(L)`: computes the offset corresponding to the pointer value in
  location `L`, which must be a `ptr addrspace(N)` type. This is generally
  (always?) `read(P) * 8`
- `read(L: loc(_, _, T))`: computes the value of type `T` obtained by
  retrieving `bitsizeof(T)` bits from location `L`. The result is undefined if
  any bit of the value retrieved is from undefined location storage or the
  offset of any bit exceeds the size of the location storage specified by
  location `L`.

:::{admonition} TODO
An LLM identified an example where we must decide whether/how rewrites must
preserve properties like the "undefined" case for `read` above:

Should `Sub(_, X, X) -> 0` be a legal rewrite? If `read(X)` may be undefined,
this "erases" that possibility. Maybe we need to separate "undefined due to
source-language/LLVM semantics" from "by-construction error in expression
form".
:::

#### Arg

```txt
DIOpArg(T: type(_), N: value(int)) -> (_, _, T)
```

Computes the location of the `N`{sup}`th` input `I` to the expression.

It is an error if `T != typeof(I)` (FIXME: this hinges on whether we can use
the LLVM type system verbatim).

#### Constant

```txt
DIOpConstant(T: type(_), V: value(T)) -> (S, 0, T)
```

`S` is an implicit location storage of size `bitsizeof(T)` and value `V`.

#### Read

```txt
DIOpRead(T: type(_), L: loc(_, _, T)) -> (S, 0, T)
```

`S` is an implicit location storage of size `bitsizeof(T)` and value `read(L)`.

#### Deref

```txt
DIOpDeref(T: type(_), P: loc(_, _, ptr addrspace(N))) -> (S, ptr2offset(P), T)
```

`S` is the memory location storage for address space `N`.

#### Offset

```txt
DIOpOffset(T: type(_), L: loc(S, N, T), M: loc(_, _, int)) -> (S, N + read(M), T)
```

:::{admonition} TODO
Define semantics of the sum `N + read(M)` for overflow, underflow, ... ?
:::

#### ZExt

```txt
DIOpZExt(T: type(int), L: loc(_, _, int)) -> (S, 0, T)
```

`S'` is an implicit location storage of size `bitsizeof(T)` and the value
`read(L)` with the high order bits filled with zero bits until it reaches
`bitsizeof(T)`.

It is an error if `bitsizeof(T) <= sizeof(typeof(L))`.

#### SExt

```txt
DIOpSExt(T: type(int), L: loc(_, _, int)) -> (S, 0, T)
```

`S'` is an implicit location storage of size `bitsizeof(T)` and the value
`read(L)` with the high order bits filled with the sign bit of `read(L)` until
it reaches `bitsizeof(T)`.

It is an error if `bitsizeof(T) <= sizeof(typeof(L))`.

#### Add

```txt
DIOpAdd(T: type(int), X: loc(_, _, T), Y: loc(_, _, T)) -> (S, 0, T)
```

`S` is an implicit location storage of size `bitsizeof(T)` and value `read(X) + read(Y)`.

#### Sub

```txt
DIOpSub(T: type(int), X: loc(_, _, T), Y: loc(_, _, T)) -> (S, 0, T)
```

`S` is an implicit location storage of size `bitsizeof(T)` and value `read(X) - read(Y)`.

#### TODO

- Similar to `Add` and `Sub` we would have a full complement of such operations
  that are 1:1 with LLVM.
  - How we reconcile the lack of signed/unsigned distinction in the LLVM
    type-system might imply the full set we implement here. For example, do we
    need `nsw`/`nuw`/... flags and signed/unsigned variants of operations?
- We had a `DIOpTypeObject` in the AMDGPU extensions definition, which was
  intended to act as a sort of `DIOpArg` but for cases where one is computing a
  "type property" like `dataLocation` or `rank` on an array. Can we just define
  the object in that case to be the first argument, so one just uses
  `DIOpArg(_, 0)` in place of `DIOpTypeObject(T)`?
- Similarly, for global variables can `DIOpArg(_, 0)` refer to the global?

### Equivalence Relations

:::{admonition} TODO
- Need to factor out some of the definitions and notations so this shares them
  with the Nodes section.
- Composites should transparently "designate" to underlying storage(?)
:::

Two bits of location storage are equivalent, denoted `B1 ~= B2`, iff:

- They designate the same bit of the same underlying storage; 
- Or, they are both bits of implicit storage with the same value.

Two locations `L1 = (S1, N1, T1)` and `L2 = (S2, N2, T2)` are equivalent,
denoted `L1 ~= L2`, iff:

- `T = T1 = T2`
- `foreach I in 0, 1, ..., bitsizeof(T)-1:`
  - `S1[N1+I] ~= S2[N2+I]`

### Rewrites

Generic peephole rewrites must preserve equivalence of the root node's output
location in every evaluation context.

:::{note}
As an example of a rewrite which does not preserve equivalence:

```txt
Add(T, X, Constant(T, 0)) -> X
```

In the LHS, the outer `Add` yields implicit location storage, but `X` refers to
location storage whose kind is unknown. Were this permitted by our equivalence
relation it would mean e.g. a debug agent would be able to mutate the located
storage where it could not before.

A corrected form of this rewrite which preserves equivalence might be:

```txt
Add(T, X, Constant(T, 0)) -> Read(T, X)
```
:::

In general, rewrites which do not maintain equivalence must be proven correct
on the basis of other analysis. This is typically in the form of:

- LLVM IR instruction semantics, in the case of `salvageDebugInfo`;
- Or, information known to a specific pass or transformation of LLVM IR.

:::{note}
As an example of a legal rewrite which does not maintain equivalence under our
definition: an IR transformation which replaces an SSA value with a constant
may change a non-implicit location into an implicit one.

In this case, the change in the DAG reflects a transform in the program, rather
than asserting location equivalence.
:::

## IR Representation Options

In any case, `#dbg_declare` and `#dbg_value` would now have no influence on the
level of indirection: the expression always unambiguously describes
indirections.

The only remaining difference is that `#dbg_declare` still applies to the
whole-function, and must be the only marker referring to the source variable
within the function.

:::{note}
The final `DILocation` argument to the debug markers is elided below for
clarity.

TODO: Remaining debug markers (i.e. `#dbg_declare_value`, `#dbg_assign`)
:::

### Metadata

:::{note}
In writing this, I think one major issue may be in writing tests against the
results? This is already an annoyance with Metadata, so maybe not critical.

One can imagine an equivalent, separate implementation which can throw away
much of the Metadata/MDNode/DINode overhead from RAUW and unused header bits.
It could also mean more easily printing the expression "inline" as
!DIExpression is today, without further complicating the Metadata rules.
:::

#### Example: Formerly-empty declare

```txt
%x = alloca i32, addrspace(5)
#dbg_declare(ptr addrspace(5) %x, !"var", !DIExpression())
```

becomes:

```txt
%x = alloca i32, addrspace(5)
#dbg_declare(ptr addrspace(5) %x, !"var", !0)
!0 = !DIOpDeref(i32, !1)
!1 = !DIOpArg(ptr addrspace(5), 0)
```

which in DWARF6 might lower to:

```
DW_TAG_variable
  DW_AT_name      ("var")
  DW_AT_type      (unsigned 32-bit integer)
  DW_AT_location  (DW_OP_fbreg -4)
```

#### Example: Formerly-empty value

```txt
%x = alloca i32, addrspace(5)
%y = load i32, ptr addrspace(5) %x
#dbg_value(i32 %y, !"var", !DIExpression())
```

becomes:

```txt
%x = alloca i32, addrspace(5)
%y = load i32, %x
#dbg_value(i32 %y, !"var", !0)
!0 = !DIOpArg(i32, 0)
```

which in DWARF6 might lower to:

```
DW_TAG_variable
  DW_AT_name      ("var")
  DW_AT_type      (unsigned 32-bit integer)
  DW_AT_location  (DW_OP_regx EAX)
```

#### Example: Multi-arg

```txt
#dbg_value(!DIArgList(i8 %x, i32 %y), !"var",
  !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_lit0, DW_OP_plus,
                DW_OP_mul, DW_OP_LLVM_arg, 1, DW_OP_plus))
```

autoupgrades to something like:

```txt
...
!0 = !DIOpPlus(generic, !1, !4)
!1 = !DIOpMul(generic, !2, !3)
!2 = !DIOpArg(generic, 0)
!3 = !DIOpPlus(generic, !4, !5)
!4 = !DIOpArg(generic, 1)
!5 = !DIOpConstant(generic, 0)
```

but when generated via optimization, we would instead retain concrete types
and more fine-grained conversions, maybe in this case something closer to:

```txt
...
!0 = !DIOpPlus(i16, !1, !5)
!1 = !DIOpMul(i16, !2, !4)
!2 = !DIOpZExt(i16, !3)
!3 = !DIOpArg(i8, 0)
!4 = !DIOpPlus(i16, !5, !7)
!5 = !DIOpTrunc(i16, !6)
!6 = !DIOpArg(i32, 1)
!7 = !DIOpConstant(i16, 0)
```

:::{admonition} TODO
The :generic: type is me hand-waving at the issue of retaining the semantics
from the old DIExpression, which mirror DWARF's concept of a generic type. It
may be enough to pick the concrete integer type with the same width as the
target's generic type in DWARF, but this would mean that relying on the type
information for these converted expressions could be misleading/unsound in some
situations (i.e. those where we have bugs today), so having a distinguished
type to reflect that would give code the opportunity to segregate the
"compatibility" path from the rest, and manage the interaction with typed
nodes.
:::

### Index-connected vector of nodes

:::{admonition} TODO
This is sort of shoehorned in alongside the original Metadata representation
sketch. I think the Metadata version can go in "alternatives" and this should
be the preferred approach.
:::

In this version, the DAG is local to each expression, and implemented in a
vector of sum-type nodes which refer to one another with 0-based indexes into
the vector.

(Without loss of generality, I'll elide types here, but they'd just be part of
the nodes, inline, or possibly indexed in another collocated table).

For example, we could encode the infix expression `$1 + ($2 * $3)` as (where
`$N` is expression argument `N`):

```
  ______________________
 /     ____________     \
/     /            \     \
0     1     2     3 \   4 \
|$1   |$2   |$3   |* 1 2|+ 0 3|
            \_____|___/     /
                  |________/
```

This would be the "at rest" representation, which is optimized/normalized with
elements de-duplicated (local to the expression), and the whole thing getting
hashed to yield a `DIExpression *`-like handle to "the whole expression, just
as a DAG encoded in a vector". The root is always the last node, except for the
case of the empty expression which has no root.

Then for edits, we can walk the DAG in post-order, applying rewrites eagerly
into a mutable, append-only buffer of immutable nodes.

Let's name the two vectors so we can unambiguously refer to them:

- old-expr: the "at rest" expression, uniqued "as a whole" and used in place of
  `DIExpression *` in the rest of the codebase. In a normalized form without
  "dead" nodes, and with deterministic node order based on the DAG it is
  encoding.
- new-expr: the append-only buffer of nodes, always in relation to a existing
  (possibly empty) old-expr, which is possibly denormalized and may have "dead"
  nodes as an artifact of editing and rewriting.

This implies there are then two namespaces of indexes:

- old-index, which identifies an element of old-expr
- new-index, which identifies an element of new-expr

We always maintain the invariant that nodes in old-expr only reference earlier
nodes in old-expr (this implies they do not contain new-indexes), and that
nodes in new-expr only reference earlier nodes in new-expr (this implies they
do not contain old-indexes). Since new nodes are immutable, this also ensures
the result is acyclic.

To bridge the two namespaces, we have a one-way mapping of
old-index-to-new-index which records the new-index of the (possibly identical,
modulo re-indexing) replacement node for each old-index. This starts empty, and
is populated as we select replacements for each old node. We maintain the
invariant that assignment to this map occurs exactly once for each node in
old-expr.

We also maintain a uniqueing table, as a hashmap from a new-expr node to its
new-index. We hash the new node structurally by its kind, types, literals, and
ordered new-index operands.

To add a new node:

- All of its operands must already be new-indexes
- Look it up in the uniqueing table:
  - If it is present, return the existing new-index
  - Otherwise, append the node to the new-expr buffer, add it to the uniqueing
    table, and return its new-index

With our invariants, we can actually implement the post-order walk seeing
every old node exactly once by just doing a linear scan of old-expr. So
our overall rewrite algorithm is roughly:

- For each old node in ascending index order:
  - Translate its operands using old-index-to-new-index, and add it to the
    new-expr, yielding a "base" new-index.
  - Try to match the rewrite being applied to the old node (note: not "base",
    as other rewrites may have resulted in it no longer matching).
    - For example, when salvaging a `Plus` operation the code in `Local.cpp`
      might rewrite `Arg(0) -> Plus(Arg(0), Arg(1))`.
    - When this matches, it receives both the old-index for the root, as well
      as the new-index "base", and must construct its RHS into new-expr. Since
      the rewrite is matched against the old-expr it is OK if it is
      non-contracting or would otherwise match its own output.
    - If the rewrite applies, it yields a new-index "rewritten";
    - Otherwise, "base" is used as "rewritten".
  - Run peepholes over "rewritten", adding candidates and remembering the root
    of the "cheapest" replacement. The exact details of e.g. ensuring
    termination, confluence, and whether/how we bound the work here are left TBD.
      - If no peepholes apply, "rewritten" is used as "cheapest".
  - Update old-index-to-new for the old node to point to "cheapest".

:::{note}
Intermediate work during selection may add nodes which are never selected, but
we don't garbage-collect them: these are still fully-fledged new nodes, which
go through uniqueing and have the same invariants as any other node. They may
subsequently be reused by other rewrites. All of the "garbage" is collected
at once after the whole process, when the "at-rest" representation is built.
:::

:::{admonition} TODO
I realized after writing that the empty old-expr case implies there is no root
on which to base the rewrite. I need to rework this a bit to distinguish the
direct "add" API where nodes are created directly without the rewrite step from
the "rewrite" APIs which will require a non-empty old-expr.
:::

After we rewrite the root of the old expr, we traverse the subgraph reachable
from its replacement to build a final "at-rest" buffer.

We implicitly drop "dead" nodes, as we just won't traverse them from the new
root, and we lower operands first, in-order, and exactly once (a
`DenseMap<unsigned, unsigned>` should be enough here, as the nodes are already
uniqued).

## API

:::{admonition} TODO
With the move away from Metadata-based operations, this needs to be updated.
The general shape should remain about the same.
:::

### Adding nodes

New nodes are added via "constructor" methods on the DAG:

```cpp
Type *I32Ty = Type::getInt32Ty();
DAG.mul(DAG.getArg(I32Ty, 0),
        DAG.plus(DAG.getArg(I32Ty, 1),
                 DAG.constant(I32Ty, 42)))
// ->
// !0 = !DIOpMul(i32, !1, !2)
// !1 = !DIOpArg(i32, 0)
// !2 = !DIOpPlus(i32, !3, !4)
// !3 = !DIOpArg(i32, 1)
// !4 = !DIOpConstant(i32, 42)
```

### Edits

Edits to an existing expr are more naturally expressed as (possibly
non-contracting) replacements for subgraphs which match a given pattern:

```cpp
// If we are salvaging a `plus 42`:
DIExpr *New = DAG.rewrite(
  Old,
  Pattern::DIOpArg(_, 1),
  [&](DIExpr *Match) {
    return DAG.plus(Match, DAG.constant(Match->type(), 42))
  }
);
```

All matches are discovered once on the pre-edit DAG rooted at `Old`, so the
result can contain the match. TODO: what about overlapping patterns?

The dependent nodes on the path from the old root to the new nodes are
re-added as-if one-by-one via post-order traversal.

### Optimizations

After any new node is created, there is a transparent
canonicalization/optimization phase. There are various knobs we can tune
here, including the set of peepholes and the strategy for applying them.

## Alternatives

- We could stick with the current `DIExpression`, and add accessory structures
  to maintain just enough type information.
- We could keep the `DIExpression` representation, but create many more
  `LLVM`-specific ops which are typed.
- We can abstract the current representation, still needing to create many new
  operations and transition to them; see
  https://discourse.llvm.org/t/rfc-for-diexpression-editing/88459/

The type-analysis we would need for the corner cases that manifest as the bugs
in many of the Motivating Issues could be done as-needed by essentially
evaluating the expression to get the information that is encoded transparently
in the graph-based approach.

It seems like we tend to expect to store and inspect expressions more
frequently than we create or edit them, so moving to a representation that
makes all the relevant information directly available might be the better
tradeoff, versus a more compact representation.

