(amdgpu-memmodel)=

# AMDGPU Memory Model

## Introduction

The {ref}`LLVM memory model<memmodel>` provides broad guarantees that are
sufficient to implement inter-thread communication via memory. But in most
communication patterns, not all memory accesses performed by a thread need to be
exposed to other threads. Even when they do need to be exposed, not all threads
may need to observe these memory accesses. This document describes the *AMDGPU
memory model* that allows the user to control how the effect of each write
operation is propagated across threads. The programmer expresses this using
**new intrinsics and metadata** as described below, and the implementation can
then choose a more efficient mechanism to complete them, such as the cache
policy bits in an AMDGPU device.

The AMDGPU memory model defines a transitive *amdgpu-happens-before* relation
for each byte. But amdgpu-happens-before by itself is not sufficient to describe
observable executions. Additionally, *availability* and *visibility* are
required to describe how the effects of write operations become observable
in other threads. This requirement is captured by *location-order*.

The AMDGPU-specific intrinsics and metadata opt out of the LLVM memory model,
and can only be explained using the AMDGPU memory model. At the same time, a
simple mapping can be used to implement these new intrinsics and metadata using
target-independent LLVM IR instructions. The LLVM *happens-before* order
implements *location-order* since LLVM IR operations have the necessary
properties. Thus, there exists a safe-by-default implementation that produces
executions that are valid in both models.

## Terminology

Operation
: An action performed by a dynamic instance. A single dynamic instance may
  perform one or more operations, and may be used interchangeably with the
  operations it performs. For example, executing a `store atomic release`
  instruction performs a store operation as well as a release operation.

Memory Access
: An operation that returns or changes the state of memory. For example:
  - non-atomic `load` and `store` instructions
  - load/store operations performed by `atomic` instructions
  - *visibility* and *availability* operations

Side-effects of an Operation
: The set of bytes that this operation accesses or can synchronize.

(amdgpu-synchronize-with)=

## `synchronize-with`

Memory synchronization relates memory operations performed by one thread with
those in other threads. This is captured by various relations in the memory
consistency model. The *synchronize-with* relation is a cross-thread relation
that enables the construction of the transitive {ref}`amdgpu-happens-before`
relation which in turn governs most of the guarantees provided by the memory
consistency model.

When an operation `X` *synchronizes-with* an operation `Y`:
- `X` and `Y` are called *synchronizing operations*,
- `X` is called a *release operation*, and,
- `Y` is called an *acquire operation*.

Whether and how an operation *synchronizes-with* another operation is defined
separately for each operation. For example:

- A `store atomic release` operation can *synchronize-with* a `load atomic
  acquire` to the same location when they both satisfy certain conditions.
- A `fence release` followed by a `store atomic` operation can similarly
  *synchronize-with* a `fence acquire` operation that is preceded by a `load
  atomic` operation to the same location.

### Synchronizing Sets of Bytes

A synchronizing operation specifies a candidate set `L` consisting of bytes in
memory whose state it can synchronize across threads (for example, using
{ref}`!"amdgpu-synchronize-as" MMRA metadata<amdgpu-fence-as>` to restrict `L`
to specific address spaces). If no such set is specified, then the operation is
assumed to synchronize all bytes.

A synchronizing operation that also performs a memory access (e.g., an atomic
operation) includes the bytes that it accesses in the set `L` of bytes that it
can synchronize.

Let `L1` and `L2` be the candidate sets of bytes that can be synchronized by
operations `X` and `Y` respectively. If `X` *synchronizes-with* `Y`, the set of
bytes `L` synchronized by that interaction is the intersection `L = L1 ∩ L2`.

Examples:

`store atomic release %ptr`
: An atomic store operation that can synchronize all bytes in memory, including
  those accessed via the `%ptr` argument.

`load atomic acquire %ptr, !mmra !{!"amdgpu-synchronize-as", !"global"}`
: An atomic load operation that can synchronize all bytes in global memory, in
  addition to the ones accessed via the `%ptr` argument.

`fence acquire, !mmra !{!"amdgpu-synchronize-as", !"local"}`
: When suitably paired with an atomic operation, a fence that can synchronize
  all bytes within the `local` address space.

(amdgpu-scopes)=

## Scopes

A *scope* is an abstract description of sets of memory accesses and
synchronizing operations in a multi-threaded execution environment. Each such
set is called an *instance* of that scope, or a *scope instance* for short.

- When an operation `X` specifies a scope `S`, it indicates the instance of
  `S` that contains `X`. This scope instance is also termed as *X's instance
  of scope S*, or just *X's scope instance* when `S` is implied by the
  context.
- When an operation does not specify a scope, it indicates the *system*
  scope defined below.

Scopes have the following properties:

- Two distinct instances of the same scope do not intersect.
- Each operation performed by the program belongs to at least one scope
  instance.
- If `S1` is a subscope of `S2`, then every instance `I1` of `S1` is a subset of
  some instance `I2` of `S2`. `I1` is said to be a *subscope instance* of `I2`.
- If two scope instances `I1` and `I2` intersect, then their intersection is
  the smaller of `I1` and `I2`.

### LLVM scopes

The LLVM Language Reference defines the following {ref}`scopes<syncscope>`:

*system scope* (empty string "")

: There exists a single instance of this scope that contains the memory accesses
  and synchronizing operations performed by all threads.

"singlethread" scope

: Each thread corresponds to a "singlethread" scope instance that contains the
  memory accesses and synchronizing operations performed by that thread.

### AMDGPU scopes

The AMDGPU target defines the following scopes:

- *system scope* (same as LLVM)
- "agent" scope
- "cluster" scope
- "workgroup" scope
- "wavefront" scope
- "singlethread" scope (same as LLVM)

These are arranged from largest scope (*system scope*) to smallest scope
("singlethread"). Every scope `S1` other than *system scope* is a *subscope* of
the scopes above it in this linear arrangement.

In addition, there exists a symbolic string
{ref}`"lds-dma"<amdgpu-lds-dma-scope>` which can refer to either "workgroup"
scope or "cluster" scope.

#### Containment

Every operation *belongs to* an instance of a specific scope:

- Operations executed by each thread *belong to* the corresponding
  "singlethread" scope instance.
- DMA operations *initiated by* each thread *belong to* a corresponding
  {ref}`"lds-dma" scope<amdgpu-lds-dma-scope>` instance.

Every operation is *contained* in the scope instance `I1` that it *belongs
to*, and therefore in every scope instance `I2` that has `I1` as a *subscope
instance*.

This affects how operations are related in *inclusive scopes* when determining
availability and visibility. For example, DMA operations require {ref}`explicit
availability and visibility<amdgpu-dma-visibility>` operations at the
corresponding "lds-dma" scope.

#### Inclusive Scopes

Two operations `X` and `Y` are said to have *inclusive
scopes* if the scope instance of each operation contains the other operation. In
that case, the *common scope instance* `S'` of `X` and `Y` is the
intersection of their scope instances. The scope corresponding to `S'` is also
termed as the *common scope* of `X` and `Y`.

## Availability and Visibility

The AMDGPU memory model is built on top of the {ref}`amdgpu-happens-before`
order, which is very similar to the LLVM {ref}`happens-before<memmodel>` order.
But **amdgpu-happens-before by itself is not sufficient** to reason about memory
consistency. Additionally, the AMDGPU model uses *availability* and *visibility*
to describe how the effects of these operations propagate to other threads.

Availability determines how *far* the effect of a write has been forwarded in
the system relative to that write. Visibility determines how *close* the effect
of the same write has reached relative to an observer operation (typically a
read).

Consider a write `W` that {ref}`amdgpu-happens-before` a read `R` from the same
byte `B`:

- `R` can potentially observe the effect of `W` **only if W is
  visible** to `R`.
- `W` can potentially be visible to `R` **only if W is first made
  available** to `R`.

:::{note}
We introduce new intrinsics because metadata cannot be used to express the
availability and visibility of ordinary
load/store operations, because the scope is necessary for correctness. In a
hypothetical operation like this:

```llvm
store ptr addrspace(1) %ptr, <4 x i32> %value, !mmra !{!"amdgcn-av", !"workgroup"}
```

If the metadata is dropped or ignored, there is no guarantee that the store
will become available at the intended scope. In implementation terms, the
store may be completed at a nearer cache than the one required for that
scope. A corresponding *load-visible* that does not access the same near
cache will fail to observe this store.
:::

(amdgpu-store-available)=

### store-available

A *store-available* operation `X` writes data to the specified bytes, and then
propagates this effect up to its scope instance `S` and also all the subscope
instances of `S` that contain `X`.

```llvm
@llvm.amdgcn.av.store.b128(%addr, %value, metadata !scope)
```

The `@llvm.amdgcn.av.store.b128` intrinsic performs a non-atomic
*store-available* operation on `%addr` with scope `!scope`. See
{ref}`amdgpu-av-load-store` for details.

```llvm
store atomic ..., [syncscope("<target-scope>")]
atomicrmw    ..., [syncscope("<target-scope>")]
cmpxchg      ..., [syncscope("<target-scope>")]
```

An atomic operation that performs a store operation is a *store-available*
operation with scope `syncscope`.

(amdgpu-load-visible)=

### load-visible

A *load-visible* operation `Y` retrieves the state of the specified bytes
from its scope instance `S`, and then loads the value at those bytes. In
addition, it makes the retrieved state visible to all subscope instances of `S`
that contain `Y`.

```llvm
%value = @llvm.amdgcn.av.load.b128(%addr, metadata !scope)
```

The `@llvm.amdgcn.av.load.b128` intrinsic performs a non-atomic *load-visible*
operation on `%addr` with scope `!scope`. See {ref}`amdgpu-av-load-store` for
details.

```llvm
%value = load atomic ..., [syncscope("<target-scope>")]
%value = atomicrmw   ..., [syncscope("<target-scope>")]
%value = cmpxchg     ..., [syncscope("<target-scope>")]
```

An atomic operation that performs a read operation is a *load-visible*
operation with scope `syncscope`.

### addrspace(3) Accesses

Any access to `addrspace(3)` (aka LDS or local) is always a *store-available* or
*load-visible* operation with scope "lds-dma".

Any access to `addrspace(0)` (aka flat or generic) is always a
*store-available* or *load-visible* operation with scope "lds-dma" if the
flat address resolves to `addrspace(3)`.

:::{note}
This is independent of the `syncscope` of an atomic access. For example, even if
a `store atomic` to `addrspace(3)` specifies a smaller `syncscope` such as
"wavefront", the effect of this write is made available at "lds-dma" scope. In
other words, the scope for atomicity is not always the same as the scope for
availability/visibility.
:::

### volatile Accesses

```llvm
store volatile ..., ptr addrspace(N) %addr
%value = load volatile ..., ptr addrspace(N) %addr
```

A non-atomic `volatile` access is a *store-available* or *load-visible*
operation with the widest scope supported by its address space:

- "system" scope for `addrspace(1)` (global) and `addrspace(0)` (generic), and
- "lds-dma" scope for `addrspace(3)` (local), since local memory is not
  observable beyond its workgroup and the DMA operations associated with that
  workgroup.

An `atomic volatile` access is not widened in this way. Its availability and
visibility remain limited to its `syncscope`, as described for atomic accesses
above.

:::{note}
The availability and visibility of volatile accesses is specific to AMDGPU, and
orthogonal to the {ref}`base semantics defined in LLVM IR<volatile>`. This fully
specifies the behavior that the {ref}`LLVM memory model<memmodel>` otherwise
leaves target-dependent for `volatile` accesses.
:::

(amdgpu-av-metadata)=

### !"amdgcn-av" metadata

```llvm
!mmra !{!"amdgcn-av", !"none"}
```

The presence of this metadata removes the ability of synchronizing operations to
establish availability and visibility, and essentially creates *non-av* synchronizing
operations.

For a synchronizing operation which itself accesses memory (e.g., `store atomic
release` or `load atomic acquire`), the metadata does not affect the
availability or the visibility of the access performed by the operation itself.
It only affects the synchronization of other memory accesses.

### make-available and make-visible

A *make-available* or *make-visible* operation propagates the state of specified
bytes from or to the specified scope respectively.

A synchronizing operation is a *make-available* or *make-visible* operation only
when specified in its definition.

```llvm
store atomic [syncscope("<target-scope>")] <ordering> [, !mmra !{!"amdgcn-av", !"none"}]
load atomic  [syncscope("<target-scope>")] <ordering> [, !mmra !{!"amdgcn-av", !"none"}]
atomicrmw    [syncscope("<target-scope>")] <ordering> [, !mmra !{!"amdgcn-av", !"none"}]
cmpxchg      [syncscope("<target-scope>")] <ordering> [, !mmra !{!"amdgcn-av", !"none"}]
fence        [syncscope("<target-scope>")] <ordering> [, !mmra !{!"amdgcn-av", !"none"}]
```

An `atomic` or `fence` operation with at least `release` ordering is a
*make-available* operation with scope `syncscope`, if it is not marked as
`!{!"amdgcn-av", !"none"}`.

An `atomic` or `fence` operation with at least `acquire` ordering is a
*make-visible* operation with scope `syncscope`, if it is not marked as
`!{!"amdgcn-av", !"none"}`.

If a *make-available* or *make-visible* operation is a synchronizing operation,
then its effect on availability and visibility is limited to the bytes
included in its *side-effects*.

```llvm
; This performs the following operations:
; - The atomic store at "agent" scope,
; - A store-available operation at "agent" scope on `%ptr`,
; - A *make-available* operation at "agent" scope that affects previous memory accesses.
store atomic syncscope("agent") release %ptr

; This performs the following operations:
; - The atomic store at "agent" scope,
; - A store-available operation at "agent" scope on `%ptr`.
; Notably, it does not perform a *make-available* operation on other memory accesses.
store atomic syncscope("agent") release %ptr, !mmra !{!"amdgcn-av", !"none"}
```

## Ordering

(amdgpu-happens-before)=

### `amdgpu-happens-before<B>`

*amdgpu-happens-before\<B\>* is defined separately for each byte `B` in memory.

An operation `X` *amdgpu-happens-before\<B\>* an operation `Y` if at least one of
the following is true:

- `X` and `Y` each include byte `B` in their side-effects and at least one of
  the following is true:
  - `X` is *program-ordered* before `Y`.
  - `X` *synchronizes-with* `Y`.
- There is an operation `Z` such that `X` *amdgpu-happens-before\<B\>* `Z`
  and `Z` *amdgpu-happens-before\<B\>* `Y`.

### Availability Operation

An operation `X` is an *availability operation* on a write `W` to a byte `B` if
at least one of the following holds:

- `X` is `W` itself, and `W` is a *store-available* operation, or,
- `X` is a *make-available* operation that follows `W` in program order and
  includes `B` in its side-effects, or,
- `X` is a *make-available* operation whose scope instance includes `W`, and
  there is an availability operation `Z` on `W` such that:

  - `Z` *amdgpu-happens-before\<B\>* `X`, and,
  - `Z`'s scope instance includes `X`.

Then `X` makes the effect of `W` on `B` available in its own scope instance
`S` and every subscope instance of `S` that also includes `W`.

### Visibility Operation

An operation `Y` is a *visibility operation* on a write `W` to a byte `B` if `Y`
is a *load-visible* operation to `B`, or a *make-visible* operation that
includes `B` in its side-effects, and at least one of the following holds:

- There exists an *availability operation* `X` on write `W` such that:
  - `X` *amdgpu-happens-before\<B\>* `Y`, and,
  - `X` and `Y` have inclusive scopes, and,
  - Their common scope instance `S` includes `W`.

  Then `Y` makes the effect of `W` on `B` visible in `S` and every subscope
  instance of `S` that includes `Y`.

- There exists a *visibility operation* `X` on write `W` such that:
  - `X` *amdgpu-happens-before\<B\>* `Y`, and,
  - `X` makes `W` visible in a scope instance `S1` that includes `Y`, and,
  - `X` is included in the scope instance `S2` of `Y`.

  Then `Y` makes the effect of `W` on `B` visible in the intersection `S`
  of `S1` and `S2`, and every subscope instance of `S` that includes `Y`.

(amdgpu-location-order)=

### Location Order

Location order relates a write `W` to a byte `B` with every other operation
`Y` that accesses `B`.

A write `W` is *location-ordered* before an access `Y` if `W` is program-ordered
before `Y`.

A write `W` is *location-ordered* before a write `W1` if there exists an
availability operation `Z` on `W` to the same byte such that:

- `Z` *amdgpu-happens-before\<B\>* `W1`, and,
- `W1` is included in `Z`'s scope instance.

A write `W` is *location-ordered* before a read `Y` if there exists a visibility
operation `Z` on write `W` to byte `B` such that:

- `Z` is `Y` itself, or,
- `Z` precedes `Y` in program order.

A read `R` is *location-ordered* before an access (read and/or write) `X` to the
same byte if it *amdgpu-happens-before\<B\>* `X`.

The AMDGPU memory model overrides the definition of each byte in the
{ref}`LLVM memory model<memmodel>` as follows.

Every (defined) read operation `R` reads a series of bytes written by
(defined) write operations. Each initialized global is assumed to have an
initial *system scoped* atomic write operation that is *location-ordered* before
any other read or write to that same byte.

For each byte `B` of a read `R`, `R` may see any write to `B`, except:

- If a write `W1` is *location-ordered* before a write `W2`, and `W2` is
  *location-ordered* before a read `R`, then `R` may not see `W1`.
- If a read `R` is *location-ordered* before a write `W3`, then `R` may not see
  `W3`.

The value returned by `R` is then defined as follows:

- If no write is *location-ordered* before a read `R`, then `R` returns
  `undef`.
- Otherwise if the set consisting of `R` and all writes that `R` may see
  contains only atomic operations with inclusive scopes that all access the
  exact same set of bytes, then `R` returns the value written by one of those
  writes.
- Otherwise, if `R` may see some write that is not *location-ordered* before
  `R`, then `R` returns `undef`.
- Otherwise, if `R` may see exactly one write `W`, then `R` returns the
  value written by `W`.
- Otherwise, `R` returns `undef`.

## Properties

:::{tip}
This section is informational.
:::

The following properties follow from the definitions above:

1. **amdgpu-happens-before is necessary for location-order, except for initial writes.**
   An access `X` is *location-ordered* before an access `Y` (on byte `B`) only
   if `X` is the initial write of an initialized global or `X`
   *amdgpu-happens-before\<B\>* `Y`.
   For accesses other than initial writes, this follows from the definition of
   location-order, availability, and visibility operations, which always
   require an *amdgpu-happens-before\<B\>* link with the preceding operation in
   the chain.
2. **A write cannot be made available in a scope that does not contain it.** The
   definition of an availability operation `X` requires that `X`'s scope
   instance includes `W` as a precondition. The operation makes `W` available
   in that scope instance and only those subscope instances that also contain
   `W`. Availability therefore cannot reach a scope instance that excludes `W`.
3. **Visibility is bounded by availability.** When a write is available in a
   scope instance, it can be made visible in that scope instance by a visibility
   operation with the corresponding scope. Subsequent *make-visible* operations
   make that write visible into narrower scope instances towards the observer.
4. **A write can be made visible in a scope instance that does not contain it.**
   The definition of a *visibility operation* anchors scope instances to the
   observer (`Y`), not to the original write. The only precondition is that for
   `Y`'s scope `S`, the write must already be visible or available in an instance
   of `S` or a subscope of `S` that contains `Y`.
5. **Availability and visibility chains.** For a write `W` to be visible to a
   read `R` anywhere in the system, the sufficient condition is a chain of
   *amdgpu-happens-before* edges that include availability and visibility
   operations with inclusive scopes. It is not necessary that `W` and `R`
   themselves have inclusive scopes. Intermediate operations can bridge scope
   gaps that the endpoints cannot satisfy directly. Such a chain passes through
   at least one availability operation and at least one visibility operation
   with inclusive scopes, such that their common scope includes both `W` and
   `R`.

(amdgpu-ordering-comparison)=

## Comparison of Ordering Relations

[This section is informational.]

```{list-table}
:header-rows: 1
:widths: 30 20 25 25

   * - Purpose
     - C++
     - Vulkan
     - AMDGPU
   * - Intra-thread ordering
     - sequenced-before<br>(transitive)
     - program-order<br>(transitive)
     - program-order<br>(transitive)
   * - Inter-thread synchronization
     - synchronizes-with
     - synchronizes-with
     - synchronizes-with
   * - Transitive inter-thread ordering
     - happens-before
     - inter-thread-happens-before\<SC\>
     - amdgpu-happens-before\<B\>
   * - Basis for visibility
     - happens-before<br>(transitive)
     - happens-before<br>(**not** transitive)
     - amdgpu-happens-before\<B\><br>(transitive)
```

The above table lines up roughly equivalent ordering relations across the
C++, Vulkan, and AMDGPU memory models.

- In C++, *happens-before* is transitive and serves directly as the basis of
  *visible side effects*.
- In Vulkan, the non-transitive *happens-before* serves as the basis of
  *location-order*.
- In AMDGPU, the transitive, per-byte *amdgpu-happens-before* serves as the
  basis of *location-order*.

### Intermediate Synchronization in Vulkan

Vulkan constructs *happens-before* as the union of *program-order* and
*inter-thread-happens-before*. The latter is parameterized over a set of storage
classes, but its union over all possible sets is included in *happens-before*.
This allows a synchronization in one storage class to establish *location-order*
between endpoints that access a different storage class.

AMDGPU instead defines *amdgpu-happens-before* as the transitive closure of
*program-order* and *synchronizes-with*, independently for each byte. The
*location-order* on AMDGPU is also per byte, and depends on this per-byte
*amdgpu-happens-before*. Thus the AMDGPU *location-order* cannot relate
operations on one byte using an *amdgpu-happens-before* relation between
operations on a different byte.

(amdgcn-av-vulkan)=

## The Vulkan Memory Model

[This section is informational.]

The AMDGPU memory model draws heavily on the Vulkan memory model. In
particular, the following instructions are equivalent.

```{list-table}
:header-rows: 1
:widths: 25 20 55

   * - LLVM
     - SPIRV
     - Available/Visible Semantics
   * - `load`
     - `OpLoad NonPrivatePointer`
     - \-
   * - `load-visible`
     - `OpLoad NonPrivatePointer`
     - `MakePointerVisible`
   * - `store`
     - `OpStore NonPrivatePointer`
     - \-
   * - `store-available`
     - `OpStore NonPrivatePointer`
     - `MakePointerAvailable`
   * - `load atomic`
     - `OpAtomicLoad`
     - `MakePointerVisible`. Also `MakeVisible` when order is at least `acquire`.
   * - `load atomic`<br>`!{!"amdgcn-av", !"none"}`
     - `OpAtomicLoad`
     - `MakePointerVisible`
   * - `store atomic`
     - `OpAtomicStore`
     - `MakePointerAvailable`. Also `MakeAvailable` when order is at least `release`.
   * - `store atomic`<br>`!{!"amdgcn-av", !"none"}`
     - `OpAtomicStore`
     - `MakePointerAvailable`
   * - `fence`
     - `OpMemoryBarrier`
     - `MakeAvailable` when order is at least `release`, and `MakeVisible` when order is at least `acquire`.
   * - `fence`<br>`!{!"amdgcn-av", !"none"}`
     - `OpMemoryBarrier`
     - \-
```

:::{note}
The above table is representative only, and does not aim to be exhaustive. In
particular, it does not list composite atomic operations like `rmw` and
`cmpxchg`. The ordering and semantics of these operations can be determined
by combining suitable rules such as:

- "`MakeAvailable` if the order is at least `release`, and the operation
  performs a store",
- "Only if it is not marked as `!{!"amdgcn-av", !"none"}`", etc.
:::

The AMDGPU memory model is a special case of the Vulkan memory model:

1. LLVM fence/atomic ordering operations have `MakeAvailable` / `MakeVisible`
   semantics by default, thus satisfying the availability and visibility chains
   required in Vulkan. Hence the LLVM memory model is a "strong" subset of the
   Vulkan memory model.
2. The AMDGPU memory model described here makes it possible to opt-out of the
   default `MakeAvailable` and `MakeVisible` semantics, and instead specify it
   on select places including the new *load-visible* and *store-available*
   operations. This expands the subset of the Vulkan memory model that can now
   be expressed in LLVM IR.
