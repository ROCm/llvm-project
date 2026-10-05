(amdgpu-dma-operations)=

# AMDGPU DMA Operations


## Introduction

DMA (or "Direct Memory Access") operations transfer data between different kinds
of memory directly without occupying registers in the invoking wave. They are
usually {ref}`asynchronous<amdgpu-async-operations>` asynchronous, and require
additional mechanisms to {ref}`track completion<amdgpu-async-completed-at>`.

All DMA operations support the same cache modifiers as ordinary load/store
operations from registers. They cannot be performed atomically.

### GFX9 LDS DMA

Each GFX9 LDS DMA instruction has a synchronous counterpart (e.g.,
`@llvm.amdgcn.load.to.lds` for `@llvm.amdgcn.load.async.to.lds`). The
synchronous variants perform the same operation, but the compiler automatically
ensures completion before their side-effects are used.
The asynchronous variants use {ref}`amdgpu-asyncmarks` to track completion.

GFX9 DMA instructions implement volatile (via `aux/cpol` bit 31) and
nontemporal (via metadata) as if they were loads from the global address space.

#### Flat/Global Addressing

```llvm
void @llvm.amdgcn.load[.async].to.lds.pN(
    ptr addrspace(N) %src,      ; base pointer to load from (per-lane)
    ptr addrspace(3) %lds_base, ; LDS base pointer (wave-uniform)
    i32 immarg %size,           ; data byte size (immediate): 1/2/4 (12/16 for gfx950)
    i32 immarg %offset,         ; offset (immediate) applied to both src and LDS address
    i32 immarg %cpol)           ; cache policy (immediate)
```

Loads data from global memory to LDS. The data size can be 1, 2, or 4 bytes
(gfx950 also allows 12 or 16 bytes). The LDS address is implicitly offset by
`4 * lane_id` bytes for sizes up to 4 bytes, and by `16 * lane_id` bytes
for larger sizes.

The `%lds_base` pointer must be wave-uniform.

The source pointer is overloaded on address space. Supported address spaces are
flat (0), global (1), and buffer fat pointer (7).

`@llvm.amdgcn.load[.async].to.lds.p7` (buffer pointer) is lowered to
`@llvm.amdgcn.raw.ptr.buffer.load[.async].lds` before instruction selection.

```llvm
void @llvm.amdgcn.global.load[.async].lds(
    ptr addrspace(1) %src,      ; global base pointer to load from (per-lane)
    ptr addrspace(3) %lds_base, ; LDS base pointer (wave-uniform)
    i32 immarg %size,           ; data byte size (immediate): 1/2/4 (12/16 for gfx950)
    i32 immarg %offset,         ; offset (immediate) applied to both global and LDS address
    i32 immarg %cpol)           ; cache policy (immediate)
```

This is identical to `@llvm.amdgcn.load[.async].to.lds.p1`.

#### Buffer Addressing

```llvm
void @llvm.amdgcn.{raw|struct}[.ptr].buffer.load[.async].lds(
    %rsrc,                      ; buffer resource descriptor (wave-uniform):
                                ;   <4 x i32> or ptr addrspace(8)
    ptr addrspace(3) %lds_base, ; LDS base pointer (wave-uniform)
    i32 immarg %size,           ; data byte size (immediate): 1/2/4 (12/16 for gfx950)
    [i32 %vindex,]              ; buffer index (per-lane, struct variants only)
    i32 %voffset,               ; offset (per-lane, included in bounds checking)
    i32 %soffset,               ; offset (wave-uniform, excluded from bounds checking)
    i32 immarg %offset,         ; offset (immediate, included in bounds checking)
    i32 immarg %cpol)           ; cache policy (immediate)
```

Loads data from a buffer resource to LDS.

The `%lds_base` pointer must be wave-uniform.

The intrinsics differ in two orthogonal ways:

- **raw** vs **struct**: The `struct` variants add a `%vindex` argument for
  indexed buffer addressing.
- **ptr** vs non-ptr: The `ptr` variants use `ptr addrspace(8)` for the
  buffer resource descriptor; the non-ptr variants use `<4 x i32>`.

### GFX1250 DMA Operations

All GFX1250 DMA operations are asynchronous and can use
{ref}`asyncmarks<amdgpu-asyncmarks>` to track completion. Some loads and tensor
operations can also use {ref}`barriers<amdgpu-async-completed-at>` as described
below. There are no synchronous variants.

(amdgpu-gfx1250-lds-dma-operations)=

#### LDS DMA Operations

GFX1250 LDS DMA instructions implement nontemporal (via metadata) as if they
were loads from the global address space. Tensor DMA instructions do not support
volatile or nontemporal.

```llvm
void @llvm.amdgcn.{global|cluster}.load.async.to.lds.b<N>(
    ptr addrspace(1) %src,      ; global base pointer to load from (per-lane)
    ptr addrspace(3) %lds_base, ; LDS base pointer (per-lane)
    i32 immarg %offset,         ; offset (immediate) applied to both global and LDS address
    i32 immarg %cpol,           ; cache policy (immediate)
    [i32 %m0])                  ; workgroup broadcast mask, cluster variants only (in M0)
```

The bit-size encoded in the name can be 8, 32, 64 or 128.

Loads data from global memory to LDS. The `%offset` is applied to both the
global and LDS addresses.

The `cluster` variants add a `%m0` argument for workgroup broadcast. The
broadcast mask selects which workgroups within a cluster participate in the load.

```llvm
void @llvm.amdgcn.global.store.async.from.lds.b<N>(
    ptr addrspace(1) %dst,      ; global base pointer to store to (per-lane)
    ptr addrspace(3) %lds_base, ; LDS base pointer to load from (per-lane)
    i32 immarg %offset,         ; offset (immediate) applied to both global and LDS address
    i32 immarg %cpol)           ; cache policy (immediate)
```

Stores data from LDS to global memory.

#### Tensor Operations

```llvm
void @llvm.amdgcn.tensor.{load.to|store.from}.lds(
    <4 x i32> %desc0,          ; D# group 0
    <8 x i32> %desc1,          ; D# group 1
    <4 x i32> %desc2,          ; D# group 2 (zero-init for D# up to 2D)
    <4 x i32> %desc3,          ; D# group 3 (zero-init for D# up to 2D)
    <8 x i32> %desc4,          ; D# group 4 (reserved, use zeroinitializer)
    i32 immarg %cpol)          ; cache policy (immediate)
```

Loads or stores data between global memory and LDS using a tensor descriptor
(D#). The descriptor is split across multiple groups. GFX1250 supports up to 4
descriptor groups; `%desc4` is reserved for future targets and must be
zero-initialized.

Despite the absence of `.async` in their names, these intrinsics are
asynchronous.

The tensor descriptor can encode an optional LDS barrier. A tensor load or
store using such a descriptor performs one {ref}`barrier
arrive<amdgpu-barrier-operations>` when it completes. The barrier is encoded in
the descriptor arguments rather than passed separately to these intrinsics.

All arguments must be wave-uniform.

(amdgpu-lds-dma-scope)=

## The "lds-dma" Scope

An LDS DMA operation or a tensor DMA operation initiated by a thread does not
belong to the corresponding instance of "singlethread" scope. Instead the DMA
operation belongs to a scope instance determined by the target. For any target,
this is an instance of a scope no larger than "cluster" scope. Operations in
LLVM IR may refer to this using the "lds-dma" symbolic string.

### Effect on Inclusive Scopes

[This section is informational.]

The symbolic mapping of "lds-dma" scope affects how the _mutually inclusive_
relation is applied to DMA operations. When "lds-dma" scope maps to "cluster"
scope, the operation does not belong to a lower scope instance such as
"workgroup".

Consider an operation `X` that specifies "workgroup" scope, and a DMA operation
`Y` initiated from the same workgroup instance. On a target that performs DMA
operations at "cluster" scope, `Y` does not belong to any "workgroup" instance.
Thus `X` and `Y` do not have inclusive scope on this target even though they are
both associated with the same workgroup instance. If the same program is
rewritten so that `X` specifies "lds-dma" scope instead, then the two operations
will always have inclusive scope, independent of target. This is also true if
`X` specified "cluster" scope, but using "lds-dma" scope is more precise, and
may result in a more efficient implementation.

(amdgpu-dma-memmodel)=

## AMDGPU DMA Memory Model

Each DMA operation ``D`` is performed in an instance of the corresponding DMA
scope ``S``. In addition, the user may specify a scope ``S'`` as an argument.
The internal operations are made available/visible at the larger of these two
scopes:

```
Availability/Visibility Scope M = max(S, S')
```

### `program-order`

When the DMA operation is performed, control flow begins from *dma_entry*
followed in *program-order* by the following operations.

```{code-block} llvm
:caption: Global to LDS Internal Sequence

dma_entry
%tmp = load-visible ptr addrspace(1) %src, M              ; non-atomic
store-available ptr addrspace(3) %dst, %tmp, "lds-dma"    ; non-atomic
dma_complete
```

```{code-block} llvm
:caption: LDS to Global Internal Sequence

dma_entry
%tmp = load-visible ptr addrspace(3) %src, "lds-dma"      ; non-atomic
store-available ptr addrspace(1) %dst, %tmp, M            ; non-atomic
dma_complete
```

### `completed-at`

A DMA operation `D` *initiated* by an instruction `X` is *completed-at* an
operation `Y` if:

- `D` is a synchronous operation and `X` is *program-ordered* before
  `Y`, or,
- `D` is an asynchronous operation and a {ref}`completion
  mechanism<amdgpu-async-completed-at>` ensures that `D` is *completed-at* `Y`.

### Synchronization

When an instruction `X` initiates a DMA operation `D`, let `S` be the set of
locations that are accessed by `D`. [Informational note --- `S` typically
contains the locations indicated by both the source and destination pointer
operands on `X`.]

`dma_entry` and `dma_complete` are both {ref}`synchronizing operations
<amdgpu-synchronizing-operation>` that synchronize the set `S`.

`X` is a synchronizing operation that synchronizes-with `dma_entry`.

If `D` is a synchronous DMA operation:
- Each memory access by `D` *inter-thread-happens-before\<S\>* `dma_complete`,
  and,
- `dma_complete` *inter-thread-happens-before\<S\>* any operation that follows
  `X` in program-order.

If `D` is *completed-at* a `wait.asyncmark()` operation `Y`, then the
`dma_complete` operation performed by `D` synchronizes-with `Y`.

If a DMA operation `D` is *completed-at* a {ref}`barrier
wait<amdgpu-barrier-operations>` operation `W` then the *dma_complete* operation
in `D` *synchronizes-with* a `fence acquire` operation `F` such that:
- `W` is *program-ordered* before `F`, and
- `D` is included in the scope instance of `F`.

(amdgpu-dma-visibility)=

### Explicit Visibility Required

[This section is informational.]

A DMA operation ``D`` is performed in an instance ``I`` of scope ``S``, but it
is not included in any subscope instances of ``I``. This means that the
availability/visibility operations implicitly performed by ``D`` **cannot** form
an *inclusive scope* relationship with those subscopes. This requires threads to
perform additional availability and visibility operations that ensure
{ref}`amdgpu-location-order` in certain cases shown below.

#### Wavefront Scope

Consider a thread that writes to global memory and then *initiates* a DMA
operation that reads from the same location. This previous write must be made
available to the scope instance that contains the DMA operation.

```llvm
call @llvm.amdgcn.av.store.b128(%global, %val, "lds-dma")               ; <--
call @llvm.amdgcn.global.load.async.to.lds(%global, %lds)
call @llvm.amdgcn.asyncmark()
call @llvm.amdgcn.wait.asyncmark(0)
%val_lds = load addrspace(3) %lds
```

Alternatively, the wave may perform a release fence specifying the "lds-dma"
scope:

```llvm
store %val, ptr %global
fence release syncscope("lds-dma")                                      ; <--
call @llvm.amdgcn.global.load.async.to.lds(%global, %lds)
call @llvm.amdgcn.asyncmark()
call @llvm.amdgcn.wait.asyncmark(0)
%val_lds = load addrspace(3) %lds
```

A similar pattern is required with a DMA operation that writes to global memory.

```llvm
call @llvm.amdgcn.global.store.async.from.lds(%global, %lds)
call @llvm.amdgcn.asyncmark()
call @llvm.amdgcn.wait.asyncmark(0)
%val = call @llvm.amdgcn.av.load.b128(%global, "lds-dma")               ; <--
```

#### Workgroup Scope

Consider the case where one wave writes to global memory and a different wave in
the same workgroup *initiates* a DMA operation that reads from the same
location. The first wave must make its store available to the "lds-dma" scope
instance that contains the DMA operation.

```llvm
; wave 1
call @llvm.amdgcn.av.store.b128(%global, %val, "lds-dma")               ; <--
store atomic release syncscope("workgroup") %flag

; wave 2
load atomic acquire syncscope("workgroup") %flag
call @llvm.amdgcn.global.load.async.to.lds(%global, %lds)
call @llvm.amdgcn.asyncmark()
call @llvm.amdgcn.wait.asyncmark(0)
%val_lds = load addrspace(3) %lds
```

Alternatively, the first wave must release its operations to a sufficiently
large scope.

```llvm
; wave 1
store %val, ptr addrspace(1) %global
store atomic release syncscope("lds-dma") %flag                         ; <--

; wave 2
load atomic acquire syncscope("lds-dma") %flag
call @llvm.amdgcn.global.load.async.to.lds(%global, %lds)
call @llvm.amdgcn.asyncmark()
call @llvm.amdgcn.wait.asyncmark(0)
%val_lds = load addrspace(3) %lds
```

Similarly, when one wave stores to global memory using a DMA operation and a
different wave from the same workgroup reads from the same location, an explicit
``make.visible`` at the DMA scope may be needed. The workgroup fence's
*MakeVisible* may not be sufficient to observe the DMA write when the DMA is not
contained in the workgroup scope instance.

```llvm
; wave 1
call @llvm.amdgcn.global.store.async.from.lds(%global, %lds)
call @llvm.amdgcn.asyncmark()
call @llvm.amdgcn.wait.asyncmark(0)
store atomic release syncscope("workgroup") %flag

; wave 2
load atomic acquire syncscope("workgroup") %flag
fence syncscope("lds-dma") acquire                                      ; <--
%val = load ptr addrspace(1) %global
```

Note how the acquire fence is used to establish visibility; it's ability to
establish happens-before is an unintended side-effect. A better replacement
would be to introduce an intrinsic with purel `MakeVisible` (e.g.,
`%llvm.amdgpu.make.visible`).
