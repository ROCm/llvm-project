(amdgpu-dma-operations)=

# AMDGPU DMA Operations

```{contents}
:local:
```

## Introduction

DMA (or "Direct Memory Access") operations transfer data between different kinds
of memory directly without occupying registers in the invoking wave. They are
usually {ref}`asynchronous<amdgpu-async-operations>` asynchronous, and require
the user to explicitly track completion using
{ref}`asyncmarks<amdgpu-async-operations>`.

All DMA operations support the same cache modifiers as ordinary load/store
operations from registers. They cannot be performed atomically.

### GFX9 DMA

Each GFX9 DMA instruction has a synchronous counterpart (e.g.,
``@llvm.amdgcn.load.to.lds`` for ``@llvm.amdgcn.load.async.to.lds``). The
synchronous variants perform the same operation, but the compiler automatically
ensures completion before their side-effects are used.

GFX9 DMA instructions implement volatile (via ``aux/cpol`` bit 31) and
nontemporal (via metadata) as if they were loads from the global address space.

**Flat/Global Addressing**

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
``4 * lane_id`` bytes for sizes up to 4 bytes, and by ``16 * lane_id`` bytes
for larger sizes.

The ``%lds_base`` pointer must be wave-uniform.

The source pointer is overloaded on address space. Supported address spaces are
flat (0), global (1), and buffer fat pointer (7).

``@llvm.amdgcn.load[.async].to.lds.p7`` (buffer pointer) is lowered to
``@llvm.amdgcn.raw.ptr.buffer.load[.async].lds`` before instruction selection.

```llvm
void @llvm.amdgcn.global.load[.async].lds(
    ptr addrspace(1) %src,      ; global base pointer to load from (per-lane)
    ptr addrspace(3) %lds_base, ; LDS base pointer (wave-uniform)
    i32 immarg %size,           ; data byte size (immediate): 1/2/4 (12/16 for gfx950)
    i32 immarg %offset,         ; offset (immediate) applied to both global and LDS address
    i32 immarg %cpol)           ; cache policy (immediate)
```

This is identical to ``@llvm.amdgcn.load[.async].to.lds.p1``.

**Buffer Addressing**

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

The ``%lds_base`` pointer must be wave-uniform.

The intrinsics differ in two orthogonal ways:

- **raw** vs **struct**: The ``struct`` variants add a ``%vindex`` argument for
  indexed buffer addressing.
- **ptr** vs non-ptr: The ``ptr`` variants use ``ptr addrspace(8)`` for the
  buffer resource descriptor; the non-ptr variants use ``<4 x i32>``.

### GFX1250

GFX1250 LDS DMA instructions implement nontemporal (via metadata) as if they
were loads from the global address space. Tensor DMA instructions do not support
volatile or nontemporal.

**Global Addressing**

```llvm
void @llvm.amdgcn.{global|cluster}.load.async.to.lds.b<N>(
    ptr addrspace(1) %src,      ; global base pointer to load from (per-lane)
    ptr addrspace(3) %lds_base, ; LDS base pointer (per-lane)
    i32 immarg %offset,         ; offset (immediate) applied to both global and LDS address
    i32 immarg %cpol,           ; cache policy (immediate)
    [i32 %m0])                  ; workgroup broadcast mask, cluster variants only (in M0)
```

The bit-size encoded in the name can be 8, 32, 64 or 128.

Loads data from global memory to LDS. The ``%offset`` is applied to both the
global and LDS addresses.

The ``cluster`` variants add a ``%m0`` argument for workgroup broadcast. The
broadcast mask selects which workgroups within a cluster participate in the load.

```llvm
void @llvm.amdgcn.global.store.async.from.lds.b<N>(
    ptr addrspace(1) %dst,      ; global base pointer to store to (per-lane)
    ptr addrspace(3) %lds_base, ; LDS base pointer to load from (per-lane)
    i32 immarg %offset,         ; offset (immediate) applied to both global and LDS address
    i32 immarg %cpol)           ; cache policy (immediate)
```

Stores data from LDS to global memory.

**Tensor Addressing**

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
descriptor groups; ``%desc4`` is reserved for future targets and must be
zero-initialized.

Despite the absence of ``.async`` in their names, these intrinsics are
asynchronous.

All arguments must be wave-uniform.

(amdgpu-dma-scopes)=

## DMA Scopes

A DMA operation initiated by a thread does not belong to the corresponding
instance of "singlethread" scope. Instead the DMA operation belongs to a
corresponding DMA scope determined by the target. The following intrinsics
return the {ref}`scope<amdgpu-scope-type>` at which each kind of DMA operation
observes memory on the current target:

```llvm
target("amdgcn.scope") @llvm.amdgcn.scope.lds.dma()
target("amdgcn.scope") @llvm.amdgcn.scope.tensor.dma()
```

These scope identifiers can be passed to any intrinsic that accepts a
{ref}`amdgpu-scope-type` argument:

```llvm
%lds_dma_scope = call target("amdgcn.scope") @llvm.amdgcn.scope.lds.dma()
call void @llvm.amdgcn.make.available(target("amdgcn.scope") %lds_dma_scope)
call void @llvm.amdgcn.make.ptr.visible(ptr %p, target("amdgcn.scope") %lds_dma_scope)
```

(amdgpu-dma-memory-model)=

## Memory Model

**TODO:** Need to carefully thread *location-order* and *happens-before*.

Each dynamic instance of a DMA *instruction* ``X`` *initiates* a DMA
*operation* ``D``. The DMA operation is performed in an instance of the
corresponding DMA scope ``S``. In addition, the user may specify a scope
``S'`` such that ``S`` is a subscope of ``S'``.

The effect of ``D`` can be modeled as the following pseudo-expansion in LLVM IR:

```llvm
; M = max(S, S')
;
%tmp = load-visible ptr %src, M     ; non-atomic
store-available ptr %dst, %tmp, M   ; non-atomic
```

(amdgpu-dma-visibility)=

### Explicit Visibility Required

[This section is informational.]

A DMA operation ``D`` is performed in an instance ``I`` of scope ``S``, but it
is not included in any subscope instances of ``I``. This means that the
{ref}`amdgpu-availability-visibility` operations performed by ``D`` **cannot**
form an *inclusive scope* relationship with those subscopes. This requires
threads to perform additional availability and visibility operations that ensure
{ref}`amdgpu-location-order` in certain cases shown below.

#### Wavefront Scope

Consider a thread that writes to global memory and then initiates a DMA
operation that reads from the same location. The two operations are related in
*happens-before*, but the DMA operation is not contained in the thread's
"singlethread" scope instance. The global write is not visible from the DMA
read; an explicit ``make.available`` at the DMA scope is needed.

```llvm
%dma_scope = call target("amdgcn.scope") @llvm.amdgcn.scope.lds.dma()

store %val, ptr %global
call @llvm.amdgcn.make.ptr.available(ptr %global, target("amdgcn.scope") %dma_scope) ; <---
call @llvm.amdgcn.global.load.async.to.lds(%global, %lds)
call @llvm.amdgcn.asyncmark()
call @llvm.amdgcn.wait.asyncmark(0)
%val_lds = load addrspace(3) %lds
```

The same result can be achieved using a {ref}`amdgpu-store-available` operation:

```llvm
%dma_scope = call target("amdgcn.scope") @llvm.amdgcn.scope.lds.dma()

call @llvm.amdgcn.global.store.available(%global, %val, target("amdgcn.scope") %dma_scope)
call @llvm.amdgcn.global.load.async.to.lds(%global, %lds)
call @llvm.amdgcn.asyncmark()
call @llvm.amdgcn.wait.asyncmark(0)
%val_lds = load addrspace(3) %lds
```

A similar pattern is required when storing to global using DMA:

```llvm
%dma_scope = call target("amdgcn.scope") @llvm.amdgcn.scope.lds.dma()

call @llvm.amdgcn.global.store.async.from.lds(%global, %lds)
call @llvm.amdgcn.asyncmark()
call @llvm.amdgcn.wait.asyncmark(0)
call @llvm.amdgcn.make.ptr.visible(ptr %global, target("amdgcn.scope") %dma_scope) ; <---
%val = load ptr %global
```

The ``make.ptr.visible`` at the DMA scope is necessary because the DMA write is
not automatically visible to the subsequent global read.

#### Workgroup Scope

Consider the case where one wave writes to global memory and a different wave in
the same workgroup initiates a DMA operation that reads from the same location.
A workgroup-scope fence can provide *happens-before* between the waves but does
not make the write available at the DMA scope. The DMA operation is not
contained in the workgroup scope instance, so the fence's *MakeAvailable* and
the DMA do not have inclusive scopes. An explicit ``make.available`` at the DMA
scope is needed.

```llvm
%dma_scope = call target("amdgcn.scope") @llvm.amdgcn.scope.lds.dma()

; wave 1
store %val, ptr addrspace(1) %global
call @llvm.amdgcn.make.ptr.available(ptr %global, target("amdgcn.scope") %dma_scope) ; <---
fence release syncscope("workgroup")

; wave 2
fence acquire syncscope("workgroup")
call @llvm.amdgcn.global.load.async.to.lds(%global, %lds)
call @llvm.amdgcn.asyncmark()
call @llvm.amdgcn.wait.asyncmark(0)
%val_lds = load addrspace(3) %lds
```

Similarly, when one wave stores to global memory using a DMA operation and a
different wave reads from the same location, an explicit ``make.visible`` at the
DMA scope is needed. The workgroup fence's *MakeVisible* cannot observe the DMA
write because the DMA is not contained in the workgroup scope instance.

```llvm
%dma_scope = call target("amdgcn.scope") @llvm.amdgcn.scope.lds.dma()

; wave 1
call @llvm.amdgcn.global.store.async.from.lds(%global, %lds)
call @llvm.amdgcn.asyncmark()
call @llvm.amdgcn.wait.asyncmark(0)
fence release syncscope("workgroup")

; wave 2
fence acquire syncscope("workgroup")
call @llvm.amdgcn.make.ptr.visible(ptr %global, target("amdgcn.scope") %dma_scope) ; <---
%val = load ptr addrspace(1) %global
```

### Implementation Details

[This section is informational.]

1. On GFX9, ``@llvm.amdgcn.scope.lds.dma()`` returns a value equivalent to
   "wavefront" scope. The LDS DMA implementation on GFX9 sees the same state of
   memory as the requesting thread, so ``make.available`` and ``make.visible``
   at this scope are no-ops.
2. On GFX1250, ``@llvm.amdgcn.scope.lds.dma()`` returns a value equivalent to
   "cluster" scope. The compiler emits a cache write-back or invalidate at
   ``SCOPE_SE`` for ``make.available`` and ``make.visible`` at this scope
   respectively.
