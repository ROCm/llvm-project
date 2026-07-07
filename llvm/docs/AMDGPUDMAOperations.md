(amdgpu-dma-operations)=

# AMDGPU DMA Operations

```{contents}
:local:
```

## Introduction

DMA (or "Direct Memory Access") operations transfer data between different kinds
of memory directly without occupying registers in the invoking wave. They are
usually {ref}`asynchronous<amdgpu-async-operations>`, and require
the user to explicitly track completion using
{ref}`asyncmarks<amdgpu-async-operations>`.

All DMA operations support the same cache modifiers as ordinary load/store
operations from registers. They cannot be performed atomically.

## GFX9 LDS DMA Instructions

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
    i32 immarg %size,           ; data byte size: 1/2/4 (12/16 for gfx950)
    i32 immarg %offset,         ; offset applied to both src and LDS address
    i32 immarg %cpol)           ; cache policy
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
    i32 immarg %size,           ; data byte size: 1/2/4 (12/16 for gfx950)
    i32 immarg %offset,         ; offset applied to both global and LDS address
    i32 immarg %cpol)           ; cache policy
```

This is identical to ``@llvm.amdgcn.load[.async].to.lds.p1``.

**Buffer Addressing**

```llvm
void @llvm.amdgcn.{raw|struct}[.ptr].buffer.load[.async].lds(
    %rsrc,                      ; buffer resource descriptor (SGPR):
                                ;   <4 x i32> or ptr addrspace(8)
    ptr addrspace(3) %lds_base, ; LDS base pointer (wave-uniform)
    i32 immarg %size,           ; data byte size: 1/2/4 (12/16 for gfx950)
    [i32 %vindex,]              ; VGPR buffer index (struct variants only)
    i32 %voffset,               ; VGPR offset (included in bounds checking)
    i32 %soffset,               ; SGPR/imm offset (excluded from bounds checking)
    i32 immarg %offset,         ; imm offset (included in bounds checking)
    i32 immarg %cpol)           ; cache policy
```

Loads data from a buffer resource to LDS.

The ``%lds_base`` pointer must be wave-uniform.

The intrinsics differ in two orthogonal ways:

- **raw** vs **struct**: The ``struct`` variants add a ``%vindex`` argument for
  indexed buffer addressing.
- **ptr** vs non-ptr: The ``ptr`` variants use ``ptr addrspace(8)`` for the
  buffer resource descriptor; the non-ptr variants use ``<4 x i32>``.

## GFX1250 Instructions

- LDS DMA instructions implement nontemporal (via metadata) as if they were
  loads from the global address space.
- Tensor instructions do not support volatile or nontemporal.

### LDS DMA Instructions

**Global Addressing**

```llvm
void @llvm.amdgcn.{global|cluster}.load.async.to.lds.b<N>(
    ptr addrspace(1) %src,      ; global base pointer to load from (per-lane)
    ptr addrspace(3) %lds_base, ; LDS base pointer (per-lane)
    i32 immarg %offset,         ; offset applied to both global and LDS address
    i32 immarg %cpol,           ; cache policy
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
    i32 immarg %offset,         ; offset applied to both global and LDS address
    i32 immarg %cpol)           ; cache policy
```

Stores data from LDS to global memory.

### Tensor Instructions

```llvm
void @llvm.amdgcn.tensor.{load.to|store.from}.lds(
    <4 x i32> %desc0,          ; D# group 0
    <8 x i32> %desc1,          ; D# group 1
    <4 x i32> %desc2,          ; D# group 2 (zero-init for D# up to 2D)
    <4 x i32> %desc3,          ; D# group 3 (zero-init for D# up to 2D)
    <8 x i32> %desc4,          ; D# group 4 (reserved, use zeroinitializer)
    i32 immarg %cpol)          ; cache policy
```

Loads or stores data between global memory and LDS using a tensor descriptor
(D#). The descriptor is split across multiple groups. GFX1250 supports up to 4
descriptor groups; ``%desc4`` is reserved for future targets and must be
zero-initialized.

Despite the absence of ``.async`` in their names, these intrinsics are
asynchronous.

All arguments must be wave-uniform.

(amdgpu-dma-memory-model)=

## Memory Model

Each dynamic instance of a DMA *instruction* ``X`` initiates a DMA *operation*
``D`` performed in the corresponding {ref}`DMA scope<amdgpu-dma-scopes>`.

- ``X`` *happens-before* ``D``.
- If ``D`` is a synchronous operation:

  - For each operation ``Y`` such that ``X`` is *program-ordered* before ``Y``,
    ``D`` *happens-before* ``Y``.

- Else ``D`` is related in *happens-before* as defined for
  {ref}`asynchronous operations<amdgpu-asyncmark-memory-model>`.

### Availability and Visibility

Each DMA operation is performed in an instance ``I`` of the corresponding
{ref}`DMA scope<amdgpu-dma-scopes>` ``S``. In addition, the user may specify a
scope ``S'`` such that ``S`` is a subscope of ``S'``. The effect of the DMA
operation can be modeled as the following sequence in LLVM IR:

```llvm
; M = max(S, S')
;
@llvm.amdgcn.make.visible(ptr %src, M)
load  ptr %src ...  ; non-atomic
store ptr %dst ...  ; non-atomic
@llvm.amdgcn.make.available(ptr %dst, M)
```

```{note}
When the DMA operation ``D`` is performed in an instance ``I`` of scope ``S``,
it is not included in any subscope instances of ``I``. This means that the
{ref}`amdgpu-availability-visibility` operations performed by ``D`` **cannot**
form an *inclusive scope* relationship with those subscopes. This requires
threads to perform additional availability and visibility operations that ensure
{ref}`location order<amdgpu-location-order>` in certain rare cases shown below.
```

#### Wavefront Scope

Consider a thread that writes to global memory and then initiates a DMA
operation that reads from the same location. The two operations are related in
*happens-before*, but the DMA operation is not contained in the "singlethread"
scope instance. The global write is not visible from the DMA read; an explicit
``make.available`` at ``lds-dma`` scope is needed to bridge this gap.

```llvm
store %val, ptr %global
call @llvm.amdgcn.make.available(%global, !"lds-dma")      ; <---
call @llvm.amdgcn.global.load.async.to.lds(%global, %lds)
call @llvm.amdgcn.asyncmark()
call @llvm.amdgcn.wait.asyncmark(0)
%val_lds = load addrspace(3) %lds
```

But such a use-case is **exceedingly rare**, where a thread writes out to global
memory, and then tries to read the same data back into LDS via a DMA operation.

The same result can be achieved using a *store-available* operation too:

```llvm
call @llvm.amdgcn.av.global.store(%global, %val, !"lds-dma")
call @llvm.amdgcn.global.load.async.to.lds(%global, %lds)
call @llvm.amdgcn.asyncmark()
call @llvm.amdgcn.wait.asyncmark(0)
%val_lds = load addrspace(3) %lds
```

A similar pattern is required when storing to global using DMA:

```llvm
call @llvm.amdgcn.global.store.async.from.lds(%global, %lds)
call @llvm.amdgcn.asyncmark()
call @llvm.amdgcn.wait.asyncmark(0)
call @llvm.amdgcn.make.visible(%global, !"lds-dma")           ; <---
%val = load ptr %global
```

The ``make.visible`` at ``lds-dma`` scope is necessary because the DMA write is
not automatically visible to the subsequent global read.

#### Workgroup Scope

Consider the case where one wave writes to global memory and a different wave in
the same workgroup initiates a DMA operation that reads from the same location.
A workgroup-scope fence can provide *happens-before* between the waves but does
not make the write available at ``lds-dma`` scope. The DMA operation is not
contained in the workgroup scope instance, so the fence's *MakeAvailable* and
the DMA do not have inclusive scopes. An explicit ``make.available`` at
``lds-dma`` scope is needed to bridge this gap.

```llvm
; wave 1
store %val, ptr addrspace(1) %global
call @llvm.amdgcn.make.available(%global, !"lds-dma")      ; <---
fence release syncscope("workgroup")

; wave 2
fence acquire syncscope("workgroup")
call @llvm.amdgcn.global.load.async.to.lds(%global, %lds)
call @llvm.amdgcn.asyncmark()
call @llvm.amdgcn.wait.asyncmark(0)
%val_lds = load addrspace(3) %lds
```

Similarly, when one wave stores to global memory using a DMA operation and a
different wave reads from the same location, an explicit ``make.visible`` at
``lds-dma`` scope is needed. The workgroup fence's *MakeVisible* cannot observe
the DMA write because the DMA is not contained in the workgroup scope instance.

```llvm
; wave 1
call @llvm.amdgcn.global.store.async.from.lds(%global, %lds)
call @llvm.amdgcn.asyncmark()
call @llvm.amdgcn.wait.asyncmark(0)
fence release syncscope("workgroup")

; wave 2
fence acquire syncscope("workgroup")
call @llvm.amdgcn.make.visible(%global, !"lds-dma")        ; <---
%val = load ptr addrspace(1) %global
```

### Implementation Details

[This section is informational.]

1. On GFX9 target, the compiler ignores the calls to
   `@llvm.amdgcn.make.available(!"lds-dma")` and
   `@llvm.amdgcn.make.visible(!"lds-dma")`. The LDS DMA implementation on GFX9
   sees the same state of memory as the requesting thread.
2. On GFX1250, the compiler emits a cache write-back or invalidate at
   `SCOPE_SE` for `@llvm.amdgcn.make.available(!"lds-dma")` and
   `@llvm.amdgcn.make.visible(!"lds-dma")` respectively.
