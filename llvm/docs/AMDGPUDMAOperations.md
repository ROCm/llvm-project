(amdgpu-dma-operations)=

# AMDGPU DMA Operations


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
`@llvm.amdgcn.load.to.lds` for `@llvm.amdgcn.load.async.to.lds`). The
synchronous variants perform the same operation, but the compiler automatically
ensures completion before their side-effects are used.

GFX9 DMA instructions implement volatile (via `aux/cpol` bit 31) and
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

The `%lds_base` pointer must be wave-uniform.

The intrinsics differ in two orthogonal ways:

- **raw** vs **struct**: The `struct` variants add a `%vindex` argument for
  indexed buffer addressing.
- **ptr** vs non-ptr: The `ptr` variants use `ptr addrspace(8)` for the
  buffer resource descriptor; the non-ptr variants use `<4 x i32>`.

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
    [i32 %mask])                ; workgroup broadcast mask, cluster variants only (wave-uniform, in M0)
```

The bit-size encoded in the name can be 8, 32, 64 or 128.

Loads data from global memory to LDS. The `%offset` is applied to both the
global and LDS addresses.

(amdgpu-cluster-broadcast-dma)=

**Cluster Broadcast**

The `cluster` variants broadcast data into the LDS of several workgroups in a
{ref}`workgroup cluster <amdgpu-clusters>`. Every workgroup in a cluster is
assigned an index, and the LDS of each workgroup in the cluster is addressable
from any workgroup in the same cluster.

The `cluster` variants add a `%mask` argument, passed in `M0`, that selects the
set of destination workgroups within the invoking workgroup's cluster and
controls the request timeout. The mask is wave-uniform. Its bits are:

- Bits `[15:0]` are the *workgroup broadcast mask*. Bit `i` corresponds to the
  workgroup with cluster index `i`: if the bit is set, that workgroup receives a
  copy of the loaded data in its LDS; if it is clear, that workgroup does not. A
  cluster therefore contains at most 16 workgroups for the purpose of this
  operation. The invoking workgroup receives the data only if its own bit is
  set.
- Bit `[16]` selects the timeout behavior. When clear, the standard timeout is
  used. When set, an *early timeout* is used: as soon as the L2 cache supplies
  the data it is returned to whichever waves have already issued their requests,
  and any waves that issue their requests later generate a new L2 request.

The operation performs a single load from global memory and writes a copy of the
loaded data into the LDS of *every* workgroup selected by the workgroup
broadcast mask. The data is written at the same LDS location (`%lds_base` plus
`%offset`) in each destination workgroup. This makes it possible to broadcast a
value from global memory into the LDS of multiple workgroups with one DMA
operation.

To be combined into a single broadcast, every destination workgroup selected by
the mask is expected to issue the same request: exactly one wave per workgroup
must perform the operation, and all such waves must supply the same `M0` value
(the same broadcast mask and address). If these conditions are not met, the
requests are not guaranteed to be combined and may instead be serviced
individually, as ordinary per-workgroup loads.

Because a *cluster broadcast* writes to the LDS of workgroups other than the
invoking one, its side-effects become available at "cluster" scope rather than
"workgroup" scope. See {ref}`amdgpu-memmodel-addrspace3` for details.

The `cluster` variants are only available on targets that have the multicast
load instructions (the `mcast-load-insts` subtarget feature, currently GFX1250).
Using them on any other target is not supported. Note that this feature is
distinct from workgroup cluster support itself: a target may support clusters
without providing these broadcast DMA instructions.

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
descriptor groups; `%desc4` is reserved for future targets and must be
zero-initialized.

Despite the absence of `.async` in their names, these intrinsics are
asynchronous.

All arguments must be wave-uniform.
