(amdgpu-dma-operations)=

# AMDGPU DMA Operations

## Introduction

DMA (or "Direct Memory Access") operations transfer data between different kinds
of memory directly without occupying registers in the invoking wave. They are
usually {ref}`asynchronous<amdgpu-async-operations>`, and require the user to
explicitly track completion using {ref}`asyncmarks<amdgpu-async-operations>`.

All DMA operations support the same cache modifiers as ordinary load/store
operations from registers. They cannot be performed atomically.

## Legacy DMA Intrinsics

| Intrinsic family | Supported targets | Required target property |
| --- | --- | --- |
| `@llvm.amdgcn.load[.async].to.lds.pN` | GFX9, GFX10 | `vmem-to-lds-load-insts` |
| `@llvm.amdgcn.global.load[.async].lds` | GFX9, GFX10 | `vmem-to-lds-load-insts` |
| `@llvm.amdgcn.raw[.ptr].buffer.load[.async].lds` | GFX9, GFX10 | `vmem-to-lds-load-insts` |
| `@llvm.amdgcn.struct[.ptr].buffer.load[.async].lds` | GFX9, GFX10 | `vmem-to-lds-load-insts` |

Most of these intrinsics have a synchronous counterpart (e.g.,
`@llvm.amdgcn.load.to.lds` for `@llvm.amdgcn.load.async.to.lds`). The
synchronous variants perform the same operation, but the compiler automatically
ensures completion before their effects are used. The asynchronous variants use
{ref}`asyncmarks<amdgpu-async-operations>` to track completion.

**Cache Policy**

Cache policies are applied to the global accesses performed as part of the DMA
operation. Setting bit 31 of the `aux`/`cpol` argument marks them as volatile.
Attaching `!nontemporal` metadata to the intrinsic call marks them as
nontemporal.

### Flat/Global Addressing

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

### Buffer Addressing

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

## LDS DMA Intrinsics

| Intrinsic family | Supported targets | Required target property | Wave size requirement |
| --- | --- | --- | --- |
| `@llvm.amdgcn.global.load.async.to.lds.b<N>` | GFX12.5x, GFX13 | `async-load-to-lds-insts` | wave32 |
| `@llvm.amdgcn.cluster.load.async.to.lds.b<N>` | GFX12.5x | `mcast-load-insts` | wave32 |
| `@llvm.amdgcn.global.store.async.from.lds.b<N>` | GFX12.5x | `async-store-from-lds-insts` | No additional requirement |

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

**Cache Policy**

Cache policies are applied to the global accesses performed as part of the DMA
operation. Attaching `!nontemporal` metadata to the intrinsic call marks them as
nontemporal. These intrinsics do not provide a volatile flag.

## Tensor Intrinsics

| Intrinsic family | Supported targets | Required target property |
| --- | --- | --- |
| `@llvm.amdgcn.tensor.load.to.lds` | GFX12.5x | `tdm-insts` |
| `@llvm.amdgcn.tensor.store.from.lds` | GFX12.5x | `tdm-insts` |

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

The descriptor carries global and LDS addresses in integer vectors. These
intrinsics may access ordinary memory; their effects are not limited to
pointer arguments or inaccessible memory.

Despite the absence of `.async` in their names, these intrinsics are
asynchronous.

All arguments must be wave-uniform.

These intrinsics do not support volatile or nontemporal.
