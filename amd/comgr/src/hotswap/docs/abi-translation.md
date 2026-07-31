# ABI Translation -- Kernel Descriptor, Kernarg, Buffer Descriptors

> **Status:** partly implemented. The raiser declares the lifted kernarg
> segment as one opaque blob the size of the source's
> `.kernarg_segment_size` and emits a fresh kernel descriptor via the
> target backend (see `raiser.cpp` §Phase 2), which covers the 80%
> path for simple Triton kernels. Sections 3.3, 5 and G2 describe the
> hidden-argument surface as shipped; the rest of this doc is still a
> proposal -- the embedded descriptors and the gates
> that make ABI mismatches fail loudly instead of silently
> miscompiling.
>
> **Scope:** gfx1250 -> gfx950 (source -> target). The same framework
> extends to gfx942, gfx90a, gfx1100 with one ISA-feature row per
> target; nothing in this axis is gfx950-specific at the principle
> level.

---

## 1. Problem in one paragraph

The translator's output is an LLVM IR module that the stock AMDGPU
backend compiles and hands to the HSA runtime as a fresh code object.
That code object has its own kernel descriptor (KD), its own kernarg
layout, and its own user-SGPR preloaded values -- all re-synthesised by
the target backend from IR-level attributes. What it must agree with
is **the host runtime's call site**: the kernarg buffer the dispatch
packet points at, the grid and workgroup dimensions the user code
supplied, and the set of implicit arguments the HSA runtime injects.
The host is still asking for the gfx1250 kernel. The target KD has to
make the gfx950 binary answer the same question. Any field the source
kernel *reads* that the target KD cannot *populate identically* is an
ABI hole, and today we have no gate for it -- the kernel runs and
silently reads garbage.

## 2. What "translation" actually means on this axis

There are three distinct surfaces, each handled differently:

| Surface | Source state | Target synthesis | Translation mode |
|---|---|---|---|
| Kernel descriptor (64-byte AMDHSA KD) | Parsed from `.note` / `.rodata` during load | Re-emitted by target backend from IR attrs | **Attribute-preserving lower-to-IR** |
| Kernarg segment | Host-side layout, shared by both ISAs | Same -- unchanged | **Layout-preserving**: one opaque blob of the source `.kernarg_segment_size`, hidden-arg slots included, so no offset moves |
| Embedded descriptors (V#, T#) | Rare; constructed at runtime from kernargs | Same -- runtime-constructed | **Pass-through** (principle); refuse on embedded constant V# |

The first two are where the real work lives. §4 and §5 cover them.
The third is included for completeness: it is a gate, not a rewrite.

## 3. Source model -- gfx1250 (gfx12) AMDHSA KD and kernarg

### 3.1 Kernel descriptor fields the raiser cares about

Of the ~20 bitfields in the gfx1250 KD, only these five affect
correctness of the raised IR (the rest are scheduling/occupancy hints
the target backend re-derives):

- `enable_wavefront_size32` -- always set on gfx1250. Tells the raiser
  the source was compiled as wave32.
- `group_segment_fixed_size` -- LDS bytes. Must be ≤ gfx950 limit
  (163,840 B) for the kernel to fit at all.
- `private_segment_fixed_size` -- scratch bytes per lane. Implicitly
  doubles on wave64 since twice as many lanes are live; the target
  backend derives this automatically *if* we propagate the original
  per-lane value as an attribute.
- `kernarg_size` -- total kernarg bytes. Passed as
  `meta.kernargSegmentSize` and must be preserved bit-exactly (the
  host runtime's dispatch packet carries this size).
- `enable_sgpr_kernarg_segment_ptr` / the user-SGPR preload bits --
  determine which user SGPRs carry which implicit pointers (kernarg
  base, dispatch packet, queue ptr, …). gfx12 and gfx9 have different
  user-SGPR layouts. See §3.3.

Everything else (VGPR/SGPR granulated counts, register usage modes,
occupancy hints, round-mode bits) is a target-backend output, not a
translation input.

### 3.2 Kernarg layout

The kernarg segment is a flat byte buffer with the same host-side layout on
both ISAs. The raiser does not resolve `s_load_*` displacements back to
individual IR-level arguments: the lifted kernel takes one opaque
`byref([N x i8])` blob of the source's `.kernarg_segment_size`, so a source
displacement addresses the same byte in the target.

Alignment needs no translation either. The lifted blob carries `align(16)`, the
AMDGPU ABI minimum, while sources typically declare
`.kernarg_segment_align: 8`. The declared value is not what the buffer gets:
ROCclr raises it to at least 128 bytes before allocating
(`kernargSegmentAlignment_ = alignUp(std::max(kernargSegmentAlignment_, 128u),
cacheLineSize)`, `rocclr/device/rocm/rockernel.cpp`), so any alignment the
lifted kernel can ask for is already satisfied.

### 3.3 Hidden arguments

gfx12 kernels consume these hidden args:

- `hidden_block_count_{x,y,z}`
- `hidden_group_size_{x,y,z}`
- `hidden_remainder_{x,y,z}`
- `hidden_grid_dims`
- `hidden_global_offset_{x,y,z}`
- `hidden_private_base`, `hidden_shared_base`
- `hidden_default_queue`, `hidden_completion_action`
- `hidden_multigrid_sync_arg`, `hidden_hostcall_buffer`, `hidden_heap_v1`
- `hidden_dynamic_lds_size`, `hidden_queue_ptr`

**None of them is translated.** A hidden-argument read is an ordinary load of
`kernarg_base + source_offset`, correct through any pointer however computed --
aliased, rebased, spilled, runtime-indexed. Three facts:

1. The application loaded the source code object, so ROCclr sizes the kernarg
   buffer from source metadata and writes each hidden field at its declared
   offset (`WriteAqlArgAt(hidden_arguments, value, it.size_, it.offset_)`,
   `clr/rocclr/device/rocm/rocvirtual.cpp`).
2. The dispatch path rewrites `kernel_object`, `private_segment_size` and, under
   a scaled dispatch, `workgroup_size_x` / `grid_size_x` -- never
   `kernarg_address` (`rocr-runtime .../core/runtime/hotswap_aql_patch.h`).
3. The lifted kernel keeps that size: one `byref([N x i8])` blob of the source
   `.kernarg_segment_size`, with `amdgpu-no-implicitarg-ptr` set so the backend
   appends no block of its own.

Only fact 3 is ours to break, and G2 checks it after codegen.

Scaled dispatch needs no exception: ROCclr fills the geometry fields from the
application's requested block before the packet is scaled, so the buffer already
holds un-scaled source geometry.

The `byref` blob appears in the lifted object's metadata as a single
`by_value` argument with no `hidden_*` entries, so the object is correct only
when substituted into the source kernel's dispatch. This is stated on the public
API. Emitting the source `.args` instead would make the object self-describing,
at the cost of either shrinking
the blob to `implicitArgsBase` and clearing the per-field `amdgpu-no-*` (which
refuses any kernel whose hidden block does not start exactly there) or rewriting
the metadata note after codegen. Neither is needed by the HotSwap runtime path.

### 3.4 User-SGPR preload layout

gfx12 preloads user SGPRs starting at a different base than gfx9 and
with a different enable-bit encoding. `raiser.cpp:279-292` already
handles this:

```263:292:amd/comgr/hotswap/raiser.cpp
  regs.storeSGPR32(B, 2, B.CreateCall(fnWorkgroupIdX, {}, "wg_id_x"));
  // ...
  if (AMDGPU::isGFX12Plus(*mc.subtargetInfo)) {
    B.CreateStore(B.CreateCall(fnWorkgroupIdX, {}, "ttmp9_wg_id"), regs.ttmp[9]);
    // wave_id = workitem_id_x / wavefront_size (32 for gfx12)
    Value *tidForTtmp = B.CreateCall(fnWorkitemIdX, {}, "ttmp8_tid");
    Value *waveId = B.CreateLShr(tidForTtmp, B.getInt32(5), "wave_id_in_wg");
    Value *ttmp8Val = B.CreateShl(waveId, B.getInt32(25), "ttmp8_val");
    B.CreateStore(ttmp8Val, regs.ttmp[8]);
  }
```

The gfx12-vs-gfx9 user-SGPR divergence is thus already absorbed at the
*read* side. The *target KD* side (telling gfx950's KD which user SGPR
carries workgroup_id_x) is the backend's job once the IR calls the
right intrinsic. This already works.

## 4. Target model -- gfx950 AMDHSA KD re-emission

### 4.0 Target-capability dispatch (KD and kernarg-slot level)

The ABI axis participates in the same project-wide
"emit native when the target supports it, decompose / synthesise only
when it does not" principle as matrix (§5.0), async / tensor-copy, and sync
(§5.0). It is subtler here because "native emit" for ABI means "the
target backend re-synthesises the KD from IR attributes" rather than
"the handler emits a specific LLVM intrinsic". What varies by target
are the user-SGPR layout, the hidden-arg block contents, and the
`enable_wavefront_size*` bit -- each of which has a capability branch.

#### 4.0.1 Capability bits

Extend `ISAProfile` with:

| New / existing bit | Backing feature | Governs |
|---|---|---|
| `waveSize` (existing) | `FeatureWavefrontSize32` | `enable_wavefront_size32` synthesis |
| `hasGFX12UserSGPRLayout` | `AMDGPU::isGFX12Plus(STI)` (existing helper) | ttmp preload init (`raiser.cpp:284-292`) |
| `hasFlatScratchArchitected` | `FeatureArchitectedFlatScratch` | interprets `hidden_private_base` |
| `hasFlatLDSArchitected` | `FeatureArchitectedSGPRs` + flat-LDS bit | interprets `hidden_shared_base` |
| `maxGroupSegmentSize` (existing profile field) | per-ISA constant | G1 LDS-budget gate |

These are not "emit vs. decompose" in the matrix sense -- they select
between **attribute values** the backend consumes, not between
handler paths. Same mechanism, different granularity.

#### 4.0.2 Dispatch pattern

The source-side read of user SGPRs already dispatches on one of these
bits -- `raiser.cpp:284-292`:

```cpp
if (AMDGPU::isGFX12Plus(*mc.subtargetInfo)) {
  // gfx12 ttmp preload layout
} else {
  // gfx9/10/11 layout
}
```

Promote the feature query to the `ISAProfile` field above so the same
pattern generalises to every axis without re-reading the subtarget
info:

```cpp
if (ctx.sourceIsa.hasGFX12UserSGPRLayout) {
  initGFX12TtmpPreload(ctx);
} else {
  initLegacyTtmpPreload(ctx);
}
```

Hidden arguments need nothing on either branch: they are read where the
runtime wrote them (§3.3, §5), so there is no per-slot decision to make.

#### 4.0.3 Consequences for same-family retarget

gfx1251 -> gfx1250 is an identity on every capability bit in §4.0.1.
The KD re-emission is bit-identical to the source (modulo backend-
rederived fields like register counts), hidden arguments pass through
untranslated as always, and no gate fires. This is why the same-family path
collapses to "raise to IR, lower to target" with no ABI-specific
work -- the capability branches above all take the identity path.

### 4.1 What the backend gives us for free

Emitting IR with `CallingConv::AMDGPU_KERNEL`, plus:

- Formal parameters typed by the raised kernarg layout
- Intrinsic calls for `workgroup_id_*`, `workitem_id_*`, dispatch
  packet pointer, queue pointer, etc.
- `"amdgpu-flat-work-group-size"` attribute pinned to the source's
  declared size (`raiser.cpp:231-232`)

…is sufficient for the gfx950 backend to synthesise a valid KD whose
user-SGPR layout, VGPR/SGPR granulated counts, LDS size, and wave64
enable bits are all self-consistent.

### 4.2 What the backend cannot infer

Two inputs only the raiser has access to:

1. **Original per-lane scratch size.** The backend allocates scratch
   from observed spills in the lowered target function. If the source
   kernel declared scratch the backend doesn't re-derive (e.g.,
   dynamic stack-like scratch allocated via `hidden_private_base`),
   we must propagate `meta.privateSegmentFixedSize` as an attribute.
2. **LDS size commitment.** The source declared a fixed LDS size; the
   target emission must at least match it (and can be larger). The
   `"amdgpu-lds-size"` / `addr-space-cast` approach ties this up at
   IR level -- TODO below.

### 4.2.1 Scratch/private-memory translation

`scratch_*` instructions are not global-memory operations with a special
base register. They are accesses to the source kernel's per-work-item
private segment: the hardware swizzles the dword offset by lane and adds
the wave's launch-time scratch base. The launch-time allocation is requested
by the KD's `private_segment_fixed_size` plus
`compute_pgm_rsrc2.ENABLE_PRIVATE_SEGMENT`; on gfx12 the source KD may also
request `enable_sgpr_flat_scratch_init`, while gfx9/gfx942 target codegen for
LLVM private memory normally emits scratch opcodes with
`.amdhsa_enable_private_segment 1` rather than a user-SGPR flat-scratch pair.

Hotswap models this at the ABI layer by creating one addrspace(5) private
frame for the source private segment, sized from the parsed source KD. A
translated `scratch_load/store` becomes a load/store through a GEP inside
that frame. This deliberately lets the target AMDGPU backend lay the source
private frame out together with any target spills and emit the target KD's
private-segment fields. Hotswap refuses a `scratch_*` instruction when the
source KD reports `private_segment_fixed_size == 0`, because inventing
scratch backing would change the source launch ABI instead of translating it.

### 4.3 What changes at ISA boundary

| Source (gfx1250) | Target (gfx950) | Strategy |
|---|---|---|
| `enable_wavefront_size32=1` | Must be wave64 | Clear bit; emit via `CallingConv::AMDGPU_KERNEL` on wave64 subtarget -- backend does the right thing |
| gfx12 user-SGPR slots for `workgroup_id_x` | gfx9 slots | Abstracted at intrinsic level; see §3.4 |
| `private_segment_size` at 32-lane wave | At 64-lane wave | Backend re-derives; we propagate via attribute |
| `group_segment_size` ≤ 327,680 | Limit 163,840 | **Refuse** if source > 163,840 (§7) |
| `kernarg_size` arbitrary | Same bit-for-bit | One opaque `byref([N x i8])` blob of the source size; post-codegen gate refuses unless the emitted segment is the same size and declares no hidden args (G2) |
| gfx12 hidden-arg block | gfx9 hidden-arg block | **Pass-through**: read from the source-populated buffer at the source offsets; no per-arg translation (§3.3, §5) |

## 5. Hidden argument compatibility

No per-field table and no per-field class: every hidden argument is read where
the runtime wrote it (§3.3). `hidden_private_base` / `hidden_shared_base` carry
the target queue's real apertures, and `hidden_global_offset_*` carries whatever
offset the caller launched with.

A kernel that *writes* a hidden slot is still a refusal case.

## 6. Embedded descriptors

### 6.1 Buffer descriptors (V#) in .rodata

V# format changed encoding between gfx11 and gfx12. Triton and
Tensilelite kernels construct V#s at runtime from kernargs -- the V#
bits are never stored as a constant in `.rodata`. We have not observed
any embedded-constant V# in the 170-kernel corpus.

**Gate:** scan `.rodata` for 128-bit-aligned values whose bits 58..63
match a known DFMT/NFMT pattern; if found, refuse. This is a
false-positive-prone heuristic, so the gate fires only when we *also*
observe a `s_load_b128` reading that address in the raised IR.

### 6.2 Runtime-constructed V# -- no translation required

The kernel assembles a V# from base+size+stride+format bits via
`s_mov_b32` / `s_lshl_b64` / `s_or_b32`. The target backend re-lowers
`buffer_load_dword` on gfx950's V# format bit layout, using whatever
the source code computed. The format-code field is the only hazard:

- gfx9 DFMT/NFMT are a (4-bit, 3-bit) split in bits [53..59].
- gfx12 folded these into a larger `OOB_SELECT` + `FORMAT` at bit [53..62].

If the source code constructs a V# with a format code that is valid
on gfx12 but meaningless on gfx9 (e.g., `FORMAT = 0x40` indicating
structured buffer), the `buffer_load` on gfx950 will return garbage.

**Gate at the VALU handler:** when the V#'s format-computing
instructions fold to a constant, the raiser flags unrecognised codes
and refuses. When the format is dynamic, emit a runtime assert on
first use (optional; off by default).

### 6.3 Image descriptors (T#)

Not used by any captured corpus kernel. Treat as refusal until a real
case appears. The scan in §6.1 extends naturally.

## 7. Principled fail-loudly gates

All gates run at raise time, before any target lowering. Each gate's
failure yields a `RaiseFailure` with a distinct reason code so the
loader's rejection report is actionable.

### G1 -- LDS budget (startup on the target subtarget)

```
if (meta.groupSegmentFixedSize > targetIsa.maxGroupSegmentSize)
  return RaiseFailure::ldsOverBudget(meta.groupSegmentFixedSize,
                                     targetIsa.maxGroupSegmentSize);
```

### G2 -- Lifted kernarg segment matches the source's (per-kernel, post-codegen)

`checkLiftedKernargSegment` refuses unless the emitted `.kernarg_segment_size`
equals the source's and the emitted metadata declares no `hidden_*` argument.
The runtime sized the buffer from source metadata and the dispatch path does not
resize it, so either mismatch means the backend added an implicit-argument block
the runtime did not allocate. This is the whole hidden-argument gate (§3.3).

Lit: `lifted_kernarg_segment.s` (gfx1151 and gfx942) pins the emitted metadata;
`hidden_arg_*.s` pin that the awkward pointer shapes lower to ordinary loads.
The refusal itself is unreachable from a code object and is unit-tested.

### G3 -- User-SGPR compatibility (startup)

For each user-SGPR enable bit the source KD sets, verify the target
ISA supports the same intrinsic (e.g., `llvm.amdgcn.dispatch.ptr`
exists across all AMDGPU subtargets, but
`llvm.amdgcn.implicit.buffer.ptr` is a gfx<11 artefact). Startup
verification over a fixed mapping table.

### G4 -- Embedded descriptor refusal (per-kernel)

§6.1 scan; refuse on any hit.

### G5 -- Kernarg alignment (per-kernel)

For each argument, verify `byteOffset % byteSize == 0`. The host
runtime pads to this anyway; a mismatch means the source metadata
is wrong, and the kernel would have UB already on gfx1250 -- refuse
rather than inherit the bug.

## 8. Decision procedure (per kernel)

```
raise(bytes, source_isa, target_isa, meta):
  run G1, G3 at startup (once per (source_isa, target_isa) pair)
  for each kernel:
    run G5 immediately
    raise to IR
    post-raise: run G4; post-codegen: run G2
    if any gate fires: return RaiseFailure
    else: emit IR; backend synthesises target KD from attrs + IR
```

No partial accepts. No "best effort" hidden-arg reads. If the kernel
reaches anything ambiguous, we reject it loudly.

## 9. What this doc does *not* cover

- **Flat-scratch -> buffer-scratch lowering.** That is a memory-ops
  concern, folded into the existing `handleFLAT` handler (gfx9 buffer
  scratch is the native flat-addrspace lowering on CDNA). The SemOps
  for flat/global/scratch loads already exist in the FLAT handler;
  cross-ISA flat-scratch->V# rewrite happens at the handler level, not
  at ABI level.
- **VGPR/SGPR count inflation.** The target backend owns this. If IR
  spills past target register budget, that is a target-backend
  failure (compile-time error), not an ABI concern.
- **Occupancy hints.** Intentionally not pinned (`raiser.cpp:234-241`
  documents why). Performance, not correctness.

## 10. Engineering tasks

In dependency order, cheapest first.

### T1 -- Add `meta.groupSegmentFixedSize` plumbing + G1

One struct field, one check in `raiseToIR` before Phase 2. 30 LoC.

### T2 -- Hidden-arg compatibility table + G2 -- **done, and smaller than planned**

Superseded. The planned table (an entry per `(arg_name, source_isa,
target_isa)`, consulted at each kernarg-load resolution) turned out to be
unnecessary: the runtime populates every hidden field at the source
offset the kernel reads, so no field needs a class and no load needs
resolving. What shipped is the §3.3 rule plus the one post-codegen
segment-size check in G2.

### T3 -- Embedded-descriptor scan (G4)

Post-raise pass over the IR walking `.rodata`-backed constants;
reject on format-code anomalies. ~80 LoC.

### T4 -- Scratch/private-segment propagation

Forward `meta.privateSegmentFixedSize` into `RaiseContext` and model real
`scratch_*` instructions as addrspace(5) private-frame accesses. The target
backend then emits the target KD's private segment request; unsupported
subcases refuse with source KD scratch fields in the structured diagnostic.

### T5 -- User-SGPR compatibility table + G3

Table of (enable-bit, available-intrinsic) × (source_isa, target_isa).
Startup check against both subtargets. ~60 LoC.

## 11. Open design questions

1. **Where does the source KD live post-load?** Currently the loader
   parses `meta.*` from the ELF note block. Do we also need the raw
   64-byte KD bytes as input to the gates, or is the parsed `meta` a
   sufficient model? Current answer: parsed `meta` is sufficient
   because every field in §3.1 is already surfaced there. The raw KD
   is discarded.
2. **Hostcall and multigrid-sync on cross-ISA translation.** Identity
   entries in §5 assume the hostcall ABI is ISA-stable. If a future
   runtime bumps the hostcall format between gfx1250 and gfx950, the
   table grows a new class: `host-incompatible` (refuse). No action
   today.
3. **Embedded V# detection false positives.** §6.1's scan is a
   heuristic. If it turns out to produce spurious refusals on any
   real kernel, move the check to first-use of the constant by a V#
   consumer (so we only flag V#s we can prove are V#s). This is a
   straightforward refinement; today's corpus has no constants
   matching the pattern at all, so we ship §6.1 as-is.
4. **Multi-target from one source.** If the same gfx1250 binary will
   be retargeted to *both* gfx942 and gfx950 in the same process, do
   the gates need per-target caching? Today the raiser runs per
   (source, target) pair, so the gates run fresh each time. No
   correctness impact; a later optimisation if needed.

## 12. Cross-axis relationship -- capability dispatch

§4.0 is the ABI-axis instance of the project-wide "emit native when
the target supports it, synthesise only when it does not" principle.
See `target-capability-dispatch.md` for the shared design and the
open implementation question (does LLVM already expose per-feature /
per-intrinsic availability we can reuse for the user-SGPR compatibility
table?). Hidden arguments are pass-through, so no table is needed for them
(sec. 3.3, 5).
