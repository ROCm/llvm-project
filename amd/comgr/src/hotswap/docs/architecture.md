# Hotswap: Architecture Overview

## What Hotswap is

Hotswap is a binary translator for AMD GPU code objects. It takes a compute
kernel compiled for one GPU instruction set and produces an equivalent code
object for another, so a kernel can run on hardware it was not built for.

Currently supported ISAs:

- Source ISA: gfx1250
- Target ISA: gfx942, gfx950

Hotswap is a compiler-side library, built into COMGR and exposed as
`amd_comgr_hotswap_transpile`. It runs at code-object load time: given a kernel
whose ISA does not match the device, it returns a translated code object to load
in its place.

## Lift to IR, then re-lower

Hotswap disassembles each source kernel, reconstructs LLVM IR, and uses the
AMDGPU backend to generate code for the target ISA.

Going through IR reuses the backend that already targets the destination:
instruction selection, register allocation, scheduling, and kernel-descriptor
emission all come for free. Hotswap only has to produce correct IR. Most of the
work is in recovering faithful IR from machine code whose meaning depends on the
source hardware's execution model.

## The pipeline

The translation stages, end to end:

```
  code object (ELF / HSACO, source ISA)
        │  read kernel code and metadata
        ▼
  ┌─────────────────────────────────────────────────────────┐
  │  per kernel: raiseToIR                                  │
  │    disassemble the source instructions                  │
  │    recover control flow                                 │
  │    translate source semantics into LLVM IR              │
  │    reject unsupported or unsafe cases                   │
  └─────────────────────────────────────────────────────────┘
        │  LLVM IR (one function per kernel)
        ▼
  optimize and lower with LLVM's AMDGPU backend
        │  target-ISA kernel objects
        ▼
  link the kernel objects
        │
        ▼
  code object (target ISA) + structured result / diagnostics
```

Orchestration is in [`pipeline.cpp`](../pipeline.cpp). The lift itself is
[`raiser.cpp`](../raiser.cpp).

The recovered IR has a few defining traits:

- **One function per kernel**, using the AMDGPU kernel calling convention.
- **Source registers are modeled as temporary memory.** `AllocaRegFile` creates
  an `alloca` for each source-ISA register slot. Instruction handlers translate
  register reads and writes into loads and stores on those slots. After all
  instructions have been translated, `PromoteMemToReg` replaces the allocas
  with SSA values and phi nodes. They are translation scaffolding, not target
  physical registers.
- **Control flow** is recovered as an LLVM basic-block graph from decoded branch
  targets. Indirect branches through a computed program counter are resolved by
  a dedicated static analysis.
- **Instruction dispatch** uses LLVM's instruction tables: the disassembler
  produces `MCInst`s, each is mapped to an architecture-neutral opcode identity,
  and routed to a format handler by its TableGen instruction flags.

Running the full backend per kernel is expensive, so two properties keep the
cost down:

- **Lazy, per kernel.** A code object can carry hundreds of kernels of which a
  program calls only a few, so a kernel is translated the first time it is
  dispatched rather than all up front.
- **Cached.** A translated kernel is keyed on the source object and the target
  ISA and reused, so the cost is paid once and skipped on later loads.

## Overview: Three problems beyond opcode remapping

If the source and target agreed on the execution model, the instruction set, and
the interface a kernel presents to the runtime, this would be opcode remapping.
They don't, and three problems account for the complexity.

### 1. The execution model differs (wave size)

An AMD GPU executes a *wavefront* of work-items in lockstep under a bitmask
called EXEC, one bit per lane. gfx1250 wavefronts are 32 lanes wide. gfx942 and
gfx950 wavefronts are 64. The source machine code bakes the 32-lane assumption
into three observable places: the width of the EXEC mask, the semantics of every
cross-lane instruction (lane shuffles, permutes, ballots, lane-id counts), and
the bit patterns of lane-id and workgroup-rank arithmetic. Translating across
the width gap requires two things.

**Modeling the source execution mask explicitly.** The source uses EXEC to
switch individual lanes on and off around side effects -- stores, atomics,
cross-lane reads. LLVM IR describes the computation from the point of view of
one work-item; it does not explicitly represent the wavefront or its EXEC mask.
Hotswap therefore tracks the source EXEC mask as a value and guards each
per-lane side effect with an explicit "is this lane active?" branch keyed on the
lane's EXEC bit. Wave-uniform state (scalar registers) is written
unconditionally. Per-lane state (vector registers, memory) is written only under
that guard. When the mask is provably all-ones the guard folds away, so uniform
code pays nothing. In the source this mechanism is named *SIMT Predicated
Execution* (SPE). Before translating, Hotswap verifies that every instruction
which writes EXEC uses this model and refuses the kernel otherwise.

**Deciding what the extra lanes do.** Going from 32 source lanes to 64 target
lanes raises the question of how source semantics map onto the wider wave. That
mapping is a policy, called a *projection*, and it is chosen per kernel:

- The default for the 32->64 case treats the target wave as two independent
  source waves stacked together (lanes 0-31 and 32-63), each with its own
  modeled execution mask. The whole kernel body runs with hardware EXEC set to
  all ones so that wide, all-lanes-must-participate operations (notably matrix
  instructions) work, while per-lane side effects stay gated by the modeled
  mask. In the source this is `WaveNativeProjection`.
- Modulo replication executes each source-lane position on two target lanes, `i`
  and `i + 32`, which consult the same bit of the modeled execution mask. It is
  valid when kernel behavior does not distinguish between the two copies, as in
  independent per-work-item computation. Lane-position-dependent behavior or
  conflicting shared-state effects require an explicit rewrite; without one,
  the kernel is refused.
- Scaled modulo replication doubles the launch size along x and makes the upper
  half of each target wave repeat the lower half's logical work-items. Hotswap
  maps their work-item IDs back to the source range, and the loader receives the
  required scale factor.
- Thread-loop projection is a fallback for wave-sensitive kernels that the
  other projections cannot represent safely. It executes the kernel body once
  per source wave, remapping work-item IDs on each iteration. Hotswap selects it
  only for specific classifier-approved cases. Kernels using workgroup barriers
  or LDS are unsupported by this approach.

A projection is only correct for kernels whose observable behavior does not
depend on the absolute lane count.

For example:

- **Wave-size independent:** a pointwise kernel whose work-items access only
  their own elements.
- **Wave-size sensitive:** a kernel that selects work by lane position or
  communicates across lanes. These operations need an explicit rewrite.

Hotswap checks known hazards with instruction classifiers and targeted data-flow
analysis. For example, under modulo replication, both copies of a source lane
could issue a returning compare-and-swap and observe different old values.
Hotswap detects these atomics and refuses them because it has no rewrite that
preserves both results.

Packing two source waves also assumes that neither must advance independently of
the other. For example, if one source wave spins until another writes a flag,
the separately scheduled source waves can make progress. Packed into one
lockstep target wave, the waiting half may prevent the producing half from
running and deadlock. The current analysis does not prove the absence of
arbitrary cross-wave dependencies. When a detected hazard has no safe projection
or rewrite, Hotswap emits no target code for that kernel and returns a
structured translation failure. In a whole-code-object COMGR request, that
per-kernel failure fails the request.

### 2. Some instructions have no target equivalent

Whole instruction classes exist on the source but not the target, and must be
re-expressed in terms the target has:

- **Matrix multiply.** gfx1250 has wave-32 matrix instructions (WMMA). The
  targets have wave-64 matrix instructions (MFMA) that distribute matrix
  elements across lanes differently. Hotswap lowers WMMA to MFMA by
  redistributing operands across lanes (via lane-permute reads), splitting the
  contraction dimension where needed, running the MFMA, and gathering the
  result back into the source layout.
- **Microscaled FP4 conversion.** A gfx1250 instruction dequantizes eight packed
  4-bit floats with a shared scale. On targets without it, Hotswap synthesizes
  the dequant as exact integer bit arithmetic on the float fields.
- **Tensor DMA.** gfx1250 has a hardware unit that moves tiled tensors between
  global memory and LDS. On targets without Tensor DMA, Hotswap links an
  embedded implementation of the tensor move into the translated module and
  inlines it. Translation fails if that helper was not included in the Hotswap
  build.
- Plus the smaller cases: scalar-float, dual-issue instruction pairs, split vs.
  combined wait counters, and the differing flat/scratch and buffer formats.

Each instruction is handled by a dedicated handler. Depending on the class, the
translation is exact, an approximation, or a refusal.

### 3. The kernel's boundary with the runtime

The runtime launches a kernel through a descriptor and a kernarg buffer. The
descriptor determines which scalar registers are preloaded with values such as
the kernarg and dispatch pointers and workgroup IDs. The kernarg buffer contains
both explicit arguments and ABI-defined hidden launch values.

**Preloaded registers.** The kernel descriptor decides which scalar registers
hold the kernarg pointer, dispatch pointer, workgroup IDs, and the rest, and the
set and order shift across generations (e.g. gfx1250 adds argument preloading
and a wider count field), so the kernarg pointer is not reliably register 0.
Hotswap uses the descriptor-derived layout to identify the corresponding
source-register allocas. At function entry it stores values from AMDGPU
intrinsics, or dwords loaded from their source kernarg offsets, into those
allocas. The handlers then read them as the source machine code would. After
promotion to SSA, the target backend assigns physical registers independently.

**Kernarg.** Explicit and hidden arguments retain their source byte offsets. The
translated kernel receives the source dispatch's kernarg buffer unchanged and
loads values directly from it.

The output descriptor is regenerated by the backend from the lifted function.
Hotswap sets the function's attributes so the regenerated argument-segment, LDS,
and workgroup sizes match the source's, and refuses if the target lowering needs
more scratch memory than the source kernel used, because that would need
adjustment of the launch parameters, which is not supported at this time.

## Planned sections

- **From machine code to IR** -- decoding, the register model, and the
  per-instruction handlers.
- **ABI reconstruction** -- how the kernel descriptor and argument layout are
  read and rebuilt.
- **Matrix and low-precision lowering** -- WMMA to MFMA, tensor moves, and FP4
  conversion.
