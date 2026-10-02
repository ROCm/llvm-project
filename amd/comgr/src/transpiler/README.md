# Transpiler

Transpiler is COMGR's AMDGPU code-object transpiler: it raises a compiled code
object to LLVM IR, re-lowers it through the stock AMDGPU backend for a different
target ISA, and relinks the result into a single merged HSACO.

The byte-level rewrite path that previously lived here — in-place ELF/MC
patching and entry trampolines, backing `amd_comgr_hotswap_rewrite` and
`amd_comgr_hotswap_rewrite_with_options` — has been removed. Those two API
entry points remain declared and exported for binary compatibility but always
fail; see `src/comgr-hotswap-stubs.cpp`. They will be dropped in Comgr v4.0.

## Directory layout

```
transpiler/
  common/     Small shared pieces: the KernelMeta ABI model and TranspilerError.
  loader/     Code-object metadata loader: ELF + MsgPack note + kernel
              descriptor parse, supplying the .text section to the decoder.
  decoder/    Per-ISA AMDGPU MC stack and the canonical-op identity the raiser
              dispatches on.
  raiser/     The transpiler proper: raises a code object to LLVM IR and
              re-lowers it for a different target ISA.
```

All four are OBJECT libraries (`transpiler::common`, `transpiler::loader`,
`transpiler::decoder`, `transpiler::raiser`) whose translation units land directly in
`amd_comgr.so`, so they can call comgr helpers without a layering inversion.
They are built only under `COMGR_ENABLE_TRANSPILER`, which is OFF by
default: the raiser, loader, and decoder consume AMDGPU target-private headers
an install-tree-only build does not expose.

There is no public C entry point for the transpiler yet; it is reachable from
the `transpile_cli` test driver used by the `test-lit/transpiler/raiser`
suite.

## Wave projection selection

The raiser selects one projection per kernel without changing the launch:

- Equal source and target wave sizes use `ReplicationProjection`, which maps
  one source wave to one target wave.
- gfx1250 wave32 to gfx942 wave64 uses `WaveNativeProjection` as a candidate.
  Consecutive source waves occupy target lanes 0-31 and 32-63. Each lane retains
  its workitem ID and its source wave's 32-bit EXEC, VCC, and scalar masks.
- Other wave-size changes receive an `unsupported-wave-projection` refusal.

WaveNative enables full hardware EXEC between predicated operations so source
scalar instructions and lane collectives can execute independently of source
EXEC. Vector writes and ordinary memory operations remain predicated by modeled
source EXEC and the lane's activity at kernel entry, including partial waves.
Ballots and lane-indexed operations stay within each source wave.

Packing is accepted only when its semantic requirements can be established:

- Source scalar control flow, scalar memory addresses, and hardware register
  writes must be uniform across the target wave. Two packed source waves cannot
  independently execute target scalar control flow or hardware effects.
- Every EXEC write must be a subset of the source EXEC at kernel entry.
- WMMA requires that same kernel-entry EXEC value, so its lowering does not
  depend on the behavior of matrix instructions under modified EXEC.
- Per-wave hardware effects such as interrupt messages and halts are refused
  at the source instruction; packing would change their execution count.

Handlers record operand requirements in `RaiseContext`. After register promotion
exposes SSA data flow, the raiser checks uniformity at definitions and uses,
EXEC containment, and SSA identity for kernel-entry EXEC requirements. An
unproven requirement is a structured refusal, even if another analysis could
prove it. These checks are selected explicitly by the projection's validation
policy, independently of its hardware EXEC scaffolding.

In the selected same-wave path, scalar entry values are target-wave uniform.
Scalar operations preserve uniformity, and native ballots and lane reads
produce uniform values when vector data enters scalar state. This path needs
no additional packing proof. WaveNative's ballot slices and source-wave lane
reads can instead produce different scalar values in the two halves, so their
uses require validation. This distinction does not depend on whether hardware
EXEC is full. Instruction-specific requirements such as zero address bits apply
to both paths.

A failed requirement ends translation of the kernel. The raiser does not retry
with another projection, replicate dispatch, or reshape the launch.

## Standalone development build

```bash
cmake -S amd/comgr/src/transpiler/raiser -B build-transpiler \
  -DLLVM_DIR=$PWD/build/lib/cmake/llvm
ninja -C build-transpiler
ctest --test-dir build-transpiler -L transpiler
```
