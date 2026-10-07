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

## Wave projection

When translating wave32 to wave64, WaveNative places two consecutive source
waves in one target wave, one in each 32-lane half. Replication places two
copies of a single source wave in those halves.

Source scalar instructions execute regardless of source EXEC, so WaveNative
runs their generated code with all target lanes enabled. For source vector
instructions that obey EXEC, register writes and memory accesses remain
controlled by the source wave's EXEC mask.

The hardware initializes EXEC to mark the lanes assigned to workitems by the
launch. A workgroup with 48 workitems starts with 48 active lanes in a 64-lane
target wave. The other 16 lanes can participate in internal calculations, but
do not correspond to launched workitems. We record which lanes were active at
kernel entry before enabling the whole wave and use that record to mask source
vector memory instructions that obey EXEC.

Source kernels may temporarily enable initially inactive lanes for wave-wide
calculations, such as scans with zero-filled unused lanes. This lowering
currently does not support such EXEC expansion. Every EXEC write must be
proven to enable only lanes that were active at kernel entry.

A source vector comparison can test whether each workitem's index is less than
a bound. All 64 target lanes participate in the ballot. Each lane contributes
a one only if its source EXEC bit is set and its index is below the bound;
otherwise it contributes zero. Target lanes 0-31 use the lower 32 bits of the
result, and lanes 32-63 use the upper 32 bits. This selects the result for each
source wave without disabling either half of the target wave.

Reading source lane 7 selects target lane 7 in the lower half and target lane
39 in the upper half.

This lowering requires the two source waves to agree on scalar branch
decisions so they follow the same control-flow path. Scalar loads execute
regardless of source EXEC. We require the same address across the target wave
because lanes without source workitems may not have a valid address of their
own.

The two source waves share the target wave's hardware control registers, so
writes to those registers must agree. Interrupt messages and halt instructions
are refused. Two source waves can each send a message, but executing the
instruction once in the target wave would send only one. Halting the target
wave would stop both source waves.

With nonzero source EXEC, WMMA uses inputs from all 32 source lanes, including
lanes whose EXEC bits are clear. This lowering requires EXEC at WMMA to match
its value at kernel entry, so a source wave containing workitems cannot reach
WMMA with EXEC zero. The entry mask may describe a partially filled wave; it
does not have to be all ones.

## Standalone development build

```bash
cmake -S amd/comgr/src/transpiler/raiser -B build-transpiler \
  -DLLVM_DIR=$PWD/build/lib/cmake/llvm
ninja -C build-transpiler
ctest --test-dir build-transpiler -L transpiler
```
