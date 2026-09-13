# HotSwap

HotSwap is COMGR's AMDGPU code-object transpiler: it raises a compiled code
object to LLVM IR, re-lowers it through the stock AMDGPU backend for a different
target ISA, and relinks the result into a single merged HSACO.

The byte-level rewrite path that previously lived here — in-place ELF/MC
patching and entry trampolines, backing `amd_comgr_hotswap_rewrite` and
`amd_comgr_hotswap_rewrite_with_options` — has been removed. Those two API
entry points remain declared and exported for binary compatibility but always
fail; see `src/comgr-hotswap-stubs.cpp`. They will be dropped in Comgr v4.0.

## Directory layout

```
hotswap/
  common/     Small shared pieces: the KernelMeta ABI model and HotswapError.
  loader/     Code-object metadata loader: ELF + MsgPack note + kernel
              descriptor parse, supplying the .text section to the decoder.
  decoder/    Per-ISA AMDGPU MC stack and the canonical-op identity the raiser
              dispatches on.
  raiser/     The transpiler proper: raises a code object to LLVM IR and
              re-lowers it for a different target ISA.
```

All four are OBJECT libraries (`hotswap::common`, `hotswap::loader`,
`hotswap::decoder`, `hotswap::raiser`) whose translation units land directly in
`amd_comgr.so`, so they can call comgr helpers without a layering inversion.
They are built only under `COMGR_ENABLE_HOTSWAP_TRANSPILE`, which is OFF by
default: the raiser, loader, and decoder consume AMDGPU target-private headers
an install-tree-only build does not expose.

There is no public C entry point for the transpiler yet; it is reachable from
the `hotswap_transpile_cli` test driver used by the `test-lit/hotswap/raiser`
suite.

## Standalone development build

```bash
cmake -S amd/comgr/src/hotswap/raiser -B build-hotswap \
  -DLLVM_DIR=$PWD/build/lib/cmake/llvm
ninja -C build-hotswap
ctest --test-dir build-hotswap -L transpiler
```
