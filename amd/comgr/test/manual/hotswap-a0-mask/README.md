# GFX1250 A0 HotSwap Mask Smoke

This directory contains manual B0-code-object inputs for testing the gfx1250
B0-to-A0 hotswap mask workarounds on an A0 machine.

The rewrite path should be invoked with explicit stepping features:

```bash
amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+
amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific-
```

`b0_mask_rewrite_canary.s` is dispatchable. It stores a sentinel through one
kernarg pointer and branches around the B0-only mask-test sites. The cold block
still statically exercises:

- `tensor_load_to_lds` D# Group 1 mask clear.
- SADDR `cluster_load*` M0 save, low-mask clear, and restore.
- Off-form `cluster_load` demotion to `global_load`.

Run the rewrite checks with in-tree tools:

```bash
CLANG=/path/to/build/bin/clang \
HOTSWAP_REWRITE=/path/to/build/bin/hotswap-rewrite \
LLVM_OBJDUMP=/path/to/build/bin/llvm-objdump \
LLVM_READELF=/path/to/build/bin/llvm-readelf \
./run_rewrite_checks.sh
```

To also dispatch the rewritten canary on an A0 machine:

```bash
c++ -std=c++17 run_hsaco_canary.cpp -I/opt/rocm/include \
  -L/opt/rocm/lib -lhsa-runtime64 -o run_hsaco_canary

RUNNER=$PWD/run_hsaco_canary ./run_rewrite_checks.sh
```

The runner dispatches `b0_mask_rewrite_canary` from the rewritten HSACO and
expects output word `0xb0a00001`.
