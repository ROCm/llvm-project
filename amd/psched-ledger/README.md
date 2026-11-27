# Scheduling ledger pass plugin (proof of concept)

`psched-ledger` is the IR half of the scheduling ledger for AMDGPU kernels. A
ledger is a JSON file of compiler advice derived from profiles. The source
decides what is computed; a ledger only selects among forms the compiler can
show to be equivalent, so it can change speed but not results.

This plugin applies, per function entry of a ledger:

- `attrs`: scheduling function attributes (a fixed whitelist),
- `hints`: existing scheduling intrinsics in the blocks that call a named
  intrinsic,
- `choose`: instruction choices (`mul24`, `wmma_bias`, `byte_negate`,
  `bit_spread`), each applied only where the equivalence is proven from the IR.

The per-region part of a ledger (instruction order and latency) is applied by
the backend in `llvm/lib/Target/AMDGPU/AMDGPUSchedLedger.cpp`.

The plugin must be built against the same LLVM build as the compiler that
loads it. The simplest way is to build it with the compiler, so that `ninja`
keeps the two in step. Add to the LLVM configure line:

```sh
-DLLVM_EXTERNAL_PROJECTS=psched-ledger \
-DLLVM_EXTERNAL_PSCHED_LEDGER_SOURCE_DIR=/path/to/llvm-project/amd/psched-ledger
```

and build the target `psched-ledger` along with `clang` and `lld`. The plugin
is then `<build>/lib/psched-ledger.so`.

It can also be built standalone against a finished build:

```sh
cmake -S amd/psched-ledger -B build-psched-ledger \
  -DLLVM_DIR=/path/to/llvm-build/lib/cmake/llvm -DCMAKE_BUILD_TYPE=Release
cmake --build build-psched-ledger
```

Use it by loading the plugin and naming the ledger files or directories:

```sh
PSCHED_LEDGER=/path/to/ledgers clang++ -x hip ... \
  -fpass-plugin=/path/to/llvm-build/lib/psched-ledger.so
```

`PSCHED_LEDGER` must also be set when device code is generated at link time,
because the backend reads it there. `PSCHED_VERBOSE=1` reports every entry
applied. No ledger entry depends on debug info.
