# AMD Hotswap Transpiler

`amd/comgr/hotswap` contains the compiler-side AMDGPU binary translation library used
by the COMGR hotswap-transpile prototype. It lifts AMDGPU code objects to LLVM
IR, lowers them through the in-tree AMDGPU backend for a target ISA, and returns
a translated HSACO.

This directory is compiler-owned and is consumed by COMGR. Runtime loaders
should call COMGR instead of including hotswap internals directly.

## Build Modes

### LLVM External Project

The intended in-tree development configuration is to build COMGR as an LLVM external project with hotswap enabled:

```bash
cmake -S llvm -B <build-dir> -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="clang;lld" \
  -DLLVM_TARGETS_TO_BUILD="X86;AMDGPU" \
  -DLLVM_EXTERNAL_PROJECTS="device-libs;comgr" \
  -DLLVM_EXTERNAL_DEVICE_LIBS_SOURCE_DIR=<llvm-project>/amd/device-libs \
  -DLLVM_EXTERNAL_COMGR_SOURCE_DIR=<llvm-project>/amd/comgr \
  -DCOMGR_ENABLE_HOTSWAP_TRANSPILE=ON \
  -DLLVM_USE_LINKER=lld
```

Build the focused targets used by the COMGR smoke test:

```bash
cmake --build <build-dir> \
  --target hotswap-transpiler amd_comgr hotswap-transpile \
           FileCheck llc llvm-mc lld llvm-readelf llvm-objdump \
  --parallel 16
```

### TheRock Compiler/COMGR Build

TheRock builds hotswap as part of COMGR. The hotswap tree is reached from
`amd/comgr/CMakeLists.txt`; it should not be listed as a separate LLVM external
project.

Before configuring TheRock, make sure the TheRock `compiler/amd-llvm` source
directory is an LLVM checkout that contains this tree at `amd/comgr/hotswap`.
Then pass the hotswap option through the `amd-comgr` subproject:

```bash
THEROCK_SRC=<therock-source-dir>
THEROCK_BUILD=<therock-build-dir>

cmake -S "${THEROCK_SRC}" -B "${THEROCK_BUILD}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DTHEROCK_AMDGPU_FAMILIES=<gfx-target-or-family> \
  -DTHEROCK_ENABLE_ALL=OFF \
  -DTHEROCK_ENABLE_COMPILER=ON \
  -Damd-llvm_CMAKE_ARGS="-DLLVM_BUILD_TOOLS=ON;-DLLVM_INSTALL_UTILS=ON" \
  -Damd-comgr_CMAKE_ARGS="-DCOMGR_ENABLE_HOTSWAP_TRANSPILE=ON;-DBUILD_TESTING=ON"
```

The `amd-llvm_CMAKE_ARGS` list builds and installs LLVM tools and utilities.
The `BUILD_TESTING=ON` entry in `amd-comgr_CMAKE_ARGS` keeps COMGR lit tests
available. The hotswap pipeline currently shells out to `llc`, `llvm-mc`, and
`ld.lld`; the focused lit test also uses `FileCheck`, `llvm-readelf`, and
`llvm-objdump`.

Build the compiler/COMGR components:

```bash
cmake --build "${THEROCK_BUILD}" --target amd-comgr --parallel 16
```

Verify that COMGR exports the hotswap entry point:

```bash
nm -D "${THEROCK_BUILD}/compiler/amd-comgr/stage/lib/libamd_comgr.so" \
  | rg ' amd_comgr_hotswap_transpile(_with_options(_v2)?)?$'
```

Run the focused COMGR lit test:

```bash
"${THEROCK_BUILD}/compiler/amd-llvm/build/bin/llvm-lit" -v \
  "${THEROCK_BUILD}/compiler/amd-comgr/build/test-lit" \
  --filter hotswap-transpile.c
```

Runtime validation should remain a separate integration test. ROCR should call
COMGR's `amd_comgr_hotswap_transpile_with_options_v2` API rather than linking
hotswap internals, so the runtime can pass explicit cache policy and report
typed cache hit/miss status from COMGR.

### Standalone Hotswap Library

For local library-only iteration, point `LLVM_DIR` at an LLVM build tree. An
LLVM install tree is not sufficient because hotswap uses AMDGPU target-private
headers and generated TableGen include files.

The build tree must also have a full `llvm-project` source checkout behind it,
not just `llvm/include` and `llvm/lib`: `hotswap-tblgen` (see
[`tablegen/`](tablegen/)) is built from `llvm/utils/TableGen/Common/*.h` and
runs over `llvm/lib/Target/AMDGPU/AMDGPU.td` to generate the raiser's canonical
opcode tables. That TableGen run adds roughly ten seconds to a clean build and
is skipped on rebuilds unless a `.td` file it read has changed.

```bash
cmake -S amd/comgr/hotswap -B amd/comgr/hotswap/build -G Ninja \
  -DLLVM_DIR=<build-dir>/lib/cmake/llvm \
  -DCMAKE_CXX_COMPILER=clang++
cmake --build amd/comgr/hotswap/build --target hotswap-transpiler
```

## COMGR Integration

COMGR enables the prototype entry point with:

```text
COMGR_ENABLE_HOTSWAP_TRANSPILE=ON
```

When enabled, COMGR links the `hotswap::transpiler` target and exports:

```c
amd_comgr_status_t amd_comgr_hotswap_transpile(
    amd_comgr_data_t input,
    const char *source_isa_name,
    const char *target_isa_name,
    amd_comgr_data_t *output);

amd_comgr_status_t amd_comgr_hotswap_transpile_with_options(
    amd_comgr_data_t input,
    const char *source_isa_name,
    const char *target_isa_name,
    const amd_comgr_hotswap_transpile_options_t *options,
    amd_comgr_data_t *output,
    amd_comgr_hotswap_transpile_result_t *result);

amd_comgr_status_t amd_comgr_hotswap_transpile_with_options_v2(
    amd_comgr_data_t input,
    const char *source_isa_name,
    const char *target_isa_name,
    const amd_comgr_hotswap_transpile_options_v2_t *options,
    amd_comgr_data_t *output,
    amd_comgr_hotswap_transpile_result_t *result);
```

The options-v2 API accepts an executable code object, returns a new executable
code object, and returns typed result metadata. COMGR owns cache lookup and writes,
but cache root, readonly/disabled policy, skip list, rules path, strict mode,
selected kernel, and opt level are explicit options passed by the caller;
cache corruption and write failures are hard errors rather than silent misses.
The options API maps to the same implementation without per-kernel or opt-level
controls.

## Validation

The focused COMGR smoke test verifies argument validation and a real
`gfx950 -> gfx942` translation of `tests/vecadd_gfx950.co`:

```bash
<build-dir>/bin/llvm-lit -v \
  <build-dir>/tools/comgr/test-lit/hotswap-transpile.c
```

Expected result:

```text
PASS: Comgr :: hotswap-transpile.c
```

A direct manual invocation is also possible:

```bash
<build-dir>/tools/comgr/test-lit/hotswap-transpile \
  amd/comgr/hotswap/tests/vecadd_gfx950.co \
  amdgcn-amd-amdhsa--gfx950 \
  amdgcn-amd-amdhsa--gfx942 \
  --output=<scratch-dir>/vecadd_gfx942.co
```

## Tests Kept In Tree

The LLVM-facing tree keeps:

- assembly-based lit fixtures under `lit_tests/`;
- focused unit tests under `tests/`;
- the small `tests/vecadd_gfx950.co` fixture used by COMGR lit coverage.

Runtime and workload-level validation should live in the runtime integration
that calls COMGR. The hotswap tree keeps compiler-unit tests, lit fixtures, and
small code-object smoke coverage.

## Current Limitations

- The pipeline still shells out to `llc`, `llvm-mc`, and `ld.lld`; an
  upstreamable runtime path should move to in-process LLVM/LLD or COMGR-owned
  entry points.
- The COMGR public entry point is provisional.
- Translation cache policy exists in the hotswap library but is not yet wired
  through the COMGR API.
- Runtime integration is expected to use COMGR as the boundary; this directory
  owns compiler-side translation.
