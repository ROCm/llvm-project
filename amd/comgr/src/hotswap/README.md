# HotSwap

HotSwap rewrites AMDGPU code objects at load time so a binary built for one
gfx1250 stepping runs correctly on another. The in-place
`amd_comgr_hotswap_rewrite` API applies a small set of stepping-specific
patches to a code object without recompiling it.

This directory ships two things:

- the **HSA_TOOLS_LIB tool** (`libamd_comgr_hotswap_tool.so`), which applies
  the rewrite automatically at runtime, and
- the **transpiler** (a sibling, raiser-based path) for the heavier cross-ISA
  case. See the bottom of this file.

## Running with the tool

Point `HSA_TOOLS_LIB` at the tool and run any HIP/HSA application unchanged:

```
HSA_TOOLS_LIB=/opt/rocm/lib/libamd_comgr_hotswap_tool.so ./my_app
```

`libhsa-runtime` loads the tool and hands it each code object before dispatch.
On a board the tool detects as gfx1250 A0, every gfx1250 code object is
rewritten in place via `amd_comgr_hotswap_rewrite`; everything else is passed
through untouched.

If a rewrite fails, the tool logs a message and forwards the original code
object, so the application still runs (just without the rewrite).

## Supported architectures

- **gfx1250, ASIC revision A0** — rewrite is armed.
- **gfx950** - coming soon
- **gfx942** - coming soon

Hotswap currently requires homogenous GPU setup. Do not try to run hotswap across multiple GPUs.

The tool auto-detects the device: it reads the agent ISA name and
`HSA_AMD_AGENT_INFO_ASIC_REVISION`, and arms the rewrite only when the target
is `gfx1250`, the revision is `0` (A0), and the revision query actually
succeeded (a failed query is never treated as A0). On any other device or
revision it passes code objects through unchanged. No environment variable is
needed to enable it.

### Stepping (A0) and the log line

`HSA_AMD_AGENT_INFO_ASIC_REVISION` reports the ASIC stepping as an integer:
revision `0` is **A0**. This rewrite targets A0 only, so the tool arms exactly
when the revision is `0` and disarms otherwise. With verbose logging the
decision prints as:

```
hotswap_tool: device=gfx1250 asic_revision=0 (valid=yes) -> A0 (rewrite armed)
```

A non-zero revision (a later stepping, e.g. B0) or a failed query prints
`-> B0/native` and the code object is forwarded unchanged. The tool does not
assume the numeric revision of any later stepping; anything other than `0` is
treated as "not A0".

## Environment variables

| Variable                   | Effect                                                        |
| -------------------------- | ------------------------------------------------------------- |
| `HSA_TOOLS_LIB`            | Standard HSA hook. Set it to this `.so` to load the tool.     |
| `HSA_HOTSWAP_TOOL_VERBOSE` | `1` enables diagnostic logging to stderr (device detection and per-code-object rewrite results). Logging only; does not change behavior. Default off. |

## Building the tool

The tool is **off by default** because most comgr consumers do not need it.
Enable it and point the build at the HSA headers:

```
cmake -S amd/comgr -B build \
  -DHOTSWAP_BUILD_TOOL=ON \
  -DHOTSWAP_TOOL_HSA_INCLUDE_ROOT=/path/to/rocr-runtime/runtime/hsa-runtime
ninja -C build amd_comgr_hotswap_tool
```

The target is skipped (with a status message) if `inc/hsa.h` is not found under
`HOTSWAP_TOOL_HSA_INCLUDE_ROOT`, so a build without HSA headers still
configures cleanly.

## Transpiler (For cross-gen)

The transpiler is the sibling to the byte-level rewrite: it raises AMDGPU code
objects into LLVM IR, re-lowers them through the stock AMDGPU backend for a
different target ISA, and relinks the result into a single merged HSACO. Where
rewrite applies in-place stepping patches, transpilation hands the entire code
object to the IR pipeline. It can be configured standalone for development:

```
cmake -S amd/comgr/hotswap -B build-hotswap \
  -DLLVM_DIR=$PWD/build/lib/cmake/llvm
ninja -C build-hotswap
ctest --test-dir build-hotswap -L transpiler
```
