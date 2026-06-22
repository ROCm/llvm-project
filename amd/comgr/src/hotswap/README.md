# HotSwap

HotSwap rewrites AMDGPU code objects at load time so that a binary built for one
gfx1250 stepping runs correctly on another. The `amd_comgr_hotswap_rewrite` API
applies a small set of stepping-specific patches in place, without recompiling
the code object.

This directory ships two COMGR-side pieces:

- The `amd_comgr_hotswap_rewrite` API, which applies stepping-specific patches
  to a single code object in place.
- The transpiler, a raiser-based path for the heavier cross-ISA case. It is
  documented at the bottom of this file.

COMGR does not ship an `HSA_TOOLS_LIB` runtime tool. Runtime interception — the
tool that loads via `HSA_TOOLS_LIB` and hands each code object to the rewrite —
is owned by rocm-systems' `libhsa-hotswap.so` (see ROCm/rocm-systems
`projects/hotswap`).

## Running at runtime

Build and install the hotswap tool from rocm-systems, then point `HSA_TOOLS_LIB`
at it and run any HIP or HSA application unchanged:

```bash
HSA_TOOLS_LIB=/opt/rocm/lib/libhsa-hotswap.so ./my_app
```

When the tool detects a gfx1250 A0 board it rewrites each gfx1250 code object in
place via `amd_comgr_hotswap_rewrite`; everything else passes through. If a
rewrite fails, the tool logs it and forwards the original code object.

## Supported architectures

| Architecture        | Status      |
| ------------------- | ----------- |
| gfx1250, ASIC rev A0 | Rewrite armed |
| gfx950              | Coming soon |
| gfx942              | Coming soon |

HotSwap currently requires a homogeneous GPU setup. Running it across multiple
GPUs is not supported.

The tool detects the device on its own. It reads the agent ISA name and
`HSA_AMD_AGENT_INFO_ASIC_REVISION`, then arms the rewrite only when the target
is gfx1250, the revision is 0 (A0), and the revision query succeeded. A failed
query is never treated as A0. On any other device or revision, code objects pass
through unchanged. No environment variable is needed to turn this on.

To confirm the rewrite is active, set `HSA_HOTSWAP_TOOL_VERBOSE=1` and look for
this line in the logs:

```bash
hotswap_tool: device=gfx1250 asic_revision=0 -> A0 (rewrite armed)
```

## Environment variables

| Variable                   | Effect                                                        |
| -------------------------- | ------------------------------------------------------------- |
| `HSA_TOOLS_LIB`            | Standard HSA hook. Set it to this `.so` to load the tool.     |
| `HSA_HOTSWAP_TOOL_VERBOSE` | Set to `1` for diagnostic logging to stderr, covering device detection and per-code-object rewrite results. Logging only; does not change behavior. Off by default. |

## Building the runtime tool

The runtime tool is built from rocm-systems (`projects/hotswap` →
`libhsa-hotswap.so`), not from COMGR. COMGR only needs
`COMGR_ENABLE_HOTSWAP_TRANSPILE=ON` to expose `amd_comgr_hotswap_rewrite` for
the rocm-systems tool to link against.

## Transpiler (cross-gen)

The transpiler is the heavier sibling to the byte-level rewrite. It raises
AMDGPU code objects into LLVM IR, re-lowers them through the stock AMDGPU backend
for a different target ISA, and relinks the result into a single merged HSACO.
The rewrite path applies in-place stepping patches; the transpiler instead hands
the whole code object to the IR pipeline. It can be built standalone for
development:

```bash
cmake -S amd/comgr/hotswap -B build-hotswap \
  -DLLVM_DIR=$PWD/build/lib/cmake/llvm
ninja -C build-hotswap
ctest --test-dir build-hotswap -L transpiler
```
