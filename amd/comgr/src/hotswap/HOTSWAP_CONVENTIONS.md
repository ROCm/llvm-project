# Hotswap Subsystem Conventions

Conventions specific to the hotswap subsystem in `amd/comgr/`. These
supplement [`AGENT_CONVENTIONS.md`](../../AGENT_CONVENTIONS.md) (general
Comgr conventions) — read that file first.

The hotswap subsystem rewrites compiled HSA code objects to apply
target-specific patches (e.g. B0-to-A0 errata workarounds for
gfx1250). Source layout:

- `src/comgr-hotswap.cpp` — public entry point
  (`amd_comgr_hotswap_rewrite`).
- `src/comgr-hotswap-elf.cpp` — ELF parsing, growth, and writing.
- `src/comgr-hotswap-llvm.cpp` — MC-layer wrappers
  (`assembleSingleInst`, `parseAsmToMCInsts`, opcode resolution,
  `LLVMState`).
- `src/comgr-hotswap-b0a0.cpp` — gfx1250 B0/A0 policy: which patches
  apply and in what order.
- `src/comgr-hotswap-patch-*.cpp` — one file per patch family
  (in-place, trampoline, WMMA hazard, VOP3PX2 src2, etc.).

## 1. Patch-pass authoring

A patch pass runs over `Ctx.Decoded[]` and may mutate `Ctx.Text`.
Several invariants must hold for stacking and re-runs to be correct.

### `Ctx.Decoded[I].Inst` is a snapshot

It is *not* re-derived from `Ctx.Text` after another patch pass writes
to it. Two consequences:

- A pass that re-reads decoded instructions whose bytes a previous
  pass mutated will read stale operands. Re-decode the byte range or
  update the cached `MCInst` after writing.
- An N-site pass that converges on the same downstream instruction
  (e.g. K splits feeding one `s_wait_dscnt`) reading the cached wait
  value and writing `original + 1` for each site will overwrite — the
  wait gets bumped by 1 instead of by K. **Tests that exercise
  multiple converging sites are mandatory** for any pass that touches
  downstream state.

### Patch passes have implicit ordering

If your pass requires another to have run first (e.g. a hazard pass
depends on in-place patches having stabilized the byte stream), state
the invariant in a comment at the top of the pass. Better: re-decode
at pass entry. Implicit pass-ordering is a recurring maintenance
hazard.

### Use named operand metadata, not positional or text-derived access

- Use `getNamedOperandIdx` for structured `MCInst` operand access.
- Never iterate "the first N register operands" — operand layouts
  change.
- Never recover semantics by parsing `MCInstPrinter` output —
  printer formatting changes.
- Compute register overlap via `MCRegisterInfo::regsOverlap`, not
  hand-rolled VGPR-range arithmetic.

### Idempotency guards check operand identity, not just mnemonic

A guard that fires when "the previous instruction was an
`s_pack_hh_b32_b16`" will mis-fire on user code that happens to
contain one. Compare both the mnemonic *and* the relevant register
operands using `MCRegisterInfo::regsOverlap`.

### Forward scans terminate on control-flow boundaries

Use `MCInstrDesc::mayAffectControlFlow` plus an `s_endpgm` opcode
comparison. Don't enumerate branch opcodes by name — `s_swappc_b64`
and other indirect transfers will be missed.

### Layer separation

- Per-target constants belong in policy modules
  (`comgr-hotswap-b0a0.cpp`) and the `RewriteConfig` struct, not in
  infra (`comgr-hotswap-elf.cpp`, `comgr-hotswap-llvm.cpp`).
- MC opcode caches resolved at `initLLVM()` belong on `LLVMState`.
- Infra carries no per-target data.

### Patch-pass return values

Distinguish "no candidates found", "candidates found and patched",
and "candidates found but not patchable". A count that conflates "no
work" with "skipped" loses information downstream callers need.

## 2. Hotswap LIT tests

Use the canonical hotswap test harness:
`test-lit/hotswap-rewrite-e2e.hip` (end-to-end), or `.s` files driven
through `test-lit/comgr-sources/hotswap-rewrite`. Don't add per-PR
custom drivers.

Specific requirements for hotswap LIT tests:

- Use `CHECK-LABEL` (or `DISASM-LABEL`) per kernel. ELF-wide `CHECK`
  lines pass even when a patch is wrongly applied to the wrong kernel
  or to both.
- Cover **every entry** of any opcode/mnemonic table the patch
  declares. If the dispatch table maps b8/b32/b64/b128, the test
  exercises all four. Single-variant coverage masks typoed entries.
- Cover both code paths when a patch has structural variants — the
  nop-sled-available path *and* the trampoline-fallback path; with-
  padding and without-padding.
- Include a negative path. The patch must correctly refuse unsupported
  shapes; verify it does.
- Prefer `CHECK-NEXT` chains over `CHECK-DAG` blocks where instruction
  order is deterministic.
- Use `mtriple`, not `-target`, in RUN lines.
- Use `%llvm-objdump --no-show-raw-insn=false` to assert encoding-bit
  changes.
- The PR description should name which call site the test forces. A
  test that runs through the plain-copy path while the PR changes the
  growth path catches nothing.

**Idempotency tests are byte-equal, not size-equal.** A second
rewrite pass can change bytes while keeping the ELF the same size.
Use full-buffer `memcmp` (`hotswap-rewrite --check-idempotent`), not
size comparison.
