# Hotswap Subsystem Conventions

Conventions specific to the hotswap subsystem in `amd/comgr/`. These
supplement [`AGENT_CONVENTIONS.md`](../../AGENT_CONVENTIONS.md) (general
Comgr conventions) — read that file first. Rules that apply to all of
Comgr (Comgr-first/LLVM-second reuse, no hardcoded opcodes, MC-layer
assembly, Windows portability, LIT-vs-gtest choice, ASAN) live there
and are not repeated here.

The hotswap subsystem rewrites compiled AMDGPU code objects to apply
target-specific patches (e.g. B0-to-A0 errata workarounds for
gfx1250). See [`README.md`](README.md) for the public API contract and
directory layout. Two transformation paths plus a shared layer:

- `rewriter/` — byte-level rewrite path (in-place ELF/MC patching and
  entry trampolines). Backs the always-on `amd_comgr_hotswap_rewrite`.
  Same-family stepping patches go here.
  - `rewriter/rewriter.cpp` — path entry / driver.
  - `rewriter/elf.cpp`, `rewriter/displacement.cpp` — ELF parsing,
    growth, writing, text displacement.
  - `rewriter/llvm.cpp` — MC-layer wrappers (`assembleSingleInst`,
    `parseAsmToMCInsts`, opcode resolution, `LLVMState`).
  - `rewriter/b0a0.cpp` — gfx1250 B0/A0 policy: which patches apply and
    in what order.
  - `rewriter/patch-*.cpp` — one file per patch family (in-place,
    trampoline, WMMA hazard/split/scale16, VOP3PX2 src2, etc.).
- `raiser/` — IR transpiler path (raise to LLVM IR, re-lower through
  the AMDGPU backend for a different target). Opt-in behind
  `COMGR_ENABLE_HOTSWAP_TRANSPILE`. Heavier cross-gen transforms.
- `common/` — path-agnostic headers shared by both paths.

## 1. Fail closed

The overriding safety rule: **produce a correct code object or refuse.
Never emit a wrong, partial, or unverifiable rewrite.** A wrong rewrite
is worse than no rewrite — it fails silently on-device.

- When any invariant a patch depends on cannot be *proven* from the
  code object at hand, take the fail-closed path. For the always-on
  API a proven-unpatchable object is returned unmodified; in strict
  mode a required-but-unemittable workaround fails the whole rewrite
  rather than returning an unpatched object.
- The final code object often lacks the information to prove a
  transform safe — there is no compiler IR, and MC call instructions
  carry no transitive callee-clobber information. When the proof needs
  information you don't have (e.g. a nested call whose clobbers you
  can't bound), leave the candidate unresolved so the fail-closed path
  is selected. Don't attempt the optimization on a hunch.
- Fail closed on any undecoded or unknown instruction. Reset the
  per-instruction analysis state, clear its target/materialization
  sets, and never carry stale analysis across an unknown slot. An
  unknown slot must never resolve to SUCCESS.
- Fixed-point / back-edge analyses must recompute per-index state on
  *every* visit; a later `Unknown` has to overwrite an earlier finite
  result (otherwise a loop back-edge poisons the input while leaving a
  stale target set in place). Add a reconvergence regression test.
- An unsupported-but-not-yet-handled case errors out rather than
  silently under-reporting. Example: a VGPR bump inside a non-kernel
  device function needs callgraph analysis to attribute occupancy —
  until that exists, error instead of under-counting.
- **Every fail-closed / early-return path logs why**, through Comgr's
  gated verbose logging (not raw `errs()`/`fprintf`). A silent
  `return false` from a planning helper loses the reason a rewrite was
  declined and makes on-device failures undebuggable.

## 2. Patch-pass authoring

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
- Exception: sub-fields the disassembler does not lift into a named
  `MCInst` operand (e.g. `byte_sel` living in OPSEL[3:2]) can only be
  read as raw bits. This is the boundary of the rule — do it, but
  comment *why* the raw read is necessary.

### Idempotency guards check operand identity, not just mnemonic

A guard that fires when "the previous instruction was an
`s_pack_hh_b32_b16`" will mis-fire on user code that happens to
contain one. Compare both the mnemonic *and* the relevant register
operands using `MCRegisterInfo::regsOverlap`.

Idempotency guards are also **bounded to the containing function/CFG** —
never scan raw `.text` across kernel boundaries. A bare tensor
construction in kernel B can otherwise match and rewrite a pattern in
kernel A. Reuse the already-built CFG / function ranges and fail closed
at control-flow boundaries.

### Update the cache after writing; dedup on full identity

- After patching, update the cached decoded operand (e.g. `setImm`) so
  shared-descriptor and converging sites are counted exactly.
- When deduplicating rewrite sites, key on the full identity tuple
  (e.g. `(name, vaddr)`), not last-value-wins on the name alone. A map
  keyed on name silently collapses distinct vaddrs. Add an `A, B, A`
  ordering test.

### Forward scans terminate on control-flow boundaries

Use `MCInstrDesc::mayAffectControlFlow` plus an `s_endpgm` opcode
comparison. Don't enumerate branch opcodes by name — `s_swappc_b64`
and other indirect transfers will be missed.

### Layer separation

- Per-target constants belong in policy modules (`rewriter/b0a0.cpp`)
  and the `RewriteConfig` struct, not in infra (`rewriter/elf.cpp`,
  `rewriter/llvm.cpp`).
- MC opcode caches resolved at `initLLVM()` belong on `LLVMState`.
- Infra carries no per-target data.
- Keep the byte-level `rewriter/` and IR `raiser/` paths independent;
  shared logic goes in `common/`, not cross-included between paths.

### Patch-pass return values

Distinguish "no candidates found", "candidates found and patched",
and "candidates found but not patchable". A count that conflates "no
work" with "skipped" loses information downstream callers need. Prefer
a typed result (`Expected<T>` / `std::optional`) over magic sentinels;
never return a magic value that a caller later string-matches or
compares against `-1`.

## 3. Register and scratch allocation

- **Never hardcode a scratch register or a fixed physical register
  pair** (`s0`, `s[100:101]`, VCC). Allocate per-kernel and prove the
  register is free via liveness. Silent clobber of a live SGPR / VCC /
  SCC produces wrong results on-device with no diagnostic and is
  merge-blocking.
- A fixed physical register is acceptable *only* with a cited ABI
  guarantee (link the ABI). "Mirrors the loader" or similar asserted
  justifications are not sufficient — cite the source.
- Track SGPRs and VCC together, and account for partial defs: a
  `vcc_lo` / `vcc_hi` write is *not* a full VCC kill. Key liveness by
  mutation generation so a re-decode after a write is observed.
- Instructions with an implicit SCC def (`s_and_b32`, etc.) require
  proving SCC is dead before use, or falling back to the at-site path.
- Record allocated scratch so the post-rewrite verification can
  cross-check what each patch claimed against what it used.

## 4. Code-object input validation

Hotswap performs ELF surgery on untrusted input. Validate at the
boundary before any rewrite reasoning.

- Require `e_machine == EM_AMDGPU` (and the expected OS/ABI and type)
  before treating an object as rewritable or data-only. Reject foreign
  or stripped objects with a *precise* error, not a degraded downstream
  "missing descriptor" result.
- Bounds-check every section's file range with overflow-safe
  arithmetic (`checkedAddUint64`, or compare via subtraction). Never
  form an end address that can wrap on malformed input.
- A byte-identical no-op SUCCESS path is dangerous — gate it hard.
  "Empty `.text` / no descriptors" is necessary but *not sufficient*
  to treat an object as data-only.
- Search both `.symtab` and `.dynsym` (a stripped code object may
  retain its kernel descriptor only in `.dynsym`), and define how
  duplicates across the two tables are handled.
- Select the decoder ISA from the ELF `e_flags`, not from the input
  filename.

## 5. Kernel metadata and descriptors

- Read required metadata fields with required-getters that error on
  absence. Never substitute a plausible default ABI value for a
  missing or malformed required field. Reject unsupported
  `amdhsa.version` before interpreting the rest.
- Locate a kernel's code via the descriptor's authoritative
  `kernel_code_entry_byte_offset` / `.symbol`, not by re-deriving the
  entry from `.name`. The loader can pair one kernel's code with
  another kernel's descriptor; `.symbol` may differ from `.name`.
- Before treating a symbol's bytes as an ABI struct, validate the
  symbol (defined, `STT_OBJECT`, correct section, expected size,
  correct alignment) and read fields as explicit little-endian, not a
  `memcpy` into a native struct. Guard the struct layout with
  `static_assert(sizeof(...) == ...)` so an upstream/downstream ABI
  drift is caught at build time.
- Cross-check fields duplicated between the metadata note and the
  kernel descriptor, and diagnose mismatches. Don't build a hybrid
  record field-by-field from whichever source is convenient.
- Parse the code object once into a reusable structure (owned metadata
  document + `StringMap`); don't re-parse ELF/MsgPack per query.

## 6. Instruction encoding and templates

Extends the general "no hardcoded opcodes" rule with hotswap
specifics.

- No two passes may recognize or encode the same instruction
  independently. Route recognition/encoding through one shared,
  named-field helper (single source of truth) and centralize any
  legacy-B0 exceptions there.
- Any pre-encoded byte template (e.g. `entry-trampoline-fast-stub.inc`)
  is generated from readable assembly by a checked-in regeneration
  tool, marked generated / do-not-edit, and guarded by an
  MC-equivalence test (`memcmp` against `llvm-mc` output) so it cannot
  silently drift from the assembler.
- Prefer generating opcode → canonical-op mappings from TableGen
  inputs (single source of truth, compile-time completeness) over
  hand-maintained macro tables.

## 7. Text growth and routing

- **Plan before writing.** Encode each candidate sequence via MC to get
  its *actual* size, reserve all routes, then modify `.text`. Don't
  assume fixed instruction widths.
- `assembleSingleInst` assembles exactly one instruction. Use a
  distinct API for multi-instruction assembly, and reject multi-line
  source on the single-instruction path.
- Don't fold two call sites through a partially-initialized shared
  struct (e.g. a fake `LLVMState`) on a fast path. Pass explicit
  fast-path values instead.
- Generated register-bank / mode transitions must inherit the hazard
  waits the original sequence relied on — e.g. drain XCNT with
  `s_wait_xcnt 0` before a changed VGPR-MSB mode; preserve VGPR-MSB
  mode when splitting WMMAs. Note that scale-prefix operands address
  VGPR bank zero and ignore the SRC MSB banks — a positive test that
  uses only mode zero will mask a bank bug.

## 8. Public API and versioning

- Prefer a generic, ISA-parameterized API
  (`amd_comgr_hotswap_rewrite(inputISA, outputISA)`, returning
  `INVALID_ARGUMENT` for unsupported pairs) over an ISA-specific name
  like `..._b0a0`. Comgr's semantic versioning means a specific entry
  point can't be removed until a major bump.
- Introduce a new public API and its version bump in the *same*
  commit, so the change cherry-picks and reverts cleanly and there is
  no window where the API exists without the matching version.
- SUCCESS means "produced a valid output code object", not "bytes
  changed". A no-op rewrite still returns SUCCESS. Document this in the
  README/API. If a caller needs to know whether a rewrite actually
  fired, expose that explicitly rather than overloading the status.
- Synthesized stub/symbol names use suffixes that are illegal in C++
  mangling (e.g. a `.`-prefixed `.stub`) to avoid colliding with
  compiler-generated symbols.

## 9. Hotswap LIT tests

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
- A negative test **pins the specific diagnostic** (enable verbose
  logs and `CHECK` the message), not just `RESULT: ERROR`. A bare
  error assertion passes on any unrelated failure.
- New or distinct behavior gets a *new* fixture, not a mutated
  existing one; cover ordering / mixing variants (e.g. global and
  cluster loads in different orders).
- A multi-site aggregation pass needs a multi-entry test that
  exercises the max/skip/zero branches *and* the failure-propagation
  path.
- Prefer `CHECK-NEXT` chains over `CHECK-DAG` blocks where instruction
  order is deterministic.
- Use `mtriple`, not `-target`, in RUN lines.
- Use `%llvm-objdump --no-show-raw-insn=false` to assert encoding-bit
  changes.
- Test the current target's fields (e.g. the gfx12 field, not a stale
  gfx11 one).
- The PR description should name which call site the test forces. A
  test that runs through the plain-copy path while the PR changes the
  growth path catches nothing.

**Idempotency tests are byte-equal, not size-equal.** A second
rewrite pass can change bytes while keeping the ELF the same size.
Use full-buffer `memcmp` (`hotswap-rewrite --check-idempotent`), not
size comparison.

**Env-gated paths must be exercised with the var set.** A decode cache
or profiler enabled only by an environment variable is untested if the
whole suite runs with it unset — run a second test process with it on.
A `PRIVATE` compile define does not reach a separately-compiled test
target.

**Cache correctness.** A decode-cache key must cover the full
`getMaxInstLength()` window *and* the remaining-byte count, and the
hit path must validate cached size against remaining bytes — an 8-byte
value ending in zero bytes otherwise collides with a 4-byte prefix at
the buffer tail. A cache-hit path must reproduce the miss path's full
result state (e.g. `DecodeSucceeded`), not just the size.

## 10. Validation bar for rewrite PRs

Because a wrong rewrite fails silently on-device, ELF-surgery and
patch PRs carry a heavier evidence bar than typical Comgr changes.

- A performance or caching refactor that claims "no functional change"
  proves byte-for-byte (or hash / idempotence) equivalence over the
  corpus. Any output diff belongs in the PR that owns that *semantic*
  change, not in the refactor.
- A semantic (numerical) change needs a trusted differential-oracle
  comparison. Translation / ELF-validity / idempotence counts do
  **not** establish numerical equivalence, and synthetic decoder
  states are not production-safety evidence.
- Patch PRs that mutate ELFs report a full-corpus transition matrix
  (no SUCCESS→FAILURE and no correctness regressions) plus ASan/UBSan
  reruns. Performance PRs include a reproducible benchmark: exact
  command, run count, median/dispersion, peak memory. Record input and
  output SHA plus the library build id so two runs are provably
  comparable.

## 11. PR structure and staged landing

- Split large hotswap changes along a natural seam (MC-layer vs
  ELF-mutation; foundation vs consumer). One reviewable concern per PR.
  A change a PR depends on lands *before* it, not bundled in.
- Landing a large feature as a series of **inert** PRs (dead code
  first, wired last) is an accepted pattern. When you do:
  - Keep inert code in its own namespace/type, separate from the wired
    weak stub it will eventually replace, so it can't accidentally
    override the production path.
  - Structure the increment so wiring it later is a no-rebuild hook-up.
  - Ship each inert PR with its own unit tests.
  - When a harness lands ahead of its logic, its fixtures must FAIL
    honestly — never fake them green. Land fixtures with the code that
    makes their expected output real.
- A stacked PR is based on the *current* head of its prerequisite and
  is a clean, dependency-free range — not a cumulative snapshot whose
  Files-changed view contains unmerged prerequisites. It must actually
  descend from the claimed prerequisite; a mechanical cherry-pick that
  skips semantic conflict resolution is not a rebase.
- Don't land a named, live field or set that has no production reader
  and whose name promises unimplemented behavior — wire it, rename it,
  or remove it. (Inert-and-clearly-named is fine; silently-inert-but-
  named-as-live is a bug.)
- Deferred follow-ups get a linked tracking issue, not a mental note.
