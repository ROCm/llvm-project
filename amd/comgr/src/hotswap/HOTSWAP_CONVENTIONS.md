# Hotswap Subsystem Conventions

Conventions specific to the hotswap subsystem in `amd/comgr/`. These
supplement [`AGENT_CONVENTIONS.md`](../../AGENT_CONVENTIONS.md) (general
Comgr conventions) — read that file first. Rules that apply to all of
Comgr (Comgr-first/LLVM-second reuse, no hardcoded opcodes, MC-layer
assembly, Windows portability, LIT-vs-gtest choice, ASAN) live there
and are not repeated here.

The hotswap subsystem raises a compiled AMDGPU code object to LLVM IR and
re-lowers it through the stock AMDGPU backend for a different target ISA. See
[`README.md`](README.md) for the directory layout. Four OBJECT libraries, all
opt-in behind `COMGR_ENABLE_HOTSWAP_TRANSPILE`:

- `loader/` — code-object metadata loader (ELF + MsgPack note + kernel
  descriptor) supplying the `.text` section.
- `decoder/` — per-ISA AMDGPU MC stack and the canonical-op identity the
  raiser dispatches on.
- `raiser/` — the transpiler proper: MC to LLVM IR, then re-lowering.
- `common/` — the `KernelMeta` ABI model and `HotswapError`.

The byte-level `rewriter/` path documented by earlier revisions of this file has
been removed; its conventions went with it.

## 1. Fail closed

The overriding safety rule: **produce a correct code object or refuse.
Never emit a wrong, partial, or unverifiable result.** A wrong result
is worse than no result — it fails silently on-device.

- When any invariant a transform depends on cannot be *proven* from the
  code object at hand, take the fail-closed path.
- Fail closed on any undecoded or unknown instruction. Reset the
  per-instruction analysis state and never carry stale analysis across
  an unknown slot. An unknown slot must never resolve to SUCCESS.
- Fixed-point / back-edge analyses must recompute per-index state on
  *every* visit; a later `Unknown` has to overwrite an earlier finite
  result (otherwise a loop back-edge poisons the input while leaving a
  stale result in place). Add a reconvergence regression test.
- An unsupported-but-not-yet-handled case errors out rather than
  silently under-reporting.
- **Every fail-closed / early-return path records why**, through
  `RaiseFailure` or Comgr's gated verbose logging (not raw
  `errs()`/`fprintf`). A silent `return false` from a planning helper
  loses the reason and makes on-device failures undebuggable.

## 2. Instruction recognition

### Use named operand metadata, not positional or text-derived access

- Use `getNamedOperandIdx` for structured `MCInst` operand access.
- Never iterate "the first N register operands" — operand layouts change.
- Never recover semantics by parsing `MCInstPrinter` output — printer
  formatting changes.
- Compute register overlap via `MCRegisterInfo::regsOverlap`, not
  hand-rolled VGPR-range arithmetic.
- Exception: sub-fields the disassembler does not lift into a named
  `MCInst` operand (e.g. `byte_sel` living in OPSEL[3:2]) can only be
  read as raw bits. This is the boundary of the rule — do it, but
  comment *why* the raw read is necessary.

### Match by opcode, not mnemonic string

Recognize an instruction through the decoder's canonical-op identity, or by
comparing `Inst.getOpcode()` against an opcode resolved once and cached. Never
match on `DI.Mnemonic` / `MCInstPrinter` strings:

- mnemonic identity is asm-level; the printer string is a formatting
  artifact that can change or alias, and the tablegen name is a
  different string again.
- it is a per-instruction string compare, usually in the middle of a
  hot dataflow/scan loop.
- it diverges from every other matcher in the subsystem.

Prefer generating opcode → canonical-op mappings from TableGen inputs (single
source of truth, compile-time completeness) over hand-maintained macro tables.

### Layer separation

- Per-target constants belong in policy modules, not in infra.
- MC opcode caches belong on the decoder's `MCState`.
- Infra carries no per-target data.

## 3. Code-object input validation

The loader parses untrusted input. Validate at the boundary before any
raising reasoning.

- Require `e_machine == EM_AMDGPU` (and the expected OS/ABI and type)
  before treating an object as loadable. Reject foreign or stripped
  objects with a *precise* error, not a degraded downstream
  "missing descriptor" result.
- Bounds-check every section's file range with overflow-safe
  arithmetic (`checkedAddUint64`, or compare via subtraction). Never
  form an end address that can wrap on malformed input.
- Search both `.symtab` and `.dynsym` (a stripped code object may
  retain its kernel descriptor only in `.dynsym`), and define how
  duplicates across the two tables are handled.
- Select the decoder ISA from the ELF `e_flags`, not from the input
  filename.

## 4. Kernel metadata and descriptors

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

## 5. Public API and versioning

- Prefer a generic, ISA-parameterized API returning `INVALID_ARGUMENT`
  for unsupported pairs over an ISA-specific entry-point name. Comgr's
  semantic versioning means a specific entry point can't be removed
  until a major bump — `amd_comgr_hotswap_rewrite` and
  `amd_comgr_hotswap_rewrite_with_options` are deprecated stubs in
  `src/comgr-hotswap-stubs.cpp` for exactly this reason, and cannot be
  deleted before v4.0.
- Introduce a new public API and its version bump in the *same*
  commit, so the change cherry-picks and reverts cleanly and there is
  no window where the API exists without the matching version.
- The transpiler has no public C entry point yet. Adding one means a
  new version node in `src/exportmap.in`, a `VERSION.txt` bump, and a
  passing `utils/check_api_consistency.py`.

## 6. Hotswap LIT tests

Use the canonical harness: `.s` fixtures assembled with `%llvm-mc` + `%ld.lld`
and driven through `%hotswap_transpile_cli`, or `.yaml` fixtures through
`%yaml2obj` for malformed-input refusals. Don't add per-PR custom drivers.
Every fixture carries `REQUIRES: comgr-has-hotswap-transpile`, since the driver
is built only under `COMGR_ENABLE_HOTSWAP_TRANSPILE`.

- Use `CHECK-LABEL` per kernel. ELF-wide `CHECK` lines pass even when a
  transform is wrongly applied to the wrong kernel or to both.
- Cover **every entry** of any opcode/mnemonic table the change
  declares. If the dispatch table maps b8/b32/b64/b128, the test
  exercises all four. Single-variant coverage masks typoed entries.
- Include a negative path. The transpiler must correctly refuse
  unsupported shapes; verify it does.
- A negative test **pins the specific diagnostic** (`CHECK` the
  message), not just that the driver exited non-zero. A bare error
  assertion passes on any unrelated failure.
- New or distinct behavior gets a *new* fixture, not a mutated
  existing one.
- Prefer `CHECK-NEXT` chains over `CHECK-DAG` blocks where order is
  deterministic.
- Use `mtriple`, not `-target`, in RUN lines.
- Test the current target's fields (e.g. the gfx12 field, not a stale
  gfx11 one).

**Because the fixtures are feature-gated, a botched rename or a build with the
option off makes them *skip*, not fail.** Check the lit summary's `Unsupported`
count, not just that there were zero failures.

**Env-gated paths must be exercised with the var set.** A decode cache
or profiler enabled only by an environment variable is untested if the
whole suite runs with it unset — run a second test process with it on.
A `PRIVATE` compile define does not reach a separately-compiled test
target.

## 7. Validation bar

- A performance or caching refactor that claims "no functional change"
  proves equivalence over the corpus. Any output diff belongs in the PR
  that owns that *semantic* change, not in the refactor.
- A semantic (numerical) change needs a trusted differential-oracle
  comparison. Translation validity and refusal counts do **not**
  establish numerical equivalence, and synthetic decoder states are not
  production-safety evidence.

## 8. PR structure and staged landing

- Split large hotswap changes along a natural seam (decoder vs raiser;
  foundation vs consumer). One reviewable concern per PR. A change a PR
  depends on lands *before* it, not bundled in.
- Landing a large feature as a series of **inert** PRs (dead code
  first, wired last) is an accepted pattern. When you do:
  - Keep inert code in its own namespace/type, separate from the wired
    stub it will eventually replace, so it can't accidentally override
    the production path.
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
