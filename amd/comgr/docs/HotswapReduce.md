# Hotswap failure reducer

`utils/hotswap/hotswap_reduce.py` turns a failing hotswap corpus bundle into a
small,
standalone reproducer. It only accepts a transformation when a caller-provided
interestingness command continues to return the configured exit code. The
original inputs are never modified and the output directory is published
atomically.

The reducer works from the outside in:

1. Delta-debug the list of code objects.
2. Delta-debug the top-level `kernels`, `cases`, and `arguments` metadata
   lists, followed by each retained case's `arguments` list.
3. Optionally remove explicitly allowed, non-allocated ELF sections with
   `llvm-objcopy`.

The section pass gets section names and flags from `llvm-readobj` JSON. It does
not parse ELF itself. Allocated sections and the default runtime-critical
section set are always protected. Section removal is disabled unless at least
one `--allow-remove-section` glob is provided.

## Interestingness command

The command is an argv vector, not a shell string. Pass the executable with
`--predicate` and repeat `--predicate-arg` for its arguments. These literal
placeholders are expanded for each candidate:

- `{bundle}`: candidate `bundle.json`
- `{input}`: the sole candidate code object (only valid with one object)
- `{metadata}`: candidate `metadata.json`
- `{workspace}`: candidate directory

Exit zero means interesting by default. Use `--interesting-exit-code` when the
reproducer intentionally reports a failure with a nonzero exit. A timeout,
launch error, or disagreement between `--predicate-runs` repetitions rejects a
candidate. Stable outcomes are cached by the complete candidate content and
predicate configuration; `--cache-file` makes that cache persistent.
The resolved predicate executable and regular-file argv items are hashed into
that identity. Repeat `--cache-dependency` for dynamically loaded libraries,
reference data, or other file inputs, and repeat `--cache-tag` for relevant
environment or configuration identity. A dependency that changes during a run
aborts the reduction instead of consulting a stale cache entry.

For example:

```console
python3 utils/hotswap/hotswap_reduce.py \
  --bundle failing-bundle.json \
  --output reduced \
  --predicate /usr/bin/python3 \
  --predicate-arg=/absolute/path/is_interesting.py \
  --predicate-arg='{bundle}' \
  --predicate-runs 3 \
  --timeout 20 \
  --allow-remove-section='.debug_*' \
  --allow-remove-section='.comment'
```

No argument is evaluated by a shell. Supplying an absolute predicate and
absolute script path makes the recorded reproduction argv directly portable.
Existing regular-file argv items are resolved to absolute paths before the
first predicate run.

## PR #3646 offline differential workflow

An A0 runtime failure is not required. The predicate can be any deterministic
offline check, which makes the reducer useful while validating the hotswap
implementation in PR #3646. For a discrepant hipBLASLt or hipSOLVER case,
first put the involved code objects and launch records in a regular bundle.
Then use one of these predicate shapes:

- A wrapper rewrites the candidate with the PR #3646 build, emits its
  structural manifest, compares it with a fixed PR #3598 reference manifest,
  and exits with the interesting code while a difference remains.
- A wrapper runs `hotswap-audit` over every retained object and exits with the
  interesting code while the audit reports an invariant violation.
- A wrapper runs `hotswap-semcheck` for the retained kernels and exits with the
  interesting code while it returns a counterexample.

For example, an offline manifest differential can be driven as:

```console
python3 utils/hotswap/hotswap_reduce.py \
  --bundle hipblaslt-discrepancy.json \
  --output reduced-3646-difference \
  --predicate /absolute/path/manifest-diff-predicate \
  --predicate-arg=/absolute/path/pr3598-reference.json \
  --predicate-arg='{bundle}' \
  --predicate-runs 2
```

The wrapper owns the meaning of the comparison and the selected build. The
reducer only sees an argv vector and an exit code, so no PR-specific opcode,
kernel name, or library test name is encoded in its reduction logic. Keep the
reference manifest outside the candidate bundle so ddmin cannot reduce the
oracle itself.

## Input and output bundle

Input bundles use this versioned JSON form. Object paths and a string-valued
`metadata` path are relative to the bundle file.

```json
{
  "format": "comgr-hotswap-reducer-bundle",
  "version": 1,
  "code_objects": [
    {"id": "hipblaslt-case-17", "path": "objects/case 17.co"}
  ],
  "metadata": {
    "kernels": [{"name": "kernel"}],
    "cases": [{"name": "case", "arguments": [1, 2, 3]}],
    "arguments": []
  }
}
```

`--code-object` can be repeated instead of providing a bundle, with optional
metadata from `--metadata`.

An inventory or test-selection pipeline can provide a NUL-delimited worklist:

```console
python3 utils/hotswap/hotswap_reduce.py \
  --worklist unique-code-objects.list \
  --metadata selection.json \
  --output reduced \
  --predicate /absolute/path/offline-predicate \
  --predicate-arg='{bundle}'
```

Every worklist entry is treated as a path, never shell text. Absolute paths
from the inventory utility work directly; relative paths are resolved from the
worklist's directory. Empty, unterminated, or duplicate entries are rejected.
A selector document can be passed directly as metadata. In addition to
`kernels`, `cases`, and `arguments`, a top-level `selected_tests` list is
delta-debugged generically; all unknown selector fields are preserved.

The output contains:

- `bundle.json`, `metadata.json`, and the retained objects under `objects/`
- `reduction-log.json`, a versioned deterministic record of every accepted or
  rejected transformation
- a final reproduction argv and interesting exit code in that log

The log also records original and final content digests. It intentionally
contains no timestamps or elapsed times so identical inputs and stable
predicates produce identical logs.

## Safety limits

- Existing output directories are never overwritten.
- Cache files cannot overwrite an input, predicate identity, dependency, or
  live inside the atomically published output directory.
- Malformed bundles and metadata fail before a predicate runs.
- Section pruning requires AMDGPU ELF input and available LLVM tools.
- Protected or allocated sections cannot be opted into removal.
- Temporary candidates are removed after interruption, timeout, or failure.
- `llvm-reduce` is not run on linked ELF code objects; it is an IR/MIR reducer.
  Use it before linking when a retained bundle also has LLVM IR or MIR source.
- Persistent cache writes are atomic but are not merged across concurrent
  reducer processes; give each concurrent process a distinct cache file.
