# Hotswap corpus inventory

`hotswap_inventory.py` finds AMDGPU ELF files below one or more corpus
roots and groups exact duplicates by SHA-256. It is intended to keep the
developer hotswap loop from rewriting identical code objects repeatedly.
It does not decode or make claims about instruction semantics.

The versioned JSON inventory is deterministic for an unchanged corpus.
Each object group names the lexicographically first path as its
representative and retains all duplicate paths for traceability. Non-ELF
files, malformed ELF files, and ELF files for other machines are listed
under `rejected`. Optional command results additionally contain measured
runtime and command output.

A frozen, newline-delimited corpus manifest can select the exact input
set. Its paths are relative to the single corpus root:

```console
$ amd/comgr/utils/hotswap/hotswap_inventory.py extracted-corpus \
    --manifest evidence/manifest.txt \
    --json-output inventory.json
```

The manifest path, SHA-256, and entry count are recorded in the report.
Duplicate, missing, absolute, root-escaping (including through symlinks),
empty, and NUL-containing manifest entries are errors.

To inventory a corpus and produce a NUL-delimited worklist:

```console
$ amd/comgr/utils/hotswap/hotswap_inventory.py corpus \
    --json-output inventory.json \
    --worklist unique-code-objects.list
```

The worklist is a byte stream of absolute paths terminated by NUL bytes.
It is safe for paths containing spaces or newlines. For example:

```console
$ xargs -0 -n1 my-hotswap-command < unique-code-objects.list
```

Place the JSON output, worklist, and cache outside the inventoried roots.
Pre-existing files below those roots are corpus inputs; the tool refuses to
replace an input through an alternate, hard-linked, or symlinked path.

The tool can run a command itself without a shell. The absolute object
path is appended to the argument vector:

```console
$ amd/comgr/utils/hotswap/hotswap_inventory.py corpus \
    --execute build/bin/hotswap-rewrite \
    --execute-arg=--check-idempotent \
    --jobs 8 \
    --timeout 120 \
    --cache-dir .hotswap-cache \
    --cache-dependency build/lib/libamd_comgr.so \
    --cache-tag candidate-commit-or-container-digest \
    --json-output results.json
```

Successful results are cached by the command identity, input content hash,
and representative path. Including the path preserves command behavior
when two corpus locations contain identical bytes. The command identity
includes the timeout, the contents of the executable, and any arguments
that name regular files. Use repeatable `--cache-dependency` options for
dynamically loaded libraries or other file inputs, and `--cache-tag` for
relevant environment, container, or configuration identity. Failed and
timed-out commands are not cached, so a later run retries them. `--jobs`
controls concurrent command execution; result records remain in
deterministic digest order. Captured standard output and standard error are
Base64-encoded in the JSON result. Every result also records monotonic
elapsed `runtime_ms`. Successful cache hits replay the original runtime so
downstream selectors can estimate a future full run. The execution summary
reports `estimated_runtime_ms` across every unique object and
`executed_runtime_ms` for work actually performed in the current run.
On POSIX systems, a timeout terminates the command's process group so
descendants that inherited output pipes cannot leave the inventory blocked.

The exit status is zero when inventory and every optional command
succeed, one when at least one optional command fails or times out, and
two for invalid arguments or an inventory, output, or cache error. Use
`--help` for all options.

## Comparing rewrite outputs

The same tool can digest the output trees produced by two implementations,
including the #3598 oracle and a #3646 candidate:

```console
$ amd/comgr/utils/hotswap/hotswap_inventory.py \
    evidence/pr3598/outputs evidence/pr3646/outputs \
    --json-output output-content.json
```

Each absolute path remains associated with its content SHA-256 in the
object groups. This makes identical output objects collapse into one
group while changed output objects remain distinct, without encoding
either PR, corpus location, or target semantics in the tool.

## Optional #3646 corpus acceptance test

`test_hotswap_inventory_acceptance.py` verifies the published 2026-07-24
corpus invariants: 2,685 aliases, 1,452 unique content hashes, and 1,233
duplicate executions skipped. It is skipped by CTest unless both of these
variables are set:

```console
$ export COMGR_HOTSWAP_CORPUS_ROOT=/path/to/hotswap-corpus
$ export COMGR_HOTSWAP_CORPUS_MANIFEST=/path/to/manifest.txt
```

The accepted frozen manifest has SHA-256
`076ea013ba466139e76074fcbb31905a5041298e7644f6e8f590bf9353e4f5b9`.
If the corpus or manifest changes, the acceptance test fails rather than
silently updating its expectations. The normal hermetic test suite uses
only small synthetic ELF fixtures.
