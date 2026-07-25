# HotSwap code-object benchmark

Batch-run the Comgr **HotSwap** gfx1250 B0→A0 rewrite (`hotswap-rewrite`) over a
directory of AMD GPU code objects (`.hsaco` / `.co`) and tabulate, per object,
the **CPU time**, **peak RSS**, and **pass/fail/timeout** into a single CSV.

Each object is translated in its own fresh process; Linux `wait4` accounting
gives that process's user+system CPU and peak resident memory. Work runs in
parallel for throughput, but rows are always **sorted by input path** before
writing, so the CSV is deterministic and two runs diff cleanly.

> ⚠️ **Disclaimer — entry trampolines are OFF by default here.** `hotswap-rewrite`
> itself enables entry trampolines, but the current gfx1250 hotswap path
> **segfaults on a large fraction of objects** when they are on. For that reason
> **these scripts default `--entry-trampolines` to OFF.** Pass `--entry-trampolines`
> only if you specifically want to reproduce the crash (expect many SIGSEGV
> failures).

## Files

| File | Purpose |
| --- | --- |
| `hotswap_bench.py` | Core runner. Defaults to `*.hsaco`. Fully flag-driven. |
| `hotswap_bench_all.py` | Thin wrapper: runs **both** `.hsaco` and `.co` with a 5-minute per-file timeout. |

## Prerequisites

- **Linux** (uses `wait4`/`/proc`; the optional memory guard uses `systemd-run` or `choom`).
- **Python 3.9+** (standard library only — no `pip` deps).
- A built **`hotswap-rewrite`** binary and **`libamd_comgr.so`** (see below).

## 1. Build the translator

From an llvm-project checkout that contains `amd/comgr`:

```bash
cd <llvm-project>
cmake -S llvm -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="clang;lld" \
  -DLLVM_TARGETS_TO_BUILD="X86;AMDGPU" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_EXTERNAL_PROJECTS="device-libs;comgr" \
  -DLLVM_EXTERNAL_DEVICE_LIBS_SOURCE_DIR="$PWD/amd/device-libs" \
  -DLLVM_EXTERNAL_COMGR_SOURCE_DIR="$PWD/amd/comgr" \
  -DCOMGR_BUILD_SHARED_LIBS=ON
ninja -C build hotswap-rewrite
```

This produces:
- `build/tools/comgr/test-lit/hotswap-rewrite`
- `build/lib/libamd_comgr.so*`

## 2. Variables you need to set

The scripts need to know **where the built tool is**. Set **one** of the
following (or pass the equivalent flags):

| Variable | Meaning | Required? |
| --- | --- | --- |
| `HOTSWAP_BUILD_DIR` | The LLVM **build** directory. The scripts derive `<dir>/tools/comgr/test-lit/hotswap-rewrite` and `<dir>/lib` from it. | **Recommended** |
| `HOTSWAP_REWRITE` | Explicit path to the `hotswap-rewrite` binary (overrides the above). | Optional |
| `HOTSWAP_LIBRARY_DIR` | Directory containing `libamd_comgr.so*`. Auto-discovered from the binary if unset. | Optional |

Equivalent command-line flags (take precedence over the env vars):
`--hotswap <binary>` and `--hotswap-library-dir <dir>`.

Auto-detection: if you run the scripts from **inside** an llvm-project checkout
that has an in-tree `build/` directory, the binary is found automatically and
no variable is needed.

The **corpus directory is a positional argument** — your choice. In our runs it
was `/home/ydeshpan/my_repos/data/hotswap-corpus`.

## 3. Run

```bash
# point at your build once
export HOTSWAP_BUILD_DIR=/path/to/llvm-project/build      # or a TheRock amd-llvm build

# all code objects (.hsaco + .co); entry trampolines are OFF by default here,
# 5-minute timeout, 16-way parallel
python3 hotswap_bench_all.py /path/to/hotswap-corpus \
  --jobs 16 \
  --memory-limit-gb 64 \
  --csv results.csv
```

`.hsaco`-only, using the core script directly:

```bash
export HOTSWAP_BUILD_DIR=/path/to/build
python3 hotswap_bench.py /path/to/hotswap-corpus --jobs 16 --csv results_hsaco.csv
```

> To *measure* the entry-trampolines crash instead, add `--entry-trampolines` —
> expect many `fail` rows with `exit_code = -11` (SIGSEGV).

## Memory safety on shared machines

A parallel run over a large corpus can spike memory. Guard it with **one** of:

- `--memory-limit-gb N` — re-exec inside a **systemd transient scope** with a
  hard aggregate `MemoryMax=N` GiB cap (`OOMPolicy=kill`); the whole run is
  killed if it exceeds the cap. Use `--systemd-scope {auto,user,system}`
  (`auto` picks a user scope when available, else a system scope as root).
- `--oom-guard` — re-exec under **`choom`** so this run is the preferred OOM
  victim under memory pressure (no hard cap). Tune with `--oom-score-adj N`.

## Useful flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--jobs, -j N` | `0` (= auto CPU count) | parallel translations; `0` selects the CPU count |
| `--timeout-seconds N` | 900 (`_all`: 300) | per-file wall-clock timeout; 0 disables |
| `--include-glob GLOB` | `*.hsaco` (repeatable) | which files to pick; `hotswap_bench_all.py` uses `*.hsaco` + `*.co` |
| `--entry-trampolines` / `--no-entry-trampolines` | **off** | kernel-entry trampoline redirection — default OFF here (see disclaimer); `--entry-trampolines` re-enables it |
| `--strict-mode` / `--no-strict-mode` | on | fail instead of returning an unpatched object |
| `--source-isa` / `--target-isa` | `amdgcn-amd-amdhsa--gfx1250` | ISA strings (same = intra-gfx1250 B0→A0) |
| `--keep-outputs DIR` | discard | retain rewritten `.co` files |
| `--csv PATH` | see scripts | output CSV path |

## Output CSV columns (one row per code object)

`input_path, filename, input_size, status, exit_code, result, cpu_seconds,
user_cpu_seconds, system_cpu_seconds, elapsed_seconds, max_rss_kib,
output_size, spawn_error`

- `status` — `pass` (exit 0), `fail` (non-zero exit), `timeout`, or `spawn_error`.
- `exit_code` — raw exit code; negative means killed by a signal (e.g. `-11` = SIGSEGV).
- `result` — the tool's `RESULT:` line (`SUCCESS` / `INVALID_ARGUMENT`) when present.
- `cpu_seconds` — user + system CPU; `max_rss_kib` — peak resident memory (KiB).

Pass/fail is **exit-code based**: `pass` == the process exited 0. The
`exit_code` and `result` columns are kept separately so failures (crashes vs.
graceful errors) can still be distinguished after the fact.
