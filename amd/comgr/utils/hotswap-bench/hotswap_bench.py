#!/usr/bin/env python3
# Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
# amd/comgr/LICENSE.TXT in this repository for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Run HotSwap gfx1250 B0-to-A0 translation over a directory of .hsaco files
and tabulate CPU time, peak RSS, and pass/fail into a single CSV.

Each input runs in a fresh process. Linux wait4 resource accounting captures
user plus system CPU time and peak resident set size for that process.

Inputs are translated in parallel (--jobs) for throughput, but rows are always
sorted by input path before writing, so the CSV is deterministic regardless of
completion order and two runs diff cleanly.

Because a parallel run over a large corpus can spike memory, the invocation can
re-exec itself under a memory guard so it never OOM-kills a shared dev machine:
  * --memory-limit-gb N wraps the run in a systemd transient scope with a hard
    aggregate MemoryMax cap (the whole process tree is killed if it exceeds N).
  * --oom-guard uses choom to bias the kernel OOM killer toward this run, so a
    machine under memory pressure sacrifices this batch instead of other work.

This is a deliberately simple, single-tool runner that exercises the Comgr
`hotswap-rewrite` lit binary over a corpus of code objects. Point it at a build
with $HOTSWAP_REWRITE / $HOTSWAP_BUILD_DIR or the --hotswap flag; see README.md
in this directory for the full list of variables and steps.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime
import functools
import hashlib
import io
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Sequence

DEFAULT_SOURCE_ISA = "amdgcn-amd-amdhsa--gfx1250"
DEFAULT_TARGET_ISA = "amdgcn-amd-amdhsa--gfx1250"

# Sentinel set on the re-exec'd child so the memory guard wraps only once.
GUARD_ENV = "HOTSWAP_HSACO_BENCH_GUARDED"

METADATA_SCHEMA_VERSION = 1

# Bounded amount of a failed translation's stderr retained per row.
STDERR_TAIL_BYTES = 4096


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read_tail(stream: Any, limit: int) -> str:
    """Return the last `limit` bytes of a binary file object as text."""
    try:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        stream.seek(max(0, size - limit))
        data = stream.read()
    except OSError:
        return ""
    text = data.decode("utf-8", "replace").strip()
    if size > limit:
        text = "...(truncated)...\n" + text
    return text


def append_jsonl(path: pathlib.Path, record: dict[str, Any]) -> None:
    """Append one result record as a JSON line, flushed so it survives an
    abrupt process death (e.g. an OOM kill) for later --resume."""
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, separators=(",", ":")))
        stream.write("\n")
        stream.flush()


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                # Tolerate a torn final line from an interrupted write.
                continue
            if isinstance(value, dict) and "input_path" in value:
                records.append(value)
    return records


def sha256_file(path: pathlib.Path) -> str | None:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def file_identity(path: pathlib.Path) -> dict[str, Any] | None:
    """Return {path, size, sha256} for a file, or None if it can't be read."""
    try:
        size = path.stat().st_size
    except OSError:
        return None
    return {"path": str(path), "size": size, "sha256": sha256_file(path)}


def find_repo_root() -> pathlib.Path | None:
    """Locate the enclosing llvm-project checkout by walking up until a dir
    containing amd/comgr is found. Returns None if not inside such a tree."""
    for parent in pathlib.Path(__file__).resolve().parents:
        if (parent / "amd" / "comgr").is_dir():
            return parent
    return None


def first_executable(candidates: Sequence[pathlib.Path | str]) -> str:
    for candidate in candidates:
        raw = str(candidate)
        if "/" not in raw:
            found = shutil.which(raw)
            if found:
                return found
            continue
        path = pathlib.Path(raw).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path.resolve())
    return ""


# hotswap-rewrite lands under one of these subpaths of an LLVM build dir,
# depending on the comgr external-project name (comgr vs amd_comgr).
HOTSWAP_BUILD_SUBPATHS = (
    ("tools", "comgr", "test-lit", "hotswap-rewrite"),
    ("tools", "amd_comgr", "test-lit", "hotswap-rewrite"),
)


def build_dir_candidates(build_dir: pathlib.Path) -> list[pathlib.Path]:
    return [build_dir.joinpath(*subpath) for subpath in HOTSWAP_BUILD_SUBPATHS]


def default_hotswap() -> str:
    """Locate the hotswap-rewrite binary. Precedence:
    1. $HOTSWAP_REWRITE (explicit binary path),
    2. $HOTSWAP_BUILD_DIR/tools/{comgr,amd_comgr}/test-lit/hotswap-rewrite,
    3. an in-tree build at <repo-root>/build/tools/{comgr,amd_comgr}/test-lit/...,
    4. `hotswap-rewrite` on PATH.
    Both the `comgr` and `amd_comgr` external-project layouts are probed.
    Override at runtime with --hotswap."""
    candidates: list[pathlib.Path | str] = []
    if value := os.environ.get("HOTSWAP_REWRITE"):
        candidates.append(value)
    if build := os.environ.get("HOTSWAP_BUILD_DIR"):
        candidates.extend(build_dir_candidates(pathlib.Path(build)))
    if (root := find_repo_root()) is not None:
        candidates.extend(build_dir_candidates(root / "build"))
    candidates.append("hotswap-rewrite")
    return first_executable(candidates)


def resolve_executable(raw: str, label: str) -> pathlib.Path:
    if not raw:
        raise ValueError(f"{label} was not found; pass --{label}")
    found = raw if "/" in raw else (shutil.which(raw) or "")
    if not found:
        raise ValueError(f"{label} executable was not found: {raw}")
    path = pathlib.Path(found).expanduser().resolve(strict=True)
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"{label} is not executable: {path}")
    return path


def discover_library_dirs(binary: pathlib.Path) -> list[pathlib.Path]:
    for ancestor in list(binary.parents)[:8]:
        for candidate in (ancestor, ancestor / "lib"):
            if candidate.is_dir() and any(candidate.glob("libamd_comgr.so*")):
                return [candidate.resolve()]
    return []


def resolve_library_dirs(
    raw_directories: Sequence[str], hotswap: pathlib.Path
) -> list[pathlib.Path]:
    values = list(raw_directories)
    if not values and (environment := os.environ.get("HOTSWAP_LIBRARY_DIR")):
        values.append(environment)
    if not values:
        return discover_library_dirs(hotswap)
    resolved: list[pathlib.Path] = []
    for raw in values:
        path = pathlib.Path(raw).expanduser().resolve(strict=True)
        if not path.is_dir():
            raise ValueError(f"HotSwap library path is not a directory: {path}")
        if path not in resolved:
            resolved.append(path)
    return resolved


def discover_inputs(
    paths: Sequence[str], recursive: bool, include_globs: Sequence[str]
) -> list[pathlib.Path]:
    discovered: set[pathlib.Path] = set()
    for raw_path in paths:
        path = pathlib.Path(raw_path).expanduser().resolve(strict=True)
        if path.is_file():
            discovered.add(path)
            continue
        if not path.is_dir():
            raise ValueError(f"input is neither a regular file nor directory: {path}")
        for include_glob in include_globs:
            iterator = (
                path.rglob(include_glob) if recursive else path.glob(include_glob)
            )
            discovered.update(
                candidate.resolve() for candidate in iterator if candidate.is_file()
            )
    return sorted(discovered)


def common_root(inputs: Sequence[pathlib.Path]) -> pathlib.Path:
    if len(inputs) == 1:
        return inputs[0].parent
    return pathlib.Path(os.path.commonpath([str(path) for path in inputs]))


def terminate_process_group(
    process: subprocess.Popen[bytes], grace: float
) -> tuple[int, Any] | None:
    if process.returncode is not None:
        return None
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        try:
            waited, status, usage = os.wait4(process.pid, os.WNOHANG)
        except InterruptedError:
            continue
        except ChildProcessError:
            return None
        if waited:
            process.returncode = os.waitstatus_to_exitcode(status)
            return status, usage
        time.sleep(0.01)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return None


def timed_process(
    command: Sequence[str],
    *,
    stdout: Any,
    stderr: Any,
    environment: dict[str, str],
    timeout_seconds: float,
) -> dict[str, Any]:
    started_ns = time.monotonic_ns()
    process = subprocess.Popen(
        list(command),
        stdout=stdout,
        stderr=stderr,
        env=environment,
        start_new_session=True,
    )
    deadline = time.monotonic() + timeout_seconds if timeout_seconds > 0 else None
    timed_out = False
    status = 0
    usage = None
    try:
        while True:
            try:
                waited, status, usage = os.wait4(process.pid, os.WNOHANG)
            except InterruptedError:
                continue
            if waited:
                break
            if deadline is not None and time.monotonic() >= deadline:
                timed_out = True
                reaped = terminate_process_group(process, 1.0)
                if reaped is not None:
                    status, usage = reaped
                else:
                    try:
                        _, status, usage = os.wait4(process.pid, 0)
                    except ChildProcessError:
                        usage = None
                break
            time.sleep(0.005)
    except KeyboardInterrupt:
        # The child runs in its own session/process group, so a terminal SIGINT
        # does not reach it; tear it (and any descendants) down explicitly
        # before propagating the interrupt.
        terminate_process_group(process, 1.0)
        raise

    if process.returncode is None:
        process.returncode = os.waitstatus_to_exitcode(status)
    elapsed_seconds = (time.monotonic_ns() - started_ns) / 1_000_000_000
    result: dict[str, Any] = {
        "return_code": process.returncode,
        "timed_out": timed_out,
        "elapsed_seconds": elapsed_seconds,
        "user_cpu_seconds": None,
        "system_cpu_seconds": None,
        "cpu_seconds": None,
        "max_rss_kib": None,
    }
    if usage is not None:
        user_seconds = float(usage.ru_utime)
        system_seconds = float(usage.ru_stime)
        result.update(
            {
                "user_cpu_seconds": user_seconds,
                "system_cpu_seconds": system_seconds,
                "cpu_seconds": user_seconds + system_seconds,
                "max_rss_kib": int(usage.ru_maxrss),
            }
        )
    return result


def hotswap_command(
    binary: pathlib.Path,
    source: pathlib.Path,
    output: pathlib.Path,
    source_isa: str,
    target_isa: str,
    *,
    entry_trampolines: bool,
    strict_mode: bool,
) -> list[str]:
    command = [str(binary), str(source), source_isa, target_isa, "--output", str(output)]
    if entry_trampolines:
        command.append("--entry-trampolines")
    if strict_mode:
        command.append("--strict-mode")
    return command


def run_one(
    *,
    source: pathlib.Path,
    hotswap: pathlib.Path,
    library_dirs: Sequence[pathlib.Path],
    source_isa: str,
    target_isa: str,
    entry_trampolines: bool,
    strict_mode: bool,
    timeout_seconds: float,
    outputs_dir: pathlib.Path | None,
    input_root: pathlib.Path,
) -> dict[str, Any]:
    relative = os.path.relpath(source, input_root)
    try:
        input_size = source.stat().st_size
    except OSError:
        input_size = None
    input_sha256 = sha256_file(source)

    if outputs_dir is not None:
        # Preserve the input's relative directory structure so retained outputs
        # map 1:1 to inputs; flattening separators to "__" would collide (e.g.
        # a/b.hsaco and a__b.hsaco) and race under parallel workers.
        output = outputs_dir / f"{relative}.co"
        output.parent.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        handle, temporary = tempfile.mkstemp(suffix=".co", prefix="hotswap-")
        os.close(handle)
        output = pathlib.Path(temporary)
        cleanup = True
    output.unlink(missing_ok=True)

    environment = dict(os.environ)
    environment["LC_ALL"] = "C"
    if library_dirs:
        previous = environment.get("LD_LIBRARY_PATH")
        prefix = os.pathsep.join(str(path) for path in library_dirs)
        environment["LD_LIBRARY_PATH"] = (
            prefix + os.pathsep + previous if previous else prefix
        )

    command = hotswap_command(
        hotswap,
        source,
        output,
        source_isa,
        target_isa,
        entry_trampolines=entry_trampolines,
        strict_mode=strict_mode,
    )

    stdout_buffer = io.BytesIO()
    stderr_tail = ""
    with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
        try:
            process_result = timed_process(
                command,
                stdout=stdout_file,
                stderr=stderr_file,
                environment=environment,
                timeout_seconds=timeout_seconds,
            )
            spawn_error = None
        except OSError as error:
            process_result = {
                "return_code": None,
                "timed_out": False,
                "elapsed_seconds": 0.0,
                "user_cpu_seconds": None,
                "system_cpu_seconds": None,
                "cpu_seconds": None,
                "max_rss_kib": None,
            }
            spawn_error = str(error)
        stdout_file.seek(0)
        stdout_buffer.write(stdout_file.read())
        # Keep a bounded stderr tail so crash diagnostics (asserts, backtraces)
        # are retained without unbounded CSV growth.
        stderr_tail = read_tail(stderr_file, STDERR_TAIL_BYTES)

    result_line = ""
    for line in stdout_buffer.getvalue().decode("utf-8", "replace").splitlines():
        if line.startswith("RESULT:"):
            result_line = line.split(":", 1)[1].strip()
            break

    output_size: int | None = None
    output_is_elf = False
    output_sha256: str | None = None
    if output.is_file():
        output_size = output.stat().st_size
        try:
            with output.open("rb") as stream:
                output_is_elf = stream.read(4) == b"\x7fELF"
        except OSError:
            output_is_elf = False
        # Hash the output before it is cleaned up so translation divergence is
        # detectable across runs.
        output_sha256 = sha256_file(output)

    if spawn_error is not None:
        status = "spawn_error"
    elif process_result["timed_out"]:
        status = "timeout"
    elif process_result["return_code"] != 0:
        status = "fail"
    elif result_line != "SUCCESS" or not output_is_elf:
        # Exit 0 but the rewrite did not report SUCCESS or wrote no valid ELF
        # (e.g. INVALID_ARGUMENT); not a successful translation.
        status = "fail"
    else:
        status = "pass"

    if cleanup:
        output.unlink(missing_ok=True)

    return {
        "input_path": relative,
        "filename": source.name,
        "input_size": input_size,
        "input_sha256": input_sha256,
        "status": status,
        "exit_code": process_result["return_code"],
        "result": result_line,
        "cpu_seconds": process_result["cpu_seconds"],
        "user_cpu_seconds": process_result["user_cpu_seconds"],
        "system_cpu_seconds": process_result["system_cpu_seconds"],
        "elapsed_seconds": process_result["elapsed_seconds"],
        "max_rss_kib": process_result["max_rss_kib"],
        "output_size": output_size,
        "output_sha256": output_sha256,
        "output_is_elf": output_is_elf,
        "spawn_error": spawn_error,
        "stderr_tail": stderr_tail,
    }


CSV_FIELDS = [
    "input_path",
    "filename",
    "input_size",
    "input_sha256",
    "status",
    "exit_code",
    "result",
    "cpu_seconds",
    "user_cpu_seconds",
    "system_cpu_seconds",
    "elapsed_seconds",
    "max_rss_kib",
    "output_size",
    "output_sha256",
    "output_is_elf",
    "spawn_error",
    "stderr_tail",
]


def write_csv(path: pathlib.Path, rows: Sequence[dict[str, Any]]) -> None:
    # Write to a sibling temp file and atomically rename, so a crash mid-write
    # never leaves a truncated CSV in place.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with tmp.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=CSV_FIELDS, extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_run_metadata(
    csv_path: pathlib.Path,
    *,
    hotswap: pathlib.Path,
    library_dirs: Sequence[pathlib.Path],
    args: argparse.Namespace,
    jobs: int,
    include_globs: Sequence[str],
    input_root: pathlib.Path,
    input_count: int,
) -> pathlib.Path:
    """Persist run-level provenance to <csv>.meta.json so two runs can be
    checked for the same tool, library, configuration, and workload scope
    (things a per-row CSV diff cannot by itself guarantee)."""
    libraries: list[dict[str, Any]] = []
    for directory in library_dirs:
        for lib in sorted(directory.glob("libamd_comgr.so*")):
            if lib.is_file() and not lib.is_symlink():
                identity = file_identity(lib)
                if identity is not None:
                    libraries.append(identity)

    command_template = hotswap_command(
        hotswap,
        pathlib.Path("INPUT"),
        pathlib.Path("OUTPUT.co"),
        args.source_isa,
        args.target_isa,
        entry_trampolines=args.entry_trampolines,
        strict_mode=args.strict_mode,
    )

    metadata = {
        "schema_version": METADATA_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "tool": {
            "hotswap_rewrite": file_identity(hotswap),
            "libraries": libraries,
        },
        "config": {
            "source_isa": args.source_isa,
            "target_isa": args.target_isa,
            "entry_trampolines": args.entry_trampolines,
            "strict_mode": args.strict_mode,
            "timeout_seconds": args.timeout_seconds,
            "jobs": jobs,
            "include_globs": list(include_globs),
            "recursive": args.recursive,
            "memory_limit_gb": args.memory_limit_gb or None,
            "oom_guard": args.oom_guard,
        },
        "command_template": command_template,
        "input_root": str(input_root),
        "input_count": input_count,
    }

    meta_path = csv_path.with_name(csv_path.name + ".meta.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return meta_path


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def apply_oom_score_adj(value: int) -> None:
    """Bias the kernel OOM killer toward this process tree. Children inherit
    oom_score_adj across fork/exec, so setting it here covers every worker and
    hotswap-rewrite descendant. Best-effort: warn but continue on failure."""
    try:
        with open("/proc/self/oom_score_adj", "w", encoding="ascii") as stream:
            stream.write(f"{value}\n")
    except OSError as error:
        print(
            f"warning: could not set oom_score_adj={value}: {error}",
            file=sys.stderr,
        )


def user_systemd_manager_available() -> bool:
    systemctl = shutil.which("systemctl")
    if not systemctl:
        return False
    return (
        subprocess.run(
            [systemctl, "--user", "show-environment"], capture_output=True
        ).returncode
        == 0
    )


def choose_systemd_scope(preference: str) -> str:
    """Resolve the systemd scope to use for the memory cap. 'auto' prefers a
    user scope, falling back to a system scope when running as root."""
    if preference == "user":
        if not user_systemd_manager_available():
            raise ValueError(
                "--systemd-scope user requires an accessible systemd user "
                "manager; start a systemd user session or use "
                "--systemd-scope system/auto"
            )
        return "user"
    if preference == "system":
        if os.geteuid() != 0:
            raise ValueError(
                "--systemd-scope system needs root (or interactive polkit "
                "auth); run as root or use --systemd-scope user"
            )
        return "system"
    # auto
    if user_systemd_manager_available():
        return "user"
    if os.geteuid() == 0:
        return "system"
    raise ValueError(
        "no usable systemd scope: no user manager and not root for a system "
        "scope. Start a systemd user session, run as root, or use --oom-guard"
    )


def reexec_under_memory_guard(args: argparse.Namespace) -> None:
    """If a memory guard was requested and is not already active, replace this
    process with one wrapped by systemd-run (hard aggregate cap) or choom (OOM
    victim bias). Returns normally when no wrapping is needed."""
    if os.environ.get(GUARD_ENV) == "1":
        return
    if not args.memory_limit_gb and not args.oom_guard:
        return
    if args.memory_limit_gb and args.oom_guard:
        raise ValueError("use only one of --memory-limit-gb or --oom-guard")

    child = [sys.executable, str(pathlib.Path(sys.argv[0]).resolve()), *sys.argv[1:]]

    if args.memory_limit_gb:
        systemd_run = shutil.which("systemd-run")
        if not systemd_run:
            raise ValueError("--memory-limit-gb requires systemd-run")
        scope = choose_systemd_scope(args.systemd_scope)
        # A transient scope puts this runner, every worker, and every
        # hotswap-rewrite descendant in one cgroup, so MemoryMax applies to
        # their aggregate memory and OOMPolicy=kill takes down the whole scope
        # if the cap is hit. OOMScoreAdjust is not a valid scope property (a
        # scope adopts existing processes rather than spawning them), so the
        # global-pressure victim bias is applied later via /proc/self.
        wrapper = [systemd_run]
        if scope == "user":
            wrapper.append("--user")
        wrapper += [
            "--scope",
            "--quiet",
            "--collect",
            "--property=OOMPolicy=kill",
            f"--property=MemoryMax={args.memory_limit_gb}G",
            "--",
        ]
        print(
            f"re-exec under systemd {scope} scope: "
            f"MemoryMax={args.memory_limit_gb}G, OOMPolicy=kill",
            file=sys.stderr,
        )
    else:
        choom = shutil.which("choom")
        if not choom:
            raise ValueError("--oom-guard requires the choom tool (util-linux)")
        wrapper = [choom, "-n", str(args.oom_score_adj), "--"]
        print(
            f"re-exec under choom: oom_score_adj={args.oom_score_adj} "
            "(preferred OOM victim; no hard cap)",
            file=sys.stderr,
        )

    os.environ[GUARD_ENV] = "1"
    os.execvp(wrapper[0], wrapper + child)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        help=".hsaco files or directories to scan",
    )
    parser.add_argument(
        "--csv",
        default="hotswap_hsaco_bench.csv",
        help="output CSV path (default: hotswap_hsaco_bench.csv)",
    )
    parser.add_argument(
        "--hotswap",
        default=default_hotswap(),
        help="hotswap-rewrite executable (default: auto-detected)",
    )
    parser.add_argument(
        "--hotswap-library-dir",
        action="append",
        default=[],
        help="prepend a directory to LD_LIBRARY_PATH for hotswap; repeatable",
    )
    parser.add_argument("--source-isa", default=DEFAULT_SOURCE_ISA)
    parser.add_argument("--target-isa", default=DEFAULT_TARGET_ISA)
    parser.add_argument(
        "--entry-trampolines",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "pass --entry-trampolines to hotswap-rewrite (default: OFF; the "
            "gfx1250 path currently segfaults with trampolines on)"
        ),
    )
    parser.add_argument(
        "--strict-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="pass --strict-mode to hotswap-rewrite (default: true)",
    )
    parser.add_argument(
        "--include-glob",
        action="append",
        default=None,
        metavar="GLOB",
        help=(
            "filename glob within directories; repeatable (default: *.hsaco). "
            "Match extensions explicitly, e.g. "
            "--include-glob '*.hsaco' --include-glob '*.co'"
        ),
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="recurse into input directories (default: true)",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=nonnegative_float,
        default=900.0,
        help="per-input timeout; zero disables (default: 900)",
    )
    parser.add_argument(
        "--jobs",
        "-j",
        type=nonnegative_int,
        default=0,
        help="parallel translations; 0 auto-detects CPU count (default: 0)",
    )
    parser.add_argument(
        "--memory-limit-gb",
        type=positive_int,
        default=0,
        help=(
            "re-exec under a systemd transient scope with this hard aggregate "
            "MemoryMax cap in GiB (whole run is killed if exceeded)"
        ),
    )
    parser.add_argument(
        "--systemd-scope",
        choices=("auto", "user", "system"),
        default="auto",
        help=(
            "systemd scope for --memory-limit-gb; auto uses a user scope when "
            "available, else a system scope as root (default: auto)"
        ),
    )
    parser.add_argument(
        "--oom-guard",
        action="store_true",
        help=(
            "re-exec under choom so this run is the preferred OOM victim under "
            "memory pressure (no hard cap; mutually exclusive with "
            "--memory-limit-gb)"
        ),
    )
    parser.add_argument(
        "--oom-score-adj",
        type=int,
        default=1000,
        help="oom_score_adj passed to choom by --oom-guard (default: 1000)",
    )
    parser.add_argument(
        "--keep-outputs",
        metavar="DIR",
        default=None,
        help="retain rewritten code objects in DIR (default: discard)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "resume from a prior run's <csv>.partial.jsonl checkpoint, skipping "
            "inputs already recorded (assumes the same inputs and configuration)"
        ),
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def execute_all(
    sources: Sequence[pathlib.Path],
    worker: Callable[..., dict[str, Any]],
    jobs: int,
    quiet: bool,
    *,
    checkpoint_path: pathlib.Path | None = None,
    total: int | None = None,
    start_index: int = 0,
) -> tuple[list[dict[str, Any]], bool]:
    """Run worker over every source, optionally in parallel. Completion order
    is arbitrary; the caller sorts before writing so output stays deterministic.
    Each completed row is appended to checkpoint_path (if given) as it finishes,
    so an OOM/crash mid-run is recoverable with --resume.
    Returns (rows, interrupted)."""
    overall_total = len(sources) if total is None else total
    rows: list[dict[str, Any]] = []
    interrupted = False

    def record(row: dict[str, Any]) -> None:
        rows.append(row)
        if checkpoint_path is not None:
            append_jsonl(checkpoint_path, row)
        if not quiet:
            cpu = row["cpu_seconds"]
            rss = row["max_rss_kib"]
            cpu_text = "n/a" if cpu is None else f"{cpu:.4f}s"
            rss_text = "n/a" if rss is None else f"{rss / 1024:.1f}MiB"
            print(
                f"[{start_index + len(rows)}/{overall_total}] {row['status']:<11} "
                f"cpu={cpu_text} rss={rss_text} {row['input_path']}",
                flush=True,
            )

    if jobs == 1:
        try:
            for source in sources:
                record(worker(source=source))
        except KeyboardInterrupt:
            interrupted = True
            print("interrupted; writing partial results", file=sys.stderr)
        return rows, interrupted

    # Each worker runs in its own process, so at most one hotswap-rewrite child
    # exists per process at a time; this keeps the wait4 accounting isolated and
    # avoids cross-process reaping races.
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=jobs)
    recorded: set[concurrent.futures.Future] = set()
    try:
        futures = {
            executor.submit(worker, source=source): source for source in sources
        }
        try:
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                recorded.add(future)
                record(row)
        except KeyboardInterrupt:
            interrupted = True
            print(
                "interrupted; stopping workers and writing partial results",
                file=sys.stderr,
            )
            for future in futures:
                future.cancel()
            # Salvage any rows that finished before the interrupt but were not
            # yet recorded by the as_completed loop.
            for future in futures:
                if future in recorded or future.cancelled() or not future.done():
                    continue
                try:
                    record(future.result())
                except BaseException:
                    pass
    finally:
        # Do not block on still-running children when interrupted; each worker's
        # timed_process already tears down its own hotswap-rewrite process group
        # on SIGINT.
        executor.shutdown(wait=not interrupted, cancel_futures=True)
    return rows, interrupted


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        # May replace this process (systemd-run / choom); returns if no guard.
        reexec_under_memory_guard(args)
        hotswap = resolve_executable(args.hotswap, "hotswap")
        library_dirs = resolve_library_dirs(args.hotswap_library_dir, hotswap)
        include_globs = args.include_glob or ["*.hsaco"]
        sources = discover_inputs(args.inputs, args.recursive, include_globs)
        if not sources:
            raise ValueError(f"no files matching {include_globs!r} were found")
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    # systemd scopes cannot set OOMScoreAdjust, so apply the victim bias here;
    # choom already applies it for the --oom-guard path.
    if args.memory_limit_gb:
        apply_oom_score_adj(args.oom_score_adj)

    jobs = args.jobs if args.jobs else (os.cpu_count() or 1)
    input_root = common_root(sources)
    outputs_dir = (
        pathlib.Path(args.keep_outputs).expanduser().resolve()
        if args.keep_outputs
        else None
    )
    if not args.quiet:
        print(f"hotswap-rewrite: {hotswap}", file=sys.stderr)
        if library_dirs:
            print(
                "LD_LIBRARY_PATH += "
                + os.pathsep.join(str(path) for path in library_dirs),
                file=sys.stderr,
            )
        print(
            f"inputs: {len(sources)} file(s) under {input_root}; jobs={jobs}",
            file=sys.stderr,
        )

    worker = functools.partial(
        run_one,
        hotswap=hotswap,
        library_dirs=library_dirs,
        source_isa=args.source_isa,
        target_isa=args.target_isa,
        entry_trampolines=args.entry_trampolines,
        strict_mode=args.strict_mode,
        timeout_seconds=args.timeout_seconds,
        outputs_dir=outputs_dir,
        input_root=input_root,
    )

    csv_path = pathlib.Path(args.csv).expanduser().resolve()
    checkpoint_path = csv_path.with_name(csv_path.name + ".partial.jsonl")

    # Resume: reuse rows already recorded in the checkpoint and only run the
    # inputs that are still missing.
    done_rows: list[dict[str, Any]] = []
    if args.resume:
        done_rows = read_jsonl(checkpoint_path)
        done_paths = {row["input_path"] for row in done_rows}
        remaining = [
            source
            for source in sources
            if os.path.relpath(source, input_root) not in done_paths
        ]
        if not args.quiet:
            print(
                f"resume: {len(done_rows)} already done, "
                f"{len(remaining)} remaining",
                file=sys.stderr,
            )
    else:
        checkpoint_path.unlink(missing_ok=True)
        remaining = list(sources)

    new_rows, interrupted = execute_all(
        remaining,
        worker,
        jobs,
        args.quiet,
        checkpoint_path=checkpoint_path,
        total=len(sources),
        start_index=len(done_rows),
    )

    rows = done_rows + new_rows
    rows.sort(key=lambda value: value["input_path"])
    write_csv(csv_path, rows)
    meta_path = write_run_metadata(
        csv_path,
        hotswap=hotswap,
        library_dirs=library_dirs,
        args=args,
        jobs=jobs,
        include_globs=include_globs,
        input_root=input_root,
        input_count=len(sources),
    )

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    summary = ", ".join(f"{status}={count}" for status, count in sorted(counts.items()))
    print(f"wrote {len(rows)} row(s) to {csv_path} ({summary})", file=sys.stderr)
    print(f"wrote run metadata to {meta_path}", file=sys.stderr)

    if interrupted:
        print(
            f"interrupted; resume with: --resume --csv {csv_path}",
            file=sys.stderr,
        )
        return 130
    # Full run finished: the checkpoint is now redundant with the CSV.
    checkpoint_path.unlink(missing_ok=True)
    return 0 if counts.get("pass", 0) == len(rows) and rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
