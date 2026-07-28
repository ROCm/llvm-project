#!/usr/bin/env python3
"""Reduce a hotswap corpus failure while preserving an external predicate.

The reducer treats the interestingness command as an argv vector. It never
passes user input through a shell.  See ``docs/HotswapReduce.md`` for the
bundle schema and examples.
"""

from __future__ import annotations

import argparse
import copy
import fnmatch
import hashlib
import json
import math
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Optional, Sequence


BUNDLE_FORMAT = "comgr-hotswap-reducer-bundle"
BUNDLE_VERSION = 1
LOG_FORMAT = "comgr-hotswap-reduction-log"
LOG_VERSION = 1
CACHE_FORMAT = "comgr-hotswap-reduction-cache"
CACHE_VERSION = 1

KNOWN_PLACEHOLDERS = ("{bundle}", "{input}", "{metadata}", "{workspace}")
DEFAULT_PROTECTED_SECTIONS = (
    "",
    ".text",
    ".note",
    ".note.*",
    ".symtab",
    ".dynsym",
    ".strtab",
    ".dynstr",
    ".shstrtab",
    ".rel",
    ".rel.*",
    ".rela",
    ".rela.*",
    ".group",
    ".rodata",
    ".data",
    ".bss",
    ".dynamic",
    ".AMDGPU.*",
)
CAPTURE_LIMIT = 4096


class ReducerError(Exception):
    """A user-facing, fail-safe reduction error."""


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r} is not supported")


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream, parse_constant=_reject_json_constant)
    except (OSError, json.JSONDecodeError, ValueError, RecursionError) as error:
        raise ReducerError(f"{path}: malformed JSON: {error}") from error


def canonical_json_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError, RecursionError) as error:
        raise ReducerError(f"value is not canonical JSON: {error}") from error
    return text.encode("ascii")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as error:
        raise ReducerError(f"could not hash {path}: {error}") from error
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_json_bytes(value))
            stream.write(b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _safe_component(value: str) -> str:
    component = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return component[:80] or "object"


def _object_output_name(index: int, object_id: str, original_name: str) -> str:
    return f"{index:04d}-{_safe_component(object_id)}-{_safe_component(original_name)}"


@dataclass(frozen=True)
class CodeObject:
    object_id: str
    original_name: str
    path: Path
    original_sha256: str
    content_sha256: Optional[str] = None
    removed_sections: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.content_sha256 is None:
            object.__setattr__(self, "content_sha256", self.original_sha256)

    @property
    def digest(self) -> str:
        return self.content_sha256 or self.original_sha256


@dataclass(frozen=True)
class Candidate:
    code_objects: tuple[CodeObject, ...]
    metadata: dict[str, Any]
    source_paths: tuple[Path, ...] = ()

    def digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"comgr-hotswap-reducer-candidate-v1\0")
        for code_object in self.code_objects:
            digest.update(canonical_json_bytes(code_object.object_id))
            digest.update(b"\0")
            digest.update(canonical_json_bytes(code_object.original_name))
            digest.update(b"\0")
            digest.update(bytes.fromhex(code_object.digest))
        digest.update(b"\0metadata\0")
        digest.update(canonical_json_bytes(self.metadata))
        return digest.hexdigest()


def _validate_metadata(metadata: Any, source: str) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        raise ReducerError(f"{source}: metadata must be a JSON object")
    for key in ("kernels", "cases", "arguments", "selected_tests"):
        if key in metadata and not isinstance(metadata[key], list):
            raise ReducerError(f"{source}: metadata.{key} must be a JSON list")
    cases = metadata.get("cases", [])
    for index, case in enumerate(cases):
        if isinstance(case, dict) and "arguments" in case:
            if not isinstance(case["arguments"], list):
                raise ReducerError(
                    f"{source}: metadata.cases[{index}].arguments must be a JSON list"
                )
    canonical_json_bytes(metadata)
    return copy.deepcopy(metadata)


def _resolve_input_path(
    path_value: Any,
    base: Path,
    description: str,
    require_within_base: bool = False,
) -> Path:
    if not isinstance(path_value, str) or not path_value:
        raise ReducerError(f"{description}: path must be a non-empty string")
    path = Path(path_value)
    if require_within_base and path.is_absolute():
        raise ReducerError(f"{description}: bundle path must be relative")
    if not path.is_absolute():
        path = base / path
    try:
        path = path.resolve(strict=True)
    except OSError as error:
        raise ReducerError(
            f"{description}: could not resolve {path}: {error}"
        ) from error
    if not path.is_file():
        raise ReducerError(f"{description}: {path} is not a regular file")
    if require_within_base and not _path_is_within(path, base.resolve()):
        raise ReducerError(f"{description}: bundle path escapes {base.resolve()}")
    return path


def load_bundle(path: Path) -> Candidate:
    bundle_path = path.resolve()
    value = load_json(bundle_path)
    if not isinstance(value, dict):
        raise ReducerError(f"{bundle_path}: bundle must be a JSON object")
    if value.get("format") != BUNDLE_FORMAT:
        raise ReducerError(f"{bundle_path}: format must be {BUNDLE_FORMAT!r}")
    if value.get("version") != BUNDLE_VERSION:
        raise ReducerError(
            f"{bundle_path}: unsupported bundle version "
            f"{value.get('version')!r}; expected {BUNDLE_VERSION}"
        )
    object_values = value.get("code_objects")
    if not isinstance(object_values, list) or not object_values:
        raise ReducerError(f"{bundle_path}: code_objects must be a non-empty JSON list")

    code_objects: list[CodeObject] = []
    seen_ids: set[str] = set()
    for index, object_value in enumerate(object_values):
        if not isinstance(object_value, dict):
            raise ReducerError(
                f"{bundle_path}: code_objects[{index}] must be a JSON object"
            )
        object_id = object_value.get("id")
        if not isinstance(object_id, str) or not object_id:
            raise ReducerError(
                f"{bundle_path}: code_objects[{index}].id must be a non-empty string"
            )
        if object_id in seen_ids:
            raise ReducerError(f"{bundle_path}: duplicate code object id {object_id!r}")
        seen_ids.add(object_id)
        object_path = _resolve_input_path(
            object_value.get("path"),
            bundle_path.parent,
            f"{bundle_path}: code_objects[{index}]",
            require_within_base=True,
        )
        code_objects.append(
            CodeObject(
                object_id=object_id,
                original_name=object_path.name,
                path=object_path,
                original_sha256=sha256_file(object_path),
            )
        )

    source_paths = [bundle_path]
    source_paths.extend(code_object.path for code_object in code_objects)
    metadata_value = value.get("metadata", {})
    if isinstance(metadata_value, str):
        metadata_path = _resolve_input_path(
            metadata_value,
            bundle_path.parent,
            f"{bundle_path}: metadata",
            require_within_base=True,
        )
        source_paths.append(metadata_path)
        metadata_value = load_json(metadata_path)
        metadata_source = str(metadata_path)
    else:
        metadata_source = f"{bundle_path}: metadata"
    metadata = _validate_metadata(metadata_value, metadata_source)
    return Candidate(tuple(code_objects), metadata, tuple(source_paths))


def load_nul_worklist(path: Path) -> list[Path]:
    """Load a NUL-delimited path list without interpreting shell syntax."""

    worklist_path = path.resolve()
    try:
        contents = worklist_path.read_bytes()
    except OSError as error:
        raise ReducerError(
            f"could not read worklist {worklist_path}: {error}"
        ) from error
    if not contents:
        raise ReducerError(f"{worklist_path}: worklist is empty")
    if not contents.endswith(b"\0"):
        raise ReducerError(f"{worklist_path}: NUL-delimited worklist must end with NUL")
    encoded_paths = contents[:-1].split(b"\0")
    if any(not value for value in encoded_paths):
        raise ReducerError(f"{worklist_path}: worklist contains an empty path")

    paths: list[Path] = []
    seen: set[Path] = set()
    for index, encoded_path in enumerate(encoded_paths):
        decoded_path = os.fsdecode(encoded_path)
        resolved_path = _resolve_input_path(
            decoded_path,
            worklist_path.parent,
            f"{worklist_path}: entry #{index + 1}",
        )
        if resolved_path in seen:
            raise ReducerError(
                f"{worklist_path}: duplicate worklist path {resolved_path}"
            )
        seen.add(resolved_path)
        paths.append(resolved_path)
    return paths


def load_inputs(
    bundle: Optional[Path],
    code_object_paths: Sequence[Path],
    worklist: Optional[Path],
    metadata_path: Optional[Path],
) -> Candidate:
    if bundle is not None:
        if code_object_paths or worklist is not None or metadata_path is not None:
            raise ReducerError(
                "--bundle cannot be combined with --code-object, --worklist, "
                "or --metadata"
            )
        return load_bundle(bundle)
    if worklist is not None:
        if code_object_paths:
            raise ReducerError("--worklist cannot be combined with --code-object")
        input_paths = load_nul_worklist(worklist)
    else:
        input_paths = list(code_object_paths)
    if not input_paths:
        raise ReducerError("one of --bundle, --code-object, or --worklist is required")

    code_objects: list[CodeObject] = []
    for index, path in enumerate(input_paths):
        object_path = _resolve_input_path(
            str(path), Path.cwd(), f"--code-object #{index + 1}"
        )
        code_objects.append(
            CodeObject(
                object_id=f"object-{index:04d}",
                original_name=object_path.name,
                path=object_path,
                original_sha256=sha256_file(object_path),
            )
        )

    source_paths = [item.path for item in code_objects]
    if worklist is not None:
        source_paths.append(worklist.resolve())
    if metadata_path is None:
        metadata: dict[str, Any] = {}
    else:
        resolved_metadata = _resolve_input_path(
            str(metadata_path), Path.cwd(), "--metadata"
        )
        source_paths.append(resolved_metadata)
        metadata = _validate_metadata(
            load_json(resolved_metadata), str(resolved_metadata)
        )
    return Candidate(tuple(code_objects), metadata, tuple(source_paths))


def materialize_candidate(candidate: Candidate, destination: Path) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    object_directory = destination / "objects"
    object_directory.mkdir()
    object_entries: list[dict[str, Any]] = []
    for index, code_object in enumerate(candidate.code_objects):
        output_name = _object_output_name(
            index, code_object.object_id, code_object.original_name
        )
        relative_path = Path("objects") / output_name
        output_path = destination / relative_path
        shutil.copyfile(code_object.path, output_path)
        copied_digest = sha256_file(output_path)
        if copied_digest != code_object.digest:
            raise ReducerError(
                f"{code_object.path}: content changed while materializing candidate"
            )
        object_entries.append(
            {
                "id": code_object.object_id,
                "path": relative_path.as_posix(),
                "sha256": code_object.digest,
                "size": output_path.stat().st_size,
            }
        )
    atomic_write_json(destination / "metadata.json", candidate.metadata)
    bundle = {
        "format": BUNDLE_FORMAT,
        "version": BUNDLE_VERSION,
        "code_objects": object_entries,
        "metadata": "metadata.json",
    }
    atomic_write_json(destination / "bundle.json", bundle)
    return bundle


def snapshot_candidate(candidate: Candidate, destination: Path) -> Candidate:
    """Copy original objects once so later predicate runs see immutable bytes."""

    destination.mkdir(parents=True)
    snapshots: list[CodeObject] = []
    for index, code_object in enumerate(candidate.code_objects):
        output_path = destination / _object_output_name(
            index, code_object.object_id, code_object.original_name
        )
        shutil.copyfile(code_object.path, output_path)
        snapshot_digest = sha256_file(output_path)
        if snapshot_digest != code_object.original_sha256:
            raise ReducerError(
                f"{code_object.path}: content changed while taking the "
                "reduction snapshot"
            )
        snapshots.append(
            replace(
                code_object,
                path=output_path,
                content_sha256=snapshot_digest,
            )
        )
    return replace(candidate, code_objects=tuple(snapshots))


@dataclass(frozen=True)
class PredicateResult:
    status: str
    exit_codes: tuple[Optional[int], ...]
    stdout: str
    stderr: str
    cached: bool = False

    @property
    def interesting(self) -> bool:
        return self.status == "interesting"

    def for_log(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "exit_codes": list(self.exit_codes),
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


class PredicateCache:
    def __init__(self, path: Optional[Path]) -> None:
        self.path = path
        self.values: dict[str, dict[str, Any]] = {}
        if path is None or not path.exists():
            return
        value = load_json(path)
        if (
            not isinstance(value, dict)
            or value.get("format") != CACHE_FORMAT
            or value.get("version") != CACHE_VERSION
            or not isinstance(value.get("entries"), dict)
        ):
            raise ReducerError(f"{path}: unsupported predicate cache format")
        self.values = value["entries"]

    def get(self, key: str) -> Optional[PredicateResult]:
        value = self.values.get(key)
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ReducerError(f"{self.path or 'predicate cache'}: corrupt entry {key}")
        status = value.get("status")
        exit_codes = value.get("exit_codes")
        if status not in ("interesting", "uninteresting") or not isinstance(
            exit_codes, list
        ):
            raise ReducerError(f"{self.path or 'predicate cache'}: corrupt entry {key}")
        if not exit_codes or not all(
            isinstance(code, int) and not isinstance(code, bool) for code in exit_codes
        ):
            raise ReducerError(f"{self.path or 'predicate cache'}: corrupt entry {key}")
        stdout = value.get("stdout", "")
        stderr = value.get("stderr", "")
        if not isinstance(stdout, str) or not isinstance(stderr, str):
            raise ReducerError(f"{self.path or 'predicate cache'}: corrupt entry {key}")
        return PredicateResult(
            status,
            tuple(exit_codes),
            stdout,
            stderr,
            cached=True,
        )

    def put(self, key: str, result: PredicateResult) -> None:
        if result.status not in ("interesting", "uninteresting"):
            return
        self.values[key] = {
            "status": result.status,
            "exit_codes": list(result.exit_codes),
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        if self.path is not None:
            atomic_write_json(
                self.path,
                {
                    "format": CACHE_FORMAT,
                    "version": CACHE_VERSION,
                    "entries": self.values,
                },
            )


class PredicateRunner:
    def __init__(
        self,
        argv_template: Sequence[str],
        interesting_exit_code: int,
        runs: int,
        timeout: float,
        work_root: Path,
        cache: PredicateCache,
        cache_dependencies: Sequence[Path] = (),
        cache_tags: Sequence[str] = (),
    ) -> None:
        if not argv_template:
            raise ReducerError("the interestingness argv must not be empty")
        if runs < 1:
            raise ReducerError("--predicate-runs must be at least 1")
        if timeout <= 0:
            raise ReducerError("--timeout must be greater than zero")
        resolved_executable = shutil.which(argv_template[0])
        if resolved_executable is None:
            raise ReducerError(
                f"could not find predicate executable {argv_template[0]!r}"
            )
        executable_path = Path(resolved_executable).resolve()
        if not executable_path.is_file():
            raise ReducerError(
                f"predicate executable is not a regular file: {executable_path}"
            )
        normalized_arguments: list[str] = []
        for argument in argv_template[1:]:
            if any(placeholder in argument for placeholder in KNOWN_PLACEHOLDERS):
                normalized_arguments.append(argument)
                continue
            argument_path = Path(argument)
            normalized_arguments.append(
                str(argument_path.resolve()) if argument_path.is_file() else argument
            )
        self.argv_template = (str(executable_path),) + tuple(normalized_arguments)
        self.interesting_exit_code = interesting_exit_code
        self.runs = runs
        self.timeout = timeout
        self.work_root = work_root
        self.cache = cache
        self._validate_placeholders()
        self.identity_files = self._collect_identity_files(
            executable_path, cache_dependencies
        )
        if any(not tag for tag in cache_tags):
            raise ReducerError("--cache-tag values must be non-empty")
        self.cache_tags = tuple(cache_tags)
        self.config_digest = hashlib.sha256(
            canonical_json_bytes(
                {
                    "argv": self.argv_template,
                    "identity_files": self.identity_files,
                    "cache_tags": self.cache_tags,
                    "interesting_exit_code": interesting_exit_code,
                    "runs": runs,
                    "timeout": timeout,
                }
            )
        ).hexdigest()

    def _collect_identity_files(
        self, executable: Path, cache_dependencies: Sequence[Path]
    ) -> tuple[dict[str, Any], ...]:
        identified_paths: list[tuple[str, Path]] = [("executable", executable)]
        for index, argument in enumerate(self.argv_template[1:], start=1):
            if any(placeholder in argument for placeholder in KNOWN_PLACEHOLDERS):
                continue
            argument_path = Path(argument)
            if argument_path.is_file():
                identified_paths.append((f"argv[{index}]", argument_path.resolve()))
        for index, dependency in enumerate(cache_dependencies):
            dependency_path = dependency.resolve()
            if not dependency_path.is_file():
                raise ReducerError(
                    f"--cache-dependency is not a regular file: {dependency_path}"
                )
            identified_paths.append((f"dependency[{index}]", dependency_path))

        identities: list[dict[str, Any]] = []
        seen_paths: set[Path] = set()
        for role, path in identified_paths:
            if path in seen_paths:
                continue
            seen_paths.add(path)
            try:
                file_mode = stat.S_IMODE(path.stat().st_mode)
            except OSError as error:
                raise ReducerError(
                    f"could not inspect predicate identity {path}: {error}"
                )
            identity: dict[str, Any] = {
                "path": str(path),
                "role": role,
                "sha256": sha256_file(path),
                "mode": file_mode,
            }
            if role == "executable":
                identity["executable"] = os.access(path, os.X_OK)
            identities.append(identity)
        return tuple(identities)

    def _verify_identity_files(self) -> None:
        for identity in self.identity_files:
            path = Path(identity["path"])
            try:
                current_mode = stat.S_IMODE(path.stat().st_mode)
            except OSError:
                current_mode = None
            executable_matches = identity.get("role") != "executable" or os.access(
                path, os.X_OK
            ) == identity.get("executable")
            if (
                not path.is_file()
                or current_mode != identity.get("mode")
                or not executable_matches
                or sha256_file(path) != identity["sha256"]
            ):
                raise ReducerError(
                    f"predicate identity changed during reduction: {path}"
                )

    def _validate_placeholders(self) -> None:
        for argument in self.argv_template:
            for placeholder in re.findall(r"\{[^{}]+\}", argument):
                if placeholder not in KNOWN_PLACEHOLDERS:
                    raise ReducerError(
                        f"unknown interestingness placeholder {placeholder!r}"
                    )

    def _cache_key(self, candidate_digest: str) -> str:
        digest = hashlib.sha256()
        digest.update(bytes.fromhex(self.config_digest))
        digest.update(bytes.fromhex(candidate_digest))
        return digest.hexdigest()

    def _expand_argv(self, workspace: Path, candidate: Candidate) -> list[str]:
        replacements = {
            "{workspace}": str(workspace),
            "{bundle}": str(workspace / "bundle.json"),
            "{metadata}": str(workspace / "metadata.json"),
        }
        if len(candidate.code_objects) == 1:
            input_name = _object_output_name(
                0,
                candidate.code_objects[0].object_id,
                candidate.code_objects[0].original_name,
            )
            replacements["{input}"] = str(workspace / "objects" / input_name)
        elif any("{input}" in argument for argument in self.argv_template):
            raise ReducerError(
                "{input} requires exactly one code object; use {bundle} "
                "while reducing a multi-object corpus"
            )
        argv: list[str] = []
        for argument in self.argv_template:
            expanded = argument
            for placeholder, value in replacements.items():
                expanded = expanded.replace(placeholder, value)
            argv.append(expanded)
        return argv

    @staticmethod
    def _normalize_capture(value: Any, workspace: Path) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            text = value.decode("utf-8", errors="replace")
        else:
            text = str(value)
        text = text.replace(str(workspace), "{workspace}")
        if len(text) > CAPTURE_LIMIT:
            text = text[:CAPTURE_LIMIT] + "\n<truncated>"
        return text

    @staticmethod
    def _terminate_process_tree(process: subprocess.Popen[bytes]) -> None:
        if os.name == "nt":
            if process.poll() is None:
                try:
                    subprocess.run(
                        [
                            "taskkill.exe",
                            "/PID",
                            str(process.pid),
                            "/T",
                            "/F",
                        ],
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                        timeout=5.0,
                    )
                except (OSError, subprocess.TimeoutExpired):
                    pass
                if process.poll() is None:
                    process.kill()
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except PermissionError:
                if process.poll() is None:
                    process.kill()
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            if process.poll() is None:
                process.kill()
            process.wait()

    def _run_predicate(
        self, argv: Sequence[str], workspace: Path
    ) -> tuple[Optional[int], str, str, bool]:
        creation_flags = (
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
        )
        with tempfile.TemporaryFile() as stdout_stream:
            with tempfile.TemporaryFile() as stderr_stream:
                try:
                    process = subprocess.Popen(
                        argv,
                        cwd=workspace,
                        stdin=subprocess.DEVNULL,
                        stdout=stdout_stream,
                        stderr=stderr_stream,
                        start_new_session=os.name != "nt",
                        creationflags=creation_flags,
                    )
                except OSError as error:
                    return (
                        None,
                        "",
                        self._normalize_capture(
                            f"could not launch predicate: {error}", workspace
                        ),
                        False,
                    )
                timed_out = False
                try:
                    try:
                        return_code = process.wait(timeout=self.timeout)
                    except subprocess.TimeoutExpired:
                        timed_out = True
                        return_code = None
                finally:
                    self._terminate_process_tree(process)
                stdout_stream.seek(0)
                stderr_stream.seek(0)
                stdout = self._normalize_capture(
                    stdout_stream.read(CAPTURE_LIMIT + 1), workspace
                )
                stderr = self._normalize_capture(
                    stderr_stream.read(CAPTURE_LIMIT + 1), workspace
                )
                return return_code, stdout, stderr, timed_out

    def evaluate(self, candidate: Candidate) -> PredicateResult:
        self._verify_identity_files()
        candidate_digest = candidate.digest()
        cache_key = self._cache_key(candidate_digest)
        cached = self.cache.get(cache_key)
        if cached is not None:
            if (
                len(cached.exit_codes) != self.runs
                or len(set(cached.exit_codes)) != 1
                or cached.interesting
                != (cached.exit_codes[0] == self.interesting_exit_code)
            ):
                raise ReducerError(
                    f"{self.cache.path or 'predicate cache'}: corrupt entry {cache_key}"
                )
            return cached

        exit_codes: list[Optional[int]] = []
        outputs: list[str] = []
        errors: list[str] = []
        for _ in range(self.runs):
            with tempfile.TemporaryDirectory(
                prefix="predicate-", dir=self.work_root
            ) as temporary_name:
                workspace = Path(temporary_name)
                materialize_candidate(candidate, workspace)
                argv = self._expand_argv(workspace, candidate)
                return_code, stdout, stderr, timed_out = self._run_predicate(
                    argv, workspace
                )
                exit_codes.append(return_code)
                outputs.append(stdout)
                errors.append(stderr)
                if timed_out:
                    return PredicateResult(
                        "timeout",
                        tuple(exit_codes),
                        "\n".join(outputs),
                        "\n".join(errors),
                    )
                if return_code is None:
                    return PredicateResult(
                        "launch-error",
                        tuple(exit_codes),
                        "\n".join(outputs),
                        "\n".join(errors),
                    )

        self._verify_identity_files()
        if len(set(exit_codes)) != 1:
            status = "flaky"
        elif exit_codes[0] == self.interesting_exit_code:
            status = "interesting"
        else:
            status = "uninteresting"
        result = PredicateResult(
            status,
            tuple(exit_codes),
            "\n".join(outputs),
            "\n".join(errors),
        )
        self.cache.put(cache_key, result)
        return result

    def reproduction_argv(self, candidate: Candidate) -> list[str]:
        replacements = {
            "{workspace}": ".",
            "{bundle}": "bundle.json",
            "{metadata}": "metadata.json",
        }
        if len(candidate.code_objects) == 1:
            replacements["{input}"] = (
                Path("objects")
                / _object_output_name(
                    0,
                    candidate.code_objects[0].object_id,
                    candidate.code_objects[0].original_name,
                )
            ).as_posix()
        argv: list[str] = []
        for argument in self.argv_template:
            expanded = argument
            for placeholder, value in replacements.items():
                expanded = expanded.replace(placeholder, value)
            argv.append(expanded)
        return argv


def ddmin(
    items: Sequence[Any],
    try_complement: Callable[[list[Any], list[Any]], bool],
    minimum_size: int = 0,
) -> list[Any]:
    """Return a 1-minimal subsequence accepted by ``try_complement``."""

    current = list(items)
    if minimum_size < 0 or minimum_size > len(current):
        raise ValueError("invalid minimum_size")
    granularity = 2
    while len(current) > minimum_size:
        granularity = min(granularity, len(current))
        reduced = False
        for part in range(granularity):
            start = part * len(current) // granularity
            end = (part + 1) * len(current) // granularity
            removed = current[start:end]
            complement = current[:start] + current[end:]
            if len(complement) < minimum_size:
                continue
            if try_complement(complement, removed):
                current = complement
                granularity = max(granularity - 1, 2)
                reduced = True
                break
        if reduced:
            continue
        if granularity == len(current):
            break
        granularity = min(len(current), granularity * 2)
    return current


@dataclass(frozen=True)
class SectionInfo:
    name: str
    allocated: bool


class ElfSectionTools:
    def __init__(
        self,
        readobj: str,
        objcopy: str,
        timeout: float,
        allow_patterns: Sequence[str],
        protect_patterns: Sequence[str],
        artifact_root: Path,
    ) -> None:
        if timeout <= 0:
            raise ReducerError("--tool-timeout must be greater than zero")
        self.readobj = self._resolve_tool(readobj, "--readobj")
        self.objcopy = self._resolve_tool(objcopy, "--objcopy")
        self.timeout = timeout
        self.allow_patterns = tuple(allow_patterns)
        self.protect_patterns = DEFAULT_PROTECTED_SECTIONS + tuple(protect_patterns)
        self.artifact_root = artifact_root
        self.artifact_digests: dict[Path, str] = {}

    @staticmethod
    def _resolve_tool(tool: str, option: str) -> str:
        resolved = shutil.which(tool)
        if resolved is None:
            raise ReducerError(f"{option}: could not find executable {tool!r}")
        return resolved

    def list_sections(self, path: Path) -> list[SectionInfo]:
        argv = [
            self.readobj,
            "--sections",
            "--elf-output-style=JSON",
            str(path),
        ]
        try:
            completed = subprocess.run(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired as error:
            raise ReducerError(
                f"llvm-readobj timed out while inspecting {path}"
            ) from error
        except OSError as error:
            raise ReducerError(
                f"could not launch llvm-readobj for {path}: {error}"
            ) from error
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", errors="replace").strip()
            raise ReducerError(
                f"llvm-readobj could not inspect {path}: "
                f"{detail or f'exit {completed.returncode}'}"
            )
        try:
            value = json.loads(
                completed.stdout.decode("utf-8"),
                parse_constant=_reject_json_constant,
            )
            file_value = value[0]
            summary = file_value["FileSummary"]
            section_values = file_value["Sections"]
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            ValueError,
            RecursionError,
            IndexError,
            KeyError,
            TypeError,
        ) as error:
            raise ReducerError(
                f"llvm-readobj returned malformed section JSON for {path}: {error}"
            ) from error
        if not isinstance(summary, dict):
            raise ReducerError(f"llvm-readobj returned no file summary for {path}")
        file_format = summary.get("Format")
        architecture = summary.get("Arch")
        if not isinstance(file_format, str) or not file_format.startswith("elf"):
            raise ReducerError(f"{path}: section reduction only supports ELF objects")
        if not isinstance(architecture, str) or architecture.lower() not in (
            "amdgcn",
            "amdgpu",
        ):
            raise ReducerError(
                f"{path}: section reduction requires an AMDGPU ELF object; "
                f"llvm-readobj reported {architecture!r}"
            )
        if not isinstance(section_values, list):
            raise ReducerError(f"llvm-readobj returned no section list for {path}")

        sections: list[SectionInfo] = []
        for index, wrapped_section in enumerate(section_values):
            try:
                section = wrapped_section["Section"]
                name = section["Name"]["Name"]
                flag_values = section["Flags"]["Flags"]
                flag_names = [flag["Name"] for flag in flag_values]
            except (KeyError, TypeError) as error:
                raise ReducerError(
                    f"llvm-readobj returned malformed section #{index} for {path}"
                ) from error
            if not isinstance(name, str) or not all(
                isinstance(flag, str) for flag in flag_names
            ):
                raise ReducerError(
                    f"llvm-readobj returned malformed section #{index} for {path}"
                )
            sections.append(SectionInfo(name, "SHF_ALLOC" in flag_names))
        return sections

    def removable_sections(self, path: Path) -> list[str]:
        sections_by_name: dict[str, list[SectionInfo]] = {}
        section_order: list[str] = []
        for section in self.list_sections(path):
            if section.name not in sections_by_name:
                sections_by_name[section.name] = []
                section_order.append(section.name)
            sections_by_name[section.name].append(section)

        removable: list[str] = []
        for section_name in section_order:
            same_name_sections = sections_by_name[section_name]
            allowed = any(
                fnmatch.fnmatchcase(section_name, pattern)
                for pattern in self.allow_patterns
            )
            protected = any(section.allocated for section in same_name_sections) or any(
                fnmatch.fnmatchcase(section_name, pattern)
                for pattern in self.protect_patterns
            )
            if allowed and not protected:
                removable.append(section_name)
        return removable

    def _verify_objcopy_output(
        self,
        input_path: Path,
        output_path: Path,
        requested_removals: Sequence[str],
    ) -> None:
        input_sections = self.list_sections(input_path)
        output_sections = self.list_sections(output_path)
        output_counts: dict[str, int] = {}
        for section in output_sections:
            output_counts[section.name] = output_counts.get(section.name, 0) + 1
        requested = set(requested_removals)
        retained_counts: dict[str, int] = {}
        for section in input_sections:
            if section.name not in requested:
                retained_counts[section.name] = retained_counts.get(section.name, 0) + 1
        remaining_requested = sorted(
            name for name in requested if output_counts.get(name, 0) != 0
        )
        missing_retained = sorted(
            name
            for name, count in retained_counts.items()
            if output_counts.get(name, 0) < count
        )
        if remaining_requested or missing_retained:
            details: list[str] = []
            if remaining_requested:
                details.append(
                    "requested sections remain: " + ", ".join(remaining_requested)
                )
            if missing_retained:
                details.append(
                    "unrequested sections disappeared: " + ", ".join(missing_retained)
                )
            raise ReducerError(
                f"llvm-objcopy output verification failed for {input_path}: "
                + "; ".join(details)
            )

    def remove_sections(
        self, code_object: CodeObject, section_names: Sequence[str]
    ) -> CodeObject:
        if not section_names:
            return code_object
        key = hashlib.sha256(
            canonical_json_bytes(
                {
                    "input": code_object.digest,
                    "remove": sorted(section_names),
                }
            )
        ).hexdigest()
        output_path = self.artifact_root / f"{key}.elf"
        if not output_path.exists():
            temporary_path = output_path.with_suffix(".tmp")
            argv = [self.objcopy]
            argv.extend(
                f"--remove-section={section_name}"
                for section_name in sorted(section_names)
            )
            argv.extend((str(code_object.path), str(temporary_path)))
            try:
                completed = subprocess.run(
                    argv,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                    timeout=self.timeout,
                )
            except subprocess.TimeoutExpired as error:
                try:
                    temporary_path.unlink()
                except FileNotFoundError:
                    pass
                raise ReducerError(
                    f"llvm-objcopy timed out while reducing {code_object.object_id!r}"
                ) from error
            except OSError as error:
                try:
                    temporary_path.unlink()
                except FileNotFoundError:
                    pass
                raise ReducerError(
                    f"could not launch llvm-objcopy for "
                    f"{code_object.object_id!r}: {error}"
                ) from error
            if completed.returncode != 0 or not temporary_path.is_file():
                try:
                    temporary_path.unlink()
                except FileNotFoundError:
                    pass
                detail = completed.stderr.decode("utf-8", errors="replace").strip()
                raise ReducerError(
                    f"llvm-objcopy could not reduce "
                    f"{code_object.object_id!r}: "
                    f"{detail or f'exit {completed.returncode}'}"
                )
            if temporary_path.stat().st_size == 0:
                temporary_path.unlink()
                raise ReducerError(
                    f"llvm-objcopy produced an empty object for "
                    f"{code_object.object_id!r}"
                )
            try:
                self._verify_objcopy_output(
                    code_object.path, temporary_path, section_names
                )
            except ReducerError:
                temporary_path.unlink()
                raise
            os.replace(temporary_path, output_path)
            self.artifact_digests[output_path] = sha256_file(output_path)
        output_digest = self.artifact_digests.get(output_path)
        if output_digest is None:
            output_digest = sha256_file(output_path)
            self.artifact_digests[output_path] = output_digest
        return replace(
            code_object,
            path=output_path,
            content_sha256=output_digest,
            removed_sections=tuple(sorted(section_names)),
        )


class Reducer:
    def __init__(
        self,
        initial: Candidate,
        runner: PredicateRunner,
        allow_empty_bundle: bool,
        section_tools: Optional[ElfSectionTools],
    ) -> None:
        self.initial = initial
        self.current = initial
        self.runner = runner
        self.allow_empty_bundle = allow_empty_bundle
        self.section_tools = section_tools
        self.transformations: list[dict[str, Any]] = []

    def _attempt(
        self,
        pass_name: str,
        transformation: dict[str, Any],
        candidate: Candidate,
    ) -> bool:
        candidate_digest = candidate.digest()
        result = self.runner.evaluate(candidate)
        accepted = result.interesting
        self.transformations.append(
            {
                "pass": pass_name,
                "transformation": transformation,
                "candidate_digest": candidate_digest,
                "accepted": accepted,
                "predicate": result.for_log(),
            }
        )
        if accepted:
            self.current = candidate
        return accepted

    def _record_tool_error(
        self, pass_name: str, transformation: dict[str, Any], error: ReducerError
    ) -> None:
        self.transformations.append(
            {
                "pass": pass_name,
                "transformation": transformation,
                "candidate_digest": self.current.digest(),
                "accepted": False,
                "predicate": {
                    "status": "tool-error",
                    "error": str(error),
                },
            }
        )

    def reduce_bundle(self) -> None:
        original = list(self.current.code_objects)

        def try_objects(kept: list[CodeObject], removed: list[CodeObject]) -> bool:
            candidate = replace(self.current, code_objects=tuple(kept))
            return self._attempt(
                "code-objects",
                {"remove_ids": [item.object_id for item in removed]},
                candidate,
            )

        minimum_size = 0 if self.allow_empty_bundle else 1
        kept = ddmin(original, try_objects, minimum_size)
        self.current = replace(self.current, code_objects=tuple(kept))

    @staticmethod
    def _set_json_path(
        value: dict[str, Any], path: Sequence[Any], replacement: Any
    ) -> None:
        target: Any = value
        for component in path[:-1]:
            target = target[component]
        target[path[-1]] = replacement

    def _reduce_metadata_list(self, path: Sequence[Any]) -> None:
        target: Any = self.current.metadata
        for component in path:
            target = target[component]
        original = list(target)
        path_text = ".".join(str(component) for component in path)

        def try_values(kept: list[Any], removed: list[Any]) -> bool:
            metadata = copy.deepcopy(self.current.metadata)
            self._set_json_path(metadata, path, kept)
            candidate = replace(self.current, metadata=metadata)
            removed_digests = [
                hashlib.sha256(canonical_json_bytes(item)).hexdigest()
                for item in removed
            ]
            return self._attempt(
                "metadata",
                {
                    "path": path_text,
                    "remove_value_digests": removed_digests,
                },
                candidate,
            )

        kept = ddmin(original, try_values)
        metadata = copy.deepcopy(self.current.metadata)
        self._set_json_path(metadata, path, kept)
        self.current = replace(self.current, metadata=metadata)

    def reduce_metadata(self) -> None:
        for key in ("kernels", "cases", "arguments", "selected_tests"):
            if key in self.current.metadata and self.current.metadata[key]:
                self._reduce_metadata_list((key,))
        cases = self.current.metadata.get("cases", [])
        for index, case in enumerate(cases):
            if (
                isinstance(case, dict)
                and isinstance(case.get("arguments"), list)
                and case["arguments"]
            ):
                self._reduce_metadata_list(("cases", index, "arguments"))

    def reduce_sections(self) -> None:
        if self.section_tools is None:
            return
        section_tools = self.section_tools
        object_index = 0
        while object_index < len(self.current.code_objects):
            base_object = self.current.code_objects[object_index]
            try:
                removable = section_tools.removable_sections(base_object.path)
            except ReducerError as error:
                raise ReducerError(
                    f"could not enumerate removable sections for "
                    f"{base_object.object_id!r}: {error}"
                ) from error
            if not removable:
                object_index += 1
                continue

            def try_sections(kept: list[str], removed: list[str]) -> bool:
                removed_sections = [
                    section for section in removable if section not in kept
                ]
                transformation = {
                    "object_id": base_object.object_id,
                    "remove_sections": sorted(removed_sections),
                    "delta_remove_sections": sorted(removed),
                }
                try:
                    reduced_object = section_tools.remove_sections(
                        base_object, removed_sections
                    )
                except ReducerError as error:
                    self._record_tool_error("elf-sections", transformation, error)
                    return False
                objects = list(self.current.code_objects)
                objects[object_index] = reduced_object
                candidate = replace(self.current, code_objects=tuple(objects))
                return self._attempt("elf-sections", transformation, candidate)

            kept = ddmin(removable, try_sections)
            removed_sections = [section for section in removable if section not in kept]
            if removed_sections:
                reduced_object = section_tools.remove_sections(
                    base_object, removed_sections
                )
                objects = list(self.current.code_objects)
                objects[object_index] = reduced_object
                self.current = replace(self.current, code_objects=tuple(objects))
            object_index += 1

    def run(self) -> Candidate:
        initial_result = self.runner.evaluate(self.initial)
        if not initial_result.interesting:
            raise ReducerError(
                "the original input is not stably interesting: "
                f"predicate status is {initial_result.status!r}"
            )
        while True:
            previous_digest = self.current.digest()
            self.reduce_bundle()
            self.reduce_metadata()
            self.reduce_sections()
            if self.current.digest() == previous_digest:
                break
        return self.current

    def make_log(self) -> dict[str, Any]:
        originals = [
            {
                "id": code_object.object_id,
                "name": code_object.original_name,
                "sha256": code_object.original_sha256,
            }
            for code_object in self.initial.code_objects
        ]
        return {
            "format": LOG_FORMAT,
            "version": LOG_VERSION,
            "initial_digest": self.initial.digest(),
            "final_digest": self.current.digest(),
            "originals": originals,
            "predicate": {
                "argv_template": list(self.runner.argv_template),
                "identity_files": list(self.runner.identity_files),
                "cache_tags": list(self.runner.cache_tags),
                "interesting_exit_code": self.runner.interesting_exit_code,
                "runs": self.runner.runs,
                "timeout_seconds": self.runner.timeout,
            },
            "section_policy": {
                "enabled": self.section_tools is not None,
                "allow": (
                    list(self.section_tools.allow_patterns)
                    if self.section_tools is not None
                    else []
                ),
                "protect": (
                    list(self.section_tools.protect_patterns)
                    if self.section_tools is not None
                    else []
                ),
                "allocated_sections_are_protected": True,
            },
            "transformations": self.transformations,
            "reproduction": {
                "working_directory": ".",
                "argv": self.runner.reproduction_argv(self.current),
                "interesting_exit_code": self.runner.interesting_exit_code,
            },
        }


def atomic_publish(
    output: Path, candidate: Candidate, reduction_log: dict[str, Any]
) -> None:
    output = output.resolve()
    if output.exists():
        raise ReducerError(f"refusing to overwrite existing output {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    try:
        materialize_candidate(candidate, temporary_path)
        atomic_write_json(temporary_path / "reduction-log.json", reduction_log)
        os.replace(temporary_path, output)
    except BaseException:
        shutil.rmtree(temporary_path, ignore_errors=True)
        raise


def _path_is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return parsed


def make_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--bundle", type=Path, help="Versioned input bundle JSON")
    input_group.add_argument(
        "--code-object",
        type=Path,
        action="append",
        help="Code object input; repeat for a corpus",
    )
    input_group.add_argument(
        "--worklist",
        type=Path,
        help="NUL-delimited code object paths from inventory or selection",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Structured launch/test metadata for --code-object or --worklist",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New output directory"
    )
    parser.add_argument(
        "--predicate",
        required=True,
        help="Interestingness executable (no shell is used)",
    )
    parser.add_argument(
        "--predicate-arg",
        action="append",
        default=[],
        help=(
            "One interestingness argv item; repeat as needed. Known "
            "placeholders: " + ", ".join(KNOWN_PLACEHOLDERS)
        ),
    )
    parser.add_argument(
        "--interesting-exit-code",
        type=int,
        default=0,
        help="Exit code meaning interesting (default: 0)",
    )
    parser.add_argument(
        "--predicate-runs",
        type=int,
        default=1,
        help="Repeat every uncached predicate; disagreement is flaky",
    )
    parser.add_argument(
        "--timeout",
        type=_positive_float,
        default=60.0,
        help="Per-predicate timeout in seconds (default: 60)",
    )
    parser.add_argument(
        "--cache-file",
        type=Path,
        help="Optional persistent content-addressed predicate cache",
    )
    parser.add_argument(
        "--cache-dependency",
        type=Path,
        action="append",
        default=[],
        help="File whose content participates in predicate cache identity",
    )
    parser.add_argument(
        "--cache-tag",
        action="append",
        default=[],
        help="Environment or configuration tag for predicate cache identity",
    )
    parser.add_argument(
        "--allow-empty-bundle",
        action="store_true",
        help="Permit reducing all code objects away",
    )
    parser.add_argument(
        "--allow-remove-section",
        action="append",
        default=[],
        metavar="GLOB",
        help=("Opt in a non-allocated ELF section glob for removal; repeat as needed"),
    )
    parser.add_argument(
        "--protect-section",
        action="append",
        default=[],
        metavar="GLOB",
        help="Additional protected ELF section glob; repeat as needed",
    )
    parser.add_argument(
        "--readobj",
        default="llvm-readobj",
        help="llvm-readobj executable for section reduction",
    )
    parser.add_argument(
        "--objcopy",
        default="llvm-objcopy",
        help="llvm-objcopy executable for section reduction",
    )
    parser.add_argument(
        "--tool-timeout",
        type=_positive_float,
        default=30.0,
        help="Per LLVM tool timeout in seconds (default: 30)",
    )
    return parser


def run_from_arguments(args: argparse.Namespace) -> tuple[Candidate, dict[str, Any]]:
    initial = load_inputs(
        args.bundle,
        args.code_object or [],
        args.worklist,
        args.metadata,
    )
    output = args.output.resolve()
    if output.exists():
        raise ReducerError(f"refusing to overwrite existing output {output}")
    cache_path = args.cache_file.resolve() if args.cache_file is not None else None
    if cache_path is not None:
        protected_inputs = {path.resolve() for path in initial.source_paths}
        protected_inputs.update(path.resolve() for path in args.cache_dependency)
        if cache_path in protected_inputs:
            raise ReducerError(
                f"refusing to overwrite reducer input with cache {cache_path}"
            )
        if _path_is_within(cache_path, output):
            raise ReducerError(
                f"--cache-file must not be inside output directory {output}"
            )

    with tempfile.TemporaryDirectory(prefix="hotswap-reduce-") as temporary_name:
        work_root = Path(temporary_name)
        initial = snapshot_candidate(initial, work_root / "originals")
        runner = PredicateRunner(
            [args.predicate] + args.predicate_arg,
            args.interesting_exit_code,
            args.predicate_runs,
            args.timeout,
            work_root,
            PredicateCache(None),
            args.cache_dependency,
            args.cache_tag,
        )
        if cache_path is not None and cache_path in {
            Path(identity["path"]) for identity in runner.identity_files
        }:
            raise ReducerError(
                f"refusing to overwrite predicate identity with cache {cache_path}"
            )
        runner.cache = PredicateCache(cache_path)
        section_tools: Optional[ElfSectionTools] = None
        if args.allow_remove_section:
            artifact_root = work_root / "artifacts"
            artifact_root.mkdir()
            section_tools = ElfSectionTools(
                args.readobj,
                args.objcopy,
                args.tool_timeout,
                args.allow_remove_section,
                args.protect_section,
                artifact_root,
            )
        reducer = Reducer(
            initial,
            runner,
            args.allow_empty_bundle,
            section_tools,
        )
        final = reducer.run()
        reduction_log = reducer.make_log()
        atomic_publish(output, final, reduction_log)
        published = load_bundle(output / "bundle.json")
        return published, reduction_log


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = make_argument_parser()
    args = parser.parse_args(argv)
    try:
        final, reduction_log = run_from_arguments(args)
    except KeyboardInterrupt:
        print("hotswap-reduce: interrupted; no output was published", file=sys.stderr)
        return 130
    except ReducerError as error:
        print(f"hotswap-reduce: error: {error}", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"hotswap-reduce: filesystem error: {error}", file=sys.stderr)
        return 1
    print(
        f"wrote {args.output} with {len(final.code_objects)} code object(s); "
        f"digest {reduction_log['final_digest']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
