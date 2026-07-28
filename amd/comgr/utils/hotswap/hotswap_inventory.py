#!/usr/bin/env python3
"""Inventory and run a command once per unique AMDGPU code object."""

import argparse
import base64
import concurrent.futures
import functools
import hashlib
import json
import os
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import time


SCHEMA = "comgr.hotswap.inventory"
SCHEMA_VERSION = 2
CACHE_SCHEMA = "comgr.hotswap.inventory.cache"
CACHE_VERSION = 2
EM_AMDGPU = 224
ELF_MAGIC = b"\x7fELF"
HASH_BLOCK_SIZE = 1024 * 1024


class InventoryError(Exception):
    """An input, filesystem, or cache error that should stop the run."""


def path_sort_key(path):
    """Sort paths by their filesystem byte representation."""
    return os.fsencode(path)


def normalized_path(path):
    """Return a deterministic absolute spelling without resolving symlinks."""
    return os.path.normpath(os.path.abspath(path))


def discover_files(roots):
    """Return sorted, distinct regular-file paths below roots.

    File symlinks are included. Directory symlinks found while walking a
    directory are not followed, which avoids cycles. An explicitly named
    symlink to a directory is traversed because the user selected that root.
    """
    discovered = set()

    def visit_directory(directory):
        try:
            with os.scandir(directory) as entries:
                sorted_entries = sorted(entries, key=lambda entry:
                                        os.fsencode(entry.name))
        except OSError as error:
            raise InventoryError("cannot read directory '{}': {}".format(
                directory, error)) from error

        for entry in sorted_entries:
            entry_path = normalized_path(entry.path)
            try:
                if entry.is_dir(follow_symlinks=False):
                    visit_directory(entry_path)
                elif entry.is_file(follow_symlinks=True):
                    discovered.add(entry_path)
                elif entry.is_symlink() and not os.path.exists(entry_path):
                    raise InventoryError(
                        "broken symlink in corpus: '{}'".format(entry_path))
            except OSError as error:
                raise InventoryError("cannot inspect '{}': {}".format(
                    entry_path, error)) from error

    if not roots:
        raise InventoryError("at least one corpus root is required")

    for root in sorted({normalized_path(root) for root in roots},
                       key=path_sort_key):
        if not os.path.lexists(root):
            raise InventoryError("corpus root does not exist: '{}'".format(
                root))
        try:
            if os.path.isdir(root):
                visit_directory(root)
            elif os.path.isfile(root):
                discovered.add(root)
            else:
                raise InventoryError(
                    "corpus root is not a regular file or directory: "
                    "'{}'".format(root))
        except OSError as error:
            raise InventoryError("cannot inspect corpus root '{}': {}".format(
                root, error)) from error

    return sorted(discovered, key=path_sort_key)


def read_manifest(root, manifest_path):
    """Read a newline-delimited list of paths relative to one corpus root."""
    normalized_root = normalized_path(root)
    if not os.path.isdir(normalized_root):
        raise InventoryError(
            "manifest root is not a directory: '{}'".format(normalized_root))
    normalized_manifest = normalized_path(manifest_path)
    try:
        with open(normalized_manifest, "rb") as stream:
            contents = stream.read()
    except OSError as error:
        raise InventoryError("cannot read manifest '{}': {}".format(
            normalized_manifest, error)) from error

    if b"\0" in contents:
        raise InventoryError(
            "manifest contains a NUL byte: '{}'".format(normalized_manifest))
    raw_entries = contents.split(b"\n")
    if raw_entries and raw_entries[-1] == b"":
        raw_entries.pop()

    paths = []
    seen = set()
    for line_number, raw_entry in enumerate(raw_entries, start=1):
        if raw_entry.endswith(b"\r"):
            raw_entry = raw_entry[:-1]
        if not raw_entry:
            raise InventoryError(
                "manifest '{}' has an empty path at line {}".format(
                    normalized_manifest, line_number))
        relative = os.fsdecode(raw_entry)
        if os.path.isabs(relative):
            raise InventoryError(
                "manifest '{}' has an absolute path at line {}".format(
                    normalized_manifest, line_number))
        path = normalized_path(os.path.join(normalized_root, relative))
        try:
            common = os.path.commonpath([normalized_root, path])
        except ValueError as error:
            raise InventoryError(
                "manifest '{}' has an invalid path at line {}".format(
                    normalized_manifest, line_number)) from error
        if common != normalized_root:
            raise InventoryError(
                "manifest '{}' escapes its corpus root at line {}".format(
                    normalized_manifest, line_number))
        if path in seen:
            raise InventoryError(
                "manifest '{}' repeats path '{}'".format(
                    normalized_manifest, relative))
        seen.add(path)
        if not os.path.isfile(path):
            raise InventoryError(
                "manifest path is not a regular file: '{}'".format(path))
        paths.append(path)

    return (
        sorted(paths, key=path_sort_key),
        {
            "path": normalized_manifest,
            "sha256": hashlib.sha256(contents).hexdigest(),
            "entries": len(paths),
        },
    )


def classify_header(header):
    """Return None for an AMDGPU ELF header or a rejection reason."""
    if len(header) < len(ELF_MAGIC) or header[:4] != ELF_MAGIC:
        return "not-elf"
    if len(header) < 16:
        return "truncated-elf-ident"

    elf_class = header[4]
    elf_data = header[5]
    ident_version = header[6]
    if elf_class not in (1, 2):
        return "unsupported-elf-class"
    if elf_data not in (1, 2):
        return "unsupported-elf-endianness"
    if ident_version != 1:
        return "unsupported-elf-ident-version"

    header_size = 52 if elf_class == 1 else 64
    if len(header) < header_size:
        return "truncated-elf-header"

    endian = "<" if elf_data == 1 else ">"
    _elf_type, machine, version = struct.unpack_from(
        endian + "HHI", header, 16)
    if version != 1:
        return "unsupported-elf-version"
    if machine != EM_AMDGPU:
        return "non-amdgpu-elf"

    header_size_offset = 40 if elf_class == 1 else 52
    declared_header_size = struct.unpack_from(
        endian + "H", header, header_size_offset)[0]
    if declared_header_size != header_size:
        return "invalid-elf-header-size"
    return None


def inspect_file(path):
    """Classify and hash one file while detecting concurrent modification."""
    try:
        with open(path, "rb") as stream:
            initial_stat = os.fstat(stream.fileno())
            if not stat.S_ISREG(initial_stat.st_mode):
                raise InventoryError(
                    "corpus path is not a regular file: '{}'".format(path))
            header = stream.read(64)
            rejection = classify_header(header)
            if rejection is not None:
                final_stat = os.fstat(stream.fileno())
                ensure_unchanged(path, initial_stat, final_stat)
                return None, rejection

            digest = hashlib.sha256()
            stream.seek(0)
            while True:
                block = stream.read(HASH_BLOCK_SIZE)
                if not block:
                    break
                digest.update(block)
            final_stat = os.fstat(stream.fileno())
            ensure_unchanged(path, initial_stat, final_stat)
            return {
                "path": path,
                "sha256": digest.hexdigest(),
                "size": initial_stat.st_size,
            }, None
    except InventoryError:
        raise
    except OSError as error:
        raise InventoryError("cannot read '{}': {}".format(path, error)) \
            from error


def ensure_unchanged(path, initial_stat, final_stat):
    """Reject a file that changed while it was being inspected."""
    initial_identity = (
        initial_stat.st_dev,
        initial_stat.st_ino,
        initial_stat.st_size,
        initial_stat.st_mtime_ns,
    )
    final_identity = (
        final_stat.st_dev,
        final_stat.st_ino,
        final_stat.st_size,
        final_stat.st_mtime_ns,
    )
    if initial_identity != final_identity:
        raise InventoryError(
            "corpus file changed while being read: '{}'".format(path))


def build_inventory(roots, manifest_path=None):
    """Build the deterministic JSON-compatible inventory."""
    manifest = None
    if manifest_path is None:
        files = discover_files(roots)
    else:
        if len(roots) != 1:
            raise InventoryError(
                "a manifest requires exactly one corpus root")
        files, manifest = read_manifest(roots[0], manifest_path)
    by_digest = {}
    rejected = []

    for path in files:
        item, reason = inspect_file(path)
        if item is None:
            rejected.append({"path": path, "reason": reason})
            continue
        digest = item["sha256"]
        if digest not in by_digest:
            by_digest[digest] = {
                "sha256": digest,
                "size": item["size"],
                "paths": [],
            }
        by_digest[digest]["paths"].append(path)

    objects = []
    for digest in sorted(by_digest):
        item = by_digest[digest]
        paths = sorted(item["paths"], key=path_sort_key)
        objects.append({
            "sha256": digest,
            "size": item["size"],
            "representative": paths[0],
            "paths": paths,
        })

    code_object_paths = sum(len(item["paths"]) for item in objects)
    duplicate_groups = sum(1 for item in objects
                           if len(item["paths"]) > 1)
    inventory = {
        "schema": SCHEMA,
        "version": SCHEMA_VERSION,
        "roots": sorted({normalized_path(root) for root in roots},
                        key=path_sort_key),
        "summary": {
            "files_examined": len(files),
            "code_object_paths": code_object_paths,
            "unique_code_objects": len(objects),
            "duplicate_paths": code_object_paths - len(objects),
            "duplicate_groups": duplicate_groups,
            "rejected_files": len(rejected),
        },
        "objects": objects,
        "rejected": sorted(rejected,
                           key=lambda item: path_sort_key(item["path"])),
    }
    if manifest is not None:
        inventory["manifest"] = manifest
    return inventory


def json_bytes(value):
    """Serialize JSON canonically."""
    return (json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) +
            "\n").encode("ascii")


def atomic_write(path, contents):
    """Atomically replace path with contents."""
    destination = normalized_path(path)
    directory = os.path.dirname(destination)
    if not os.path.isdir(directory):
        raise InventoryError(
            "output directory does not exist: '{}'".format(directory))
    temporary_path = None
    try:
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=".hotswap-inventory-", dir=directory)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
    except OSError as error:
        raise InventoryError("cannot write '{}': {}".format(
            destination, error)) from error
    finally:
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass


def write_worklist(path, objects):
    """Write representative paths as a NUL-delimited byte stream."""
    contents = b"".join(
        os.fsencode(item["representative"]) + b"\0" for item in objects)
    atomic_write(path, contents)


def protect_inventory_inputs(inventory, output_paths):
    """Refuse to overwrite an input code object or the selected manifest."""
    protected = set()
    for item in inventory["objects"]:
        protected.update(item["paths"])
    protected.update(item["path"] for item in inventory["rejected"])
    if "manifest" in inventory:
        protected.add(inventory["manifest"]["path"])
    for output_path in output_paths:
        if output_path is None or output_path == "-":
            continue
        normalized_output = normalized_path(output_path)
        if normalized_output in protected:
            raise InventoryError(
                "refusing to overwrite inventory input '{}'".format(
                    normalized_output))


def resolve_program(program):
    """Resolve a command executable before computing its cache identity."""
    if os.path.dirname(program):
        resolved = normalized_path(program)
        if not os.path.isfile(resolved):
            raise InventoryError(
                "command executable does not exist: '{}'".format(program))
        if not os.access(resolved, os.X_OK):
            raise InventoryError(
                "command executable is not executable: '{}'".format(program))
        return resolved
    resolved = shutil.which(program)
    if resolved is None:
        raise InventoryError(
            "command executable was not found on PATH: '{}'".format(program))
    return normalized_path(resolved)


def hash_regular_file(path):
    """Hash a regular file used as part of command identity."""
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as stream:
            while True:
                block = stream.read(HASH_BLOCK_SIZE)
                if not block:
                    break
                digest.update(block)
    except OSError as error:
        raise InventoryError("cannot hash command file '{}': {}".format(
            path, error)) from error
    return digest.hexdigest()


def command_identity(argv, dependencies, tags):
    """Return a content-sensitive key for an executable and file arguments."""
    files = []
    for index, argument in enumerate(argv):
        candidate = normalized_path(argument)
        if os.path.isfile(candidate):
            files.append({
                "kind": "argv",
                "index": index,
                "path": candidate,
                "sha256": hash_regular_file(candidate),
            })
    normalized_dependencies = []
    for dependency in dependencies:
        path = normalized_path(dependency)
        if not os.path.isfile(path):
            raise InventoryError(
                "cache dependency is not a regular file: '{}'".format(path))
        normalized_dependencies.append(path)
    for path in sorted(set(normalized_dependencies), key=path_sort_key):
        files.append({
            "kind": "dependency",
            "path": path,
            "sha256": hash_regular_file(path),
        })
    identity = {"argv": argv, "files": files, "tags": tags}
    return hashlib.sha256(json_bytes(identity)).hexdigest()


def read_cache_entry(path, command_key, digest):
    """Read and validate one successful cached result."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as stream:
            entry = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise InventoryError("cannot read cache entry '{}': {}".format(
            path, error)) from error

    if not isinstance(entry, dict):
        raise InventoryError(
            "cache entry '{}' is not a JSON object".format(path))
    expected = {
        "schema": CACHE_SCHEMA,
        "version": CACHE_VERSION,
        "command_key": command_key,
        "sha256": digest,
    }
    for key, value in expected.items():
        if entry.get(key) != value:
            raise InventoryError(
                "cache entry '{}' has invalid {}".format(path, key))
    if entry.get("returncode") != 0:
        raise InventoryError(
            "cache entry '{}' is not a successful result".format(path))
    runtime_ms = entry.get("runtime_ms")
    if (isinstance(runtime_ms, bool) or not isinstance(runtime_ms, int) or
            runtime_ms < 0):
        raise InventoryError(
            "cache entry '{}' has invalid runtime_ms".format(path))
    for key in ("stdout_base64", "stderr_base64"):
        value = entry.get(key)
        if not isinstance(value, str):
            raise InventoryError(
                "cache entry '{}' has invalid {}".format(path, key))
        try:
            base64.b64decode(value.encode("ascii"), validate=True)
        except (ValueError, UnicodeError) as error:
            raise InventoryError(
                "cache entry '{}' has invalid {}".format(path, key)) \
                from error
    return entry


def elapsed_runtime_ms(start_time):
    """Convert monotonic elapsed time to a nonnegative integer."""
    return int(round(max(0.0, time.monotonic() - start_time) * 1000))


def run_one_command(item, argv_prefix, command_key, timeout,
                    normalized_cache):
    """Run or load the result for one unique content digest."""
    representative = item["representative"]
    digest = item["sha256"]
    argv = argv_prefix + [representative]
    cache_path = None
    cached = None
    if normalized_cache is not None:
        cache_path = os.path.join(
            normalized_cache, command_key, digest + ".json")
        cached = read_cache_entry(cache_path, command_key, digest)
    if cached is not None:
        return {
            "sha256": digest,
            "path": representative,
            "argv": argv,
            "status": "passed",
            "cached": True,
            "returncode": 0,
            "runtime_ms": cached["runtime_ms"],
            "stdout_base64": cached["stdout_base64"],
            "stderr_base64": cached["stderr_base64"],
        }

    start_time = time.monotonic()
    try:
        completed = subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
            shell=False)
        result = {
            "sha256": digest,
            "path": representative,
            "argv": argv,
            "status": ("passed" if completed.returncode == 0
                       else "failed"),
            "cached": False,
            "returncode": completed.returncode,
            "runtime_ms": elapsed_runtime_ms(start_time),
            "stdout_base64": base64.b64encode(
                completed.stdout).decode("ascii"),
            "stderr_base64": base64.b64encode(
                completed.stderr).decode("ascii"),
        }
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout if error.stdout is not None else b""
        stderr = error.stderr if error.stderr is not None else b""
        result = {
            "sha256": digest,
            "path": representative,
            "argv": argv,
            "status": "timed-out",
            "cached": False,
            "returncode": None,
            "runtime_ms": elapsed_runtime_ms(start_time),
            "stdout_base64": base64.b64encode(stdout).decode("ascii"),
            "stderr_base64": base64.b64encode(stderr).decode("ascii"),
        }
    except OSError as error:
        result = {
            "sha256": digest,
            "path": representative,
            "argv": argv,
            "status": "launch-error",
            "cached": False,
            "returncode": None,
            "runtime_ms": elapsed_runtime_ms(start_time),
            "error": str(error),
            "stdout_base64": "",
            "stderr_base64": "",
        }

    if cache_path is not None and result["status"] == "passed":
        cache_entry = {
            "schema": CACHE_SCHEMA,
            "version": CACHE_VERSION,
            "command_key": command_key,
            "sha256": digest,
            "returncode": 0,
            "runtime_ms": result["runtime_ms"],
            "stdout_base64": result["stdout_base64"],
            "stderr_base64": result["stderr_base64"],
        }
        atomic_write(cache_path, json_bytes(cache_entry))
    return result


def run_command(inventory, program, arguments, timeout, cache_dir, jobs,
                cache_dependencies, cache_tags):
    """Run argv plus each unique representative, without invoking a shell."""
    resolved_program = resolve_program(program)
    argv_prefix = [resolved_program] + arguments
    command_key = command_identity(
        argv_prefix, cache_dependencies, cache_tags)

    normalized_cache = None
    if cache_dir is not None:
        normalized_cache = normalized_path(cache_dir)
        try:
            os.makedirs(os.path.join(normalized_cache, command_key),
                        exist_ok=True)
        except OSError as error:
            raise InventoryError("cannot create cache directory '{}': {}"
                                 .format(normalized_cache, error)) from error

    run_one = functools.partial(
        run_one_command,
        argv_prefix=argv_prefix,
        command_key=command_key,
        timeout=timeout,
        normalized_cache=normalized_cache)
    if jobs == 1:
        results = [run_one(item) for item in inventory["objects"]]
    else:
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=jobs) as executor:
            results = list(executor.map(run_one, inventory["objects"]))

    return {
        "command": argv_prefix,
        "command_key": command_key,
        "cache_dependencies": sorted(
            {normalized_path(path) for path in cache_dependencies},
            key=path_sort_key),
        "cache_tags": cache_tags,
        "timeout_seconds": timeout,
        "jobs": jobs,
        "results": results,
        "summary": {
            "total": len(results),
            "passed": sum(1 for result in results
                          if result["status"] == "passed"),
            "failed": sum(1 for result in results
                          if result["status"] != "passed"),
            "cache_hits": sum(1 for result in results if result["cached"]),
            "estimated_runtime_ms": sum(
                result["runtime_ms"] for result in results),
            "executed_runtime_ms": sum(
                result["runtime_ms"] for result in results
                if not result["cached"]),
        },
    }


def create_argument_parser():
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Inventory AMDGPU ELF code objects, deduplicate them by SHA-256, "
            "and optionally run one command per unique object."))
    parser.add_argument(
        "roots", metavar="ROOT", nargs="+",
        help="file or directory to inventory recursively")
    parser.add_argument(
        "--json-output", metavar="PATH", default="-",
        help="write the JSON report to PATH (default: standard output)")
    parser.add_argument(
        "--worklist", metavar="PATH",
        help="write unique representative paths as NUL-delimited bytes")
    parser.add_argument(
        "--manifest", metavar="PATH",
        help=("inventory only newline-delimited paths from PATH, relative to "
              "one corpus root"))
    parser.add_argument(
        "--execute", metavar="PROGRAM",
        help="run PROGRAM once per unique object; the path is appended")
    parser.add_argument(
        "--execute-arg", metavar="ARG", action="append", default=[],
        help=("pass ARG before the object path; repeat as needed (use "
              "--execute-arg=--flag for arguments beginning with '-')"))
    parser.add_argument(
        "--timeout", metavar="SECONDS", type=float,
        help="terminate each command after this many seconds")
    parser.add_argument(
        "--cache-dir", metavar="PATH",
        help="reuse successful results by command and input content hash")
    parser.add_argument(
        "--cache-dependency", metavar="PATH", action="append", default=[],
        help="hash PATH into the command cache key; repeat as needed")
    parser.add_argument(
        "--cache-tag", metavar="TEXT", action="append", default=[],
        help="include TEXT in the command cache key; repeat as needed")
    parser.add_argument(
        "--jobs", metavar="COUNT", type=int, default=1,
        help="run up to COUNT commands concurrently (default: 1)")
    return parser


def validate_arguments(parser, arguments):
    """Validate relationships not expressible with argparse declarations."""
    if arguments.timeout is not None and arguments.timeout <= 0:
        parser.error("--timeout must be greater than zero")
    if arguments.jobs <= 0:
        parser.error("--jobs must be greater than zero")
    if arguments.execute is None:
        if arguments.execute_arg:
            parser.error("--execute-arg requires --execute")
        if arguments.timeout is not None:
            parser.error("--timeout requires --execute")
        if arguments.cache_dir is not None:
            parser.error("--cache-dir requires --execute")
        if arguments.cache_dependency:
            parser.error("--cache-dependency requires --execute")
        if arguments.cache_tag:
            parser.error("--cache-tag requires --execute")
        if arguments.jobs != 1:
            parser.error("--jobs requires --execute")
    if arguments.worklist == "-":
        parser.error("--worklist requires a file path, not standard output")
    if arguments.manifest is not None and len(arguments.roots) != 1:
        parser.error("--manifest requires exactly one corpus root")
    if (arguments.worklist is not None and
            arguments.json_output != "-" and
            normalized_path(arguments.worklist) ==
            normalized_path(arguments.json_output)):
        parser.error("--worklist and --json-output must be different files")


def main(argv=None):
    """Command-line entry point."""
    parser = create_argument_parser()
    arguments = parser.parse_args(argv)
    validate_arguments(parser, arguments)

    try:
        inventory = build_inventory(arguments.roots, arguments.manifest)
        protect_inventory_inputs(
            inventory, [arguments.worklist, arguments.json_output])
        if arguments.worklist is not None:
            write_worklist(arguments.worklist, inventory["objects"])

        command_failed = False
        if arguments.execute is not None:
            execution = run_command(
                inventory,
                arguments.execute,
                arguments.execute_arg,
                arguments.timeout,
                arguments.cache_dir,
                arguments.jobs,
                arguments.cache_dependency,
                arguments.cache_tag)
            inventory["execution"] = execution
            command_failed = execution["summary"]["failed"] != 0

        report = json_bytes(inventory)
        if arguments.json_output == "-":
            sys.stdout.buffer.write(report)
            sys.stdout.buffer.flush()
        else:
            atomic_write(arguments.json_output, report)
        return 1 if command_failed else 0
    except (InventoryError, OSError) as error:
        print("hotswap-inventory: error: {}".format(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
