#!/usr/bin/env python3
"""Hermetic tests for utils/hotswap/hotswap_inventory.py."""

import base64
import hashlib
import importlib.util
import json
import os
import stat
import struct
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


SCRIPT = (Path(__file__).resolve().parents[2] / "utils" / "hotswap" /
          "hotswap_inventory.py")
SPEC = importlib.util.spec_from_file_location("hotswap_inventory", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
INVENTORY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INVENTORY)


def make_elf(payload=b"", machine=INVENTORY.EM_AMDGPU, elf_class=2,
             endian="<", ident_version=1, elf_version=1):
    """Build the smallest header accepted by the inventory classifier."""
    header_size = 52 if elf_class == 1 else 64
    header = bytearray(header_size)
    header[:4] = INVENTORY.ELF_MAGIC
    header[4] = elf_class
    header[5] = 1 if endian == "<" else 2
    header[6] = ident_version
    struct.pack_into(endian + "HHI", header, 16, 3, machine, elf_version)
    header_size_offset = 40 if elf_class == 1 else 52
    struct.pack_into(endian + "H", header, header_size_offset, header_size)
    return bytes(header) + payload


class HotswapInventoryTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def write(self, relative_path, contents):
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
        return path

    def run_cli(self, *arguments, env=None, cwd=None):
        command = [sys.executable, str(SCRIPT)] + [
            str(argument) for argument in arguments
        ]
        return subprocess.run(
            command,
            cwd=str(cwd or self.root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False)

    def parse_report(self, completed):
        return json.loads(completed.stdout.decode("ascii"))

    def make_command(self, body):
        script = self.write("command.py", textwrap.dedent(body).encode())
        return [sys.executable, "--execute-arg", script]

    def test_empty_corpus(self):
        report = INVENTORY.build_inventory([str(self.root)])
        self.assertEqual(report["schema"], INVENTORY.SCHEMA)
        self.assertEqual(report["version"], 2)
        self.assertEqual(report["objects"], [])
        self.assertEqual(report["rejected"], [])
        self.assertEqual(
            report["summary"],
            {
                "files_examined": 0,
                "code_object_paths": 0,
                "unique_code_objects": 0,
                "duplicate_paths": 0,
                "duplicate_groups": 0,
                "rejected_files": 0,
            })

    def test_recurses_and_deduplicates_by_content_not_suffix(self):
        contents = make_elf(b"same")
        first = self.write("z/object.hsaco", contents)
        second = self.write("a/no-suffix", contents)
        unique = self.write("nested/deeper/object.txt", make_elf(b"unique"))
        self.write("nested/not-an-elf.hsaco", b"plain text")

        report = INVENTORY.build_inventory([str(self.root)])

        self.assertEqual(report["summary"]["files_examined"], 4)
        self.assertEqual(report["summary"]["code_object_paths"], 3)
        self.assertEqual(report["summary"]["unique_code_objects"], 2)
        self.assertEqual(report["summary"]["duplicate_paths"], 1)
        self.assertEqual(report["summary"]["duplicate_groups"], 1)
        groups = {item["sha256"]: item for item in report["objects"]}
        duplicate = groups[hashlib.sha256(contents).hexdigest()]
        expected_paths = sorted(
            [str(first), str(second)], key=os.fsencode)
        self.assertEqual(duplicate["paths"], expected_paths)
        self.assertEqual(duplicate["representative"], expected_paths[0])
        self.assertIn(str(unique), [
            item["representative"] for item in report["objects"]
        ])

    def test_rejects_non_elf_other_machine_and_malformed_elf(self):
        plain = self.write("plain", b"not elf")
        host = self.write("host", make_elf(machine=62))
        truncated = self.write("truncated", INVENTORY.ELF_MAGIC + b"\x02")
        invalid_size = bytearray(make_elf())
        struct.pack_into("<H", invalid_size, 52, 0)
        invalid = self.write("invalid-size", invalid_size)

        report = INVENTORY.build_inventory([str(self.root)])
        reasons = {item["path"]: item["reason"]
                   for item in report["rejected"]}

        self.assertEqual(reasons[str(plain)], "not-elf")
        self.assertEqual(reasons[str(host)], "non-amdgpu-elf")
        self.assertEqual(reasons[str(truncated)], "truncated-elf-ident")
        self.assertEqual(reasons[str(invalid)], "invalid-elf-header-size")

    def test_accepts_32_bit_and_big_endian_headers(self):
        little32 = self.write(
            "little32", make_elf(b"32", elf_class=1, endian="<"))
        big64 = self.write(
            "big64", make_elf(b"64", elf_class=2, endian=">"))

        report = INVENTORY.build_inventory([str(self.root)])

        self.assertEqual(report["summary"]["unique_code_objects"], 2)
        self.assertEqual(
            {item["representative"] for item in report["objects"]},
            {str(little32), str(big64)})

    def test_repeated_and_overlapping_roots_do_not_repeat_paths(self):
        code_object = self.write("nested/object", make_elf())
        report = INVENTORY.build_inventory([
            str(self.root),
            str(self.root),
            str(code_object.parent),
            str(code_object),
        ])
        self.assertEqual(report["summary"]["files_examined"], 1)
        self.assertEqual(report["summary"]["code_object_paths"], 1)

    def test_multiple_output_roots_are_grouped_by_content_digest(self):
        shared = make_elf(b"shared output")
        oracle_shared = self.write("oracle/output/shared", shared)
        candidate_shared = self.write("candidate/output/shared", shared)
        oracle_changed = self.write(
            "oracle/output/changed", make_elf(b"oracle"))
        candidate_changed = self.write(
            "candidate/output/changed", make_elf(b"candidate"))

        report = INVENTORY.build_inventory([
            str(self.root / "candidate" / "output"),
            str(self.root / "oracle" / "output"),
        ])

        groups = {item["sha256"]: item for item in report["objects"]}
        shared_group = groups[hashlib.sha256(shared).hexdigest()]
        self.assertEqual(
            shared_group["paths"],
            sorted([str(oracle_shared), str(candidate_shared)],
                   key=os.fsencode))
        self.assertEqual(
            {item["representative"] for item in report["objects"]
             if len(item["paths"]) == 1},
            {str(oracle_changed), str(candidate_changed)})

    def test_manifest_selects_exact_files_and_records_its_digest(self):
        first = self.write("corpus/first", make_elf(b"first"))
        second = self.write("corpus/nested/second", make_elf(b"second"))
        self.write("corpus/unlisted", make_elf(b"unlisted"))
        manifest_contents = b"nested/second\nfirst\n"
        manifest = self.write("manifest.txt", manifest_contents)

        report = INVENTORY.build_inventory(
            [str(self.root / "corpus")], str(manifest))

        self.assertEqual(report["summary"]["files_examined"], 2)
        self.assertEqual(report["summary"]["unique_code_objects"], 2)
        self.assertEqual(
            {item["representative"] for item in report["objects"]},
            {str(first), str(second)})
        self.assertEqual(
            report["manifest"],
            {
                "path": str(manifest),
                "sha256": hashlib.sha256(manifest_contents).hexdigest(),
                "entries": 2,
            })

    def test_manifest_errors_are_specific(self):
        corpus = self.root / "corpus"
        corpus.mkdir()
        self.write("corpus/object", make_elf())
        cases = {
            "empty": b"object\n\n",
            "duplicate": b"object\nobject\n",
            "absolute": os.fsencode(str(corpus / "object")) + b"\n",
            "escape": b"../manifest.txt\n",
            "missing": b"missing\n",
            "nul": b"object\0suffix\n",
        }
        for name, contents in cases.items():
            with self.subTest(name=name):
                manifest = self.write(
                    "manifests/{}.txt".format(name), contents)
                completed = self.run_cli(
                    corpus, "--manifest", manifest)
                self.assertEqual(completed.returncode, 2)
                self.assertIn(b"manifest", completed.stderr)

    def test_manifest_requires_one_root(self):
        first_root = self.root / "first-root"
        second_root = self.root / "second-root"
        first_root.mkdir()
        second_root.mkdir()
        manifest = self.write("manifest.txt", b"object\n")
        completed = self.run_cli(
            first_root, second_root, "--manifest", manifest)
        self.assertEqual(completed.returncode, 2)
        self.assertIn(b"exactly one corpus root", completed.stderr)

    @unittest.skipIf(
        os.name == "nt" or not hasattr(os, "symlink"),
        "symlink creation may require Windows developer mode")
    def test_file_symlink_is_duplicate_and_directory_symlink_is_not_followed(
            self):
        target = self.write("real/object", make_elf())
        file_link = self.root / "file-link"
        file_link.symlink_to(target)
        directory_link = self.root / "directory-link"
        directory_link.symlink_to(target.parent, target_is_directory=True)

        report = INVENTORY.build_inventory([str(self.root)])

        self.assertEqual(report["summary"]["files_examined"], 2)
        self.assertEqual(report["summary"]["code_object_paths"], 2)
        self.assertEqual(report["summary"]["unique_code_objects"], 1)
        self.assertEqual(
            report["objects"][0]["paths"],
            sorted([str(target), str(file_link)], key=os.fsencode))

    @unittest.skipIf(
        os.name == "nt" or not hasattr(os, "symlink"),
        "symlink creation may require Windows developer mode")
    def test_explicit_directory_symlink_root_is_traversed(self):
        target = self.write("real/object", make_elf())
        link = self.root / "root-link"
        link.symlink_to(target.parent, target_is_directory=True)

        report = INVENTORY.build_inventory([str(link)])

        self.assertEqual(report["summary"]["code_object_paths"], 1)
        self.assertEqual(
            report["objects"][0]["representative"],
            str(link / target.name))

    @unittest.skipIf(
        os.name == "nt" or not hasattr(os, "symlink"),
        "symlink creation may require Windows developer mode")
    def test_broken_symlink_fails_loudly(self):
        (self.root / "broken").symlink_to(self.root / "missing")
        completed = self.run_cli(self.root)
        self.assertEqual(completed.returncode, 2)
        self.assertIn(b"broken symlink", completed.stderr)
        self.assertEqual(completed.stdout, b"")

    def test_json_is_byte_for_byte_deterministic(self):
        self.write("z", make_elf(b"z"))
        self.write("a", make_elf(b"a"))
        first = self.run_cli(self.root)
        second = self.run_cli(self.root)
        self.assertEqual(first.returncode, 0)
        self.assertEqual(first.stdout, second.stdout)
        self.assertTrue(first.stdout.endswith(b"\n"))

    def test_worklist_is_nul_safe_and_uses_representatives(self):
        first = self.write("space name\nline", make_elf(b"one"))
        self.write("duplicate", first.read_bytes())
        second = self.write("literal;$(touch PWNED)", make_elf(b"two"))
        worklist = self.root / "worklist"

        completed = self.run_cli(self.root, "--worklist", worklist)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        paths = worklist.read_bytes().split(b"\0")
        self.assertEqual(paths[-1], b"")
        report = self.parse_report(completed)
        expected = [
            os.fsencode(item["representative"])
            for item in report["objects"]
        ]
        self.assertEqual(paths[:-1], expected)
        self.assertIn(os.fsencode(second), paths)
        self.assertFalse((self.root / "PWNED").exists())

    def test_command_runs_once_per_unique_object_without_a_shell(self):
        duplicate_contents = make_elf(b"duplicate")
        self.write("first", duplicate_contents)
        self.write("second", duplicate_contents)
        dangerous = self.write("$(touch PWNED)", make_elf(b"unique"))
        log = self.root / "command-log"
        command = self.make_command(
            """
            import json
            import os
            import sys
            with open(os.environ["INVENTORY_LOG"], "a", encoding="utf-8") as f:
                f.write(json.dumps(sys.argv[-1]) + "\\n")
            print("processed")
            """)
        environment = os.environ.copy()
        environment["INVENTORY_LOG"] = str(log)

        completed = self.run_cli(
            self.root,
            "--execute", command[0],
            "--execute-arg", command[2],
            env=environment)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = self.parse_report(completed)
        self.assertEqual(report["execution"]["summary"]["total"], 2)
        self.assertEqual(report["execution"]["summary"]["passed"], 2)
        logged_paths = [json.loads(line)
                        for line in log.read_text().splitlines()]
        self.assertEqual(
            logged_paths,
            [item["representative"] for item in report["objects"]])
        self.assertIn(str(dangerous), logged_paths)
        self.assertFalse((self.root / "PWNED").exists())
        for result in report["execution"]["results"]:
            self.assertEqual(
                base64.b64decode(result["stdout_base64"]), b"processed\n")

    def test_command_failure_is_recorded_and_returns_one(self):
        self.write("object", make_elf())
        command = self.make_command(
            """
            import sys
            print("bad", file=sys.stderr)
            sys.exit(7)
            """)

        completed = self.run_cli(
            self.root,
            "--execute", command[0],
            "--execute-arg", command[2])

        self.assertEqual(completed.returncode, 1)
        result = self.parse_report(completed)["execution"]["results"][0]
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["returncode"], 7)
        self.assertIsInstance(result["runtime_ms"], int)
        self.assertGreaterEqual(result["runtime_ms"], 0)
        self.assertEqual(
            base64.b64decode(result["stderr_base64"]), b"bad\n")

    def test_parallel_jobs_preserve_deterministic_result_order(self):
        for index in range(5):
            self.write(
                "object-{}".format(index),
                make_elf(str(index).encode("ascii")))
        command = self.make_command(
            """
            import sys
            print(sys.argv[-1])
            """)

        completed = self.run_cli(
            self.root,
            "--execute", command[0],
            "--execute-arg", command[2],
            "--jobs", "3")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = self.parse_report(completed)
        execution = report["execution"]
        self.assertEqual(execution["jobs"], 3)
        self.assertEqual(
            [result["path"] for result in execution["results"]],
            [item["representative"] for item in report["objects"]])
        for result in execution["results"]:
            self.assertEqual(
                base64.b64decode(result["stdout_base64"]).decode().strip(),
                result["path"])

    def test_timeout_is_recorded_and_returns_one(self):
        self.write("object", make_elf())
        command = self.make_command(
            """
            import time
            time.sleep(10)
            """)

        completed = self.run_cli(
            self.root,
            "--execute", command[0],
            "--execute-arg", command[2],
            "--timeout", "0.05")

        self.assertEqual(completed.returncode, 1)
        result = self.parse_report(completed)["execution"]["results"][0]
        self.assertEqual(result["status"], "timed-out")
        self.assertIsNone(result["returncode"])
        self.assertGreaterEqual(result["runtime_ms"], 40)

    def test_success_cache_hits_and_input_changes_invalidate(self):
        code_object = self.write("object", make_elf(b"first"))
        counter = self.root / "counter"
        cache = self.root / "cache"
        command = self.make_command(
            """
            import os
            from pathlib import Path
            counter = Path(os.environ["INVENTORY_COUNTER"])
            value = int(counter.read_text()) if counter.exists() else 0
            counter.write_text(str(value + 1))
            """)
        environment = os.environ.copy()
        environment["INVENTORY_COUNTER"] = str(counter)
        arguments = [
            self.root,
            "--execute", command[0],
            "--execute-arg", command[2],
            "--cache-dir", cache,
        ]

        first = self.run_cli(*arguments, env=environment)
        second = self.run_cli(*arguments, env=environment)
        code_object.write_bytes(make_elf(b"changed"))
        third = self.run_cli(*arguments, env=environment)

        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(third.returncode, 0, third.stderr)
        self.assertEqual(counter.read_text(), "2")
        self.assertEqual(
            self.parse_report(first)["execution"]["summary"]["cache_hits"], 0)
        self.assertEqual(
            self.parse_report(second)["execution"]["summary"]["cache_hits"], 1)
        self.assertEqual(
            self.parse_report(third)["execution"]["summary"]["cache_hits"], 0)
        first_execution = self.parse_report(first)["execution"]
        second_execution = self.parse_report(second)["execution"]
        self.assertEqual(
            first_execution["results"][0]["runtime_ms"],
            second_execution["results"][0]["runtime_ms"])
        self.assertEqual(
            second_execution["summary"]["estimated_runtime_ms"],
            second_execution["results"][0]["runtime_ms"])
        self.assertEqual(
            second_execution["summary"]["executed_runtime_ms"], 0)

    def test_failed_command_is_not_cached(self):
        self.write("object", make_elf())
        counter = self.root / "counter"
        cache = self.root / "cache"
        command = self.make_command(
            """
            import os
            import sys
            from pathlib import Path
            counter = Path(os.environ["INVENTORY_COUNTER"])
            value = int(counter.read_text()) if counter.exists() else 0
            counter.write_text(str(value + 1))
            sys.exit(1)
            """)
        environment = os.environ.copy()
        environment["INVENTORY_COUNTER"] = str(counter)
        arguments = [
            self.root,
            "--execute", command[0],
            "--execute-arg", command[2],
            "--cache-dir", cache,
        ]

        first = self.run_cli(*arguments, env=environment)
        second = self.run_cli(*arguments, env=environment)

        self.assertEqual(first.returncode, 1)
        self.assertEqual(second.returncode, 1)
        self.assertEqual(counter.read_text(), "2")

    def test_corrupt_cache_fails_loudly(self):
        self.write("object", make_elf())
        cache = self.root / "cache"
        command = self.make_command("print('ok')\n")
        arguments = [
            self.root,
            "--execute", command[0],
            "--execute-arg", command[2],
            "--cache-dir", cache,
        ]
        first = self.run_cli(*arguments)
        self.assertEqual(first.returncode, 0, first.stderr)
        cache_entries = list(cache.rglob("*.json"))
        self.assertEqual(len(cache_entries), 1)
        cache_entries[0].write_text("{not json")

        second = self.run_cli(*arguments)

        self.assertEqual(second.returncode, 2)
        self.assertIn(b"cannot read cache entry", second.stderr)
        self.assertEqual(second.stdout, b"")

    def test_non_object_cache_entry_fails_loudly(self):
        self.write("object", make_elf())
        cache = self.root / "cache"
        command = self.make_command("print('ok')\n")
        arguments = [
            self.root,
            "--execute", command[0],
            "--execute-arg", command[2],
            "--cache-dir", cache,
        ]
        first = self.run_cli(*arguments)
        self.assertEqual(first.returncode, 0, first.stderr)
        cache_entry = next(cache.rglob("*.json"))
        cache_entry.write_text("[]")

        second = self.run_cli(*arguments)

        self.assertEqual(second.returncode, 2)
        self.assertIn(b"is not a JSON object", second.stderr)

    def test_stale_cache_version_is_rejected(self):
        self.write("object", make_elf())
        cache = self.root / "cache"
        command = self.make_command("print('ok')\n")
        arguments = [
            self.root,
            "--execute", command[0],
            "--execute-arg", command[2],
            "--cache-dir", cache,
        ]
        first = self.run_cli(*arguments)
        self.assertEqual(first.returncode, 0, first.stderr)
        cache_entry = next(cache.rglob("*.json"))
        entry = json.loads(cache_entry.read_text())
        self.assertEqual(entry["version"], INVENTORY.CACHE_VERSION)
        entry["version"] = INVENTORY.CACHE_VERSION - 1
        cache_entry.write_text(json.dumps(entry))

        second = self.run_cli(*arguments)

        self.assertEqual(second.returncode, 2)
        self.assertIn(b"invalid version", second.stderr)

    def test_cache_runtime_requires_nonnegative_integer(self):
        self.write("object", make_elf())
        cache = self.root / "cache"
        command = self.make_command("print('ok')\n")
        arguments = [
            self.root,
            "--execute", command[0],
            "--execute-arg", command[2],
            "--cache-dir", cache,
        ]
        first = self.run_cli(*arguments)
        self.assertEqual(first.returncode, 0, first.stderr)
        cache_entry = next(cache.rglob("*.json"))
        entry = json.loads(cache_entry.read_text())
        for invalid_runtime in (-1, 1.5, True, None):
            with self.subTest(invalid_runtime=invalid_runtime):
                entry["runtime_ms"] = invalid_runtime
                cache_entry.write_text(json.dumps(entry))
                completed = self.run_cli(*arguments)
                self.assertEqual(completed.returncode, 2)
                self.assertIn(b"invalid runtime_ms", completed.stderr)

    def test_command_file_content_change_invalidates_cache_identity(self):
        self.write("object", make_elf())
        cache = self.root / "cache"
        command_script = self.write("worker.py", b"print('first')\n")
        arguments = [
            self.root,
            "--execute", sys.executable,
            "--execute-arg", command_script,
            "--cache-dir", cache,
        ]

        first = self.run_cli(*arguments)
        command_script.write_bytes(b"print('second')\n")
        second = self.run_cli(*arguments)

        first_execution = self.parse_report(first)["execution"]
        second_execution = self.parse_report(second)["execution"]
        self.assertNotEqual(
            first_execution["command_key"], second_execution["command_key"])
        self.assertEqual(second_execution["summary"]["cache_hits"], 0)
        self.assertEqual(
            base64.b64decode(
                second_execution["results"][0]["stdout_base64"]),
            b"second\n")

    def test_cache_dependency_and_tag_change_command_identity(self):
        self.write("object", make_elf())
        dependency = self.write("libcandidate.so", b"version one")
        command = self.make_command("print('ok')\n")

        def run(tag):
            return self.run_cli(
                self.root,
                "--execute", command[0],
                "--execute-arg", command[2],
                "--cache-dependency", dependency,
                "--cache-tag", tag)

        first = self.parse_report(run("configuration-one"))["execution"]
        dependency.write_bytes(b"version two")
        second = self.parse_report(run("configuration-one"))["execution"]
        third = self.parse_report(run("configuration-two"))["execution"]

        self.assertNotEqual(first["command_key"], second["command_key"])
        self.assertNotEqual(second["command_key"], third["command_key"])
        self.assertEqual(
            second["cache_dependencies"], [str(dependency)])
        self.assertEqual(third["cache_tags"], ["configuration-two"])

    def test_unknown_flag_and_invalid_option_relationships_are_rejected(self):
        for arguments in (
                [self.root, "--unknown"],
                [self.root, "--execute-arg", "x"],
                [self.root, "--cache-dir", "cache"],
                [self.root, "--cache-dependency", "dependency"],
                [self.root, "--cache-tag", "tag"],
                [self.root, "--timeout", "0"],
                [self.root, "--timeout", "1"],
                [self.root, "--jobs", "0"],
                [self.root, "--jobs", "2"],
                [self.root, "--worklist", "-"]):
            with self.subTest(arguments=arguments):
                completed = self.run_cli(*arguments)
                self.assertEqual(completed.returncode, 2)
                self.assertIn(b"error:", completed.stderr)

    def test_missing_root_and_missing_output_directory_fail_loudly(self):
        missing = self.run_cli(self.root / "missing")
        self.assertEqual(missing.returncode, 2)
        self.assertIn(b"does not exist", missing.stderr)

        output = self.run_cli(
            self.root, "--json-output", self.root / "missing" / "report.json")
        self.assertEqual(output.returncode, 2)
        self.assertIn(b"output directory does not exist", output.stderr)

    def test_same_json_and_worklist_path_is_rejected(self):
        output = self.root / "output"
        completed = self.run_cli(
            self.root, "--json-output", output, "--worklist", output)
        self.assertEqual(completed.returncode, 2)
        self.assertIn(b"must be different", completed.stderr)

    def test_output_cannot_overwrite_input_or_manifest(self):
        code_object = self.write("corpus/object", make_elf())
        manifest = self.write("manifest.txt", b"object\n")
        for arguments in (
                [self.root / "corpus", "--json-output", code_object],
                [self.root / "corpus", "--worklist", code_object],
                [
                    self.root / "corpus",
                    "--manifest", manifest,
                    "--json-output", manifest,
                ]):
            with self.subTest(arguments=arguments):
                completed = self.run_cli(*arguments)
                self.assertEqual(completed.returncode, 2)
                self.assertIn(
                    b"refusing to overwrite inventory input",
                    completed.stderr)
                self.assertEqual(code_object.read_bytes(), make_elf())

    def test_help_documents_execution_and_worklist(self):
        completed = self.run_cli("--help")
        self.assertEqual(completed.returncode, 0)
        self.assertIn(b"--worklist", completed.stdout)
        self.assertIn(b"--manifest", completed.stdout)
        self.assertIn(b"--execute", completed.stdout)
        self.assertIn(b"--cache-dir", completed.stdout)
        self.assertIn(b"--cache-dependency", completed.stdout)
        self.assertIn(b"--cache-tag", completed.stdout)
        self.assertIn(b"--jobs", completed.stdout)

    def test_rejects_non_regular_explicit_root(self):
        if not hasattr(os, "mkfifo"):
            self.skipTest("FIFOs unavailable")
        fifo = self.root / "fifo"
        os.mkfifo(fifo)
        mode = fifo.stat().st_mode
        self.assertTrue(stat.S_ISFIFO(mode))
        completed = self.run_cli(fifo)
        self.assertEqual(completed.returncode, 2)
        self.assertIn(b"not a regular file or directory", completed.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
