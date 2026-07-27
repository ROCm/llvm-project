#!/usr/bin/env python3
# Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
# amd/comgr/LICENSE.TXT in this repository for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Unit tests for hotswap_bench.py.

These use small stub executables and temporary directories so the whole suite
runs without a GPU or a real hotswap-rewrite / libamd_comgr.so. Run with:

    python3 -m unittest -v          # from this directory
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import hotswap_bench as bench  # noqa: E402

# Stub bodies emulating hotswap-rewrite behaviours. Each writes to --output
# and/or prints a RESULT: line as the real driver would.
STUB_SUCCESS = """
import sys
args = sys.argv[1:]
out = args[args.index("--output") + 1] if "--output" in args else None
if out:
    with open(out, "wb") as handle:
        handle.write(b"\\x7fELFstub-output")
print("RESULT: SUCCESS")
"""

STUB_NO_RESULT = """
raise SystemExit(0)
"""

STUB_INVALID_ARGUMENT = """
print("RESULT: INVALID_ARGUMENT")
raise SystemExit(0)
"""

STUB_FAIL = """
import sys
print("boom: assertion failed in rewrite", file=sys.stderr)
print("FAILED: unexpected error status ERROR")
raise SystemExit(1)
"""

STUB_SLEEP = """
import time
time.sleep(30)
"""


def make_stub(directory: pathlib.Path, name: str, body: str) -> pathlib.Path:
    path = directory / name
    path.write_text("#!/usr/bin/env python3\n" + body)
    path.chmod(0o755)
    return path


class TempCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = pathlib.Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def run_one(self, stub_body: str, source: pathlib.Path, **overrides):
        stub = make_stub(self.tmp, "stub", stub_body)
        kwargs = dict(
            source=source,
            hotswap=stub,
            library_dirs=[],
            source_isa="amdgcn-amd-amdhsa--gfx1250",
            target_isa="amdgcn-amd-amdhsa--gfx1250",
            entry_trampolines=False,
            strict_mode=True,
            timeout_seconds=30.0,
            outputs_dir=None,
            input_root=source.parent,
        )
        kwargs.update(overrides)
        return bench.run_one(**kwargs)


class ClassificationTests(TempCase):
    def _input(self, name: str = "a.hsaco", data: bytes = b"input-bytes"):
        path = self.tmp / name
        path.write_bytes(data)
        return path

    def test_success_is_pass_with_hashes(self):
        row = self.run_one(STUB_SUCCESS, self._input())
        self.assertEqual(row["status"], "pass")
        self.assertEqual(row["result"], "SUCCESS")
        self.assertTrue(row["output_is_elf"])
        self.assertTrue(row["input_sha256"])
        self.assertTrue(row["output_sha256"])

    def test_exit0_without_result_or_output_is_fail(self):
        row = self.run_one(STUB_NO_RESULT, self._input())
        self.assertEqual(row["status"], "fail")
        self.assertEqual(row["exit_code"], 0)
        self.assertFalse(row["output_is_elf"])

    def test_invalid_argument_is_fail(self):
        row = self.run_one(STUB_INVALID_ARGUMENT, self._input())
        self.assertEqual(row["status"], "fail")
        self.assertEqual(row["result"], "INVALID_ARGUMENT")

    def test_nonzero_exit_is_fail_and_keeps_stderr_tail(self):
        row = self.run_one(STUB_FAIL, self._input())
        self.assertEqual(row["status"], "fail")
        self.assertEqual(row["exit_code"], 1)
        self.assertIn("assertion failed", row["stderr_tail"])

    def test_timeout(self):
        row = self.run_one(STUB_SLEEP, self._input(), timeout_seconds=1.0)
        self.assertEqual(row["status"], "timeout")


class DiscoveryTests(TempCase):
    def test_matches_only_requested_extensions(self):
        corpus = self.tmp / "corpus"
        (corpus / "sub").mkdir(parents=True)
        for name in ("a.hsaco", "b.co", "sub/c.hsaco"):
            (corpus / name).write_bytes(b"x")
        # Stray files that a broad glob would wrongly pick up.
        for name in ("results.csv", "header.host", "backup.hsaco.bak"):
            (corpus / name).write_bytes(b"x")

        found = bench.discover_inputs([str(corpus)], True, ["*.hsaco", "*.co"])
        relative = sorted(str(p.relative_to(corpus)) for p in found)
        self.assertEqual(relative, ["a.hsaco", "b.co", "sub/c.hsaco"])


class OutputNamingTests(TempCase):
    def test_keep_outputs_is_collision_free(self):
        root = self.tmp / "corpus"
        (root / "a").mkdir(parents=True)
        nested = root / "a" / "b.hsaco"
        flat = root / "a__b.hsaco"
        nested.write_bytes(b"n")
        flat.write_bytes(b"f")
        outputs = self.tmp / "outputs"

        row_nested = self.run_one(
            STUB_SUCCESS, nested, outputs_dir=outputs, input_root=root
        )
        row_flat = self.run_one(
            STUB_SUCCESS, flat, outputs_dir=outputs, input_root=root
        )
        self.assertEqual(row_nested["status"], "pass")
        self.assertEqual(row_flat["status"], "pass")
        self.assertTrue((outputs / "a" / "b.hsaco.co").is_file())
        self.assertTrue((outputs / "a__b.hsaco.co").is_file())


class ParserTests(unittest.TestCase):
    def test_jobs_zero_accepted(self):
        args = bench.build_parser().parse_args(["input", "-j", "0"])
        self.assertEqual(args.jobs, 0)

    def test_jobs_negative_rejected(self):
        with self.assertRaises(SystemExit):
            bench.build_parser().parse_args(["input", "-j", "-1"])

    def test_entry_trampolines_default_off(self):
        args = bench.build_parser().parse_args(["input"])
        self.assertFalse(args.entry_trampolines)


class ProcessGroupTests(unittest.TestCase):
    def test_terminate_process_group_kills_child(self):
        process = subprocess.Popen(["sleep", "30"], start_new_session=True)
        try:
            bench.terminate_process_group(process, 1.0)
            self.assertIsNotNone(process.returncode)
        finally:
            if process.returncode is None:
                process.kill()
                process.wait()


class MainTests(TempCase):
    def _corpus(self):
        corpus = self.tmp / "corpus"
        corpus.mkdir()
        (corpus / "b.co").write_bytes(b"bbbb")
        (corpus / "a.hsaco").write_bytes(b"aaaa")
        return corpus

    def _read_csv(self, path: pathlib.Path):
        import csv

        with path.open(newline="") as handle:
            return list(csv.DictReader(handle))

    def test_end_to_end_csv_sorted_with_provenance_and_sidecar(self):
        stub = make_stub(self.tmp, "stub", STUB_SUCCESS)
        corpus = self._corpus()
        csv_path = self.tmp / "out.csv"
        code = bench.main(
            [
                str(corpus),
                "--hotswap",
                str(stub),
                "--include-glob",
                "*.hsaco",
                "--include-glob",
                "*.co",
                "-j",
                "1",
                "--csv",
                str(csv_path),
                "--quiet",
            ]
        )
        self.assertEqual(code, 0)
        rows = self._read_csv(csv_path)
        # Deterministic sort by input_path.
        self.assertEqual([r["input_path"] for r in rows], ["a.hsaco", "b.co"])
        self.assertTrue(all(r["input_sha256"] for r in rows))
        self.assertTrue(all(r["output_sha256"] for r in rows))
        # Sidecar with tool identity.
        meta = json.loads((self.tmp / "out.csv.meta.json").read_text())
        self.assertEqual(meta["tool"]["hotswap_rewrite"]["path"], str(stub))
        self.assertIn("sha256", meta["tool"]["hotswap_rewrite"])
        # Checkpoint removed after a complete run.
        self.assertFalse((self.tmp / "out.csv.partial.jsonl").exists())

    def test_resume_skips_recorded_inputs(self):
        stub = make_stub(self.tmp, "stub", STUB_SUCCESS)
        corpus = self._corpus()
        csv_path = self.tmp / "out.csv"
        checkpoint = self.tmp / "out.csv.partial.jsonl"
        # Seed a.hsaco as already done with a sentinel result so we can tell it
        # was reused rather than re-run.
        seeded = {
            "input_path": "a.hsaco",
            "filename": "a.hsaco",
            "status": "pass",
            "result": "SEEDED",
        }
        bench.append_jsonl(checkpoint, seeded)

        code = bench.main(
            [
                str(corpus),
                "--hotswap",
                str(stub),
                "--include-glob",
                "*.hsaco",
                "--include-glob",
                "*.co",
                "-j",
                "1",
                "--resume",
                "--csv",
                str(csv_path),
                "--quiet",
            ]
        )
        self.assertEqual(code, 0)
        rows = {r["input_path"]: r for r in self._read_csv(csv_path)}
        self.assertEqual(set(rows), {"a.hsaco", "b.co"})
        # a.hsaco came from the checkpoint (sentinel), b.co was actually run.
        self.assertEqual(rows["a.hsaco"]["result"], "SEEDED")
        self.assertEqual(rows["b.co"]["result"], "SUCCESS")


if __name__ == "__main__":
    unittest.main()
