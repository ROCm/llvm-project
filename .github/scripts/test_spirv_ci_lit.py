"""Exercise the translator CI gate without building LLVM."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import spirv_ci_lit


class LitGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.build = self.root / "build with spaces"
        (self.build / "bin").mkdir(parents=True)
        self.output = self.root / "github-output"
        self.results = self.root / "results.json"

    def report(self, code: str) -> dict[str, list[dict[str, str]]]:
        return {"tests": [{"name": "LLVM_SPIRV :: example.spvasm", "code": code}]}

    def run_driver(
        self,
        code: str = "PASS",
        exit_code: int = 0,
        *,
        pr_code: str | None = None,
        write_report: bool = True,
        rebuild: bool = False,
        build_error: bool = False,
        lit_name: str = "llvm-lit",
    ) -> subprocess.CompletedProcess[str]:
        report = self.report(code)
        fake_lit = (
            "import json, os, pathlib, shlex, sys\n"
            "sys.argv.extend(shlex.split(os.environ.get('LIT_OPTS', '')))\n"
        )
        if write_report:
            fake_lit += (
                "pathlib.Path(sys.argv[sys.argv.index('--output') + 1]).write_text("
                f"json.dumps({report!r}))\n"
            )
        fake_lit += f"sys.exit({exit_code})\n"
        (self.build / "bin" / lit_name).write_text(fake_lit, encoding="utf-8")
        command = [
            sys.executable,
            str(Path(spirv_ci_lit.__file__)),
            "--build-dir",
            str(self.build),
            "--results",
            str(self.results),
        ]
        if pr_code is not None:
            pr = self.root / "pr.json"
            pr.write_text(json.dumps(self.report(pr_code)), encoding="utf-8")
            command.extend(["--pr-results", str(pr)])
        if rebuild:
            source = self.root / "llvm-project/llvm"
            source.mkdir(parents=True)
            target_command = (
                '"${CMAKE_COMMAND}" -E false'
                if build_error
                else f'"{Path(sys.executable).as_posix()}" "${{CMAKE_BINARY_DIR}}/bin/llvm-lit"'
            )
            (source / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 3.20)\n"
                "project(LitGate NONE)\n"
                f"add_custom_target(check-amd-llvm-spirv COMMAND {target_command} VERBATIM)\n",
                encoding="utf-8",
            )
            command.append("--rebuild")
        return subprocess.run(
            command,
            cwd=self.root,
            env=dict(os.environ, GITHUB_OUTPUT=str(self.output)),
            capture_output=True,
            text=True,
            check=False,
        )

    def test_pass_skips_baseline(self) -> None:
        result = self.run_driver()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.output.read_text(), "has_failures=false\n")

    def test_test_failure_requests_baseline(self) -> None:
        result = self.run_driver("FAIL", 1)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.output.read_text(), "has_failures=true\n")

    def test_windows_launcher(self) -> None:
        result = self.run_driver(lit_name="llvm-lit.py")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.output.read_text(), "has_failures=false\n")

    def test_existing_failure_is_allowed(self) -> None:
        result = self.run_driver("FAIL", 1, pr_code="FAIL")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_new_failure_is_rejected(self) -> None:
        result = self.run_driver(pr_code="FAIL")
        self.assertEqual(result.returncode, 1)
        self.assertIn("New translator test failure", result.stdout)

    def test_baseline_build_and_report_with_spaces_in_path(self) -> None:
        result = self.run_driver("FAIL", 1, pr_code="FAIL", rebuild=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("No new translator lit failures", result.stdout)

    def test_baseline_build_failure_is_rejected(self) -> None:
        result = self.run_driver(pr_code="FAIL", rebuild=True, build_error=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.results.exists())

    def test_unexpected_pass_requests_baseline(self) -> None:
        result = self.run_driver("XPASS", 1)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.output.read_text(), "has_failures=true\n")

    def test_infrastructure_results_are_rejected(self) -> None:
        for code in ["UNRESOLVED", "TIMEOUT", "SKIPPED", "UNKNOWN"]:
            with self.subTest(code=code):
                self.assertNotEqual(self.run_driver(code, 1).returncode, 0)
                self.assertFalse(self.output.exists())

    def test_exit_code_must_match_report(self) -> None:
        for code, status in [("PASS", 1), ("FAIL", 0), ("FAIL", 2)]:
            with self.subTest(code=code, status=status):
                self.assertNotEqual(self.run_driver(code, status).returncode, 0)
                self.assertFalse(self.output.exists())

    def test_stale_report_cannot_mask_missing_report(self) -> None:
        self.results.write_text(json.dumps(self.report("PASS")), encoding="utf-8")
        self.assertNotEqual(self.run_driver(write_report=False).returncode, 0)
        self.assertFalse(self.results.exists())

    def test_invalid_reports_are_rejected(self) -> None:
        for report in [
            {"tests": []},
            self.report("UNSUPPORTED"),
            {"tests": [{"name": "another suite", "code": "PASS"}]},
            {"tests": self.report("PASS")["tests"] * 2},
            {},
        ]:
            with self.subTest(report=report):
                self.results.write_text(json.dumps(report), encoding="utf-8")
                with self.assertRaises((ValueError, KeyError)):
                    spirv_ci_lit.read_failures(self.results)


if __name__ == "__main__":
    unittest.main()
