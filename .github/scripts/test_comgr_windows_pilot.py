"""Verify pilot failure propagation using fake tools, without LLVM builds."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


DRIVER = Path(__file__).with_name("comgr_windows_pilot.py")


class PilotTest(unittest.TestCase):
    def run_driver(self, arm: str, failure: str = "") -> tuple[list[str], dict[str, object]]:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tools = root / "tools"
            tools.mkdir()
            units = root / "build-comgr/test-unit"
            units.mkdir(parents=True)
            (root / "build-comgr/amd_comgr.dll").write_text("original")
            common = """
echo "$phase" >> events
if [ "$FAIL_PHASE" = "$phase" ]; then exit 17; fi
"""
            commands = {
                "tools/ninja": """
case "$*" in
  *check-comgr*) phase=check-comgr ;;
  *) phase=build ;;
esac
""" + common + "echo linked > build-comgr/amd_comgr.dll\n",
                "tools/python": "phase=lit\n" + common,
                "tools/ctest": "phase=ctest\n" + common,
                "build-comgr/test-unit/first.exe": "phase=unit1\n" + common,
                "build-comgr/test-unit/second.exe": "phase=unit2\n" + common,
            }
            for name, command in commands.items():
                path = root / name
                path.write_text("#!/bin/bash\nset -e\n" + command)
                path.chmod(0o755)
            environment = dict(
                os.environ,
                PATH=str(tools) + os.pathsep + os.environ["PATH"],
                FAIL_PHASE=failure,
                PILOT_ARTIFACT_RUN="123",
                PILOT_SOURCE_REF="test-source",
                PILOT_TRANSLATOR_REF="test-translator",
                LC_ALL="C",
            )
            result = subprocess.run(
                [sys.executable, str(DRIVER), "--arm", arm, "--trial", "1",
                 "--output", "results"],
                cwd=root, env=environment, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 1 if failure else 0,
                             result.stdout + result.stderr)
            payload = json.loads((root / "results/result.json").read_text())
            self.assertEqual(payload["returncode"], 17 if failure else 0)
            self.assertEqual(payload["arm"], arm)
            self.assertGreater(payload["elapsed_seconds"], 0)
            if arm == "C" and not failure:
                self.assertEqual((units / "amd_comgr.dll").read_text(), "linked\n")
            return (root / "events").read_text().splitlines(), payload

    def test_baseline_runs_only_check_target(self) -> None:
        events, result = self.run_driver("A")
        self.assertEqual(events, ["check-comgr"])
        self.assertEqual(list(result["phase_seconds"]), ["check-comgr"])

    def test_direct_runs_every_suite(self) -> None:
        events, result = self.run_driver("B")
        self.assertEqual(events, ["build", "lit", "unit1", "unit2", "ctest"])
        self.assertEqual(list(result["phase_seconds"]),
                         ["build", "lit", "unit", "ctest"])

    def test_full_workaround_runs_every_suite(self) -> None:
        events, result = self.run_driver("C")
        self.assertEqual(events, ["build", "lit", "unit1", "unit2", "ctest"])
        self.assertEqual(list(result["phase_seconds"]),
                         ["build", "dll-replacement", "lit", "unit", "ctest"])

    def test_baseline_failure_is_preserved(self) -> None:
        events, result = self.run_driver("A", "check-comgr")
        self.assertEqual(events, ["check-comgr"])
        self.assertEqual(result["last_phase"], "check-comgr")

    def test_build_failure_prevents_tests(self) -> None:
        events, result = self.run_driver("C", "build")
        self.assertEqual(events, ["build"])
        self.assertEqual(result["last_phase"], "build")

    def test_lit_failure_prevents_later_suites(self) -> None:
        events, result = self.run_driver("C", "lit")
        self.assertEqual(events, ["build", "lit"])
        self.assertEqual(result["last_phase"], "lit")

    def test_unit_failure_stops_loop_and_ctest(self) -> None:
        events, result = self.run_driver("C", "unit1")
        self.assertEqual(events, ["build", "lit", "unit1"])
        self.assertEqual(result["last_phase"], "unit")

    def test_ctest_failure_is_preserved(self) -> None:
        events, result = self.run_driver("B", "ctest")
        self.assertEqual(events, ["build", "lit", "unit1", "unit2", "ctest"])
        self.assertEqual(result["last_phase"], "ctest")


if __name__ == "__main__":
    unittest.main()

