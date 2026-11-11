"""Run one Comgr Windows pilot trial without preloading its DLL."""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import time


@dataclass
class TrialResult:
    arm: str
    trial: int
    started_at: str
    elapsed_seconds: float
    returncode: int
    last_phase: str
    phase_seconds: dict[str, float]
    environment: dict[str, str]


def trial_script(arm: str) -> str:
    script = r"""
run_phase() {
  local phase="$1"
  shift
  printf 'COMGR_PILOT_START %s\n' "$phase"
  TIMEFORMAT="COMGR_PILOT_SECONDS $phase %3R"
  time "$@"
}
run_units() {
  for test in build-comgr/test-unit/*.exe; do
    "$test"
  done
}
replace_dlls() {
  cp build-comgr/amd_comgr.dll build-comgr/amd_comgr.ready.dll
  mv -f build-comgr/amd_comgr.ready.dll build-comgr/amd_comgr.dll
  cp build-comgr/amd_comgr.dll build-comgr/test-unit/amd_comgr.ready.dll
  mv -f build-comgr/test-unit/amd_comgr.ready.dll build-comgr/test-unit/amd_comgr.dll
}
"""
    if arm == "A":
        return script + "run_phase check-comgr ninja -C build-comgr check-comgr\n"
    if arm not in ("B", "C"):
        raise ValueError(f"Unknown arm: {arm}")
    script += "run_phase build ninja -C build-comgr all\n"
    if arm == "C":
        script += "run_phase dll-replacement replace_dlls\n"
    return script + """
run_phase lit python build/bin/llvm-lit.py build-comgr/test-lit -v
run_phase unit run_units
run_phase ctest ctest --test-dir build-comgr --output-on-failure
"""


def run_trial(arm: str, trial: int, output: Path) -> TrialResult:
    if not 1 <= trial <= 10:
        raise ValueError("The pilot has exactly ten trials per arm")
    bash = shutil.which("bash")
    if bash is None:
        raise RuntimeError("Git Bash is required")
    required = ("PILOT_ARTIFACT_RUN", "PILOT_SOURCE_REF", "PILOT_TRANSLATOR_REF")
    environment = {name: os.environ[name] for name in required}
    for name in ("RUNNER_NAME", "COMPUTERNAME", "ImageOS", "ImageVersion",
                 "VCToolsVersion", "WindowsSDKVersion", "GITHUB_SHA",
                 "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT"):
        if name in os.environ:
            environment[name] = os.environ[name]
    environment["platform"] = platform.platform()
    environment["python"] = platform.python_version()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "result.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    script = trial_script(arm)
    (output / "trial.sh").write_text(script, encoding="utf-8")
    started_at = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    phase_seconds: dict[str, float] = {}
    last_phase = ""
    with (output / "trial.log").open("x", encoding="utf-8") as log:
        with subprocess.Popen(
            [bash, "--noprofile", "--norc", "-e", "-o", "pipefail", "-c", script],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        ) as process:
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                print(line, end="", flush=True)
                if line.startswith("COMGR_PILOT_START "):
                    last_phase = line.split()[1]
                match = re.fullmatch(
                    r"COMGR_PILOT_SECONDS (\S+) ([0-9]+\.[0-9]+)\n", line
                )
                if match:
                    name, seconds = match.groups()
                    if name in phase_seconds:
                        raise ValueError(f"Repeated phase: {name}")
                    phase_seconds[name] = float(seconds)
            returncode = process.wait()
    result = TrialResult(
        arm, trial, started_at, time.perf_counter() - start, returncode,
        last_phase, phase_seconds, environment,
    )
    expected = ["check-comgr"] if arm == "A" else ["build", "lit", "unit", "ctest"]
    if arm == "C":
        expected.insert(1, "dll-replacement")
    if returncode == 0 and list(phase_seconds) != expected:
        raise RuntimeError(f"Incomplete successful trial: {phase_seconds}")
    payload = json.dumps(asdict(result), indent=2) + "\n"
    result_path.write_text(payload, encoding="utf-8")
    if json.loads(result_path.read_text(encoding="utf-8")) != asdict(result):
        raise RuntimeError("Trial result did not round-trip")
    ninja_log = Path("build-comgr/.ninja_log")
    if ninja_log.is_file():
        shutil.copyfile(ninja_log, output / "ninja-log.txt")
    print("COMGR_PILOT_RESULT " + json.dumps(asdict(result)), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("A", "B", "C"), required=True)
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_trial(args.arm, args.trial, args.output)
    return 0 if result.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

