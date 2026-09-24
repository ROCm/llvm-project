#!/usr/bin/env python3
"""Run prebuilt translator tests and compare only valid lit results."""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


def read_failures(path: Path) -> set[tuple[str, str]]:
    tests = json.loads(path.read_text(encoding="utf-8"))["tests"]
    if not isinstance(tests, list) or not tests:
        raise ValueError(f"{path}: no test results")
    failures: set[tuple[str, str]] = set()
    names: set[str] = set()
    executed = False
    for test in tests:
        name, code = test["name"], test["code"]
        if not isinstance(name, str) or not name.startswith("LLVM_SPIRV :: "):
            raise ValueError(f"{path}: invalid translator test name {name!r}")
        if name in names:
            raise ValueError(f"{path}: duplicate test {name}")
        names.add(name)
        if code not in {"PASS", "FLAKYPASS", "XFAIL", "UNSUPPORTED", "FAIL", "XPASS"}:
            raise ValueError(f"{path}: incomplete or invalid result {code!r}: {name}")
        executed |= code != "UNSUPPORTED"
        if code in {"FAIL", "XPASS"}:
            failures.add((name, code))
    if not executed:
        raise ValueError(f"{path}: all tests unsupported")
    return failures


def run_lit(
    build_dir: Path, results: Path, rebuild: bool = False
) -> set[tuple[str, str]]:
    # Never accept a report left over from an earlier invocation.
    results.unlink(missing_ok=True)
    lit_args = ["-sv", "--no-progress-bar", "--output", str(results.resolve())]
    if rebuild:
        # Keep the baseline's exact CMake test dependencies, including new tools.
        subprocess.run(
            [
                "cmake",
                "-G",
                "Ninja",
                "-S",
                "llvm-project/llvm",
                "-B",
                str(build_dir),
            ],
            check=True,
        )
        command = ["ninja", "-C", str(build_dir), "check-amd-llvm-spirv"]
    else:
        lit = build_dir / "bin/llvm-lit"
        if not lit.is_file():
            lit = lit.with_suffix(".py")
        if not lit.is_file():
            raise FileNotFoundError(f"No llvm-lit launcher in {build_dir / 'bin'}")
        command = [
            sys.executable,
            str(lit),
            *lit_args,
            str(build_dir / "projects/SPIRV-LLVM-Translator/test"),
        ]
    env = dict(os.environ, LIT_OPTS=shlex.join(lit_args)) if rebuild else None
    result = subprocess.run(command, env=env, check=False)
    failures = read_failures(results)
    if result.returncode != (1 if failures else 0):
        raise ValueError(f"lit exit code {result.returncode} disagrees with {results}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--pr-results", type=Path)
    parser.add_argument(
        "--rebuild", action="store_true", help="Build baseline test dependencies"
    )
    args = parser.parse_args()

    failures = run_lit(args.build_dir, args.results, args.rebuild)
    if args.pr_results is None:
        with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as output:
            output.write(f"has_failures={str(bool(failures)).lower()}\n")
        print(f"PR head: {len(failures)} translator test failure(s)")
        return 0

    new_failures = read_failures(args.pr_results) - failures
    if new_failures:
        for name, code in sorted(new_failures):
            print(f"::error::New translator test failure: {code}: {name}")
        return 1
    print("No new translator lit failures vs amd-staging baseline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
