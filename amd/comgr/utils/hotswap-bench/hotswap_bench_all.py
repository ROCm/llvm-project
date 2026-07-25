#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run HotSwap gfx1250 B0-to-A0 translation over ALL code objects (.hsaco AND
.co) in a directory, with a 5-minute per-file timeout.

This is a thin convenience wrapper around hotswap_bench.py that changes only
the defaults:
  * --include-glob matches both .co and .hsaco,
  * --timeout-seconds is 300 (5 minutes),
  * --csv defaults to hotswap_bench_all.csv.

The translator binary and its library are resolved by hotswap_bench.py itself
(via $HOTSWAP_REWRITE / $HOTSWAP_BUILD_DIR / --hotswap; see README.md), so this
wrapper stays free of any machine-specific paths.

Every other flag (--jobs, --memory-limit-gb, --oom-guard, --source-isa, ...) is
inherited from hotswap_bench.py and can be overridden on the command line;
anything passed explicitly wins over these defaults.

Example:
  HOTSWAP_BUILD_DIR=/path/to/build \\
    python3 hotswap_bench_all.py /path/to/corpus --jobs 16 --memory-limit-gb 64
"""

from __future__ import annotations

import pathlib
import sys

# Make the sibling module importable regardless of the caller's working dir.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import hotswap_bench as bench  # noqa: E402

# Defaults injected only when the flag is not already supplied by the caller.
DEFAULTS: dict[str, str] = {
    "--include-glob": "*.[hc][so]*",
    "--timeout-seconds": "300",
    "--csv": "hotswap_bench_all.csv",
}


def apply_defaults(argv: list[str]) -> list[str]:
    result = list(argv)
    for flag, value in DEFAULTS.items():
        if not any(arg == flag or arg.startswith(flag + "=") for arg in result):
            result += [flag, value]
    return result


def main() -> int:
    # Rewrite sys.argv in place so hotswap_bench's argument parsing and its
    # memory-guard re-exec (which reads sys.argv) both see the same values.
    sys.argv = [sys.argv[0], *apply_defaults(sys.argv[1:])]
    return bench.main()


if __name__ == "__main__":
    raise SystemExit(main())
