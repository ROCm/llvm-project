#!/usr/bin/env python3
"""A script to generate FileCheck statements for Fortran runtime funcs.

This script can be used to update
flang/test/Transforms/verify-known-runtime-functions.fir
whenever new recognized Fortran runtime functions are added
into flang/Optimizer/Transforms/RuntimeFunctions.inc
or any of the recognized functions changes its signature.
"""

# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import os
import re
import sys

ADVERT_BEGIN = "// NOTE: Assertions have been autogenerated by "
ADVERT_END = """
// The script allows updating Flang LIT test
// flang/test/Transforms/verify-known-runtime-functions.fir,
// which is intended to verify signatures of Fortran runtime
// functions recognized in flang/Optimizer/Transforms/RuntimeFunctions.inc
// table. If new function is added into the table or
// an existing function changes its signature,
// the SetRuntimeCallAttributesPass may need to be updated
// to properly handle it. Once the pass is verified to work,
// one can update this test using the following output:
//   echo "module {}" | fir-opt --gen-runtime-calls-for-test | \\
//   generate-checks-for-runtime-funcs.py
"""

CHECK_RE_STR = "func.func.*@_Fortran.*"
CHECK_RE = re.compile(CHECK_RE_STR)

CHECK_NOT_STR = "// CHECK-NOT: func.func"
CHECK_STR = "// CHECK:"
CHECK_NEXT_STR = "// CHECK-NEXT:"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "input", nargs="?", type=argparse.FileType("r"), default=sys.stdin
    )
    args = parser.parse_args()
    input_lines = [l.rstrip() for l in args.input]
    args.input.close()

    repo_path = os.path.join(os.path.dirname(__file__), "..", "..", "..")
    script_name = os.path.relpath(__file__, repo_path)
    autogenerated_note = ADVERT_BEGIN + script_name + "\n" + ADVERT_END

    output = sys.stdout
    output.write(autogenerated_note + "\n")

    output_lines = []
    output_lines.append(CHECK_NOT_STR)
    check_prefix = CHECK_STR
    for input_line in input_lines:
        if not input_line:
            continue

        m = CHECK_RE.match(input_line.lstrip())
        if m:
            output_lines.append(check_prefix + " " + input_line.lstrip())
            check_prefix = CHECK_NEXT_STR

    output_lines.append(CHECK_NOT_STR)
    for line in output_lines:
        output.write(line + "\n")

    output.close()


if __name__ == "__main__":
    main()
