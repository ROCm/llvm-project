#===- enumerate-isa-check.py ---------------------------------------------===#
#
# Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
# amd/comgr/LICENSE.TXT in this repository for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===#
#
# Enumerates every ISA reported by Comgr (via the isa-enumeration tool),
# compiles the given OpenCL source for each one as both a relocatable (-c)
# and a shared object (-shared), and verifies amd_comgr_get_data_isa_name()
# round-trips the ISA name (via the get-data-isa-name-linux tool).
#
#===----------------------------------------------------------------------===#

import subprocess
import sys


def run(cmd):
    """Run cmd (a list of args), echoing it.  Exit nonzero on failure so lit
    reports the test as failed and the failing command is visible."""
    print("+ " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit("command failed (exit %d): %s"
                 % (result.returncode, " ".join(cmd)))


def main(argv):
    if len(argv) != 4:
        sys.exit("Usage: enumerate-isa-check.py <clang> <source.cl> "
                 "<tmp-prefix>")

    clang, source, tmp_prefix = argv[1], argv[2], argv[3]
    obj = tmp_prefix + ".o"
    shared = tmp_prefix + ".so"

    # isa-enumeration prints one ISA name (with optional feature suffixes)
    # per line.
    enumerated = subprocess.run(["isa-enumeration"], check=True,
                                capture_output=True, text=True)
    isas = [line for line in enumerated.stdout.splitlines() if line]
    if not isas:
        sys.exit("isa-enumeration produced no ISAs")

    common = [clang, "-target", "amdgcn-amd-amdhsa", "-nogpulib", "-nogpuinc"]

    for isa in isas:
        # The GPU name is everything after the final "--" (vendor/OS are
        # empty here), including any feature suffix such as ":sramecc+",
        # which clang accepts on -mcpu.
        gpu = isa.rsplit("--", 1)[-1]

        run(common + ["-mcpu=" + gpu, "-c", source, "-o", obj])
        run(common + ["-mcpu=" + gpu, "-shared", source, "-o", shared])
        run(["get-data-isa-name-linux", obj, shared, isa])


if __name__ == "__main__":
    main(sys.argv)
