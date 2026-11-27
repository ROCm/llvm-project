# ===- lit.cfg.py - Scheduling ledger plugin test configuration --------------=== #
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------=== #

import os

import lit.formats

config.name = "AMD scheduling ledger plugin"
config.test_format = lit.formats.ShTest()
config.suffixes = [".ll"]
config.excludes = ["Inputs"]
config.test_source_root = config.psched_ledger_source_root
config.test_exec_root = config.psched_ledger_obj_root

if not config.psched_ledger_amdgpu_available:
    config.unsupported = True

config.environment["PATH"] = os.pathsep.join(
    [config.llvm_tools_dir, config.environment.get("PATH", "")]
)
# A ledger from the developer's environment must not leak into the tests.
for var in ("PSCHED_LEDGER", "PSCHED_VERBOSE", "PSCHED_DUMP_IR"):
    config.environment.pop(var, None)


def tool(name):
    return os.path.join(config.llvm_tools_dir, name).replace("\\", "/")


config.substitutions.extend(
    [
        ("%psched-plugin", config.psched_ledger_plugin.replace("\\", "/")),
        ("%opt", tool("opt")),
        ("%FileCheck", tool("FileCheck")),
    ]
)
