# ===- lit.cfg.py - AMD SQTT marker test configuration --------------------=== #
#
# Part of AMD SQTT Marker, under the MIT License. See
# amd/sqtt-marker/LICENSE.txt for license information.
# SPDX-License-Identifier: MIT
#
# ===----------------------------------------------------------------------=== #

import os

import lit.formats

config.name = "AMD SQTT marker"
config.test_format = lit.formats.ShTest()
config.suffixes = [".c", ".hip", ".ll"]
config.excludes = ["Inputs", "legacy"]
config.test_source_root = config.sqtt_marker_source_root
config.test_exec_root = config.sqtt_marker_obj_root

if config.sqtt_marker_amdgpu_available:
    config.available_features.add("amdgpu-registered-target")
if config.sqtt_marker_clang_tests_available:
    config.available_features.add("sqtt-marker-has-clang")
if config.sqtt_marker_offload_tools_available:
    config.available_features.add("sqtt-marker-has-offload-tools")

config.environment["PATH"] = os.pathsep.join(
    [config.llvm_tools_dir, config.environment.get("PATH", "")]
)


def tool(name):
    return os.path.join(config.llvm_tools_dir, name).replace("\\", "/")


config.substitutions.extend(
    [
        ("%sqtt-marker-plugin", config.sqtt_marker_plugin.replace("\\", "/")),
        ("%sqtt-marker-include", config.sqtt_marker_include_dir.replace("\\", "/")),
        ("%host-triple", config.llvm_host_triple),
        ("%clang", tool("clang")),
        ("%clang-offload-bundler", tool("clang-offload-bundler")),
        ("%opt", tool("opt")),
        ("%llvm-objcopy", tool("llvm-objcopy")),
        ("%FileCheck", tool("FileCheck")),
        ("%python", config.python_executable.replace("\\", "/")),
    ]
)
