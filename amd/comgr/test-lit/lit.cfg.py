import os
import platform
import re

import lit.formats
import lit.util

config.name = "Comgr"
config.suffixes = {".hip", ".cl", ".c", ".cpp"}
config.test_format = lit.formats.ShTest(True)

config.excludes = ["comgr-sources"]

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = config.my_obj_root

if not config.comgr_disable_spirv:
    config.available_features.add("comgr-has-spirv")

if platform.system() == "Windows":
    config.available_features.add("system-windows")
elif platform.system() == "Linux":
    config.available_features.add("system-linux")

# By default, disable the cache for the tests.
# Test for the cache must explicitly enable this variable.
config.environment['AMD_COMGR_CACHE'] = "0"

# Resolve tool paths at configure time with forward slashes.  On Windows,
# os.path.join may return paths with backslashes, which break when written
# into bash scripts (e.g. "bin\clang" -> "binclang").
def _fwd(*parts):
    return os.path.join(*parts).replace("\\", "/")

# %-prefixed substitutions for LLVM tools (used as %clang, %llvm-dis, etc.)
config.substitutions.append(('%clang', _fwd(config.llvm_tools_dir, 'clang')))
config.substitutions.append(('%llvm-dis', _fwd(config.llvm_tools_dir, 'llvm-dis')))
config.substitutions.append(('%llvm-objdump', _fwd(config.llvm_tools_dir, 'llvm-objdump')))
config.substitutions.append(('%FileCheck', _fwd(config.llvm_tools_dir, 'FileCheck')))
config.substitutions.append(('%amd-llvm-spirv', _fwd(config.llvm_tools_dir, 'amd-llvm-spirv')))

# Resolve bare tool names used in RUN lines.  On Windows, shell PATH
# resolution via os.path.join introduces backslashes; resolving here
# with forward slashes avoids that.
_tool_dirs = os.pathsep.join([config.llvm_tools_dir, config.comgr_obj_dir])

def _resolve_tool(name):
    path = lit.util.which(name, _tool_dirs)
    if path:
        return path.replace("\\", "/")
    return name

_bare_tools = [
    "unbundle", "source-to-bc-with-dev-libs", "compile-opencl-minimal",
    "compile-hip-minimal", "spirv-translator", "spirv-to-reloc",
    "source-to-spirv", "lookup-code-object",
    "get-version", "status-string", "data-action",
]

for _name in _bare_tools:
    _resolved = _resolve_tool(_name)
    # Match the tool name at word boundaries but NOT when preceded by a path
    # separator (inside a path) or followed by '.' or '-' (inside a filename
    # like spirv-translator.cl).  Modeled after LLVM's ToolSubst patterns.
    _pattern = r"(?<![/\\])\b" + re.escape(_name) + r"\b(?![.-])"
    config.substitutions.append((_pattern, _resolved))
