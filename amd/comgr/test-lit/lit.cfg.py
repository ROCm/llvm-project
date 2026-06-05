import os
import platform

import lit.formats
import subprocess

config.name = "Comgr"
config.suffixes = {".hip", ".cl", ".c", ".cpp", ".s"}
config.test_format = lit.formats.ShTest(True)

config.excludes = ["comgr-sources"]

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = config.my_obj_root

if config.comgr_spirv_backend_available:
    config.available_features.add("comgr-has-spirv-backend")
if config.comgr_spirv_translator_available:
    config.available_features.add("comgr-has-spirv-translator")

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

isa_tool = _fwd(config.my_obj_root, 'isa-enumeration')
try:
    out = subprocess.check_output([isa_tool], text=True)
    config.amd_isas = [line.strip() for line in out.splitlines() if line.strip()]
except (OSError, subprocess.CalledProcessError):
    config.amd_isas = []

def _isa_runs(clang, drv):
    cmds = []
    for isa in config.amd_isas:
        gpu = isa.split('--')[-1]
    cmds.append(
        f"{clang} -target amdgcn-amd-amdhsa -mcpu={gpu} -nogpulib -nogpuinc "
        f"-c %S/get-data-isa-name.cl -o %t.o && "
        f"{clang} -target amdgcn-amd-amdhsa -mcpu={gpu} -nogpulib -nogpuinc "
        f"-shared %S/get-data-isa-name.cl -o %t.so && "
        f"{drv} %t.o %t.so {isa}"
    )
    return " && ".join(cmds) if cmds else "true"

#config.substitutions.append(('%run_all_isas',
#    _isa_runs(_fwd(config.llvm_tools_dir, 'clang'),
#        _fwd(config.my_obj_root, 'get-data-isa-name'))))

# %-prefixed substitutions for LLVM tools (used as %clang, %llvm-dis, etc.)
config.substitutions.append(('%clang', _fwd(config.llvm_tools_dir, 'clang')))
config.substitutions.append(('%llvm-dis', _fwd(config.llvm_tools_dir, 'llvm-dis')))
config.substitutions.append(('%llvm-objdump', _fwd(config.llvm_tools_dir, 'llvm-objdump')))
config.substitutions.append(('%llvm-readelf', _fwd(config.llvm_tools_dir, 'llvm-readelf')))
config.substitutions.append(('%FileCheck', _fwd(config.llvm_tools_dir, 'FileCheck')))
config.substitutions.append(('%amd-llvm-spirv', _fwd(config.llvm_tools_dir, 'amd-llvm-spirv')))
