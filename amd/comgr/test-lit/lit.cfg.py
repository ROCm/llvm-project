import os
import platform

import lit.formats

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

if config.comgr_enable_hotswap_transpile:
    config.available_features.add("comgr-hotswap-transpile")

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
config.substitutions.append(('%llvm-readelf', _fwd(config.llvm_tools_dir, 'llvm-readelf')))
config.substitutions.append(('%FileCheck', _fwd(config.llvm_tools_dir, 'FileCheck')))
config.substitutions.append(('%amd-llvm-spirv', _fwd(config.llvm_tools_dir, 'amd-llvm-spirv')))
config.substitutions.append(('%not', _fwd(config.llvm_tools_dir, 'not')))
# %llvm_mc bakes in the AMDGPU triple + object filetype so each hotswap-
# transpile/ RUN line only has to vary -mcpu=<isa>; %ld_lld is the raw
# linker invoked with -shared to produce an amdhsa code object the
# raise_cli driver consumes.
config.substitutions.append(
    ('%llvm_mc',
     _fwd(config.llvm_tools_dir, 'llvm-mc') +
         ' -triple=amdgcn-amd-amdhsa -filetype=obj'))
config.substitutions.append(('%ld_lld', _fwd(config.llvm_tools_dir, 'ld.lld')))
# %raise_cli resolves to the raise_cli driver built alongside test-lit.
# Fixtures may also invoke it as a bare `raise_cli` (resolved via PATH
# from `comgr_obj_dir` in lit.site.cfg.py); both spellings are supported.
config.substitutions.append(('%raise_cli', _fwd(config.comgr_obj_dir, 'raise_cli')))
