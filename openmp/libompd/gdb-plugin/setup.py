from setuptools import setup, Extension, find_packages

import os
import shlex
import subprocess

dir_path = os.path.dirname(os.path.realpath(__file__))
omp_include_dir = os.environ.get("LIBOMP_INCLUDE_DIR", dir_path)
python_include_dir = os.environ.get("PYTHON_HEADERS", dir_path)
llvm_include_dir = os.environ.get("LLVM_MAIN_INCLUDE_DIR", dir_path)
llvm_install_include = os.environ.get("LLVM_INCLUDE_DIR", "")
llvm_include_dirs = [d for d in os.environ.get("LLVM_INCLUDE_DIRS", "").split(";") if d]
llvm_library_dir = os.environ.get("LLVM_LIBRARY_DIR", "")
llvm_config = os.environ.get("LLVM_CONFIG", "")
link_llvm_dylib = os.environ.get("LLVM_LINK_LLVM_DYLIB", "").upper() in (
    "1",
    "ON",
    "TRUE",
    "YES",
)


def llvm_config_flags(*args):
    if not llvm_config or not os.path.isfile(llvm_config):
        return []
    try:
        out = subprocess.check_output([llvm_config] + list(args), text=True)
    except (OSError, subprocess.CalledProcessError):
        return []
    return shlex.split(out)


include_dirs = [omp_include_dir]
for d in [llvm_install_include, llvm_include_dir] + llvm_include_dirs:
    if d and d not in include_dirs:
        include_dirs.append(d)

library_dirs = [llvm_library_dir] if llvm_library_dir else []
# Prefer llvm-config so AOMP's versioned libLLVM name is used.
extra_link_args = llvm_config_flags("--ldflags", "--libs", "support", "--system-libs")
libraries = ["dl"]
if not extra_link_args:
    libraries.append("LLVM" if link_llvm_dylib else "LLVMSupport")

print("find_packages : ", find_packages())
setup(
    name="ompd",
    version="1.0",
    py_modules=["loadompd"],
    setup_requires=["wheel"],
    packages=find_packages(),
    ext_modules=[
        Extension(
            "ompd.ompdModule",
            [
                dir_path + "/ompdModule.c",
                dir_path + "/ompdAPITests.c",
                dir_path + "/ompdDLService.cpp",
            ],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            runtime_library_dirs=["$ORIGIN:$ORIGIN/../lib"],
            libraries=libraries,
            extra_link_args=extra_link_args,
            language="c++",
        )
    ],
)
