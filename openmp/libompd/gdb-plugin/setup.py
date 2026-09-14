from setuptools import setup, Extension, find_packages

import os

dir_path = os.path.dirname(os.path.realpath(__file__))
omp_include_dir = os.environ.get("LIBOMP_INCLUDE_DIR", dir_path)
python_include_dir = os.environ.get("PYTHON_HEADERS", dir_path)
llvm_include_dir = os.environ.get("LLVM_MAIN_INCLUDE_DIR", dir_path)
llvm_include_dirs = [d for d in os.environ.get("LLVM_INCLUDE_DIRS", "").split(";") if d]
llvm_library_dir = os.environ.get("LLVM_LIBRARY_DIR", "")
link_llvm_dylib = os.environ.get("LLVM_LINK_LLVM_DYLIB", "").upper() in (
    "1",
    "ON",
    "TRUE",
    "YES",
)

include_dirs = [omp_include_dir, llvm_include_dir] + llvm_include_dirs
library_dirs = [llvm_library_dir] if llvm_library_dir else []
llvm_lib = "LLVM" if link_llvm_dylib else "LLVMSupport"

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
            libraries=["dl", llvm_lib],
        )
    ],
)
