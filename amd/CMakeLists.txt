# See for https://openmp.llvm.org/SupportAndFAQ.html for instructions on how
# to build offload with CMake.

cmake_minimum_required(VERSION 3.20.0)
if("${CMAKE_VERSION}" VERSION_LESS "3.31.0")
  message(WARNING
    "Your CMake version is ${CMAKE_VERSION}. Starting with LLVM 24, the "
    "minimum version of CMake required to build LLVM will become 3.31.0, and "
    "using an older CMake will become an error. Please upgrade your CMake to "
    "at least 3.31.0 now to avoid issues in the future.")
endif()

add_subdirectory(tools)
