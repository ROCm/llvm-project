cmake_minimum_required(VERSION 3.20)
project(hotswap-translation-cache LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- LLVM ---------------------------------------------------------------------
if(NOT TARGET LLVMSupport)
  find_package(LLVM REQUIRED CONFIG)
endif()
if(NOT COMMAND llvm_update_compile_flags)
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
  include(AddLLVM)
endif()

include_directories(${LLVM_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})
add_definitions(${LLVM_DEFINITIONS})

# --- Library ------------------------------------------------------------------
# OBJECT library so its translation units land directly in amd_comgr.so when
# `target_link_libraries(amd_comgr PRIVATE hotswap::translation_cache)` runs
# in the parent CMakeLists.txt. Depends only on LLVM: the transpile pipeline
# result type (pipeline.h) and the AMDGPU kernel-name query
# (code-object-utils) are the module's only cross-file surface, and both are
# LLVM-only, so the cache carries no comgr metadata-layer coupling.
add_library(hotswap-translation-cache OBJECT
  translation-cache.cpp
  code-object-utils.cpp
)

if(NOT TARGET hotswap::translation_cache)
  add_library(hotswap::translation_cache ALIAS hotswap-translation-cache)
endif()

# Match LLVM's compile flags (no-rtti, no-exceptions); mixing RTTI-on TUs with
# RTTI-off LLVM libs produces undefined-reference errors for `typeinfo`
# symbols on any class with a virtual method.
llvm_update_compile_flags(hotswap-translation-cache)

set_target_properties(hotswap-translation-cache
  PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Public include root (src/) so consumers can
# `#include "hotswap/cache/translation-cache.h"`.
target_include_directories(hotswap-translation-cache PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
)

target_link_libraries(hotswap-translation-cache
  PUBLIC
    LLVMBinaryFormat
    LLVMCore
    LLVMObject
    LLVMSupport
    LLVMTargetParser
)
