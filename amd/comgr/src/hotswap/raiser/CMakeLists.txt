cmake_minimum_required(VERSION 3.20)
project(hotswap-raiser LANGUAGES CXX)

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
# `target_link_libraries(amd_comgr PRIVATE hotswap::raiser)` runs (gated
# on COMGR_ENABLE_HOTSWAP_TRANSPILE in the parent CMakeLists.txt). This puts
# the hotswap raiser and the rest of comgr in the same TU set so hotswap
# files can call comgr-metadata helpers directly without a layering inversion
# or a separate static archive.
add_library(hotswap-raiser OBJECT
  raiser.cpp
)

if(NOT TARGET hotswap::raiser)
  add_library(hotswap::raiser ALIAS hotswap-raiser)
endif()

# Match LLVM's compile flags (no-rtti, no-exceptions) — every LLVM type we
# consume relies on LLVM's hand-rolled RTTI rather than C++ typeid; mixing
# RTTI-on TUs with RTTI-off LLVM libs produces undefined-reference errors
# for `typeinfo` symbols on any class with a virtual method.
llvm_update_compile_flags(hotswap-raiser)

set_target_properties(hotswap-raiser PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Public include root (src/) so consumers can `#include "hotswap/raiser/raiser.h"`.
target_include_directories(hotswap-raiser PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
)

# Follow COMGR's selected LLVM linkage mode: the single LLVM shared library in a
# dylib build, otherwise the individual static components, so amd_comgr never
# links a component both statically and through libLLVM.
if(LLVM_LINK_LLVM_DYLIB)
  set(hotswap_raiser_llvm_libs LLVM)
else()
  set(hotswap_raiser_llvm_libs LLVMCore LLVMSupport LLVMTargetParser)
endif()

# PRIVATE keeps these out of the link interface, but CMake still records them as
# $<LINK_ONLY:> for an OBJECT library; clear the interface so consumers get the
# objects only (see ../CMakeLists.txt).
target_link_libraries(hotswap-raiser
  PRIVATE
    ${hotswap_raiser_llvm_libs}
)
set_target_properties(hotswap-raiser PROPERTIES INTERFACE_LINK_LIBRARIES "")
