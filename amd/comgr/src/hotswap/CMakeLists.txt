cmake_minimum_required(VERSION 3.20)
project(hotswap-transpiler LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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
# `target_link_libraries(amd_comgr PRIVATE hotswap::transpiler)` runs (gated
# on COMGR_ENABLE_HOTSWAP_TRANSPILE in the parent CMakeLists.txt). This puts
# the hotswap raiser and the rest of comgr in the same TU set so hotswap
# files can call comgr-metadata helpers directly without a layering inversion
# or a separate static archive.
add_library(hotswap-transpiler OBJECT
  raiser.cpp
)

if(NOT TARGET hotswap::transpiler)
  add_library(hotswap::transpiler ALIAS hotswap-transpiler)
endif()

# Match LLVM's compile flags (no-rtti, no-exceptions) — every LLVM type we
# consume relies on LLVM's hand-rolled RTTI rather than C++ typeid; mixing
# RTTI-on TUs with RTTI-off LLVM libs produces undefined-reference errors
# for `typeinfo` symbols on any class with a virtual method.
llvm_update_compile_flags(hotswap-transpiler)

set_target_properties(hotswap-transpiler PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Public include root so consumers can `#include "hotswap/raiser.hpp"`.
target_include_directories(hotswap-transpiler PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
)

target_link_libraries(hotswap-transpiler
  PUBLIC
    LLVMCore
    LLVMSupport
    LLVMTargetParser
)

# HSA_TOOLS_LIB hotswap tool (libamd_comgr_hotswap_tool.so): ROCR loads it via
# HSA_TOOLS_LIB and it rewrites code objects through the comgr hotswap API.
# Off by default - it is an opt-in artifact most comgr consumers do not need.
# When enabled it requires the HSA headers (inc/hsa.h, from
# HOTSWAP_TOOL_HSA_INCLUDE_ROOT); configuration errors if they are not found.
option(HOTSWAP_BUILD_TOOL "Build the HSA_TOOLS_LIB hotswap tool" OFF)
if(HOTSWAP_BUILD_TOOL)
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR
      "HOTSWAP_BUILD_TOOL is only supported on Linux (it requires the HSA "
      "runtime); set HOTSWAP_BUILD_TOOL=OFF on this platform.")
  endif()
  find_path(HOTSWAP_TOOL_HSA_INC inc/hsa.h HINTS "${HOTSWAP_TOOL_HSA_INCLUDE_ROOT}")
  if(NOT HOTSWAP_TOOL_HSA_INC)
    message(FATAL_ERROR
      "HOTSWAP_BUILD_TOOL is ON but inc/hsa.h was not found. Set "
      "HOTSWAP_TOOL_HSA_INCLUDE_ROOT to the rocr-runtime hsa-runtime directory, "
      "or set HOTSWAP_BUILD_TOOL=OFF.")
  endif()
  # amd/comgr/src/hotswap binary dir -> amd/comgr binary dir (generated amd_comgr.h).
  get_filename_component(_comgr_bin "${CMAKE_CURRENT_BINARY_DIR}/../.." ABSOLUTE)
  add_library(amd_comgr_hotswap_tool SHARED comgr-hotswap-tool.cpp)
  set_target_properties(amd_comgr_hotswap_tool PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON)
  target_include_directories(amd_comgr_hotswap_tool PRIVATE
    "${HOTSWAP_TOOL_HSA_INC}"
    "${_comgr_bin}/include")  # generated amd_comgr.h
  target_link_libraries(amd_comgr_hotswap_tool PRIVATE amd_comgr)
  install(TARGETS amd_comgr_hotswap_tool
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT amd-comgr)
endif()
