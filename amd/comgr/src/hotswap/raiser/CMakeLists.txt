cmake_minimum_required(VERSION 3.20)

# No project() here: this is an add_subdirectory component of amd_comgr, not a
# standalone build. A nested project() re-runs the superbuild dependency provider
# and re-imports LLVM, which links a second copy of the AMDGPU static libraries
# into amd_comgr and double-registers their cl::opts under a non-folding linker.

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
  raise_failure.cpp
  reg-file.cpp
  wave-projection.cpp
  user-sgpr-layout.cpp
  register-state.cpp
  raise-context.cpp
  op-resolver.cpp
  handle-sop1.cpp
  handle-sopp.cpp
)

if(NOT TARGET hotswap::raiser)
  add_library(hotswap::raiser ALIAS hotswap-raiser)
endif()

# Match LLVM's compile flags (no-rtti, no-exceptions) — every LLVM type we
# consume relies on LLVM's hand-rolled RTTI rather than C++ typeid; mixing
# RTTI-on TUs with RTTI-off LLVM libs produces undefined-reference errors
# for `typeinfo` symbols on any class with a virtual method.
llvm_update_compile_flags(hotswap-raiser)
# llvm_update_compile_flags does not reliably carry LLVM's -fno-rtti to a
# standalone find_package(LLVM) consumer. RaiseFailure derives from
# llvm::ErrorInfo, so mirror LLVM_ENABLE_RTTI explicitly.
if(NOT LLVM_ENABLE_RTTI)
  if(MSVC)
    target_compile_options(hotswap-raiser PRIVATE /GR-)
  else()
    target_compile_options(hotswap-raiser PRIVATE -fno-rtti)
  endif()
endif()

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

# The wave projection lowers decoded instructions to IR, so it links the decoder
# OBJECT library and inherits its curated LLVM component set (the AMDGPU MC/Desc/
# Utils libraries the projection's register queries need).
# PRIVATE keeps these out of the link interface, but CMake still records them as
# $<LINK_ONLY:> for an OBJECT library; clear the interface so consumers get the
# objects only (see ../CMakeLists.txt).
target_link_libraries(hotswap-raiser
  PRIVATE
    hotswap::decoder
    ${hotswap_raiser_llvm_libs}
)
set_target_properties(hotswap-raiser PROPERTIES INTERFACE_LINK_LIBRARIES "")

# The wave projection consumes AMDGPU target-private headers (AMDGPUMCTargetDesc.h,
# AMDGPUBaseInfo.h). SIDefines.h and its siblings are hand-written headers that
# live only in the LLVM source tree; the TableGen-generated AMDGPUGen*.inc they
# transitively include live only in the LLVM build tree. Neither is exposed by an
# install tree, so locate both.
#
# Source: an in-tree monorepo build exposes it as LLVM_MAIN_SRC_DIR, a
# find_package(LLVM) consumer as LLVM_BUILD_MAIN_SRC_DIR, and the ROCm superbuild
# sets neither but keeps amd/comgr next to llvm/ under a shared root. Probe for
# the header rather than trust one variable, so a wrong or empty root fails the
# configure loudly.
set(HOTSWAP_AMDGPU_SRC_DIR "")
foreach(HotswapLLVMRoot
    "${LLVM_MAIN_SRC_DIR}"
    "${LLVM_BUILD_MAIN_SRC_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../llvm")
  if(HotswapLLVMRoot AND
     EXISTS "${HotswapLLVMRoot}/lib/Target/AMDGPU/SIDefines.h")
    get_filename_component(HOTSWAP_AMDGPU_SRC_DIR
      "${HotswapLLVMRoot}/lib/Target/AMDGPU" ABSOLUTE)
    break()
  endif()
endforeach()
if(NOT HOTSWAP_AMDGPU_SRC_DIR)
  message(FATAL_ERROR
    "hotswap-raiser: cannot find the LLVM AMDGPU target sources "
    "(SIDefines.h); set LLVM_MAIN_SRC_DIR to the llvm source directory.")
endif()
# Generated headers: the TableGen-generated AMDGPUGen*.inc live in the LLVM build
# tree, never an install prefix, so in an installed LLVMConfig.cmake
# LLVM_BINARY_DIR (the install prefix) carries none of them. Probe, in order:
#   1. LLVM_BUILD_BINARY_DIR, if a consumer points us straight at its LLVM build;
#   2. LLVM_BINARY_DIR, correct for an in-tree monorepo build;
#   3. the ROCm/TheRock superbuild layout, where comgr consumes the installed
#      LLVM under <root>/dist/lib/llvm while the producing build sits beside it
#      at <root>/build (LLVM_BINARY_DIR is .../dist/lib/llvm there).
# Validate every header the projection pulls in so an unrecognised layout fails
# the configure loudly.
set(HOTSWAP_AMDGPU_GEN_DIR "")
foreach(HotswapLLVMBuild
    "${LLVM_BUILD_BINARY_DIR}"
    "${LLVM_BINARY_DIR}"
    "${LLVM_BINARY_DIR}/../../../build")
  if(HotswapLLVMBuild AND
     EXISTS "${HotswapLLVMBuild}/lib/Target/AMDGPU/AMDGPUGenInstrInfo.inc")
    get_filename_component(HOTSWAP_AMDGPU_GEN_DIR
      "${HotswapLLVMBuild}/lib/Target/AMDGPU" ABSOLUTE)
    break()
  endif()
endforeach()
foreach(HotswapGenInc
    AMDGPUGenInstrInfo.inc
    AMDGPUGenRegisterInfo.inc
    AMDGPUGenSubtargetInfo.inc)
  if(NOT EXISTS "${HOTSWAP_AMDGPU_GEN_DIR}/${HotswapGenInc}")
    message(FATAL_ERROR
      "hotswap-raiser: cannot find the generated AMDGPU headers "
      "(${HotswapGenInc}); pass -DLLVM_BUILD_BINARY_DIR=<llvm build dir>.")
  endif()
endforeach()

target_include_directories(hotswap-raiser PRIVATE
  "${HOTSWAP_AMDGPU_SRC_DIR}"
  "${HOTSWAP_AMDGPU_SRC_DIR}/Utils"
  "${HOTSWAP_AMDGPU_GEN_DIR}"
)
