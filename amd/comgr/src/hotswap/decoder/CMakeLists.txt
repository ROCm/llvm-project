cmake_minimum_required(VERSION 3.20)
project(hotswap-decoder LANGUAGES CXX)

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
# The per-ISA AMDGPU MC stack (mc-state) plus the architecture-neutral
# instruction identity the raiser dispatches on (canonical-op) and the MCInst
# format helpers (amdgpu-formats). Its own OBJECT library so the MC/decoder
# role stays distinct from the raiser and the code-object loader.

# Follow COMGR's selected LLVM linkage mode: in a dylib build link the single
# `LLVM` shared library, otherwise the individual static components. Linking
# fixed component libraries unconditionally can pull both the dylib and the
# static components into a dylib build, risking duplicate symbols. Kept PUBLIC
# so consumers inherit the same set when they link this OBJECT library.
if(LLVM_LINK_LLVM_DYLIB)
  set(HOTSWAP_LLVM_LIBS LLVM)
else()
  set(HOTSWAP_LLVM_LIBS
    LLVMAMDGPUAsmParser
    LLVMAMDGPUCodeGen
    LLVMAMDGPUDesc
    LLVMAMDGPUDisassembler
    LLVMAMDGPUInfo
    LLVMAMDGPUUtils
    LLVMAnalysis
    LLVMAsmPrinter
    LLVMBinaryFormat
    LLVMCodeGen
    LLVMCore
    LLVMMC
    LLVMMCDisassembler
    LLVMMCParser
    LLVMObject
    LLVMPasses
    LLVMSupport
    LLVMTarget
    LLVMTargetParser
    LLVMTransformUtils)
endif()

add_library(hotswap-decoder OBJECT
  mc-state.cpp
  amdgpu-formats.cpp
  canonical-op.cpp
)

if(NOT TARGET hotswap::decoder)
  add_library(hotswap::decoder ALIAS hotswap-decoder)
endif()

llvm_update_compile_flags(hotswap-decoder)
set_target_properties(hotswap-decoder PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Public include root (src/) so consumers can `#include "hotswap/..."`.
target_include_directories(hotswap-decoder PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
)

# The MC stack consumes AMDGPU target-private headers (AMDGPUMCTargetDesc.h,
# AMDGPUBaseInfo.h, SIDefines.h), which transitively include the TableGen-
# generated AMDGPUGen*.inc files. Those live in the LLVM source and build trees
# and are not exposed by an install tree.
# SIDefines.h and its siblings are hand-written AMDGPU headers that live only
# in the LLVM source tree; the build tree carries just the generated *.inc.
# Locate that source across the layouts comgr builds in: an in-tree monorepo
# build exposes it as LLVM_MAIN_SRC_DIR, a find_package(LLVM) consumer as
# LLVM_BUILD_MAIN_SRC_DIR, and the ROCm superbuild sets neither but keeps
# amd/comgr next to llvm/ under a shared root. Probe for the header rather than
# trust one variable, so a wrong or empty root fails the configure loudly.
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
    "hotswap-decoder: cannot find the LLVM AMDGPU target sources "
    "(SIDefines.h); set LLVM_MAIN_SRC_DIR to the llvm source directory.")
endif()
target_include_directories(hotswap-decoder PRIVATE
  "${HOTSWAP_AMDGPU_SRC_DIR}"
  "${HOTSWAP_AMDGPU_SRC_DIR}/Utils"
  "${LLVM_BINARY_DIR}/lib/Target/AMDGPU"
)

target_link_libraries(hotswap-decoder PUBLIC ${HOTSWAP_LLVM_LIBS})
