cmake_minimum_required(VERSION 3.20)

# No project() here: this is an add_subdirectory component of amd_comgr, not a
# standalone build. A nested project() re-runs the superbuild dependency provider
# and re-imports LLVM, which links a second copy of the AMDGPU static libraries
# into amd_comgr and double-registers their cl::opts under a non-folding linker.

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
# Reads an AMDGPU code object's metadata (MsgPack notes + kernel descriptor)
# into the neutral common/kernel-meta.h model the raiser consumes. Built as its
# own OBJECT library, separate from the raiser, so a client that only needs
# `raiseToIR` (e.g. the raiser unit test) does not pull in the loader's
# comgr-metadata / comgr-symbol dependencies.

# Follow COMGR's selected LLVM linkage mode: in a dylib build link the single
# `LLVM` shared library, otherwise the individual static components. Linking
# fixed component libraries unconditionally can pull both the dylib and the
# static components into a dylib build, risking duplicate symbols.
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

add_library(hotswap-loader OBJECT
  code-object-utils.cpp
)

if(NOT TARGET hotswap::loader)
  add_library(hotswap::loader ALIAS hotswap-loader)
endif()

llvm_update_compile_flags(hotswap-loader)
# llvm_update_compile_flags does not reliably carry LLVM's -fno-rtti to a
# standalone find_package(LLVM) consumer, and the loader raises HotswapError
# (an llvm::ErrorInfo). Mirror LLVM_ENABLE_RTTI explicitly so we do not emit
# typeinfo an RTTI-off LLVM cannot resolve at link time.
if(NOT LLVM_ENABLE_RTTI)
  if(MSVC)
    target_compile_options(hotswap-loader PRIVATE /GR-)
  else()
    target_compile_options(hotswap-loader PRIVATE -fno-rtti)
  endif()
endif()
set_target_properties(hotswap-loader PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS No
  POSITION_INDEPENDENT_CODE ON)

# Match LLVM's RTTI setting explicitly. llvm_update_compile_flags only adds
# -fno-rtti when it detects a GCC-compatible compiler, which is not set in this
# subproject scope, so against an RTTI-disabled libLLVM the ErrorInfo<HotswapError>
# instantiation would emit C++ typeinfo referencing llvm::ErrorInfoBase, a symbol
# that library never defines.
if(NOT LLVM_ENABLE_RTTI)
  if(MSVC)
    target_compile_options(hotswap-loader PRIVATE /GR-)
  else()
    target_compile_options(hotswap-loader PRIVATE -fno-rtti)
  endif()
endif()

# Public include root (src/) so consumers can `#include "hotswap/..."`.
target_include_directories(hotswap-loader PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
)

# The decoder consumes AMDGPU target-private headers (SIDefines.h,
# SIInstrInfo.h, AMDGPUBaseInfo.h), which transitively include the
# TableGen-generated AMDGPUGen*.inc files. Those live in the LLVM source and
# build trees and are not exposed by an install tree.
# comgr-metadata.h -> comgr.h -> amd_comgr.h, configured into comgr's binary
# include/ directory (src/hotswap/loader -> src/hotswap -> src -> comgr root).
get_filename_component(HOTSWAP_COMGR_BINARY_DIR
  "${CMAKE_CURRENT_BINARY_DIR}/../../.." ABSOLUTE)
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
    "hotswap-loader: cannot find the LLVM AMDGPU target sources "
    "(SIDefines.h); set LLVM_MAIN_SRC_DIR to the llvm source directory.")
endif()

# Generated headers: the TableGen-generated AMDGPUGen*.inc live in the LLVM
# build tree, never an install prefix, so in an installed LLVMConfig.cmake
# LLVM_BINARY_DIR (the install prefix) carries none of them. Probe, in order:
#   1. LLVM_BUILD_BINARY_DIR, if a consumer points us straight at its LLVM build;
#   2. LLVM_BINARY_DIR, correct for an in-tree monorepo build;
#   3. the ROCm/TheRock superbuild layout, where comgr consumes the installed
#      LLVM under <root>/dist/lib/llvm while the producing build sits beside it
#      at <root>/build (LLVM_BINARY_DIR is .../dist/lib/llvm there).
# Validate every header the MC stack pulls in so an unrecognised layout fails
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
      "hotswap-loader: cannot find the generated AMDGPU headers "
      "(${HotswapGenInc}); pass -DLLVM_BUILD_BINARY_DIR=<llvm build dir>.")
  endif()
endforeach()

target_include_directories(hotswap-loader PRIVATE
  "${HOTSWAP_AMDGPU_SRC_DIR}"
  "${HOTSWAP_AMDGPU_SRC_DIR}/Utils"
  "${HOTSWAP_AMDGPU_GEN_DIR}"
  "${HOTSWAP_COMGR_BINARY_DIR}/include"
)

# PRIVATE keeps these out of the link interface, but CMake still records them as
# $<LINK_ONLY:> for an OBJECT library; clear the interface so consumers get the
# objects only (see ../CMakeLists.txt).
target_link_libraries(hotswap-loader PRIVATE ${HOTSWAP_LLVM_LIBS})
set_target_properties(hotswap-loader PROPERTIES INTERFACE_LINK_LIBRARIES "")
