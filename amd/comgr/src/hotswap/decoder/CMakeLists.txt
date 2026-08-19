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
# The per-ISA AMDGPU MC stack (mc-state) plus the architecture-neutral
# instruction identity the raiser dispatches on (canonical-op) and the MCInst
# format helpers (amdgpu-formats). Its own OBJECT library so the MC/decoder
# role stays distinct from the raiser and the code-object loader.

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

add_library(hotswap-decoder OBJECT
  mc-state.cpp
  amdgpu-mc-tables.cpp
  amdgpu-formats.cpp
  canonical-op.cpp
  isa-profile.cpp
  opcode-map.cpp
  decode.cpp
)

if(NOT TARGET hotswap::decoder)
  add_library(hotswap::decoder ALIAS hotswap-decoder)
endif()

llvm_update_compile_flags(hotswap-decoder)
# llvm_update_compile_flags does not reliably carry LLVM's -fno-rtti to a
# standalone find_package(LLVM) consumer, and the MC stack raises HotswapError
# (an llvm::ErrorInfo). Mirror LLVM_ENABLE_RTTI explicitly so we do not emit
# typeinfo an RTTI-off LLVM cannot resolve at link time.
if(NOT LLVM_ENABLE_RTTI)
  if(MSVC)
    target_compile_options(hotswap-decoder PRIVATE /GR-)
  else()
    target_compile_options(hotswap-decoder PRIVATE -fno-rtti)
  endif()
endif()
set_target_properties(hotswap-decoder PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Public include root (src/) so consumers can `#include "hotswap/..."`.
target_include_directories(hotswap-decoder PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
)

# The MC stack consumes AMDGPU target-private headers (AMDGPUMCTargetDesc.h,
# AMDGPUBaseInfo.h, SIDefines.h). SIDefines.h and its siblings are hand-written
# headers that live only in the LLVM source tree; the TableGen-generated
# AMDGPUGen*.inc they transitively include live only in the LLVM build tree.
# Neither is exposed by an install tree, so locate both.
#
# Source: an in-tree monorepo build exposes it as LLVM_MAIN_SRC_DIR, a
# find_package(LLVM) consumer as LLVM_BUILD_MAIN_SRC_DIR, and the ROCm
# superbuild sets neither but keeps amd/comgr next to llvm/ under a shared root.
# Probe for the header rather than trust one variable, so a wrong or empty root
# fails the configure loudly.
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
      "hotswap-decoder: cannot find the generated AMDGPU headers "
      "(${HotswapGenInc}); pass -DLLVM_BUILD_BINARY_DIR=<llvm build dir>.")
  endif()
endforeach()

# comgr.h (pulled in for COMGR::ensureLLVMInitialized) includes the configured
# amd_comgr.h from comgr's binary include/ tree
# (src/hotswap/decoder -> src/hotswap -> src -> comgr root).
get_filename_component(HOTSWAP_COMGR_BINARY_DIR
  "${CMAKE_CURRENT_BINARY_DIR}/../../.." ABSOLUTE)

target_include_directories(hotswap-decoder PRIVATE
  "${HOTSWAP_AMDGPU_SRC_DIR}"
  "${HOTSWAP_AMDGPU_SRC_DIR}/Utils"
  "${HOTSWAP_AMDGPU_GEN_DIR}"
  "${HOTSWAP_COMGR_BINARY_DIR}/include"
)

# Consumers choose their LLVM linkage, so do not propagate the OBJECT library's
# private dependencies through $<LINK_ONLY:>.
target_link_libraries(hotswap-decoder PRIVATE ${HOTSWAP_LLVM_LIBS})
set_target_properties(hotswap-decoder PROPERTIES INTERFACE_LINK_LIBRARIES "")
