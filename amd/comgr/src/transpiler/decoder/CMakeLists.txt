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
# The per-ISA AMDGPU MC stack (mc-state) plus the architecture-neutral
# instruction identity the raiser dispatches on (canonical-op) and the MCInst
# format helpers (amdgpu-formats). Its own OBJECT library so the MC/decoder
# role stays distinct from the raiser and the code-object loader.

# Follow COMGR's selected LLVM linkage mode: in a dylib build link the single
# `LLVM` shared library, otherwise the individual static components. Linking
# fixed component libraries unconditionally can pull both the dylib and the
# static components into a dylib build, risking duplicate symbols.
if(LLVM_LINK_LLVM_DYLIB)
  set(TRANSPILER_LLVM_LIBS LLVM)
else()
  set(TRANSPILER_LLVM_LIBS
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

add_library(transpiler-decoder OBJECT
  mc-state.cpp
  amdgpu-mc-tables.cpp
  amdgpu-formats.cpp
  canonical-op.cpp
  opcode-map.cpp
  decode.cpp
)

if(NOT TARGET transpiler::decoder)
  add_library(transpiler::decoder ALIAS transpiler-decoder)
endif()

llvm_update_compile_flags(transpiler-decoder)
# llvm_update_compile_flags does not reliably carry LLVM's -fno-rtti to a
# standalone find_package(LLVM) consumer, and the MC stack raises TranspilerError
# (an llvm::ErrorInfo). Mirror LLVM_ENABLE_RTTI explicitly so we do not emit
# typeinfo an RTTI-off LLVM cannot resolve at link time.
if(NOT LLVM_ENABLE_RTTI)
  if(MSVC)
    target_compile_options(transpiler-decoder PRIVATE /GR-)
  else()
    target_compile_options(transpiler-decoder PRIVATE -fno-rtti)
  endif()
endif()
set_target_properties(transpiler-decoder PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS No
  POSITION_INDEPENDENT_CODE ON)

# Public include root (src/) so consumers can `#include "transpiler/..."`.
target_include_directories(transpiler-decoder PUBLIC
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
set(TRANSPILER_AMDGPU_SRC_DIR "")
foreach(TranspilerLLVMRoot
    "${LLVM_MAIN_SRC_DIR}"
    "${LLVM_BUILD_MAIN_SRC_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../llvm")
  if(TranspilerLLVMRoot AND
     EXISTS "${TranspilerLLVMRoot}/lib/Target/AMDGPU/SIDefines.h")
    get_filename_component(TRANSPILER_AMDGPU_SRC_DIR
      "${TranspilerLLVMRoot}/lib/Target/AMDGPU" ABSOLUTE)
    break()
  endif()
endforeach()
if(NOT TRANSPILER_AMDGPU_SRC_DIR)
  message(FATAL_ERROR
    "transpiler-decoder: cannot find the LLVM AMDGPU target sources "
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
set(TRANSPILER_AMDGPU_GEN_DIR "")
foreach(TranspilerLLVMBuild
    "${LLVM_BUILD_BINARY_DIR}"
    "${LLVM_BINARY_DIR}"
    "${LLVM_BINARY_DIR}/../../../build")
  if(TranspilerLLVMBuild AND
     EXISTS "${TranspilerLLVMBuild}/lib/Target/AMDGPU/AMDGPUGenInstrInfo.inc")
    get_filename_component(TRANSPILER_AMDGPU_GEN_DIR
      "${TranspilerLLVMBuild}/lib/Target/AMDGPU" ABSOLUTE)
    break()
  endif()
endforeach()
foreach(TranspilerGenInc
    AMDGPUGenInstrInfo.inc
    AMDGPUGenRegisterInfo.inc
    AMDGPUGenSubtargetInfo.inc)
  if(NOT EXISTS "${TRANSPILER_AMDGPU_GEN_DIR}/${TranspilerGenInc}")
    message(FATAL_ERROR
      "transpiler-decoder: cannot find the generated AMDGPU headers "
      "(${TranspilerGenInc}); pass -DLLVM_BUILD_BINARY_DIR=<llvm build dir>.")
  endif()
endforeach()

# comgr.h (pulled in for COMGR::ensureLLVMInitialized) includes the configured
# amd_comgr.h from comgr's binary include/ tree
# (src/transpiler/decoder -> src/transpiler -> src -> comgr root).
get_filename_component(TRANSPILER_COMGR_BINARY_DIR
  "${CMAKE_CURRENT_BINARY_DIR}/../../.." ABSOLUTE)

target_include_directories(transpiler-decoder PRIVATE
  "${TRANSPILER_AMDGPU_SRC_DIR}"
  "${TRANSPILER_AMDGPU_SRC_DIR}/Utils"
  "${TRANSPILER_AMDGPU_GEN_DIR}"
  "${TRANSPILER_COMGR_BINARY_DIR}/include"
)

# Consumers choose their LLVM linkage, so do not propagate the OBJECT library's
# private dependencies through $<LINK_ONLY:>.
target_link_libraries(transpiler-decoder PRIVATE ${TRANSPILER_LLVM_LIBS})
set_target_properties(transpiler-decoder PROPERTIES INTERFACE_LINK_LIBRARIES "")
