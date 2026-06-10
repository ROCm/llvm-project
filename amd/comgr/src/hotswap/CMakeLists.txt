cmake_minimum_required(VERSION 3.20)
project(hotswap-transpiler LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Paths to LLVM ------------------------------------------------------------
# Hotswap needs an LLVM *build tree* (not an install tree) because the raiser
# uses AMDGPU target-private headers (SIDefines.h, SIInstrInfo.h,
# AMDGPUBaseInfo.h) which transitively #include TableGen-generated
# AMDGPUGen*.inc files. Those .inc files are produced by llvm-tblgen during
# the LLVM build and are not copied by the `install` target, so we
# require an LLVM *build tree* rather than an install tree.
#
# We support three configurations:
#
#   1. In-tree LLVM build. Hotswap was added via add_subdirectory() from
#      COMGR's CMake via add_subdirectory(). LLVM has
#      already populated LLVM_MAIN_SRC_DIR / LLVM_BINARY_DIR and the LLVM
#      library targets; AddLLVM.cmake is loaded. No find_package needed.
#
#   2. add_subdirectory() from another out-of-tree project (comgr, ROCR,
#      ...) that already called find_package(LLVM). Variables are
#      populated; the find_package call below becomes a no-op.
#
#   3. Standalone (cmake -S amd/comgr/hotswap -B build). We need
#      find_package(LLVM) to resolve LLVM_DIR. Since hotswap now lives at
#      <llvm-project>/amd/comgr/hotswap, default LLVM_DIR to the conventional
#      <llvm-project>/build tree if the user did not pass one explicitly.
if(TARGET LLVMSupport AND DEFINED LLVM_MAIN_SRC_DIR)
  # Case 1. Mirror the variable name find_package(LLVM) would have set
  # against a build tree, so the rest of this file does not have to
  # branch.
  set(LLVM_BUILD_MAIN_SRC_DIR "${LLVM_MAIN_SRC_DIR}")
else()
  # Cases 2 and 3.
  if(NOT DEFINED LLVM_DIR OR NOT LLVM_DIR)
    # amd/comgr/hotswap/ -> amd/comgr/ -> amd/ -> <llvm-project>/
    set(_hotswap_default_llvm_build
        "${CMAKE_CURRENT_SOURCE_DIR}/../../../build/lib/cmake/llvm")
    if(EXISTS "${_hotswap_default_llvm_build}/LLVMConfig.cmake")
      get_filename_component(_hotswap_default_llvm_build
                             "${_hotswap_default_llvm_build}" ABSOLUTE)
      set(LLVM_DIR "${_hotswap_default_llvm_build}" CACHE PATH
          "Path to LLVMConfig.cmake (auto-detected from <llvm-project>/build)")
      message(STATUS
        "LLVM_DIR not set; defaulting to ${LLVM_DIR}")
    endif()
    unset(_hotswap_default_llvm_build)
  endif()
  find_package(LLVM REQUIRED CONFIG)
endif()

if(NOT DEFINED LLVM_BUILD_MAIN_SRC_DIR OR NOT LLVM_BUILD_MAIN_SRC_DIR OR
   NOT DEFINED LLVM_BINARY_DIR OR NOT LLVM_BINARY_DIR)
  message(FATAL_ERROR
    "hotswap-transpiler requires an LLVM build tree (not an install tree). "
    "The install tree does not expose AMDGPU target-private headers "
    "(SIDefines.h, AMDGPUBaseInfo.h, AMDGPUGen*.inc). "
    "Pass -DLLVM_DIR=<llvm-build>/lib/cmake/llvm or build LLVM at "
    "<llvm-project>/build so it can be auto-detected.")
endif()

if(DEFINED LLVM_DIR AND LLVM_DIR)
  message(STATUS "Using LLVM from: ${LLVM_DIR}")
endif()
message(STATUS "LLVM source tree: ${LLVM_BUILD_MAIN_SRC_DIR}")
message(STATUS "LLVM build tree:  ${LLVM_BINARY_DIR}")
if(DEFINED LLVM_PACKAGE_VERSION)
  message(STATUS "LLVM version:     ${LLVM_PACKAGE_VERSION}")
endif()

# In the in-tree case (1), AddLLVM is already loaded by the parent LLVM
# build and llvm_update_compile_flags is available. Only pull it in
# explicitly for cases (2) and (3).
if(NOT COMMAND llvm_update_compile_flags)
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
  include(AddLLVM)
endif()

# --- Include paths ------------------------------------------------------------
include_directories(${LLVM_INCLUDE_DIRS})

# --- Link paths ---------------------------------------------------------------
link_directories(${LLVM_LIBRARY_DIRS})

# --- LLVM definitions ---------------------------------------------------------
add_definitions(${LLVM_DEFINITIONS})

# --- TDM emulation runtime helper (HIP -> bitcode -> embedded array) --------
# When `hipcc` is available we build `runtime/tdm.hip` to LLVM bitcode for
# gfx942 and embed the bytes via `cmake/EmbedBinary.cmake`. The transpiler
# parses and link-merges the bitcode into each raised IR module that uses
# the cross-target VIMAGE TENSOR helper (see `tdm-runtime.{h,cpp}` and the
# raiser's cross-target VIMAGE handling). When hipcc is missing the embedded
# blob is empty (one-byte stub in tdm-runtime.cpp), `tdmRuntimeAvailable()`
# returns false, and the cross-target VIMAGE handler keeps its loud refusal.
# This preserves the no-hipcc build behaviour exactly.
find_program(HIPCC hipcc)
set(HOTSWAP_TDM_RUNTIME_GENERATED_CPP "")
if(HIPCC)
  set(_tdm_src "${CMAKE_CURRENT_SOURCE_DIR}/runtime/tdm.hip")
  set(_tdm_bc  "${CMAKE_CURRENT_BINARY_DIR}/tdm_runtime.gfx942.bc")
  set(_tdm_cpp "${CMAKE_CURRENT_BINARY_DIR}/tdm_runtime_blob.cpp")

  add_custom_command(
    OUTPUT ${_tdm_bc}
    COMMAND ${HIPCC}
            --offload-arch=gfx942
            -O2 -emit-llvm -c
            -fgpu-rdc
            --cuda-device-only
            -o ${_tdm_bc} ${_tdm_src}
    DEPENDS ${_tdm_src}
    COMMENT "Building TDM emulation runtime bitcode (gfx942)"
    VERBATIM)

  add_custom_command(
    OUTPUT ${_tdm_cpp}
    COMMAND ${CMAKE_COMMAND}
            -DINPUT=${_tdm_bc}
            -DOUTPUT=${_tdm_cpp}
            -DSYMBOL=tdm_runtime_bc
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedBinary.cmake
    DEPENDS ${_tdm_bc} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedBinary.cmake
    COMMENT "Embedding TDM runtime bitcode into C++ blob"
    VERBATIM)

  set(HOTSWAP_TDM_RUNTIME_GENERATED_CPP ${_tdm_cpp})
  set(HOTSWAP_HAVE_TDM_RUNTIME 1)
  message(STATUS "TDM runtime helper enabled (hipcc=${HIPCC})")
else()
  set(HOTSWAP_HAVE_TDM_RUNTIME 0)
  message(STATUS "TDM runtime helper DISABLED (hipcc not found); "
                 "cross-target gfx1250->gfx942 TENSOR ops will keep refusing")
endif()

# Surface the flag to other subdirectories (e.g. test-lit, which gates the
# `tdm-runtime` lit feature on it so the tensor_*_lds.s fixtures only run
# when hipcc was present at configure time).
set(HOTSWAP_HAVE_TDM_RUNTIME ${HOTSWAP_HAVE_TDM_RUNTIME} CACHE INTERNAL
    "Hotswap built with TDM emulation runtime (hipcc available at configure time)")

# --- Raiser + Pipeline library ------------------------------------------------
# OBJECT library so its translation units land directly in amd_comgr.so when
# `target_link_libraries(amd_comgr PRIVATE hotswap::transpiler)` runs (gated
# on COMGR_ENABLE_HOTSWAP_TRANSPILE in the parent CMakeLists.txt). This puts
# the hotswap raiser and the rest of comgr in the same TU set so hotswap
# files can call comgr-metadata helpers directly without a layering inversion
# or a separate static archive.
add_library(hotswap-transpiler OBJECT
  raiser.cpp
  code-object-utils.cpp
  mc-state.cpp
  hotswap-error.cpp
  canonical-op.cpp
  opcode-map.cpp
  decode.cpp
  raise-failure.cpp
  raise-context.cpp
  reg-file.cpp
  kernarg-layout.cpp
  user-sgpr-layout.cpp
  wave-projection.cpp
  setpc-analysis.cpp
  handle-sopp.cpp
  handle-sopc.cpp
  handle-sop1.cpp
  handle-sop2.cpp
  handle-sopk.cpp
  canonical-op-attrs.cpp
  handle-smem.cpp
  source-hidden-args.cpp
  handle-flat.cpp
  flat-addr.cpp
  handle-mubuf.cpp
  mubuf-addr.cpp
  handle-ds.cpp
  handle-valu.cpp
  handle-valu-f16-utils.cpp
  handle-valu-output-mods.cpp
  handle-valu-small-ops.cpp
  handle-valu-cross-lane.cpp
  handle-valu-vcmp.cpp
  handle-valu-vop3p.cpp
  wmma-lowering.cpp
  mxfp4-dequant.cpp
  handle-mfma.cpp
  handle-vopd.cpp
  handle-vimage.cpp
  ocml-runtime.cpp
  tdm-runtime.cpp
  wave-size-obstruction.cpp
  rewrite-cross-lane-divergent.cpp
  c5-predicate-chain-classifier.cpp
  pipeline.cpp
  translation-cache.cpp
  ${HOTSWAP_TDM_RUNTIME_GENERATED_CPP}
)

if(NOT TARGET hotswap::transpiler)
  add_library(hotswap::transpiler ALIAS hotswap-transpiler)
endif()

target_compile_definitions(hotswap-transpiler PRIVATE
  HOTSWAP_HAVE_TDM_RUNTIME=${HOTSWAP_HAVE_TDM_RUNTIME}
)

# Match LLVM's compile flags (no-rtti, no-exceptions, etc.). LLVM is built
# with `LLVM_ENABLE_RTTI=OFF` and `LLVM_ENABLE_EH=OFF` by default, and every
# public LLVM type we consume -- Value, Type, Instruction, MCInst, ... -- relies
# on LLVM's hand-rolled RTTI (`isa<>` / `cast<>` / `dyn_cast<>`) rather than
# C++ `typeid` / `dynamic_cast`. Mixing an RTTI-on TU with RTTI-off LLVM libs
# produces undefined-reference errors for `typeinfo` symbols on any class with
# a virtual method. `llvm_update_compile_flags` (from AddLLVM) is the canonical
# way to stay in lockstep with libLLVM: it reads `LLVM_ENABLE_RTTI` /
# `LLVM_ENABLE_EH` from the LLVM CMake package and propagates the matching
# `-fno-rtti` / `-fno-exceptions` (or MSVC equivalents) to the target.
llvm_update_compile_flags(hotswap-transpiler)

# AMDGPU target-private headers (not exposed by the LLVM install tree).
# SIDefines.h, SIInstrInfo.h -> source tree Target/AMDGPU/{,Utils}
# AMDGPUGenInstrInfo.inc etc. -> build tree Target/AMDGPU
# These are PRIVATE because no public header (pipeline.hpp, transpiler.hpp)
# exposes AMDGPU-private LLVM types, so consumers (e.g. ROCR) do not need
# these paths to compile against us.
get_filename_component(HOTSWAP_COMGR_BINARY_DIR
  "${CMAKE_CURRENT_BINARY_DIR}/../.." ABSOLUTE)
target_include_directories(hotswap-transpiler PRIVATE
  "${LLVM_BUILD_MAIN_SRC_DIR}/lib/Target/AMDGPU"
  "${LLVM_BUILD_MAIN_SRC_DIR}/lib/Target/AMDGPU/Utils"
  "${LLVM_BINARY_DIR}/lib/Target/AMDGPU"
  # Hotswap calls into comgr-metadata helpers (`metadata::walkElfMetadataIntoDoc`,
  # `metadata::getElfIsaNameFromBuffer`), which transitively include
  # `comgr.h` -> `amd_comgr.h`. The latter is configured from
  # `amd/comgr/include/amd_comgr.h.in` into Comgr's binary `include/` dir.
  "${CMAKE_CURRENT_SOURCE_DIR}/.."
  "${HOTSWAP_COMGR_BINARY_DIR}/include"
)

# Allow this static library to be linked into a shared library
# (libhsa-runtime64.so when consumed from ROCR).
set_target_properties(hotswap-transpiler PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Public headers consumed by parent projects (pipeline.hpp, transpiler.hpp).
# Use the hotswap/ root so consumers can write `#include "hotswap/pipeline.hpp"`
# or `#include "hotswap/transpiler.hpp"`, matching existing ROCR usage.
target_include_directories(hotswap-transpiler PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
)

# `llc`, `llvm-mc`, `llvm-objdump` etc. live in the build tree's bin/.
# ROCR's trap-handler subproject may require a ROCm Clang package even when
# Hotswap itself is validated against a local LLVM build, so keep the tool path
# overrideable rather than deriving it unconditionally from the LLVM package
# used for in-process linkage.
set(HOTSWAP_TRANSPILER_LLVM_TOOLS_DIR "${LLVM_TOOLS_BINARY_DIR}" CACHE PATH
  "Directory containing llc, llvm-mc, llvm-objdump, and related LLVM tools used by Hotswap")
target_compile_definitions(hotswap-transpiler PRIVATE
  LLVM_TOOLS_DIR="${HOTSWAP_TRANSPILER_LLVM_TOOLS_DIR}"
)

target_link_libraries(hotswap-transpiler
  PUBLIC
    # LLVM AMDGPU components, sorted alphabetically.
    LLVMAMDGPUAsmParser
    LLVMAMDGPUCodeGen
    LLVMAMDGPUDesc
    LLVMAMDGPUDisassembler
    LLVMAMDGPUInfo
    LLVMAMDGPUUtils
    # LLVM CodeGen
    LLVMAsmPrinter
    LLVMCodeGen
    LLVMCodeGenTypes
    LLVMGlobalISel
    LLVMMIRParser
    LLVMSelectionDAG
    LLVMTarget
    # LLVM MC
    LLVMMC
    LLVMMCDisassembler
    LLVMMCParser
    # LLVM Core + Support
    LLVMAggressiveInstCombine
    LLVMAnalysis
    LLVMAsmParser
    LLVMBinaryFormat
    LLVMBitReader
    LLVMBitstreamReader
    LLVMBitWriter
    LLVMCFGuard
    LLVMCore
    LLVMDebugInfoCodeView
    LLVMDebugInfoDWARF
    LLVMDebugInfoMSF
    LLVMDebugInfoPDB
    LLVMDemangle
    LLVMFrontendOffloading
    LLVMFrontendOpenMP
    LLVMInstCombine
    LLVMInstrumentation
    LLVMipo
    LLVMIRReader
    LLVMLinker
    LLVMObjCARCOpts
    LLVMObject
    LLVMPasses
    LLVMProfileData
    LLVMRemarks
    LLVMScalarOpts
    LLVMSupport
    LLVMSymbolize
    LLVMTargetParser
    LLVMTextAPI
    LLVMTransformUtils
    LLVMVectorize
)

# Per-file gtests live under amd/comgr/test-unit/ and follow the 3-tier
# llvm_gtest / find_package(GTest CONFIG) / LLVM-installed fallback discovery
# there. No standalone gtest target lives in this subdirectory.
