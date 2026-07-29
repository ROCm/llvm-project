cmake_minimum_required(VERSION 3.20)

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

# --- Library ------------------------------------------------------------------
# OBJECT library so its translation units land directly in amd_comgr.so when
# `target_link_libraries(amd_comgr PRIVATE hotswap::rewriter)` runs in the
# parent CMakeLists.txt. This is the byte-level rewrite path backing the
# always-on amd_comgr_hotswap_rewrite API.
#
# Unlike the raiser, the rewriter consumes comgr's private header (comgr.h) and
# the configure-time-generated amd_comgr.h, so it is not standalone-buildable
# and deliberately does not declare its own project() — it relies on the
# enclosing amd_comgr project scope (PROJECT_SOURCE_DIR / PROJECT_BINARY_DIR).
add_library(hotswap-rewriter OBJECT
  rewriter.cpp
  b0a0.cpp
  displacement.cpp
  entry-trampoline.cpp
  entry-trampoline-fast.cpp
  occupancy.cpp
  patch-trampoline.cpp
  elf.cpp
  llvm.cpp
  patch-f32-to-e5m3.cpp
  patch-inplace.cpp
  patch-vop3px2-src2.cpp
  patch-wmma-hazard.cpp
  patch-wmma-scale16.cpp
  patch-wmma-split.cpp
  profile.cpp
)

if(NOT TARGET hotswap::rewriter)
  add_library(hotswap::rewriter ALIAS hotswap-rewriter)
endif()

# Match LLVM's compile flags (no-rtti, no-exceptions); mixing RTTI-on TUs with
# RTTI-off LLVM libs produces undefined-reference errors for `typeinfo` symbols
# on any class with a virtual method.
llvm_update_compile_flags(hotswap-rewriter)

set_target_properties(hotswap-rewriter PROPERTIES POSITION_INDEPENDENT_CODE ON)

# These TUs land directly in amd_comgr and define exported amd_comgr_hotswap_*
# API functions, so they must be built with AMD_COMGR_EXPORT for AMD_COMGR_API
# to resolve to dllexport (not dllimport) on Windows.
target_compile_definitions(hotswap-rewriter PRIVATE AMD_COMGR_EXPORT)

# Include roots: LLVM headers, comgr's src/ (comgr.h, and the src/-relative
# hotswap/common/*.h shared headers), and the generated amd_comgr.h under the
# comgr build tree.
target_include_directories(hotswap-rewriter PUBLIC
  ${LLVM_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
)

# The rewriter is a pure MC-layer byte rewriter: disassemble (MCDisassembler),
# assemble/encode (MC, MCParser, MCCodeEmitter), read ELF (Object,
# BinaryFormat), and query AMDGPU MC descriptions and named operands
# (AMDGPU{Desc,Info,Utils,AsmParser,Disassembler}). It uses no LLVM IR or
# codegen, so Core and AMDGPUCodeGen are intentionally omitted; Core is still
# pulled in transitively where needed.
llvm_map_components_to_libnames(hotswap_rewriter_llvm_libs
  BinaryFormat
  MC
  MCDisassembler
  MCParser
  Object
  Support
  TargetParser
  AMDGPUAsmParser
  AMDGPUDesc
  AMDGPUDisassembler
  AMDGPUInfo
  AMDGPUUtils)

target_link_libraries(hotswap-rewriter PUBLIC ${hotswap_rewriter_llvm_libs})
