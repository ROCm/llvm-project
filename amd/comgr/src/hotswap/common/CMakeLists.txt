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
# Path-agnostic pieces shared across the transpile path. Most of common/ is
# header-only (kernel-meta.h); hotswap-error carries one out-of-line symbol
# (HotswapError::ID), so it is compiled into a small OBJECT library both the
# loader and the decoder consume, keeping the error type in one place rather
# than coupling those directories to each other.
add_library(hotswap-common OBJECT
  hotswap-error.cpp
)

if(NOT TARGET hotswap::common)
  add_library(hotswap::common ALIAS hotswap-common)
endif()

llvm_update_compile_flags(hotswap-common)
# llvm_update_compile_flags does not reliably carry LLVM's -fno-rtti to a
# standalone find_package(LLVM) consumer, and HotswapError derives from
# llvm::ErrorInfo. Mirror LLVM_ENABLE_RTTI explicitly so we do not emit typeinfo
# an RTTI-off LLVM cannot resolve at link time.
if(NOT LLVM_ENABLE_RTTI)
  if(MSVC)
    target_compile_options(hotswap-common PRIVATE /GR-)
  else()
    target_compile_options(hotswap-common PRIVATE -fno-rtti)
  endif()
endif()
set_target_properties(hotswap-common PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS No
  POSITION_INDEPENDENT_CODE ON)

# Match LLVM's RTTI setting explicitly. llvm_update_compile_flags only adds
# -fno-rtti when it detects a GCC-compatible compiler, which is not set in this
# subproject scope, so against an RTTI-disabled libLLVM the ErrorInfo<HotswapError>
# vtable would emit C++ typeinfo referencing llvm::ErrorInfoBase, a symbol that
# library never defines.
if(NOT LLVM_ENABLE_RTTI)
  if(MSVC)
    target_compile_options(hotswap-common PRIVATE /GR-)
  else()
    target_compile_options(hotswap-common PRIVATE -fno-rtti)
  endif()
endif()

# Public include root (src/) so consumers can `#include "hotswap/common/...".
target_include_directories(hotswap-common PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
)

# Follow COMGR's selected LLVM linkage mode: the single LLVM shared library in a
# dylib build, otherwise static LLVMSupport, so amd_comgr never links Support
# both statically and through libLLVM.
if(LLVM_LINK_LLVM_DYLIB)
  set(hotswap_common_llvm_libs LLVM)
else()
  set(hotswap_common_llvm_libs LLVMSupport)
endif()

# PRIVATE keeps these out of the link interface, but CMake still records them as
# $<LINK_ONLY:> for an OBJECT library; clear the interface so consumers get the
# objects only (see ../CMakeLists.txt).
target_link_libraries(hotswap-common
  PRIVATE
    ${hotswap_common_llvm_libs}
)
set_target_properties(hotswap-common PROPERTIES INTERFACE_LINK_LIBRARIES "")
