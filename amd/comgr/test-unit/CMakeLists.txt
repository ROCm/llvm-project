# GTest discovery for Comgr unit tests. Three sources, in priority order:
#
# 1. In-tree llvm_gtest target (LLVM monorepo build with utils/unittest/).
# 2. Vanilla GTest CMake package (find_package(GTest CONFIG)). Used by
#    superproject builds (e.g. TheRock) that supply googletest via the
#    standard GTest:: imported targets. This avoids forcing the LLVM
#    build to enable LLVM_INSTALL_GTEST just to test Comgr.
# 3. LLVM-installed gtest (-DLLVM_INSTALL_GTEST=ON on the LLVM build).
#    Legacy fallback for installs that ship llvm_gtest alongside LLVM.
#
# If none is available, skip the test-unit suite with a warning rather
# than failing configure.
if(TARGET llvm_gtest)
  set(COMGR_GTEST_LIBS llvm_gtest_main llvm_gtest)
  set(COMGR_GTEST_INCLUDE_DIRS "")
else()
  find_package(GTest CONFIG QUIET)
  if(GTest_FOUND)
    set(COMGR_GTEST_LIBS GTest::gtest_main GTest::gtest)
    set(COMGR_GTEST_INCLUDE_DIRS "")
  else()
    find_library(COMGR_LLVM_GTEST_LIB llvm_gtest
      PATHS ${LLVM_LIBRARY_DIRS} NO_DEFAULT_PATH)
    find_library(COMGR_LLVM_GTEST_MAIN_LIB llvm_gtest_main
      PATHS ${LLVM_LIBRARY_DIRS} NO_DEFAULT_PATH)
    find_path(COMGR_LLVM_GTEST_INCLUDE_DIR gtest/gtest.h
      PATHS ${LLVM_INCLUDE_DIRS} PATH_SUFFIXES llvm-gtest NO_DEFAULT_PATH)
    if(COMGR_LLVM_GTEST_LIB AND COMGR_LLVM_GTEST_MAIN_LIB
        AND COMGR_LLVM_GTEST_INCLUDE_DIR)
      set(COMGR_GTEST_LIBS
        ${COMGR_LLVM_GTEST_MAIN_LIB} ${COMGR_LLVM_GTEST_LIB})
      set(COMGR_GTEST_INCLUDE_DIRS ${COMGR_LLVM_GTEST_INCLUDE_DIR})
    else()
      message(WARNING
        "Comgr test-unit skipped: no llvm_gtest target, no GTest "
        "CMake package, and no LLVM-installed gtest found at "
        "${LLVM_INCLUDE_DIRS}/llvm-gtest. Provide one of: in-tree LLVM "
        "with utils/unittest/, find_package(GTest), or "
        "-DLLVM_INSTALL_GTEST=ON on the LLVM build.")
      return()
    endif()
  endif()
endif()

# Pin C++17 and mirror LLVM's RTTI setting (avoids type_info link errors
# against libLLVM*).
function(comgr_configure_test_target target)
  set_target_properties(${target} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED Yes
    CXX_EXTENSIONS No)
  if(NOT LLVM_ENABLE_RTTI)
    if(MSVC)
      target_compile_options(${target} PRIVATE /GR-)
    else()
      target_compile_options(${target} PRIVATE -fno-rtti)
    endif()
  endif()
endfunction()

# Apply LLVM header paths to the hotswap unit-test binaries.
#
# Some superproject / wrapper flows finish assigning LLVM_INCLUDE_DIRS only
# after this directory's CMakeLists.txt is parsed (or print it from a
# different phase than the expansion of ${LLVM_INCLUDE_DIRS} in
# target_include_directories). That yields builds where compile lines miss
# -I.../llvm/include and fail on llvm/ADT/StringRef.h.
#
# CMake 3.19+: defer to the end of processing amd/comgr/CMakeLists.txt so
# LLVM_INCLUDE_DIRS is read when the deferred call runs, not when this file
# is first parsed.
function(_comgr_test_unit_apply_llvm_include_dirs)
  if(NOT LLVM_INCLUDE_DIRS)
    message(FATAL_ERROR
      "Comgr test-unit: LLVM_INCLUDE_DIRS is empty when applying include "
      "directories to Hotswap* tests. Fix the parent amd/comgr/CMakeLists.txt "
      "so find_package(LLVM ...) (or equivalent) runs before add_subdirectory"
      "(test-unit), or ensure the LLVM package exports LLVM_INCLUDE_DIRS.")
  endif()
  target_include_directories(HotswapElfTests PRIVATE ${LLVM_INCLUDE_DIRS})
  target_include_directories(HotswapMCTests PRIVATE ${LLVM_INCLUDE_DIRS})
endfunction()

# -- HotswapElfTests ----------------------------------------------------------
#
# Lightweight tests for the ELF layer in comgr-hotswap-elf.cpp. Needs only
# llvm::object for ELF parsing; no MC state constructed.

add_executable(HotswapElfTests
  HotswapElfTest.cpp
  ../src/comgr-hotswap-elf.cpp
  # COMGR::env::shouldEmitVerboseLogs() is referenced by the inline
  # COMGR::hotswap::log() helper in comgr-hotswap-internal.h; link its
  # definition so non-inlined builds (e.g. -O0 / ASan) resolve.
  ../src/comgr-env.cpp)

llvm_map_components_to_libnames(COMGR_TEST_UNIT_ELF_LIBS
  Object Support TargetParser)

target_link_libraries(HotswapElfTests PRIVATE
  ${COMGR_GTEST_LIBS}
  ${COMGR_TEST_UNIT_ELF_LIBS}
  ${LLVM_PTHREAD_LIB})

target_include_directories(HotswapElfTests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../src
  ${COMGR_GTEST_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../include>)

comgr_configure_test_target(HotswapElfTests)

# -- HotswapMCTests -----------------------------------------------------------
#
# MC-layer tests need the AMDGPU backend (TargetRegistry, instruction
# definitions, disassembler, asm parser, and code emitter) so the test
# binary can stand up a real LLVMState for gfx1250 and exercise the
# assemble / decode / encode primitives.

add_executable(HotswapMCTests
  HotswapMCTest.cpp
  ../src/comgr-hotswap-elf.cpp
  ../src/comgr-hotswap-llvm.cpp
  ../src/comgr-hotswap-patch-vop3px2-src2.cpp
  ../src/comgr-hotswap-patch-wmma-hazard.cpp
  ../src/comgr-env.cpp)

llvm_map_components_to_libnames(COMGR_TEST_UNIT_MC_LIBS
  MC
  MCDisassembler
  MCParser
  Object
  Support
  TargetParser
  AMDGPUAsmParser
  AMDGPUCodeGen
  AMDGPUDesc
  AMDGPUDisassembler
  AMDGPUInfo)

target_link_libraries(HotswapMCTests PRIVATE
  ${COMGR_GTEST_LIBS}
  ${COMGR_TEST_UNIT_MC_LIBS}
  ${LLVM_PTHREAD_LIB})

target_include_directories(HotswapMCTests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../src
  ${COMGR_GTEST_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../include>)

comgr_configure_test_target(HotswapMCTests)

# LLVM includes: deferred when possible (see _comgr_test_unit_apply_llvm_include_dirs).
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.19")
  cmake_language(DEFER DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
    CALL _comgr_test_unit_apply_llvm_include_dirs)
else()
  if(NOT LLVM_INCLUDE_DIRS)
    message(FATAL_ERROR
      "Comgr test-unit requires LLVM_INCLUDE_DIRS (CMake >= 3.19 supports "
      "deferring this; upgrade CMake or fix parent configure order).")
  endif()
  _comgr_test_unit_apply_llvm_include_dirs()
endif()

# Register both test binaries with the test-unit / check-comgr plumbing.
add_custom_target(test-unit
  COMMAND $<TARGET_FILE:HotswapElfTests>
  COMMAND $<TARGET_FILE:HotswapMCTests>)
add_dependencies(test-unit HotswapElfTests HotswapMCTests)
add_dependencies(check-comgr test-unit)

