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
  ${LLVM_INCLUDE_DIRS}
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
  ${LLVM_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/../src
  ${COMGR_GTEST_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../include>)

comgr_configure_test_target(HotswapMCTests)

# -- HotswapToolTests ---------------------------------------------------------
#
# Pure device-detection helpers for the HSA_TOOLS_LIB tool
# (hotswap/comgr-hotswap-tool-detect.h): gfx-target extraction and the A0
# rewrite gate. Header-only, so no comgr/LLVM sources are compiled in.

add_executable(HotswapToolTests
  HotswapToolTest.cpp)

llvm_map_components_to_libnames(COMGR_TEST_UNIT_TOOL_LIBS BinaryFormat Support)

target_link_libraries(HotswapToolTests PRIVATE
  ${COMGR_GTEST_LIBS}
  ${COMGR_TEST_UNIT_TOOL_LIBS}
  ${LLVM_PTHREAD_LIB})

# Disable gtest's LLVM-type printers; the tool helpers need no LLVM type support.
target_compile_definitions(HotswapToolTests PRIVATE GTEST_NO_LLVM_SUPPORT=true)

target_include_directories(HotswapToolTests PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/../src
  ${COMGR_GTEST_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../include>)

comgr_configure_test_target(HotswapToolTests)

# -- HotswapToolLoaderTests ---------------------------------------------------
#
# Drives the tool's HSA_TOOLS_LIB entry points (OnLoad/OnUnload) against a stub
# HSA API table. Compiles the tool TU directly, so it needs the HSA headers and
# amd_comgr; gated on COMGR_BUILD_HOTSWAP_TOOL like the tool itself.
if(COMGR_BUILD_HOTSWAP_TOOL AND HOTSWAP_TOOL_HSA_INC)
  add_executable(HotswapToolLoaderTests
    HotswapToolLoaderTest.cpp
    ../src/hotswap/comgr-hotswap-tool.cpp)

  llvm_map_components_to_libnames(COMGR_TEST_UNIT_LOADER_LIBS
    BinaryFormat Support)

  target_link_libraries(HotswapToolLoaderTests PRIVATE
    ${COMGR_GTEST_LIBS}
    amd_comgr
    ${COMGR_TEST_UNIT_LOADER_LIBS}
    ${LLVM_PTHREAD_LIB})

  target_compile_definitions(HotswapToolLoaderTests PRIVATE
    GTEST_NO_LLVM_SUPPORT=true)

  target_include_directories(HotswapToolLoaderTests PRIVATE
    ${LLVM_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    "${HOTSWAP_TOOL_HSA_INC}"
    ${COMGR_GTEST_INCLUDE_DIRS}
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../include>)

  comgr_configure_test_target(HotswapToolLoaderTests)
endif()

# -- DemangleSymbolNameTest ---------------------------------------------------

add_executable(DemangleSymbolNameTest
  DemangleSymbolNameTest.cpp)

target_link_libraries(DemangleSymbolNameTest PRIVATE
  ${COMGR_GTEST_LIBS}
  amd_comgr
  ${LLVM_PTHREAD_LIB})

# Gtest's LLVM support brings in some printers for LLVM types (StringRef, etc.).
# When building in-tree, the included headers bring in some code that depend on LLVM's abi-breaking checks.
# This adds a dependency with LLVM's Support library.
# We could link against Support, but since we're not using these printers we can just disable them.
target_compile_definitions(DemangleSymbolNameTest PRIVATE GTEST_NO_LLVM_SUPPORT=true)

target_include_directories(DemangleSymbolNameTest PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/../src
  ${COMGR_GTEST_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../include>)

comgr_configure_test_target(DemangleSymbolNameTest)

# amd_comgr is a shared lib; the test-unit target runs the exe directly
# without the PATH setup the lit suite uses, so place the DLL next to it.
if(WIN32 AND COMGR_BUILD_SHARED_LIBS)
  add_custom_command(TARGET DemangleSymbolNameTest POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      $<TARGET_FILE:amd_comgr> $<TARGET_FILE_DIR:DemangleSymbolNameTest>)
endif()

# Register every test binary with the test-unit / check-comgr plumbing.
add_custom_target(test-unit
  COMMAND $<TARGET_FILE:HotswapElfTests>
  COMMAND $<TARGET_FILE:HotswapMCTests>
  COMMAND $<TARGET_FILE:HotswapToolTests>)
add_dependencies(test-unit HotswapElfTests HotswapMCTests HotswapToolTests)
add_dependencies(check-comgr test-unit)
