# Unit tests for the hotswap rewriter (byte-level B0-to-A0 rewrite path). The
# test binaries compile the rewriter TUs directly (rather than linking
# amd_comgr) so they can exercise internal, non-exported entry points.
#
# ${PROJECT_SOURCE_DIR}/src and ${PROJECT_BINARY_DIR}/include resolve to the
# comgr source and generated-header roots regardless of this directory's depth.

# -- HotswapElfTests ----------------------------------------------------------
#
# Lightweight tests for the ELF layer in hotswap/rewriter/elf.cpp. Needs only
# llvm::object for ELF parsing; no MC state constructed.

add_executable(HotswapElfTests
  HotswapElfTest.cpp
  ${PROJECT_SOURCE_DIR}/src/hotswap/rewriter/elf.cpp
  # COMGR::env::shouldEmitVerboseLogs() is referenced by the inline
  # COMGR::hotswap::log() helper in hotswap/rewriter/internal.h; link its
  # definition so non-inlined builds (e.g. -O0 / ASan) resolve.
  ${PROJECT_SOURCE_DIR}/src/comgr-env.cpp)

llvm_map_components_to_libnames(COMGR_TEST_UNIT_ELF_LIBS
  BinaryFormat Object Support TargetParser)

target_link_libraries(HotswapElfTests PRIVATE
  ${COMGR_GTEST_LIBS}
  ${COMGR_TEST_UNIT_ELF_LIBS}
  ${LLVM_PTHREAD_LIB})

target_include_directories(HotswapElfTests PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/src
  ${COMGR_GTEST_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>)

comgr_configure_test_target(HotswapElfTests)

# -- HotswapMCTests -----------------------------------------------------------
#
# MC-layer tests need the AMDGPU backend (TargetRegistry, instruction
# definitions, disassembler, asm parser, and code emitter) so the test
# binary can stand up a real LLVMState for gfx1250 and exercise the
# assemble / decode / encode primitives.

add_executable(HotswapMCTests
  HotswapOccupancyTest.cpp
  HotswapMCTest.cpp
  ${PROJECT_SOURCE_DIR}/src/hotswap/rewriter/b0a0.cpp
  ${PROJECT_SOURCE_DIR}/src/hotswap/rewriter/displacement.cpp
  ${PROJECT_SOURCE_DIR}/src/hotswap/rewriter/entry-trampoline.cpp
  ${PROJECT_SOURCE_DIR}/src/hotswap/rewriter/entry-trampoline-fast.cpp
  ${PROJECT_SOURCE_DIR}/src/hotswap/rewriter/elf.cpp
  ${PROJECT_SOURCE_DIR}/src/hotswap/rewriter/llvm.cpp
  ${PROJECT_SOURCE_DIR}/src/hotswap/rewriter/occupancy.cpp
  ${PROJECT_SOURCE_DIR}/src/hotswap/rewriter/patch-f32-to-e5m3.cpp
  ${PROJECT_SOURCE_DIR}/src/hotswap/rewriter/patch-inplace.cpp
  ${PROJECT_SOURCE_DIR}/src/hotswap/rewriter/patch-trampoline.cpp
  ${PROJECT_SOURCE_DIR}/src/hotswap/rewriter/patch-vop3px2-src2.cpp
  ${PROJECT_SOURCE_DIR}/src/hotswap/rewriter/patch-wmma-hazard.cpp
  ${PROJECT_SOURCE_DIR}/src/hotswap/rewriter/patch-wmma-scale16.cpp
  ${PROJECT_SOURCE_DIR}/src/hotswap/rewriter/patch-wmma-split.cpp
  ${PROJECT_SOURCE_DIR}/src/comgr-env.cpp
  # HotswapProfile bodies are out-of-line in hotswap/rewriter/profile.cpp; its
  # flush() merges into Comgr TimeStatistics (time-stat.cpp / mergeStats()).
  ${PROJECT_SOURCE_DIR}/src/hotswap/rewriter/profile.cpp
  ${PROJECT_SOURCE_DIR}/src/time-stat/time-stat.cpp)

llvm_map_components_to_libnames(COMGR_TEST_UNIT_MC_LIBS
  BinaryFormat
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
  comgr-test-llvm-init
  ${COMGR_TEST_UNIT_MC_LIBS}
  ${LLVM_PTHREAD_LIB})

target_include_directories(HotswapMCTests PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/src
  ${COMGR_GTEST_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>)

comgr_configure_test_target(HotswapMCTests)

# -- HotswapProfileTests ------------------------------------------------------
#
# The profiler is declared in hotswap/rewriter/internal.h with out-of-line
# bodies in hotswap/rewriter/profile.cpp; it is gated only at run time by
# AMD_COMGR_TIME_STATISTICS. This dedicated target exercises the HotswapProfile
# session logic directly (accumulation and flush) for both disabled and enabled
# sessions.

add_executable(HotswapProfileTests
  HotswapProfileTest.cpp
  # COMGR::env::* is referenced by the inline COMGR::hotswap::log() helper and by
  # the profiler gating, both pulled in via hotswap/rewriter/internal.h.
  ${PROJECT_SOURCE_DIR}/src/comgr-env.cpp
  # HotswapProfile bodies are out-of-line in hotswap/rewriter/profile.cpp; its
  # flush() merges into Comgr TimeStatistics (time-stat.cpp / mergeStats()).
  ${PROJECT_SOURCE_DIR}/src/hotswap/rewriter/profile.cpp
  ${PROJECT_SOURCE_DIR}/src/time-stat/time-stat.cpp)

llvm_map_components_to_libnames(COMGR_TEST_UNIT_PROFILE_LIBS
  BinaryFormat Object Support TargetParser)

target_link_libraries(HotswapProfileTests PRIVATE
  ${COMGR_GTEST_LIBS}
  ${COMGR_TEST_UNIT_PROFILE_LIBS}
  ${LLVM_PTHREAD_LIB})

target_include_directories(HotswapProfileTests PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/src
  ${COMGR_GTEST_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>)

comgr_configure_test_target(HotswapProfileTests)
