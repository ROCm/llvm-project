# Unit tests for the hotswap raiser (ISA-to-IR transpiler scaffolding).

# -- RaiserScaffoldingTests ---------------------------------------------------
#
# Pins the bare-bones raiser scaffolding contract (`hotswap/raiser/raiser.h`):
# empty input produces a verifyModule-clean `llvm::Module` with a single
# `AMDGPU_KERNEL` `ret void` function, plus the failure paths for missing
# kernel descriptor / malformed ISA.

add_executable(RaiserScaffoldingTests
  RaiserScaffoldingTest.cpp)

# hotswap::raiser keeps its LLVM dependencies PRIVATE, so name them here. It also
# carries the wave-projection objects, which call into hotswap::decoder; link it
# too and inherit its curated LLVM component set.
llvm_map_components_to_libnames(COMGR_TEST_UNIT_RAISER_LIBS
  Core Support TargetParser)

target_link_libraries(RaiserScaffoldingTests PRIVATE
  ${COMGR_GTEST_LIBS}
  hotswap::raiser
  hotswap::decoder
  hotswap::common
  ${COMGR_TEST_UNIT_RAISER_LIBS}
  ${LLVM_PTHREAD_LIB})

target_include_directories(RaiserScaffoldingTests PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/src
  ${COMGR_GTEST_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>)

comgr_configure_test_target(RaiserScaffoldingTests)

# -- WaveProjectionTests ------------------------------------------------------
#
# Pins the per-projection contract on WaveProjection and its subclasses (EXEC
# storage width, source waves per target wave, doubled-dispatch factor, the
# exec/mbcnt predicates) and the wrapAsWWMValue emission behaviour.

add_executable(WaveProjectionTests
  WaveProjectionTest.cpp)

# The wave projection lives in hotswap::raiser and calls into hotswap::decoder
# (ISAProfile and the AMDGPU register queries); link both. hotswap::decoder
# carries the curated LLVM component set the projection needs.
target_link_libraries(WaveProjectionTests PRIVATE
  ${COMGR_GTEST_LIBS}
  hotswap::raiser
  hotswap::decoder
  hotswap::common
  ${LLVM_PTHREAD_LIB})

target_include_directories(WaveProjectionTests PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/src
  ${COMGR_GTEST_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>)

comgr_configure_test_target(WaveProjectionTests)
