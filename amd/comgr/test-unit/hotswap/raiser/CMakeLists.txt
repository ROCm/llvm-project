# Unit tests for the hotswap raiser (ISA-to-IR transpiler scaffolding).

# -- RaiserScaffoldingTests ---------------------------------------------------
#
# Pins the bare-bones raiser scaffolding contract (`hotswap/raiser/raiser.h`):
# empty input produces a verifyModule-clean `llvm::Module` with a single
# `AMDGPU_KERNEL` `ret void` function, plus the failure paths for missing
# kernel descriptor / malformed ISA.

add_executable(RaiserScaffoldingTests
  RaiserScaffoldingTest.cpp)

# hotswap::raiser carries the wave-projection objects, which call into
# hotswap::decoder; link both. Both keep their LLVM dependencies PRIVATE, so
# name the LLVM components here (COMGR_HOTSWAP_TEST_LLVM_LIBS, ../CMakeLists.txt).
target_link_libraries(RaiserScaffoldingTests PRIVATE
  ${COMGR_GTEST_LIBS}
  hotswap::raiser
  hotswap::decoder
  hotswap::common
  ${COMGR_HOTSWAP_TEST_LLVM_LIBS}
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
# (ISAProfile and the AMDGPU register queries); link both, plus the LLVM
# components the OBJECT libraries keep PRIVATE (COMGR_HOTSWAP_TEST_LLVM_LIBS,
# ../CMakeLists.txt).
target_link_libraries(WaveProjectionTests PRIVATE
  ${COMGR_GTEST_LIBS}
  hotswap::raiser
  hotswap::decoder
  hotswap::common
  ${COMGR_HOTSWAP_TEST_LLVM_LIBS}
  ${LLVM_PTHREAD_LIB})

target_include_directories(WaveProjectionTests PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/src
  ${COMGR_GTEST_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>)

comgr_configure_test_target(WaveProjectionTests)

# -- RaiseFailureTests --------------------------------------------------------
#
# Pins the RaiseFailure value type: the shape factories, the optional
# format/offset accessors, and the stable reasonString / log diagnostic text.

add_executable(RaiseFailureTests
  RaiseFailureTest.cpp)

# RaiseFailure lives in hotswap::raiser. That OBJECT library also carries the
# reg file and wave projection, which reach into hotswap::decoder, so link both.
target_link_libraries(RaiseFailureTests PRIVATE
  ${COMGR_GTEST_LIBS}
  hotswap::raiser
  hotswap::decoder
  hotswap::common
  ${LLVM_PTHREAD_LIB})

target_include_directories(RaiseFailureTests PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/src
  ${COMGR_GTEST_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>)

comgr_configure_test_target(RaiseFailureTests)

# -- RegFileTests -------------------------------------------------------------
#
# Round-trips values through AllocaRegFile's ParsedReg dispatch, then promotes
# the allocas to SSA and constant-folds to confirm the value survives. Needs
# TransformUtils (PromoteMemToReg) and Analysis (constant folding) on top of the
# component set hotswap::decoder carries.

add_executable(RegFileTests
  RegFileTest.cpp)

# Dylib build: link libLLVM, not static archives (see RaiserScaffoldingTests).
if(LLVM_LINK_LLVM_DYLIB)
  set(COMGR_TEST_UNIT_REGFILE_LIBS LLVM)
else()
  llvm_map_components_to_libnames(COMGR_TEST_UNIT_REGFILE_LIBS
    Analysis TransformUtils)
endif()

target_link_libraries(RegFileTests PRIVATE
  ${COMGR_GTEST_LIBS}
  hotswap::raiser
  hotswap::decoder
  hotswap::common
  ${COMGR_TEST_UNIT_REGFILE_LIBS}
  ${LLVM_PTHREAD_LIB})

target_include_directories(RegFileTests PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/src
  ${COMGR_GTEST_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>)

comgr_configure_test_target(RegFileTests)

# -- KernargAbiLayoutTests ----------------------------------------------------
#
# Covers the source kernel-argument ABI reconstruction: the user-SGPR layout
# derived from the kernel descriptor, hidden-argument byte classification, and
# the IR synthesis of source hidden argument values.

add_executable(KernargAbiLayoutTests
  KernargAbiLayoutTest.cpp)

target_link_libraries(KernargAbiLayoutTests PRIVATE
  ${COMGR_GTEST_LIBS}
  hotswap::raiser
  hotswap::decoder
  hotswap::common
  ${LLVM_PTHREAD_LIB})

target_include_directories(KernargAbiLayoutTests PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/src
  ${COMGR_GTEST_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>)

comgr_configure_test_target(KernargAbiLayoutTests)
