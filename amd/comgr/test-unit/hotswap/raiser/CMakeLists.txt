# Unit tests for the hotswap raiser (ISA-to-IR transpiler scaffolding).

# -- RaiserScaffoldingTests ---------------------------------------------------
#
# Pins the bare-bones raiser scaffolding contract (`hotswap/raiser/raiser.h`):
# empty input produces a verifyModule-clean `llvm::Module` with a single
# `AMDGPU_KERNEL` `ret void` function, plus the failure paths for missing
# kernel descriptor / malformed ISA.

add_executable(RaiserScaffoldingTests
  RaiserScaffoldingTest.cpp)

# The wave-projection objects require decoder symbols.
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

# The wave-projection objects require decoder symbols.
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
  ${COMGR_HOTSWAP_TEST_LLVM_LIBS}
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
# the allocas to SSA and constant-folds to confirm the value survives.

add_executable(RegFileTests
  RegFileTest.cpp)

target_link_libraries(RegFileTests PRIVATE
  ${COMGR_GTEST_LIBS}
  hotswap::raiser
  hotswap::decoder
  hotswap::common
  ${COMGR_HOTSWAP_TEST_LLVM_LIBS}
  ${LLVM_PTHREAD_LIB})

target_include_directories(RegFileTests PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/src
  ${COMGR_GTEST_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>)

comgr_configure_test_target(RegFileTests)

# -- RegisterStateTests -------------------------------------------------------
#
# Covers the source architectural registers as the raiser sees them: operand
# decode and reads, EXEC-predicated writes, and the per-SGPR shadow storage
# that carries wave masks and source-wave values across blocks.

add_executable(RegisterStateTests
  RegisterStateTest.cpp)

target_link_libraries(RegisterStateTests PRIVATE
  ${COMGR_GTEST_LIBS}
  hotswap::raiser
  hotswap::decoder
  hotswap::common
  ${COMGR_HOTSWAP_TEST_LLVM_LIBS}
  ${LLVM_PTHREAD_LIB})

target_include_directories(RegisterStateTests PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/src
  ${COMGR_GTEST_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>)

comgr_configure_test_target(RegisterStateTests)

# -- RaiseContextTests --------------------------------------------------------
#
# Covers the per-kernel context itself: source-offset block lookup.

add_executable(RaiseContextTests
  RaiseContextTest.cpp)

target_link_libraries(RaiseContextTests PRIVATE
  ${COMGR_GTEST_LIBS}
  hotswap::raiser
  hotswap::decoder
  hotswap::common
  ${COMGR_HOTSWAP_TEST_LLVM_LIBS}
  ${LLVM_PTHREAD_LIB})

target_include_directories(RaiseContextTests PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/src
  ${COMGR_GTEST_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>)

comgr_configure_test_target(RaiseContextTests)

# -- OpResolverTests ----------------------------------------------------------
#
# Covers the per-instruction operand accessor: the failures it propagates out of
# the register reads it resolves through.

add_executable(OpResolverTests
  OpResolverTest.cpp)

target_link_libraries(OpResolverTests PRIVATE
  ${COMGR_GTEST_LIBS}
  hotswap::raiser
  hotswap::decoder
  hotswap::common
  ${COMGR_HOTSWAP_TEST_LLVM_LIBS}
  ${LLVM_PTHREAD_LIB})

target_include_directories(OpResolverTests PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/src
  ${COMGR_GTEST_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>)

comgr_configure_test_target(OpResolverTests)

# -- UserSgprLayoutTests ------------------------------------------------------
#
# Covers the user-SGPR layout the raiser derives from the kernel descriptor:
# the canonical entry order, the kernarg preload decode, and the descriptors it
# refuses.

add_executable(UserSgprLayoutTests
  UserSgprLayoutTest.cpp)

target_link_libraries(UserSgprLayoutTests PRIVATE
  ${COMGR_GTEST_LIBS}
  hotswap::raiser
  hotswap::decoder
  hotswap::common
  ${COMGR_HOTSWAP_TEST_LLVM_LIBS}
  ${LLVM_PTHREAD_LIB})

target_include_directories(UserSgprLayoutTests PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/src
  ${COMGR_GTEST_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>)

comgr_configure_test_target(UserSgprLayoutTests)
