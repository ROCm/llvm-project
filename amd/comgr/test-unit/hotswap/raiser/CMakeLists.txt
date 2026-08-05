# Unit tests for the hotswap raiser (ISA-to-IR transpiler scaffolding).

# -- RaiserScaffoldingTests ---------------------------------------------------
#
# Pins the bare-bones raiser scaffolding contract (`hotswap/raiser/raiser.h`):
# empty text with a valid ISA produces a verifyModule-clean `llvm::Module` with
# a single `AMDGPU_KERNEL` `ret void` function, plus the failure path for a
# non-AMDGPU source ISA.

# standalone-init.cpp defines COMGR::ensureLLVMInitialized against the test's own
# LLVM (comgr-object-utils references it), rather than the copy inside amd_comgr.
add_executable(RaiserScaffoldingTests
  RaiserScaffoldingTest.cpp
  ${PROJECT_SOURCE_DIR}/test-shared/standalone-init.cpp)

# The raiser OBJECT library depends on its sibling hotswap libraries (decoder MC
# helpers, common error/metadata types, loader code-object utilities) and on the
# shared comgr metadata/symbol helpers (parseTargetIdentifier, getMetadataRoot,
# lookupSymbolByName) in comgr-object-utils. Link the same set as
# hotswap_transpile_cli; --gc-sections drops the object-utils code the raiser
# never reaches.
target_link_libraries(RaiserScaffoldingTests PRIVATE
  ${COMGR_GTEST_LIBS}
  hotswap::raiser
  hotswap::decoder
  hotswap::loader
  hotswap::common
  comgr-object-utils
  ${LLVM_PTHREAD_LIB})
if(NOT WIN32)
  target_link_options(RaiserScaffoldingTests PRIVATE -Wl,--gc-sections)
endif()

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

# -- RaiseContextTests --------------------------------------------------------

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

# -- KernargAbiLayoutTests ----------------------------------------------------
#
# Covers the source kernel-argument ABI reconstruction: the user-SGPR layout
# derived from the kernel descriptor, hidden-argument byte classification, and
# the IR synthesis of source hidden argument values.

add_executable(KernargAbiLayoutTests
  KernargAbiLayoutTest.cpp)

# The raiser OBJECT library references the shared comgr metadata/symbol helpers
# (parseTargetIdentifier and friends) in comgr-object-utils and the loader's
# code-object utilities; link the same set as RaiserScaffoldingTests and let
# --gc-sections drop what these layout tests never reach.
target_link_libraries(KernargAbiLayoutTests PRIVATE
  ${COMGR_GTEST_LIBS}
  hotswap::raiser
  hotswap::decoder
  hotswap::loader
  hotswap::common
  comgr-object-utils
  ${LLVM_PTHREAD_LIB})
if(NOT WIN32)
  target_link_options(KernargAbiLayoutTests PRIVATE -Wl,--gc-sections)
endif()

target_include_directories(KernargAbiLayoutTests PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/src
  ${COMGR_GTEST_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>)

comgr_configure_test_target(KernargAbiLayoutTests)
