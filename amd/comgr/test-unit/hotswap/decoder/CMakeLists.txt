# Unit tests for the hotswap decoder (AMDGPU MC stack, canonical-op identity,
# ISA profile, and the decoded-instruction model).

# -- DecoderTests -------------------------------------------------------------

add_executable(DecoderTests
  DecoderTest.cpp)

target_link_libraries(DecoderTests PRIVATE
  ${COMGR_GTEST_LIBS}
  hotswap::decoder
  hotswap::common
  ${COMGR_HOTSWAP_TEST_LLVM_LIBS}
  ${LLVM_PTHREAD_LIB})

target_include_directories(DecoderTests PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/src
  ${COMGR_GTEST_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>)

comgr_configure_test_target(DecoderTests)
