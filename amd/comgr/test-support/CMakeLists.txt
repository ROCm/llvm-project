# Objects shared by the test binaries under test-lit and test-unit. Those link
# the hotswap OBJECT libraries directly rather than amd_comgr, so they have to
# supply the pieces of Comgr those objects call into.
#
# An OBJECT library rather than a source listed per target: the consumers span
# both test directories, so this is the one place that compiles the registration
# once and hands the same object to each of them.
add_library(comgr-test-llvm-init OBJECT llvm-init.cpp)

set_target_properties(comgr-test-llvm-init PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS No
  POSITION_INDEPENDENT_CODE ON)

target_include_directories(comgr-test-llvm-init PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/../src
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../include>)

# LLVM is built without RTTI; match it so nothing here is emitted against the
# -fno-rtti Support library.
if(NOT LLVM_ENABLE_RTTI)
  if(MSVC)
    target_compile_options(comgr-test-llvm-init PRIVATE /GR-)
  else()
    target_compile_options(comgr-test-llvm-init PRIVATE -fno-rtti)
  endif()
endif()
