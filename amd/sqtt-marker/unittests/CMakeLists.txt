# ===- CMakeLists.txt - AMD SQTT marker unit tests ------------------------=== #
#
# Part of AMD SQTT Marker, under the MIT License. See
# amd/sqtt-marker/LICENSE.txt for license information.
# SPDX-License-Identifier: MIT
#
# ===----------------------------------------------------------------------=== #

set(LLVM_LINK_COMPONENTS
  Analysis
  Core
  Passes
  Support
  TargetParser
  TransformUtils
)

set(SQTT_MARKER_UNIT_SOURCES
  SQTTMarkerTest.cpp
)

if(COMMAND add_llvm_unittest AND TARGET UnitTests)
  add_llvm_unittest(SQTTMarkerTests ${SQTT_MARKER_UNIT_SOURCES})
  set_output_directory(SQTTMarkerTests
    BINARY_DIR "${LLVM_BINARY_DIR}/unittests/SQTTMarker"
    LIBRARY_DIR "${LLVM_BINARY_DIR}/unittests/SQTTMarker")
else()
  find_package(GTest REQUIRED)
  add_executable(SQTTMarkerTests ${SQTT_MARKER_UNIT_SOURCES})
  llvm_map_components_to_libnames(SQTT_MARKER_TEST_LLVM_LIBS
    ${LLVM_LINK_COMPONENTS})
  target_link_libraries(SQTTMarkerTests PRIVATE
    GTest::gtest_main
    ${SQTT_MARKER_TEST_LLVM_LIBS}
  )
  include(GoogleTest)
  gtest_discover_tests(SQTTMarkerTests)
endif()

target_link_libraries(SQTTMarkerTests PRIVATE SQTTMarkerObjects)

target_include_directories(SQTTMarkerTests PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../lib"
  ${LLVM_INCLUDE_DIRS}
)
if(SQTT_MARKER_STANDALONE)
  target_compile_options(SQTTMarkerTests PRIVATE
    ${SQTT_MARKER_LLVM_DEFINITIONS})
endif()
target_compile_features(SQTTMarkerTests PRIVATE cxx_std_17)
if(TARGET intrinsics_gen)
  add_dependencies(SQTTMarkerTests intrinsics_gen)
endif()

if(COMMAND llvm_update_compile_flags)
  llvm_update_compile_flags(SQTTMarkerTests)
endif()
