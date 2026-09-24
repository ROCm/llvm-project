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

if(TARGET llvm_gtest)
    # Build as an external project
    set(GTEST_LIBS llvm_gtest_main llvm_gtest)
    set(GTEST_INCLUDE_DIR "")
else()
    # Build as a separate pass
    find_library(LLVM_GTEST_MAIN_LIB llvm_gtest PATHS ${LLVM_LIBRARY_DIRS} NO_DEFAULT_PATH)
    find_library(LLVM_GTEST_LIB llvm_gtest PATHS ${LLVM_LIBRARY_DIRS} NO_DEFAULT_PATH)
    find_path(LLVM_GTEST_INCLUDE_DIR gtest/gtest.h PATHS ${LLVM_INCLUDE_DIRS} PATH_SUFFIXES llvm-gtest NO_DEFAULT_PATH)
    if(LLVM_GTEST_LIB AND LLVM_GTEST_MAIN_LIB AND LLVM_GTEST_INCLUDE_DIR)
        set(GTEST_LIBS ${LLVM_GTEST_MAIN_LIB} ${LLVM_GTEST_LIB})
        set(GTEST_INCLUDE_DIR "${LLVM_GTEST_INCLUDE_DIR}")
    else()
        find_package(GTest REQUIRED)
        set(GTEST_LIBS GTest::gtest_main GTest::gtest)
        set(GTEST_INCLUDE_DIR "")
    endif()
endif()

add_executable(SQTTMarkerTests ${SQTT_MARKER_UNIT_SOURCES})
llvm_map_components_to_libnames(SQTT_MARKER_TEST_LLVM_LIBS
  ${LLVM_LINK_COMPONENTS})
target_link_libraries(SQTTMarkerTests PRIVATE
    ${GTEST_LIBS}
    ${SQTT_MARKER_TEST_LLVM_LIBS}
    SQTTMarkerObjects
)
include(GoogleTest)
gtest_discover_tests(SQTTMarkerTests)

target_include_directories(SQTTMarkerTests PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../lib"
  ${LLVM_INCLUDE_DIRS}
  ${GTEST_INCLUDE_DIR}
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
