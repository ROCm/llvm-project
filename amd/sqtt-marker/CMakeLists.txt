# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.20)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(amd-sqtt-marker VERSION 0.1.0 LANGUAGES C CXX)
  include(GNUInstallDirs)
  include(CTest)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)

  find_package(LLVM REQUIRED CONFIG)
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
  include(AddLLVM)

  set(SQTT_MARKER_STANDALONE ON)
else()
  set(SQTT_MARKER_STANDALONE OFF)
endif()

if(SQTT_MARKER_STANDALONE)
  set(SQTT_MARKER_BUILD_TESTS_DEFAULT ${BUILD_TESTING})
else()
  set(SQTT_MARKER_BUILD_TESTS_DEFAULT ${LLVM_INCLUDE_TESTS})
endif()
option(SQTT_MARKER_BUILD_TESTS "Build AMD SQTT marker tests"
       ${SQTT_MARKER_BUILD_TESTS_DEFAULT})

set(SQTT_MARKER_SOURCES
  lib/SQTTAddressTrace.cpp
  lib/SQTTAutomatic.cpp
  lib/SQTTFuncMap.cpp
  lib/SQTTFunctions.cpp
  lib/SQTTMarkers.cpp
  lib/SQTTNamedMarkers.cpp
  lib/SQTTPass.cpp
)

add_library(SQTTMarkerPass MODULE ${SQTT_MARKER_SOURCES})
if(TARGET intrinsics_gen)
  add_dependencies(SQTTMarkerPass intrinsics_gen)
endif()
target_include_directories(SQTTMarkerPass PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/lib"
  ${LLVM_INCLUDE_DIRS}
)
if(SQTT_MARKER_STANDALONE)
  target_compile_definitions(SQTTMarkerPass PRIVATE ${LLVM_DEFINITIONS})
endif()
target_compile_features(SQTTMarkerPass PRIVATE cxx_std_17)

if(WIN32)
  if(TARGET LLVM)
    target_link_libraries(SQTTMarkerPass PRIVATE LLVM)
  else()
    llvm_map_components_to_libnames(SQTT_MARKER_LLVM_LIBS
      Analysis Core Passes Support TargetParser TransformUtils)
    target_link_libraries(SQTTMarkerPass PRIVATE ${SQTT_MARKER_LLVM_LIBS})
  endif()
endif()

if(COMMAND llvm_update_compile_flags)
  llvm_update_compile_flags(SQTTMarkerPass)
elseif(NOT MSVC)
  target_compile_options(SQTTMarkerPass PRIVATE -fno-rtti)
endif()

# The plugin resolves LLVM symbols from the compiler process that loads it.
set_target_properties(SQTTMarkerPass PROPERTIES
  NO_SONAME ON
  PREFIX "lib"
  OUTPUT_NAME "sqtt-marker"
  POSITION_INDEPENDENT_CODE ON
  LIBRARY_OUTPUT_DIRECTORY "${LLVM_LIBRARY_OUTPUT_INTDIR}"
  RUNTIME_OUTPUT_DIRECTORY "${LLVM_RUNTIME_OUTPUT_INTDIR}"
)

install(TARGETS SQTTMarkerPass
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT sqtt-marker
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT sqtt-marker
)
install(DIRECTORY include/
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  COMPONENT sqtt-marker
)
install(FILES README.md docs/SQTTMarkerFormat.md LICENSE.txt
  DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/amd-sqtt-marker"
  COMPONENT sqtt-marker
)

if(COMMAND add_llvm_install_targets AND NOT LLVM_ENABLE_IDE)
  add_llvm_install_targets(install-sqtt-marker
    DEPENDS SQTTMarkerPass
    COMPONENT sqtt-marker)
endif()

if(SQTT_MARKER_BUILD_TESTS)
  add_subdirectory(test)
  add_subdirectory(unittests)
endif()

if(SQTT_MARKER_STANDALONE)
  set(CPACK_PACKAGE_NAME "amd-sqtt-marker")
  set(CPACK_PACKAGE_VENDOR "Advanced Micro Devices, Inc.")
  set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
  set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
      "LLVM pass plugin and C header for AMD SQTT markers")
  set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.txt")
  include(CPack)
endif()
