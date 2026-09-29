# ===- CMakeLists.txt - AMD SQTT marker build -----------------------------=== #
#
# Part of AMD SQTT Marker, under the MIT License. See
# amd/sqtt-marker/LICENSE.txt for license information.
# SPDX-License-Identifier: MIT
#
# ===----------------------------------------------------------------------=== #

cmake_minimum_required(VERSION 3.20)

set(SQTT_MARKER_VERSION 0.1.0)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(amd-sqtt-marker VERSION ${SQTT_MARKER_VERSION} LANGUAGES C CXX)
  include(GNUInstallDirs)
  include(CTest)

  find_package(LLVM REQUIRED CONFIG)
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
  include(AddLLVM)
  separate_arguments(SQTT_MARKER_LLVM_DEFINITIONS NATIVE_COMMAND
                     "${LLVM_DEFINITIONS}")

  set(SQTT_MARKER_STANDALONE ON)
else()
  set(SQTT_MARKER_STANDALONE OFF)
endif()

if(SQTT_MARKER_STANDALONE)
  set(SQTT_MARKER_BUILD_TESTS_DEFAULT OFF)
else()
  set(SQTT_MARKER_BUILD_TESTS_DEFAULT ${LLVM_INCLUDE_TESTS})
endif()
option(SQTT_MARKER_BUILD_TESTS "Build AMD SQTT marker tests"
       ${SQTT_MARKER_BUILD_TESTS_DEFAULT})

set(SQTT_MARKER_SOURCES
  lib/SQTTAddressTrace.cpp
  lib/SQTTAutomatic.cpp
  lib/SQTTConfig.cpp
  lib/SQTTFuncMap.cpp
  lib/SQTTFunctions.cpp
  lib/SQTTMarkers.cpp
  lib/SQTTNamedMarkers.cpp
  lib/SQTTPass.cpp
)

add_library(SQTTMarkerObjects OBJECT ${SQTT_MARKER_SOURCES})
if(TARGET intrinsics_gen)
  add_dependencies(SQTTMarkerObjects intrinsics_gen)
endif()
target_include_directories(SQTTMarkerObjects PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/lib"
  ${LLVM_INCLUDE_DIRS}
)
if(SQTT_MARKER_STANDALONE)
  target_compile_options(SQTTMarkerObjects PRIVATE
    ${SQTT_MARKER_LLVM_DEFINITIONS})
endif()
target_compile_features(SQTTMarkerObjects PRIVATE cxx_std_17)
target_compile_definitions(SQTTMarkerObjects PRIVATE
  SQTT_MARKER_VERSION_STRING="${SQTT_MARKER_VERSION}")
set_target_properties(SQTTMarkerObjects PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(SQTTMarkerPass MODULE $<TARGET_OBJECTS:SQTTMarkerObjects>)

if(WIN32)
  set_target_properties(SQTTMarkerPass PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
  if(TARGET LLVM)
    target_link_libraries(SQTTMarkerPass PRIVATE LLVM)
  else()
    llvm_map_components_to_libnames(SQTT_MARKER_LLVM_LIBS
      Analysis Core Passes Support TargetParser TransformUtils)
    target_link_libraries(SQTTMarkerPass PRIVATE ${SQTT_MARKER_LLVM_LIBS})
  endif()
endif()

if(COMMAND llvm_update_compile_flags)
  llvm_update_compile_flags(SQTTMarkerObjects)
elseif(NOT MSVC)
  target_compile_options(SQTTMarkerObjects PRIVATE -fno-rtti)
endif()

# The plugin resolves LLVM symbols from the compiler process that loads it.
set_target_properties(SQTTMarkerPass PROPERTIES
  NO_SONAME ON
  OUTPUT_NAME "sqtt-marker"
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
  if(SQTT_MARKER_STANDALONE)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
  endif()
  add_subdirectory(test)
  add_subdirectory(unittests)
endif()

if(SQTT_MARKER_STANDALONE)
  set(CPACK_PACKAGE_NAME "amd-sqtt-marker")
  set(CPACK_PACKAGE_VENDOR "Advanced Micro Devices, Inc.")
  set(CPACK_PACKAGE_CONTACT
      "ROCm Compiler Support <rocm.compiler.support@amd.com>")
  set(CPACK_PACKAGE_VERSION "${SQTT_MARKER_VERSION}")
  set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
      "LLVM pass plugin and C header for AMD SQTT markers")
  set(CPACK_PACKAGE_HOMEPAGE_URL
      "https://github.com/ROCm/llvm-project/tree/amd-staging/amd/sqtt-marker")
  set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.txt")
  set(CPACK_GENERATOR "TGZ" CACHE STRING "Default package format")

  set(CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT")
  set(CPACK_DEBIAN_PACKAGE_MAINTAINER "${CPACK_PACKAGE_CONTACT}")
  set(CPACK_DEBIAN_PACKAGE_DEPENDS "rocm-llvm")
  set(CPACK_DEBIAN_PACKAGE_HOMEPAGE "${CPACK_PACKAGE_HOMEPAGE_URL}")

  set(CPACK_RPM_FILE_NAME "RPM-DEFAULT")
  set(CPACK_RPM_PACKAGE_LICENSE "MIT")
  set(CPACK_RPM_PACKAGE_REQUIRES "rocm-llvm")
  include(CPack)
endif()
