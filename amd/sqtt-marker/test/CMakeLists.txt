# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

if(NOT LLVM_TOOLS_BINARY_DIR)
  get_filename_component(SQTT_MARKER_LLVM_PREFIX "${LLVM_DIR}/../../.." ABSOLUTE)
  set(LLVM_TOOLS_BINARY_DIR "${SQTT_MARKER_LLVM_PREFIX}/bin")
endif()

if(SQTT_MARKER_STANDALONE AND NOT EXISTS "${LLVM_EXTERNAL_LIT}")
  set(LLVM_EXTERNAL_LIT "${CMAKE_CURRENT_SOURCE_DIR}/run-lit.py"
      CACHE FILEPATH "Command used to spawn lit" FORCE)
endif()

get_target_property(SQTT_MARKER_PLUGIN_DIR SQTTMarkerPass LIBRARY_OUTPUT_DIRECTORY)
if(NOT SQTT_MARKER_PLUGIN_DIR)
  set(SQTT_MARKER_PLUGIN_DIR "${PROJECT_BINARY_DIR}")
endif()
set(SQTT_MARKER_PLUGIN_FILE
  "${SQTT_MARKER_PLUGIN_DIR}/${CMAKE_SHARED_MODULE_PREFIX}sqtt-marker${CMAKE_SHARED_MODULE_SUFFIX}")
set(SQTT_MARKER_AMDGPU_AVAILABLE 0)
if("AMDGPU" IN_LIST LLVM_TARGETS_TO_BUILD)
  set(SQTT_MARKER_AMDGPU_AVAILABLE 1)
endif()

configure_file(lit.site.cfg.py.in lit.site.cfg.py @ONLY)

set(SQTT_MARKER_TEST_DEPENDS SQTTMarkerPass)
foreach(tool FileCheck clang clang-offload-bundler llvm-objcopy opt)
  if(TARGET ${tool})
    list(APPEND SQTT_MARKER_TEST_DEPENDS ${tool})
  endif()
endforeach()

add_lit_testsuite(check-sqtt-marker "Running AMD SQTT marker tests"
  "${CMAKE_CURRENT_BINARY_DIR}"
  DEPENDS ${SQTT_MARKER_TEST_DEPENDS}
)
