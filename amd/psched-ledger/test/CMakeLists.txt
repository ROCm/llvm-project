# ===- CMakeLists.txt - Scheduling ledger plugin lit tests -------------------=== #
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------=== #

# Only wired up when the plugin is built as part of LLVM (LLVM_EXTERNAL_PROJECTS),
# where lit, opt and FileCheck are targets of the same build.

set(PSCHED_LEDGER_PLUGIN_FILE
  "${LLVM_LIBRARY_DIR}/psched-ledger${LLVM_PLUGIN_EXT}")
set(PSCHED_LEDGER_AMDGPU_AVAILABLE 0)
if("AMDGPU" IN_LIST LLVM_TARGETS_TO_BUILD)
  set(PSCHED_LEDGER_AMDGPU_AVAILABLE 1)
endif()

configure_file(lit.site.cfg.py.in lit.site.cfg.py @ONLY)

set(PSCHED_LEDGER_TEST_DEPENDS psched-ledger)
foreach(tool FileCheck opt not)
  if(TARGET ${tool})
    list(APPEND PSCHED_LEDGER_TEST_DEPENDS ${tool})
  endif()
endforeach()

add_lit_testsuite(check-psched-ledger "Running scheduling ledger plugin tests"
  "${CMAKE_CURRENT_BINARY_DIR}"
  DEPENDS ${PSCHED_LEDGER_TEST_DEPENDS}
)
