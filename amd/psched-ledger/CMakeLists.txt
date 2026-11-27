# ===- CMakeLists.txt - Scheduling ledger pass plugin ---------------------=== #
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------=== #
#
# The plugin must be built against the same LLVM build as the compiler that
# loads it. Two ways:
#
# With the compiler (recommended: `ninja` then keeps the two in step). Add to
# the LLVM configure line
#   -DLLVM_EXTERNAL_PROJECTS=psched-ledger
#   -DLLVM_EXTERNAL_PSCHED_LEDGER_SOURCE_DIR=<llvm-project>/amd/psched-ledger
# and build the target `psched-ledger`; the result is <build>/lib/psched-ledger.so.
#
# Standalone, against a finished build:
#   cmake -S amd/psched-ledger -B <dir> -DLLVM_DIR=<llvm build>/lib/cmake/llvm
#   cmake --build <dir>

cmake_minimum_required(VERSION 3.20)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(amd-psched-ledger LANGUAGES C CXX)
  find_package(LLVM REQUIRED CONFIG)
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
  include(AddLLVM)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
  separate_arguments(PSCHED_LEDGER_LLVM_DEFINITIONS NATIVE_COMMAND
                     "${LLVM_DEFINITIONS}")
  add_definitions(${PSCHED_LEDGER_LLVM_DEFINITIONS})
  if(NOT LLVM_ENABLE_RTTI)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
  endif()
endif()

add_llvm_pass_plugin(psched-ledger lib/LedgerPass.cpp)
