#===-- unittests/Runtime/CMakeLists.txt ------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

set(_runtime_test_sources
  AccessTest.cpp
  Allocatable.cpp
  ArrayConstructor.cpp
  Assign.cpp
  BufferTest.cpp
  CharacterTest.cpp
  CommandTest.cpp
  Complex.cpp
  CrashHandlerFixture.cpp
  Derived.cpp
  Descriptor.cpp
  Exceptions.cpp
  ExternalIOTest.cpp
  Format.cpp
  InputExtensions.cpp
  Inquiry.cpp
  LeadingZeroTest.cpp
  ListInputTest.cpp
  LogicalFormatTest.cpp
  Matmul.cpp
  MatmulTranspose.cpp
  MiscIntrinsic.cpp
  Namelist.cpp
  Numeric.cpp
  NumericalFormatTest.cpp
  Pointer.cpp
  Ragged.cpp
  Random.cpp
  Reduction.cpp
  RuntimeCrashTest.cpp
  Stop.cpp
  Support.cpp
  Time.cpp
  TemporaryStack.cpp
  Transformational.cpp
  TypeCode.cpp
)

set(_runtime_test_link_libs flang_rt.runtime.unittest)

# OmpAllocator.cpp + flang_rt.openmp.unittest are only meaningful when the
# OpenMP host runtime has been built (FLANG_RT_OPENMP_KMPC_ALLOC controls
# that knob in flang-rt/lib/openmp/CMakeLists.txt).  When it has not been
# built we simply skip the OMP-allocator test and its support library so
# the rest of the runtime unittests still build cleanly.
if (TARGET flang_rt.openmp.static)
  list(APPEND _runtime_test_sources OmpAllocator.cpp)
  list(APPEND _runtime_test_link_libs flang_rt.openmp.unittest)
endif ()

add_flangrt_unittest(RuntimeTests
  ${_runtime_test_sources}

  LINK_LIBS
    ${_runtime_test_link_libs}
)
target_compile_definitions(RuntimeTests PRIVATE NOT_EXE="${LLVM_TOOLS_DIR}/not${CMAKE_EXECUTABLE_SUFFIX}")

# OmpAllocator.cpp defines fake __kmpc_alloc / __kmpc_aligned_alloc /
# __kmpc_free / __kmpc_global_thread_num symbols that flang_rt.openmp's
# omp_kmpc_alloc.cpp resolves at runtime via dlsym(RTLD_DEFAULT, ...).
# We need *those four* symbols exported as dynamic symbols, but ideally
# *only* those four -- if every C++ symbol in the test binary is exported,
# unrelated test libraries with their own __kmpc_* overlays (or LLVM's
# global symbol interposition heuristics) can win the lookup race and the
# test ends up using the wrong allocator.
#
# Where the linker supports it (ELF / gold / lld via
# --dynamic-list), restrict the dynamic-symbol set to the four __kmpc_*
# names listed in OmpAllocatorKmpcSymbols.list.  Other platforms fall
# back on the legacy "export everything" behaviour, which is still
# correct -- just slightly looser than ideal.
if (TARGET flang_rt.openmp.static)
  if (UNIX AND NOT APPLE)
    target_link_options(RuntimeTests PRIVATE
      "LINKER:--dynamic-list=${CMAKE_CURRENT_SOURCE_DIR}/OmpAllocatorKmpcSymbols.list")
    set_property(TARGET RuntimeTests APPEND PROPERTY LINK_DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/OmpAllocatorKmpcSymbols.list")
  else ()
    set_target_properties(RuntimeTests PROPERTIES ENABLE_EXPORTS ON)
  endif ()
endif ()

if (FLANG_RT_INCLUDE_CUF)
  add_subdirectory(CUDA)
endif ()
