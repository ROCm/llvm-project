#===-- unittests/Runtime/CMakeLists.txt ------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

add_flangrt_unittest(RuntimeTests
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
  OmpAllocator.cpp
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

  LINK_LIBS
    flang_rt.runtime.unittest
    flang_rt.openmp.unittest
)
target_compile_definitions(RuntimeTests PRIVATE NOT_EXE="${LLVM_TOOLS_DIR}/not${CMAKE_EXECUTABLE_SUFFIX}")

# OmpAllocator.cpp defines fake __kmpc_alloc / __kmpc_free / __kmpc_global_thread_num
# symbols that flang_rt.openmp's omp-allocator.cpp resolves at runtime via
# dlsym(RTLD_DEFAULT, ...).  For that to succeed the executable must export its
# dynamic symbols; enable that here (no-op on platforms where it doesn't apply).
set_target_properties(RuntimeTests PROPERTIES ENABLE_EXPORTS ON)

if (FLANG_RT_INCLUDE_CUF)
  add_subdirectory(CUDA)
endif ()
