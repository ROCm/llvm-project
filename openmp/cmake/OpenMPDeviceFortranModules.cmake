#===-- openmp/cmake/OpenMPDeviceFortranModules.cmake ---------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#
#
# Device-side OpenMP Fortran modules (omp_lib.mod) use iso_c_binding and other
# intrinsic modules produced by the host flang-rt build. Device runtimes
# sub-builds are typically configured with LLVM_ENABLE_RUNTIMES=openmp only, so
# config-Fortran.cmake does not wire up flang-rt-mod automatically.
#
#===------------------------------------------------------------------------===#

include_guard(DIRECTORY)

function(openmp_get_host_flang_rt_module_dir out_var)
  if (LIBOMP_HOST_FLANG_RT_MODULE_DIR)
    set(${out_var} "${LIBOMP_HOST_FLANG_RT_MODULE_DIR}" PARENT_SCOPE)
    return()
  endif()

  if (NOT LLVM_BINARY_DIR OR NOT LLVM_HOST_TRIPLE)
    set(${out_var} "" PARENT_SCOPE)
    return()
  endif()

  include(GetClangResourceDir)
  include(GetToolchainDirs)
  include(ExtendPath)

  get_clang_resource_dir(_resource_dir PREFIX "${LLVM_BINARY_DIR}")
  get_toolchain_module_subdir(_mod_subdir)
  extend_path(_host_mod_dir "${_resource_dir}" "${_mod_subdir}/${LLVM_HOST_TRIPLE}")

  set(${out_var} "${_host_mod_dir}" PARENT_SCOPE)
endfunction()

function(openmp_require_host_flang_rt_modules_for_device_mod target)
  if (NOT "${LLVM_DEFAULT_TARGET_TRIPLE}" MATCHES "^amdgcn|^nvptx|^spirv")
    return()
  endif()

  # Same sub-build also builds flang-rt: flang_module_target() wiring is enough.
  if ("flang-rt" IN_LIST LLVM_ENABLE_RUNTIMES)
    return()
  endif()

  openmp_get_host_flang_rt_module_dir(_host_mod_dir)
  if (NOT _host_mod_dir)
    message(FATAL_ERROR
      "Building device-side ${target} for ${LLVM_DEFAULT_TARGET_TRIPLE} requires "
      "host Flang-RT intrinsic modules (e.g. iso_c_binding.mod). "
      "Set -DLIBOMP_HOST_FLANG_RT_MODULE_DIR=<path-to-host-modules> or ensure the "
      "host runtimes build (flang-rt-mod) completes before the device runtimes build.")
  endif()

  target_compile_options(${target} PRIVATE
    "$<$<COMPILE_LANGUAGE:Fortran>:-fintrinsic-modules-path=${_host_mod_dir}>")
endfunction()
