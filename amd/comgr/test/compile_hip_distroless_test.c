//===- compile_hip_distroless_test.c --------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Verify that the embedded libc++ injection path still works when forced via
// `AMD_COMGR_USE_EMBEDDED_LIBCXX=force`. This simulates a distroless / driver-
// only host where no system C++ headers exist; the env override bypasses the
// detection probe and exercises the same code path that runs on a truly bare
// system.
//
// Source uses only headers from the embedded subset (LIBCXX_USER_HEADERS in
// cmake/LibcxxHeaders.cmake): type_traits, limits, tuple, cstdint, cstddef,
// initializer_list, concepts. Headers outside that subset would fail because
// the embedded set is partial.
//
//===----------------------------------------------------------------------===//

#include "amd_comgr.h"
#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32) || defined(_WIN64)
// MSVC: setenv is POSIX. _putenv_s is the closest equivalent and ignores
// the overwrite flag (always overwrites), which matches our usage here.
#define setenv(name, value, overwrite) _putenv_s((name), (value))
#endif

const char *HipSource =
    "#define __global__ __attribute__((global))\n"
    "#define __device__ __attribute__((device))\n"
    "\n"
    "#include <type_traits>\n"
    "#include <limits>\n"
    "#include <tuple>\n"
    "#include <cstdint>\n"
    "\n"
    "static_assert(std::is_integral<int>::value, \"int is integral\");\n"
    "static_assert(std::numeric_limits<std::int32_t>::digits == 31,\n"
    "              \"int32 digits\");\n"
    "static_assert(std::tuple_size<std::tuple<int, float>>::value == 2,\n"
    "              \"tuple size\");\n"
    "\n"
    "extern \"C\" __global__ void test_kernel(int *out) {\n"
    "    std::tuple<int, float> t{42, 3.14f};\n"
    "    out[0] = std::get<0>(t);\n"
    "}\n";

int main(int Argc, char *Argv[]) {
  // Force the embedded path even when system C++ headers are present, to
  // mirror what comgr does on a distroless host without invasive sysroot setup.
  setenv("AMD_COMGR_USE_EMBEDDED_LIBCXX", "force", 1);

  amd_comgr_data_t DataSource;
  amd_comgr_data_set_t DataSetIn, DataSetBc;
  amd_comgr_action_info_t ActionInfo;
  amd_comgr_status_t Status;

  const char *CompileOptions[] = {"-std=c++17", "-nogpuinc"};
  size_t CompileOptionsCount =
      sizeof(CompileOptions) / sizeof(CompileOptions[0]);

  Status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &DataSource);
  checkError(Status, "amd_comgr_create_data");
  Status = amd_comgr_set_data(DataSource, strlen(HipSource), HipSource);
  checkError(Status, "amd_comgr_set_data");
  Status = amd_comgr_set_data_name(DataSource, "test_distroless.hip");
  checkError(Status, "amd_comgr_set_data_name");

  Status = amd_comgr_create_data_set(&DataSetIn);
  checkError(Status, "amd_comgr_create_data_set");
  Status = amd_comgr_data_set_add(DataSetIn, DataSource);
  checkError(Status, "amd_comgr_data_set_add");

  Status = amd_comgr_create_action_info(&ActionInfo);
  checkError(Status, "amd_comgr_create_action_info");
  Status =
      amd_comgr_action_info_set_language(ActionInfo, AMD_COMGR_LANGUAGE_HIP);
  checkError(Status, "amd_comgr_action_info_set_language");
  Status = amd_comgr_action_info_set_isa_name(ActionInfo,
                                              "amdgcn-amd-amdhsa--gfx906");
  checkError(Status, "amd_comgr_action_info_set_isa_name");
  Status = amd_comgr_action_info_set_option_list(ActionInfo, CompileOptions,
                                                 CompileOptionsCount);
  checkError(Status, "amd_comgr_action_info_set_option_list");

  Status = amd_comgr_create_data_set(&DataSetBc);
  checkError(Status, "amd_comgr_create_data_set");

  Status = amd_comgr_do_action(
      AMD_COMGR_ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC, ActionInfo,
      DataSetIn, DataSetBc);
  checkError(Status, "amd_comgr_do_action (compile to BC)");

  printf("compile_hip_distroless_test PASSED\n");

  amd_comgr_destroy_action_info(ActionInfo);
  amd_comgr_release_data(DataSource);
  amd_comgr_destroy_data_set(DataSetIn);
  amd_comgr_destroy_data_set(DataSetBc);

  return 0;
}
