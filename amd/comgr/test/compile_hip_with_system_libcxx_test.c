//===- compile_hip_with_system_libcxx_test.c ------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Verify that when system libc++ is installed at the standard path
// (`/usr/include/c++/v1/__config_site`), comgr's auto-detection takes the
// libc++ branch of `detectSystemCxxHeadersOnDisk` and refrains from
// `-idirafter` injection. Companion to `compile_hip_stdlibcxx_conflict_test`,
// which covers the libstdc++ branch and the actual end-to-end compile.
//
// This test asserts the detection behavior (verbose log) rather than
// compile success: clang's `-stdlib=libc++` resolution does not redirect to
// `/usr/include/c++/v1`, so a clean compile would require a more invasive
// test setup. The detection branch firing is what matters for the fix.
//
// Skipped on hosts without system libc++ headers (e.g. minimal CI images).
//
//===----------------------------------------------------------------------===//

#include "amd_comgr.h"
#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32) || defined(_WIN64)
// The probed paths (/usr/include/c++/v1/__config_site) are Linux-only;
// the Windows toolchain layout has no analogous system libc++ to detect.
int main(int Argc, char *Argv[]) {
  printf("compile_hip_with_system_libcxx_test SKIPPED (not applicable on "
         "Windows)\n");
  return 0;
}
#else

#include <sys/stat.h>

static int fileExists(const char *Path) {
  struct stat St;
  return stat(Path, &St) == 0;
}

// Search every LOG entry in DataSet for Needle; fail with id if absent.
static void requireLogContains(const char *Id, amd_comgr_data_set_t DataSet,
                               const char *Needle) {
  size_t Count;
  amd_comgr_status_t Status =
      amd_comgr_action_data_count(DataSet, AMD_COMGR_DATA_KIND_LOG, &Count);
  checkError(Status, "amd_comgr_action_data_count");

  int Found = 0;
  for (size_t I = 0; I < Count && !Found; ++I) {
    amd_comgr_data_t Data;
    Status = amd_comgr_action_data_get_data(DataSet, AMD_COMGR_DATA_KIND_LOG, I,
                                            &Data);
    checkError(Status, "amd_comgr_action_data_get_data");

    size_t Size;
    Status = amd_comgr_get_data(Data, &Size, NULL);
    checkError(Status, "amd_comgr_get_data");

    char *Bytes = (char *)malloc(Size + 1);
    if (!Bytes)
      fail("malloc");
    Status = amd_comgr_get_data(Data, &Size, Bytes);
    checkError(Status, "amd_comgr_get_data");
    Bytes[Size] = '\0';

    if (strstr(Bytes, Needle))
      Found = 1;

    free(Bytes);
    amd_comgr_release_data(Data);
  }

  if (!Found)
    fail("%s: expected log substring \"%s\" not found", Id, Needle);
}

const char *HipSource =
    "#define __global__ __attribute__((global))\n"
    "#define __device__ __attribute__((device))\n"
    "\n"
    "extern \"C\" __global__ void test_kernel(int *out) { out[0] = 1; }\n";

int main(int Argc, char *Argv[]) {
  if (!fileExists("/usr/include/c++/v1/__config_site") &&
      !fileExists("/usr/local/include/c++/v1/__config_site")) {
    printf("compile_hip_with_system_libcxx_test SKIPPED "
           "(system libc++ not installed)\n");
    return 0;
  }

  // Need verbose logs for the "Embedded libc++ headers: ..." line.
  setenv("AMD_COMGR_EMIT_VERBOSE_LOGS", "1", 1);

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
  Status = amd_comgr_set_data_name(DataSource, "test_system_libcxx.hip");
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
  Status = amd_comgr_action_info_set_logging(ActionInfo, true);
  checkError(Status, "amd_comgr_action_info_set_logging");

  Status = amd_comgr_create_data_set(&DataSetBc);
  checkError(Status, "amd_comgr_create_data_set");

  // Compile outcome is not asserted; we only care that detection ran and
  // logged the skip decision.
  amd_comgr_do_action(AMD_COMGR_ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC,
                      ActionInfo, DataSetIn, DataSetBc);

  requireLogContains("compile_hip_with_system_libcxx_test", DataSetBc,
                     "Embedded libc++ headers: skipped");

  printf("compile_hip_with_system_libcxx_test PASSED\n");

  amd_comgr_destroy_action_info(ActionInfo);
  amd_comgr_release_data(DataSource);
  amd_comgr_destroy_data_set(DataSetIn);
  amd_comgr_destroy_data_set(DataSetBc);

  return 0;
}

#endif // !_WIN32
