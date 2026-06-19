//===- compile_hip_stdlibcxx_conflict_test.c ------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Regression test for: embedded libc++ headers conflict with system libstdc++
// on RHEL/manylinux environments (ROCm/llvm-project#2445).
//
// On systems where gcc-toolset libstdc++ is installed at clang's default search
// path, comgr's -idirafter injection of embedded libc++ causes a mixed-header
// chain:
//   system libstdc++ <array>/<exception> -> libc++ stddef.h -> #include_next
//   fails under -nogpuinc (no next stddef.h on device path).
//
// This test compiles HIP source that includes headers known to trigger the
// conflict (<array>, <stdexcept>, <exception>) without passing -nostdinc++.
// It must succeed: either the system does not have conflicting libstdc++
// headers, or comgr skips embedded -idirafter injection and prevents the
// mixed-header chain.
//
//===----------------------------------------------------------------------===//

#include "amd_comgr.h"
#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32) || defined(_WIN64)
// The libstdc++/libc++ header conflict only manifests on Linux distributions
// shipping libstdc++ at clang's default search paths (RHEL/manylinux/Ubuntu).
// Windows has no equivalent toolchain layout, so the entire scenario is N/A.
int main(int Argc, char *Argv[]) {
  printf("compile_hip_stdlibcxx_conflict_test SKIPPED (not applicable on "
         "Windows)\n");
  return 0;
}
#else

#include <dirent.h>
#include <sys/stat.h>

static int fileExists(const char *Path) {
  struct stat St;
  return stat(Path, &St) == 0;
}

// Probe whether a clang resource directory containing builtin headers
// (specifically stdarg.h) is reachable on disk at one of the layouts comgr
// auto-detects. Without one, libstdc++'s <wchar.h> #include <stdarg.h>
// chain has nothing to resolve to (the embedded VFS subset only carries
// stddef.h derivatives). In that environment the conflict test cannot
// distinguish "libcxx-skip is broken" from "no clang builtins on disk".
static int hasClangBuiltinHeadersOnDisk(void) {
  // Honor LLVM_PATH first if set.
  const char *LLVMPath = getenv("LLVM_PATH");
  const char *Roots[] = {LLVMPath ? LLVMPath : "",
                         "/opt/rocm/llvm",
                         "/usr",
                         NULL};
  for (int I = 0; Roots[I]; ++I) {
    if (Roots[I][0] == '\0')
      continue;
    char ClangDir[512];
    snprintf(ClangDir, sizeof(ClangDir), "%s/lib/clang", Roots[I]);
    DIR *D = opendir(ClangDir);
    if (!D)
      continue;
    struct dirent *E;
    while ((E = readdir(D)) != NULL) {
      if (E->d_name[0] == '.')
        continue;
      char Probe[1024];
      snprintf(Probe, sizeof(Probe), "%s/%s/include/stdarg.h", ClangDir,
               E->d_name);
      if (fileExists(Probe)) {
        closedir(D);
        return 1;
      }
    }
    closedir(D);
  }
  return 0;
}

// Mirror of detectSystemCxxHeadersOnDisk in comgr-compiler.cpp: returns true
// when either system libc++ or libstdc++ headers are installed at standard
// paths. Used to skip this test on truly bare hosts (no system C++ headers
// at all), where the embedded subset cannot resolve <array>/<stdexcept>.
static int hasSystemCxxHeaders(void) {
  if (fileExists("/usr/include/c++/v1/__config_site") ||
      fileExists("/usr/local/include/c++/v1/__config_site"))
    return 1;
  DIR *D = opendir("/usr/include/c++");
  if (!D)
    return 0;
  struct dirent *E;
  int Found = 0;
  while ((E = readdir(D)) != NULL) {
    if (E->d_name[0] == '.')
      continue;
    char Probe[512];
    snprintf(Probe, sizeof(Probe), "/usr/include/c++/%s/cstddef", E->d_name);
    if (fileExists(Probe)) {
      Found = 1;
      break;
    }
  }
  closedir(D);
  return Found;
}

// HIP source using headers that trigger the libstdc++/libc++ conflict.
// <array>, <stdexcept>, <exception> all transitively pull in <stddef.h>
// via libstdc++ on RHEL/manylinux/Ubuntu, which then collides with libc++'s
// VFS-mapped stddef.h doing #include_next with no successor under -nogpuinc.
//
// We deliberately do NOT throw on device; devices can't throw. The point is
// that #including these host C++ headers must not break parsing.
const char *HipSource =
    "#define __global__ __attribute__((global))\n"
    "#define __device__ __attribute__((device))\n"
    "\n"
    "#include <array>\n"
    "#include <stdexcept>\n"
    "#include <exception>\n"
    "\n"
    "static_assert(std::tuple_size<std::array<int, 3>>::value == 3,\n"
    "              \"array size\");\n"
    "\n"
    "extern \"C\" __global__ void test_kernel(int *out) {\n"
    "    std::array<int, 3> a = {1, 2, 3};\n"
    "    out[0] = a[0] + a[1] + a[2];\n"
    "}\n";

// Print log data from a data set for diagnostics on failure.
static void printLogs(amd_comgr_data_set_t DataSet) {
  size_t Count;
  amd_comgr_status_t Status =
      amd_comgr_action_data_count(DataSet, AMD_COMGR_DATA_KIND_LOG, &Count);
  if (Status != AMD_COMGR_STATUS_SUCCESS)
    return;
  for (size_t i = 0; i < Count; i++) {
    amd_comgr_data_t Data;
    Status = amd_comgr_action_data_get_data(DataSet, AMD_COMGR_DATA_KIND_LOG, i,
                                            &Data);
    if (Status != AMD_COMGR_STATUS_SUCCESS)
      continue;
    size_t Size;
    Status = amd_comgr_get_data(Data, &Size, NULL);
    if (Status != AMD_COMGR_STATUS_SUCCESS) {
      amd_comgr_release_data(Data);
      continue;
    }
    char *Bytes = (char *)malloc(Size + 1);
    if (!Bytes) {
      amd_comgr_release_data(Data);
      continue;
    }
    Status = amd_comgr_get_data(Data, &Size, Bytes);
    if (Status == AMD_COMGR_STATUS_SUCCESS) {
      Bytes[Size] = '\0';
      fprintf(stderr, "comgr log:\n%s\n", Bytes);
    }
    free(Bytes);
    amd_comgr_release_data(Data);
  }
}

int main(int Argc, char *Argv[]) {
  // <array>/<stdexcept>/<exception> are not in the embedded libc++ subset,
  // so on a host with neither libstdc++ nor libc++ this test cannot
  // distinguish "fix is broken" from "no host C++ stdlib at all". Skip.
  if (!hasSystemCxxHeaders()) {
    printf("compile_hip_stdlibcxx_conflict_test SKIPPED "
           "(no system C++ headers found)\n");
    return 0;
  }

  // If no clang resource dir is reachable on disk, libstdc++'s
  // <wchar.h> -> <stdarg.h> chain cannot resolve regardless of comgr's
  // libcxx-skip behavior. Skip rather than report a misleading failure.
  if (!hasClangBuiltinHeadersOnDisk()) {
    printf("compile_hip_stdlibcxx_conflict_test SKIPPED "
           "(no clang builtin headers on disk; set LLVM_PATH or install "
           "clang at /opt/rocm/llvm or /usr)\n");
    return 0;
  }

  amd_comgr_data_t DataSource;
  amd_comgr_data_set_t DataSetIn, DataSetBc;
  amd_comgr_action_info_t ActionInfo;
  amd_comgr_status_t Status;

  // No -nostdinc++ here: that is the point of this test. Comgr must handle
  // the libstdc++/libc++ conflict internally by skipping its embedded
  // -idirafter fallback when system C++ headers are present.
  const char *CompileOptions[] = {"-std=c++17", "-nogpuinc"};
  size_t CompileOptionsCount =
      sizeof(CompileOptions) / sizeof(CompileOptions[0]);

  Status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &DataSource);
  checkError(Status, "amd_comgr_create_data");
  Status = amd_comgr_set_data(DataSource, strlen(HipSource), HipSource);
  checkError(Status, "amd_comgr_set_data");
  Status = amd_comgr_set_data_name(DataSource, "test_conflict.hip");
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
  if (Status != AMD_COMGR_STATUS_SUCCESS) {
    printLogs(DataSetBc);
    fail("amd_comgr_do_action (compile to BC) -- "
         "likely libstdc++/libc++ header conflict (ROCm#2445): "
         "comgr must skip embedded -idirafter when system headers exist");
  }

  printf("compile_hip_stdlibcxx_conflict_test PASSED\n");

  amd_comgr_destroy_action_info(ActionInfo);
  amd_comgr_release_data(DataSource);
  amd_comgr_destroy_data_set(DataSetIn);
  amd_comgr_destroy_data_set(DataSetBc);

  return 0;
}

#endif // !_WIN32
