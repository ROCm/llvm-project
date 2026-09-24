//===- compile-hip-cxx-headers.c ------------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "amd_comgr.h"
#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32) || defined(_WIN64)
#define setenv(name, value, overwrite) _putenv_s((name), (value))
#else
#include <dirent.h>
#include <sys/stat.h>
extern int setenv(const char *, const char *, int);
#endif

enum TestMode {
  TEST_DISTROLESS,
  TEST_LIBSTDCXX_CONFLICT,
  TEST_SYSTEM_LIBCXX,
  TEST_GCC_TOOLCHAIN
};

static int fileExists(const char *Path) {
#if defined(_WIN32) || defined(_WIN64)
  return 0;
#else
  struct stat St;
  return stat(Path, &St) == 0;
#endif
}

static int hasClangBuiltinHeadersOnDisk(void) {
#if defined(_WIN32) || defined(_WIN64)
  return 0;
#else
  const char *LLVMPath = getenv("LLVM_PATH");
  const char *Roots[] = {LLVMPath ? LLVMPath : "", "/opt/rocm/llvm", "/usr",
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
#endif
}

static int scanLibstdcxxHeaders(const char *CxxRoot) {
#if defined(_WIN32) || defined(_WIN64)
  return 0;
#else
  DIR *D = opendir(CxxRoot);
  if (!D)
    return 0;
  struct dirent *E;
  int Found = 0;
  while ((E = readdir(D)) != NULL) {
    if (E->d_name[0] == '.')
      continue;
    if (!strcmp(E->d_name, "v1"))
      continue;
    char Probe[512];
    snprintf(Probe, sizeof(Probe), "%s/%s/cstddef", CxxRoot, E->d_name);
    if (fileExists(Probe)) {
      Found = 1;
      break;
    }
  }
  closedir(D);
  return Found;
#endif
}

static int hasSystemLibstdcxxHeaders(void) {
  return scanLibstdcxxHeaders("/usr/include/c++") ||
         scanLibstdcxxHeaders("/usr/local/include/c++");
}

static int hasSystemLibcxxHeaders(void) {
  return fileExists("/usr/include/c++/v1/cstddef") ||
         fileExists("/usr/local/include/c++/v1/cstddef");
}

static void printLogs(amd_comgr_data_set_t DataSet) {
  size_t Count;
  amd_comgr_status_t Status =
      amd_comgr_action_data_count(DataSet, AMD_COMGR_DATA_KIND_LOG, &Count);
  if (Status != AMD_COMGR_STATUS_SUCCESS)
    return;
  for (size_t I = 0; I < Count; ++I) {
    amd_comgr_data_t Data;
    Status = amd_comgr_action_data_get_data(DataSet, AMD_COMGR_DATA_KIND_LOG, I,
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
      printf("%s", Bytes);
    }
    free(Bytes);
    amd_comgr_release_data(Data);
  }
}

static int logContains(amd_comgr_data_set_t DataSet, const char *Needle) {
  size_t Count;
  amd_comgr_status_t Status =
      amd_comgr_action_data_count(DataSet, AMD_COMGR_DATA_KIND_LOG, &Count);
  if (Status != AMD_COMGR_STATUS_SUCCESS)
    return 0;
  for (size_t I = 0; I < Count; ++I) {
    amd_comgr_data_t Data;
    Status = amd_comgr_action_data_get_data(DataSet, AMD_COMGR_DATA_KIND_LOG, I,
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
    int Found = 0;
    if (Status == AMD_COMGR_STATUS_SUCCESS) {
      Bytes[Size] = '\0';
      Found = strstr(Bytes, Needle) != NULL;
    }
    free(Bytes);
    amd_comgr_release_data(Data);
    if (Found)
      return 1;
  }
  return 0;
}

static enum TestMode parseMode(const char *Arg) {
  if (!strcmp(Arg, "--mode=distroless"))
    return TEST_DISTROLESS;
  if (!strcmp(Arg, "--mode=libstdcxx-conflict"))
    return TEST_LIBSTDCXX_CONFLICT;
  if (!strcmp(Arg, "--mode=system-libcxx"))
    return TEST_SYSTEM_LIBCXX;
  if (!strcmp(Arg, "--mode=gcc-toolchain"))
    return TEST_GCC_TOOLCHAIN;
  fail("unknown mode: %s", Arg);
  return TEST_DISTROLESS;
}

int main(int argc, char *argv[]) {
  if (argc < 3)
    fail("Usage: compile-hip-cxx-headers --mode=<mode> input.hip "
         "[sysroot gcc-toolchain]");

  enum TestMode Mode = parseMode(argv[1]);
  if (Mode == TEST_GCC_TOOLCHAIN && argc != 5)
    fail("Usage for --mode=gcc-toolchain: compile-hip-cxx-headers "
         "--mode=gcc-toolchain input.hip sysroot gcc-toolchain");
  if (Mode != TEST_GCC_TOOLCHAIN && argc != 3)
    fail("Usage: compile-hip-cxx-headers --mode=<mode> input.hip");

  if (Mode == TEST_LIBSTDCXX_CONFLICT && !hasSystemLibstdcxxHeaders()) {
    printf("RESULT: SKIPPED no system libstdc++ headers found\n");
    return 0;
  }
  if (Mode == TEST_LIBSTDCXX_CONFLICT && !hasClangBuiltinHeadersOnDisk()) {
    printf("RESULT: SKIPPED no clang builtin headers on disk\n");
    return 0;
  }
  if (Mode == TEST_SYSTEM_LIBCXX && !hasSystemLibcxxHeaders()) {
    printf("RESULT: SKIPPED no system libc++ headers found\n");
    return 0;
  }

  setenv("AMD_COMGR_EMIT_VERBOSE_LOGS", "1", 1);
  if (Mode == TEST_DISTROLESS)
    setenv("AMD_COMGR_USE_EMBEDDED_LIBCXX", "force", 1);
  else
    setenv("AMD_COMGR_USE_EMBEDDED_LIBCXX", "auto", 1);

  const char *DefaultOptions[] = {"-std=c++17", "-nogpuinc"};
  const char *LibcxxOptions[] = {"-std=c++17", "-stdlib=libc++", "-nogpuinc"};
  char SysrootOpt[1024];
  char GccToolchainOpt[1024];
  const char *GccToolchainOptions[] = {"-std=c++17", "-nogpuinc", SysrootOpt,
                                       GccToolchainOpt};
  const char **CompileOptions = DefaultOptions;
  size_t CompileOptionsCount =
      sizeof(DefaultOptions) / sizeof(DefaultOptions[0]);
  if (Mode == TEST_SYSTEM_LIBCXX) {
    CompileOptions = LibcxxOptions;
    CompileOptionsCount = sizeof(LibcxxOptions) / sizeof(LibcxxOptions[0]);
  } else if (Mode == TEST_GCC_TOOLCHAIN) {
    snprintf(SysrootOpt, sizeof(SysrootOpt), "--sysroot=%s", argv[3]);
    snprintf(GccToolchainOpt, sizeof(GccToolchainOpt), "--gcc-toolchain=%s",
             argv[4]);
    CompileOptions = GccToolchainOptions;
    CompileOptionsCount =
        sizeof(GccToolchainOptions) / sizeof(GccToolchainOptions[0]);
  }

  char *BufSource;
  size_t SizeSource = setBuf(argv[2], &BufSource);

  amd_comgr_data_t DataSource;
  amd_comgr_data_set_t DataSetIn, DataSetBc;
  amd_comgr_action_info_t ActionInfo;

  amd_comgr_(create_data(AMD_COMGR_DATA_KIND_SOURCE, &DataSource));
  amd_comgr_(set_data(DataSource, SizeSource, BufSource));
  amd_comgr_(set_data_name(DataSource, "test_cxx_headers.hip"));
  amd_comgr_(create_data_set(&DataSetIn));
  amd_comgr_(data_set_add(DataSetIn, DataSource));
  amd_comgr_(create_action_info(&ActionInfo));
  amd_comgr_(action_info_set_language(ActionInfo, AMD_COMGR_LANGUAGE_HIP));
  amd_comgr_(action_info_set_isa_name(ActionInfo, "amdgcn-amd-amdhsa--gfx906"));
  amd_comgr_(action_info_set_option_list(ActionInfo, CompileOptions,
                                         CompileOptionsCount));
  amd_comgr_(action_info_set_logging(ActionInfo, true));
  amd_comgr_(create_data_set(&DataSetBc));

  amd_comgr_status_t Status = amd_comgr_do_action(
      AMD_COMGR_ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC, ActionInfo,
      DataSetIn, DataSetBc);
  printLogs(DataSetBc);

  if (Status != AMD_COMGR_STATUS_SUCCESS)
    fail("amd_comgr_do_action failed");

  if ((Mode == TEST_DISTROLESS || Mode == TEST_GCC_TOOLCHAIN) &&
      !logContains(DataSetBc, "Embedded libc++ headers: active"))
    fail("expected embedded libc++ active log");
  if (Mode != TEST_DISTROLESS && Mode != TEST_GCC_TOOLCHAIN &&
      !logContains(DataSetBc, "Embedded libc++ headers: skipped"))
    fail("expected embedded libc++ skipped log");
  if (Mode == TEST_SYSTEM_LIBCXX && !logContains(DataSetBc, "c++/v1"))
    fail("expected system libc++ detection path in log");

  printf("RESULT: PASS\n");

  amd_comgr_(destroy_action_info(ActionInfo));
  amd_comgr_(release_data(DataSource));
  amd_comgr_(destroy_data_set(DataSetIn));
  amd_comgr_(destroy_data_set(DataSetBc));
  free(BufSource);
  return 0;
}
