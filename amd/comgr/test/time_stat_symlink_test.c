//===- time_stat_symlink_test.c -------------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Regression test for the AMD_COMGR_TIME_STATISTICS perf log being opened with
// O_NOFOLLOW. A symlink pre-planted at the log path must not be followed, so
// the symlink target must be left untouched. The perf log defaults to a
// CWD-relative "PerfStatsLog.txt"; we deliberately do NOT set
// AMD_COMGR_REDIRECT_LOGS, since that env var also drives a separate
// (append-mode, non-O_NOFOLLOW) log open and would confound the test.
//
// O_NOFOLLOW is POSIX-only, so on Windows this test is a no-op.

// Expose POSIX.1-2008 declarations (mkdtemp, setenv/unsetenv, symlink,
// lstat, PATH_MAX) under -std=c99, which otherwise builds in strict ISO
// mode with no GNU/POSIX extensions visible. Must precede every header
// include, since common.h transitively pulls in <sys/stat.h>/<unistd.h>.
#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#ifdef __APPLE__
#define _DARWIN_C_SOURCE
#endif
#endif

#include "amd_comgr.h"
#include "common.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef _WIN32
#include <limits.h>
#include <sys/stat.h>
#include <unistd.h>

static const char Sentinel[] = "CANARY_SENTINEL_DO_NOT_TRUNCATE\n";
static const char *PerfLogName = "PerfStatsLog.txt";

// A minimal OpenCL kernel; the compile only needs to reach amd_comgr_do_action,
// which is where the perf log is opened.
static const char *KernelSource =
    "void kernel k(__global int *p) { *p = 1; }\n";

static void writeSentinel(const char *Path) {
  FILE *F = fopen(Path, "w");
  if (!F) {
    perror("fopen canary");
    exit(1);
  }
  if (fwrite(Sentinel, 1, sizeof(Sentinel) - 1, F) != sizeof(Sentinel) - 1) {
    perror("fwrite canary");
    exit(1);
  }
  fclose(F);
}

static void checkSentinelIntact(const char *Path) {
  char Buf[256] = {0};
  FILE *F = fopen(Path, "r");
  if (!F) {
    perror("fopen canary (read)");
    exit(1);
  }
  size_t N = fread(Buf, 1, sizeof(Buf) - 1, F);
  fclose(F);

  if (N != sizeof(Sentinel) - 1 || memcmp(Buf, Sentinel, N) != 0) {
    printf("FAILED: canary file was modified through the symlink "
           "(perf log open followed the symlink)\n");
    exit(1);
  }
}

int main(void) {
  char TmpDir[] = "/tmp/comgr_time_stat_XXXXXX";
  if (!mkdtemp(TmpDir)) {
    perror("mkdtemp");
    return 1;
  }

  char Canary[PATH_MAX];
  snprintf(Canary, sizeof(Canary), "%s/canary.txt", TmpDir);
  writeSentinel(Canary);

  // Enable the perf log; leave AMD_COMGR_REDIRECT_LOGS unset so the log path
  // is the default CWD-relative PerfStatsLog.txt.
  setenv("AMD_COMGR_TIME_STATISTICS", "1", 1);
  unsetenv("AMD_COMGR_REDIRECT_LOGS");
  setenv("AMD_COMGR_CACHE", "0", 1);

  // Run inside the temp dir and plant the symlink at the default log path.
  if (chdir(TmpDir)) {
    perror("chdir");
    return 1;
  }
  if (symlink(Canary, PerfLogName)) {
    perror("symlink");
    return 1;
  }

  // Trigger a compile so amd_comgr_do_action initializes the perf log.
  amd_comgr_data_t DataSource;
  amd_comgr_data_set_t DataSetIn, DataSetBc;
  amd_comgr_action_info_t DataAction;
  amd_comgr_status_t Status;

  Status = amd_comgr_create_data_set(&DataSetIn);
  checkError(Status, "amd_comgr_create_data_set");
  Status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &DataSource);
  checkError(Status, "amd_comgr_create_data");
  Status = amd_comgr_set_data(DataSource, strlen(KernelSource), KernelSource);
  checkError(Status, "amd_comgr_set_data");
  Status = amd_comgr_set_data_name(DataSource, "source.cl");
  checkError(Status, "amd_comgr_set_data_name");
  Status = amd_comgr_data_set_add(DataSetIn, DataSource);
  checkError(Status, "amd_comgr_data_set_add");

  Status = amd_comgr_create_action_info(&DataAction);
  checkError(Status, "amd_comgr_create_action_info");
  Status = amd_comgr_action_info_set_language(DataAction,
                                              AMD_COMGR_LANGUAGE_OPENCL_1_2);
  checkError(Status, "amd_comgr_action_info_set_language");
  Status = amd_comgr_action_info_set_isa_name(DataAction,
                                              "amdgcn-amd-amdhsa--gfx900");
  checkError(Status, "amd_comgr_action_info_set_isa_name");

  Status = amd_comgr_create_data_set(&DataSetBc);
  checkError(Status, "amd_comgr_create_data_set");
  Status = amd_comgr_do_action(AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC,
                               DataAction, DataSetIn, DataSetBc);
  checkError(Status, "amd_comgr_do_action");

  // The perf log open must have refused the symlink: the target is unchanged
  // and the symlink itself is still a symlink (not replaced by a regular file).
  checkSentinelIntact(Canary);

  struct stat St;
  if (lstat(PerfLogName, &St) != 0 || !S_ISLNK(St.st_mode)) {
    printf("FAILED: %s is no longer a symlink after perf log init\n",
           PerfLogName);
    return 1;
  }

  Status = amd_comgr_release_data(DataSource);
  checkError(Status, "amd_comgr_release_data");
  Status = amd_comgr_destroy_data_set(DataSetIn);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_destroy_data_set(DataSetBc);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_destroy_action_info(DataAction);
  checkError(Status, "amd_comgr_destroy_action_info");

  unlink(PerfLogName);
  unlink(Canary);
  rmdir(TmpDir);

  printf("time_stat_symlink_test passed\n");
  return 0;
}

#else // _WIN32

int main(void) {
  // O_NOFOLLOW has no Windows equivalent; nothing to exercise here.
  printf("time_stat_symlink_test skipped on Windows\n");
  return 0;
}

#endif
