//===- hotswap-transpile.c - Hotswap transpile test driver ---------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Driver for amd_comgr_hotswap_transpile_with_options_v2, the hotswap-backed
// entry point that reports typed cache/proof metadata.
//
// Mirrors the call/return shape of hotswap-rewrite.c for the validation
// paths so the two entry points stay in lockstep at the comgr boundary.
// The "real HSACO" path (positional invocation with a code-object file) is
// gated on the caller supplying a known-good HSACO; in that case it just
// asserts that the call returns SUCCESS and emits a non-empty output.
//
// The hotswap pipeline uses in-process codegen/linking. The real-HSACO lit
// path checks that the public API produces a target executable and structured
// result metadata.
//
//===----------------------------------------------------------------------===//

#include "amd_comgr.h"
#include "common.h"

#include <assert.h>
#include <stddef.h>

static const char *lookup_status_name(
    amd_comgr_hotswap_cache_lookup_status_t Status) {
  switch (Status) {
  case AMD_COMGR_HOTSWAP_CACHE_LOOKUP_DISABLED:
    return "disabled";
  case AMD_COMGR_HOTSWAP_CACHE_LOOKUP_BYPASSED:
    return "bypassed";
  case AMD_COMGR_HOTSWAP_CACHE_LOOKUP_MISS:
    return "miss";
  case AMD_COMGR_HOTSWAP_CACHE_LOOKUP_HIT:
    return "hit";
  case AMD_COMGR_HOTSWAP_CACHE_LOOKUP_INVALID:
    return "invalid";
  }
  return "invalid";
}

static const char *write_status_name(
    amd_comgr_hotswap_cache_write_status_t Status) {
  switch (Status) {
  case AMD_COMGR_HOTSWAP_CACHE_WRITE_NOT_ATTEMPTED:
    return "not_attempted";
  case AMD_COMGR_HOTSWAP_CACHE_WRITE_SUCCESS:
    return "success";
  case AMD_COMGR_HOTSWAP_CACHE_WRITE_FAILED:
    return "failed";
  }
  return "failed";
}

// Receives a temporary result string owned by with_result_string.
typedef void (*result_string_callback_t)(const char *Value, size_t Size,
                                         void *UserData);

// Fetch Field, call Callback with its temporary string, then free it.
static void with_result_string(
    amd_comgr_hotswap_transpile_result_t Result,
    amd_comgr_hotswap_transpile_result_string_t Field,
    result_string_callback_t Callback, void *UserData) {
  assert(Callback && "result string callback must be non-null");
  size_t Size = 0;
  amd_comgr_(hotswap_transpile_result_get_string(Result, Field, &Size, NULL));
  if (Size == 0)
    fail("hotswap result string field %d returned zero size", (int)Field);
  char *Buffer = (char *)malloc(Size);
  if (Buffer == NULL)
    fail("malloc(%zu) for result string field %d failed", Size, (int)Field);
  amd_comgr_(hotswap_transpile_result_get_string(Result, Field, &Size, Buffer));
  Callback(Buffer, Size, UserData);
  free(Buffer);
}

// Print a temporary result string passed by with_result_string.
static void print_result_string(const char *Value, size_t Size, void *UserData) {
  assert(Size > 0 && "result string size includes terminating NUL");
  FILE *Output = (FILE *)UserData;
  assert(Output && "result string output stream must be non-null");
  if (fwrite(Value, 1, Size - 1, Output) != Size - 1)
    fail("short write while printing result string");
}

static void print_result_if_present(amd_comgr_hotswap_transpile_result_t Result) {
  if (!Result.handle)
    return;

  bool Success = false;
  bool CacheHit = false;
  int64_t Lifted = 0;
  int64_t Total = 0;
  amd_comgr_hotswap_cache_lookup_status_t Lookup =
      AMD_COMGR_HOTSWAP_CACHE_LOOKUP_DISABLED;
  amd_comgr_hotswap_cache_write_status_t Write =
      AMD_COMGR_HOTSWAP_CACHE_WRITE_NOT_ATTEMPTED;

  amd_comgr_(hotswap_transpile_result_get_info(
      Result, AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_SUCCESS, &Success));
  amd_comgr_(hotswap_transpile_result_get_info(
      Result, AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_CACHE_HIT, &CacheHit));
  amd_comgr_(hotswap_transpile_result_get_info(
      Result, AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_CACHE_LOOKUP, &Lookup));
  amd_comgr_(hotswap_transpile_result_get_info(
      Result, AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_CACHE_WRITE, &Write));
  amd_comgr_(hotswap_transpile_result_get_info(
      Result, AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_LIFTED_COUNT, &Lifted));
  amd_comgr_(hotswap_transpile_result_get_info(
      Result, AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_TOTAL_COUNT, &Total));
  int64_t ScaledFactor = 1;
  amd_comgr_(hotswap_transpile_result_get_info(
      Result, AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_SCALED_DISPATCH_FACTOR,
      &ScaledFactor));

  printf("RESULT_INFO: success=%d cache_hit=%d cache_lookup=%s cache_write=%s "
         "source_gfx=",
         Success ? 1 : 0, CacheHit ? 1 : 0, lookup_status_name(Lookup),
         write_status_name(Write));
  with_result_string(Result, AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_SOURCE_GFX,
                     print_result_string, stdout);
  printf(" target_gfx=");
  with_result_string(Result, AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_TARGET_GFX,
                     print_result_string, stdout);
  printf(" kernel_name=");
  with_result_string(Result, AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_KERNEL_NAME,
                     print_result_string, stdout);
  printf(" lifted=%lld total=%lld scaled_dispatch_factor=%lld fail_reason=",
         (long long)Lifted, (long long)Total, (long long)ScaledFactor);
  with_result_string(Result, AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_FAIL_REASON,
                     print_result_string, stdout);
  printf(" cache_key=");
  with_result_string(Result, AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_CACHE_KEY,
                     print_result_string, stdout);
  printf("\n");
}

static int parse_opt_level_arg(const char *Arg) {
  if (Arg[0] != '-' || Arg[1] != 'O' || Arg[2] < '0' || Arg[2] > '3' ||
      Arg[3] != '\0')
    return -1;
  return Arg[2] - '0';
}

int main(int argc, char *argv[]) {
  // No args -> NULL-pointer validation. Mirrors hotswap-rewrite.c's first
  // mode so the two entry points have the same negative-test surface.
  if (argc < 2) {
    amd_comgr_data_t dummy_output;
    amd_comgr_data_t dummy_input = {0};
    amd_comgr_status_t Status =
        amd_comgr_hotswap_transpile(dummy_input, NULL, NULL, &dummy_output);
    if (Status != AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT)
      fail("transpile with NULL args: expected INVALID_ARGUMENT, got %d",
           (int)Status);
    printf("NULL_ARGS: INVALID_ARGUMENT\n");
    return 0;
  }

  if (argc < 4)
    fail("usage: hotswap-transpile <elf_file> <source_isa> <target_isa> "
         "[--zero-size|--wrong-kind|--use-options-api|--use-plain-api|"
         "--test-bad-options-version|--test-null-kernel-name-option|"
         "--test-empty-kernel-name-option|--test-invalid-opt-level] "
         "[--output=<path>] [-O0|-O1|-O2|-O3]");

  const char *ElfFile = argv[1];
  const char *SourceISA = argv[2];
  const char *TargetISA = argv[3];
  int ZeroSize = 0;
  int WrongKind = 0;
  int UseOptionsApi = 0;
  int BadOptionsVersion = 0;
  int TestNullKernelNameOption = 0;
  int TestEmptyKernelNameOption = 0;
  int TestInvalidOptLevel = 0;
  int OptLevel = -1;
  // Optional path to dump the transpiled bytes to. lit tests use this to
  // hand the output to llvm-readelf / llvm-objdump for ISA-level smoke
  // checks; the validation paths leave it NULL and only inspect stdout.
  const char *OutputPath = NULL;
  static const char ZeroSizeOpt[] = "--zero-size";
  static const char WrongKindOpt[] = "--wrong-kind";
  static const char UseOptionsApiOpt[] = "--use-options-api";
  // The plain entry point takes no result object, so it is the only way to
  // reach the "caller cannot discover a scaled dispatch" refusal.
  static const char UsePlainApiOpt[] = "--use-plain-api";
  static const char BadOptionsVersionOpt[] = "--test-bad-options-version";
  static const char TestNullKernelNameOpt[] = "--test-null-kernel-name-option";
  static const char TestEmptyKernelNameOpt[] = "--test-empty-kernel-name-option";
  static const char TestInvalidOptLevelOpt[] = "--test-invalid-opt-level";
  static const char OutputPrefix[] = "--output=";
  int UsePlainApi = 0;
  for (int i = 4; i < argc; i++) {
    int ParsedOptLevel = parse_opt_level_arg(argv[i]);
    if (strncmp(argv[i], ZeroSizeOpt, sizeof(ZeroSizeOpt)) == 0)
      ZeroSize = 1;
    else if (strncmp(argv[i], WrongKindOpt, sizeof(WrongKindOpt)) == 0)
      WrongKind = 1;
    else if (strncmp(argv[i], UseOptionsApiOpt, sizeof(UseOptionsApiOpt)) == 0)
      UseOptionsApi = 1;
    else if (strncmp(argv[i], UsePlainApiOpt, sizeof(UsePlainApiOpt)) == 0)
      UsePlainApi = 1;
    else if (strncmp(argv[i], BadOptionsVersionOpt,
                     sizeof(BadOptionsVersionOpt)) == 0)
      BadOptionsVersion = 1;
    else if (strncmp(argv[i], TestNullKernelNameOpt,
                     sizeof(TestNullKernelNameOpt)) == 0)
      TestNullKernelNameOption = 1;
    else if (strncmp(argv[i], TestEmptyKernelNameOpt,
                     sizeof(TestEmptyKernelNameOpt)) == 0)
      TestEmptyKernelNameOption = 1;
    else if (strncmp(argv[i], TestInvalidOptLevelOpt,
                     sizeof(TestInvalidOptLevelOpt)) == 0)
      TestInvalidOptLevel = 1;
    else if (strncmp(argv[i], OutputPrefix, sizeof(OutputPrefix) - 1) == 0)
      OutputPath = argv[i] + sizeof(OutputPrefix) - 1;
    else if (ParsedOptLevel >= 0)
      OptLevel = ParsedOptLevel;
    else
      fail("unknown option: %s", argv[i]);
  }

  char *ElfBuf;
  size_t ElfSize = (size_t)setBuf(ElfFile, &ElfBuf);

  amd_comgr_data_t InputData;
  // --wrong-kind: feed BC instead of EXECUTABLE. Exercises the data-kind
  // gate in amd_comgr_hotswap_transpile_with_options_v2, which mirrors the
  // gate in the byte-level rewriter.
  amd_comgr_data_kind_t Kind =
      WrongKind ? AMD_COMGR_DATA_KIND_BC : AMD_COMGR_DATA_KIND_EXECUTABLE;
  amd_comgr_(create_data(Kind, &InputData));
  if (!ZeroSize)
    amd_comgr_(set_data(InputData, ElfSize, ElfBuf));

  amd_comgr_hotswap_transpile_options_v2_t OptionsV2;
  memset(&OptionsV2, 0, sizeof(OptionsV2));
  OptionsV2.cache_directory = getenv("HSA_HOTSWAP_CACHE_DIR");
  OptionsV2.cache_skip_kernels = getenv("HSA_HOTSWAP_CACHE_SKIP_KERNELS");
  OptionsV2.hotswap_rules_path = getenv("HSA_HOTSWAP_RULES");
  OptionsV2.version =
      BadOptionsVersion ? 999 : AMD_COMGR_HOTSWAP_TRANSPILE_OPTIONS_VERSION_2;
  OptionsV2.kernel_name = getenv("HSA_HOTSWAP_TRANSLATE_KERNEL");
  if (OptionsV2.kernel_name && OptionsV2.kernel_name[0] != '\0')
    OptionsV2.flags |= AMD_COMGR_HOTSWAP_TRANSPILE_OPTIONS_V2_USE_KERNEL_NAME;
  if (TestNullKernelNameOption) {
    OptionsV2.flags |= AMD_COMGR_HOTSWAP_TRANSPILE_OPTIONS_V2_USE_KERNEL_NAME;
    OptionsV2.kernel_name = NULL;
  }
  if (TestEmptyKernelNameOption) {
    OptionsV2.flags |= AMD_COMGR_HOTSWAP_TRANSPILE_OPTIONS_V2_USE_KERNEL_NAME;
    OptionsV2.kernel_name = "";
  }
  if (OptLevel >= 0) {
    OptionsV2.flags |= AMD_COMGR_HOTSWAP_TRANSPILE_OPTIONS_V2_USE_OPT_LEVEL;
    OptionsV2.opt_level = (uint32_t)OptLevel;
  }
  if (TestInvalidOptLevel) {
    OptionsV2.flags |= AMD_COMGR_HOTSWAP_TRANSPILE_OPTIONS_V2_USE_OPT_LEVEL;
    OptionsV2.opt_level = 4;
  }
  if (getenv("HSA_HOTSWAP_CACHE_DISABLE"))
    OptionsV2.flags |= AMD_COMGR_HOTSWAP_TRANSPILE_OPTIONS_V2_CACHE_DISABLE;
  if (getenv("HSA_HOTSWAP_CACHE_READONLY"))
    OptionsV2.flags |= AMD_COMGR_HOTSWAP_TRANSPILE_OPTIONS_V2_CACHE_READONLY;
  if (getenv("HSA_HOTSWAP_STRICT"))
    OptionsV2.flags |= AMD_COMGR_HOTSWAP_TRANSPILE_OPTIONS_V2_STRICT;
  if (getenv("HSA_HOTSWAP_ASSUME_HIP_GLOBAL_OFFSET_ZERO"))
    OptionsV2.flags |=
        AMD_COMGR_HOTSWAP_TRANSPILE_OPTIONS_V2_ASSUME_HIP_GLOBAL_OFFSET_ZERO;

  amd_comgr_data_t OutputData = {0};
  amd_comgr_hotswap_transpile_result_t ResultData = {0};
  amd_comgr_status_t Status;
  if (UsePlainApi) {
    Status = amd_comgr_hotswap_transpile(InputData, SourceISA, TargetISA,
                                         &OutputData);
  } else if (UseOptionsApi) {
    amd_comgr_hotswap_transpile_options_t Options;
    memset(&Options, 0, sizeof(Options));
    Options.size = sizeof(Options);
    Options.cache_directory = OptionsV2.cache_directory;
    Options.cache_skip_kernels = OptionsV2.cache_skip_kernels;
    Options.hotswap_rules_path = OptionsV2.hotswap_rules_path;
    if (OptionsV2.flags & AMD_COMGR_HOTSWAP_TRANSPILE_OPTIONS_V2_CACHE_DISABLE)
      Options.flags |= AMD_COMGR_HOTSWAP_TRANSPILE_OPTIONS_CACHE_DISABLE;
    if (OptionsV2.flags & AMD_COMGR_HOTSWAP_TRANSPILE_OPTIONS_V2_CACHE_READONLY)
      Options.flags |= AMD_COMGR_HOTSWAP_TRANSPILE_OPTIONS_CACHE_READONLY;
    if (OptionsV2.flags & AMD_COMGR_HOTSWAP_TRANSPILE_OPTIONS_V2_STRICT)
      Options.flags |= AMD_COMGR_HOTSWAP_TRANSPILE_OPTIONS_STRICT;
    if (OptionsV2.flags &
        AMD_COMGR_HOTSWAP_TRANSPILE_OPTIONS_V2_ASSUME_HIP_GLOBAL_OFFSET_ZERO)
      Options.flags |=
          AMD_COMGR_HOTSWAP_TRANSPILE_OPTIONS_ASSUME_HIP_GLOBAL_OFFSET_ZERO;
    Status = amd_comgr_hotswap_transpile_with_options(
        InputData, SourceISA, TargetISA, &Options, &OutputData, &ResultData);
  } else {
    Status = amd_comgr_hotswap_transpile_with_options_v2(
        InputData, SourceISA, TargetISA, &OptionsV2, &OutputData, &ResultData);
  }

  if (Status == AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT) {
    printf("RESULT: INVALID_ARGUMENT\n");
    print_result_if_present(ResultData);
    if (ResultData.handle)
      amd_comgr_(destroy_hotswap_transpile_result(ResultData));
    amd_comgr_(release_data(InputData));
    free(ElfBuf);
    return 0;
  }

  // The hotswap pipeline reports per-kernel failures as
  // AMD_COMGR_STATUS_ERROR (e.g. unsupported instruction, missing kernels,
  // backend compile failure). Surface this as a distinct line so lit can
  // assert it independently from INVALID_ARGUMENT.
  if (Status == AMD_COMGR_STATUS_ERROR) {
    printf("RESULT: ERROR\n");
    print_result_if_present(ResultData);
    if (ResultData.handle)
      amd_comgr_(destroy_hotswap_transpile_result(ResultData));
    amd_comgr_(release_data(InputData));
    free(ElfBuf);
    return 0;
  }

  if (Status != AMD_COMGR_STATUS_SUCCESS)
    fail("unexpected error status %d", (int)Status);

  size_t OutSize = 0;
  amd_comgr_(get_data(OutputData, &OutSize, NULL));

  if (OutSize == 0)
    fail("transpile produced empty output on supposedly successful call");

  if (OutputPath != NULL) {
    char *OutBuf = (char *)malloc(OutSize);
    if (OutBuf == NULL)
      fail("malloc(%zu) for output buffer failed", OutSize);
    amd_comgr_(get_data(OutputData, &OutSize, OutBuf));
    FILE *OutFile = fopen(OutputPath, "wb");
    if (OutFile == NULL)
      fail("cannot open output path '%s' for writing", OutputPath);
    if (fwrite(OutBuf, 1, OutSize, OutFile) != OutSize)
      fail("short write to '%s'", OutputPath);
    fclose(OutFile);
    free(OutBuf);
  }

  print_result_if_present(ResultData);
  if (ResultData.handle)
    amd_comgr_(destroy_hotswap_transpile_result(ResultData));
  amd_comgr_(release_data(OutputData));
  amd_comgr_(release_data(InputData));
  free(ElfBuf);

  printf("RESULT: SUCCESS bytes=%zu\n", OutSize);
  return 0;
}
