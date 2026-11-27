//===- assemble-llvm-options.c --------------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "amd_comgr.h"
#include "common.h"

static void assemble(amd_comgr_action_info_t Action, amd_comgr_data_set_t Input,
                     const char *Options[], size_t NumOptions,
                     amd_comgr_status_t ExpectedStatus,
                     const char *ExpectedDiagnostic) {
  amd_comgr_(action_info_set_option_list(Action, Options, NumOptions));
  amd_comgr_data_set_t Output;
  amd_comgr_(create_data_set(&Output));
  amd_comgr_status_t Status = amd_comgr_do_action(
      AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE, Action, Input, Output);
  if (Status != ExpectedStatus)
    fail("assembly returned %d, expected %d", Status, ExpectedStatus);

  size_t Count;
  amd_comgr_(
      action_data_count(Output, AMD_COMGR_DATA_KIND_RELOCATABLE, &Count));
  if (Count != (ExpectedStatus == AMD_COMGR_STATUS_SUCCESS ? 1 : 0))
    fail("unexpected number of relocatable objects: %zu", Count);

  if (ExpectedDiagnostic) {
    amd_comgr_data_t Log;
    amd_comgr_(action_data_get_data(Output, AMD_COMGR_DATA_KIND_LOG, 0, &Log));
    size_t Size;
    amd_comgr_(get_data(Log, &Size, NULL));
    char *Text = malloc(Size + 1);
    if (!Text)
      fail("malloc");
    amd_comgr_(get_data(Log, &Size, Text));
    Text[Size] = '\0';
    if (!strstr(Text, ExpectedDiagnostic))
      fail("missing diagnostic '%s' in log:\n%s", ExpectedDiagnostic, Text);
    free(Text);
    amd_comgr_(release_data(Log));
  }

  amd_comgr_(destroy_data_set(Output));
}

int main(void) {
  const char Source[] = ".text\n"
                        ".macro inner\n"
                        "s_endpgm\n"
                        ".endm\n"
                        ".macro outer\n"
                        "inner\n"
                        ".endm\n"
                        "outer\n";
  amd_comgr_data_t Data;
  amd_comgr_(create_data(AMD_COMGR_DATA_KIND_SOURCE, &Data));
  amd_comgr_(set_data(Data, sizeof(Source) - 1, Source));
  amd_comgr_(set_data_name(Data, "source.s"));
  amd_comgr_data_set_t Input;
  amd_comgr_(create_data_set(&Input));
  amd_comgr_(data_set_add(Input, Data));
  amd_comgr_(release_data(Data));

  amd_comgr_action_info_t Action;
  amd_comgr_(create_action_info(&Action));
  amd_comgr_(action_info_set_isa_name(Action, "amdgcn-amd-amdhsa--gfx900"));
  amd_comgr_(action_info_set_logging(Action, true));

  // CLR passes this library option during runtime compilation.
  const char *Internalize[] = {"-mllvm", "-amdgpu-internalize-symbols"};
  assemble(Action, Input, Internalize, 2, AMD_COMGR_STATUS_SUCCESS, NULL);

  // The assembler must use the parsed context, including earlier options.
  const char *Depth[] = {"-mllvm", "-asm-macro-max-nesting-depth=1", "-mllvm",
                         "-amdgpu-internalize-symbols"};
  const char *DepthDiagnostic =
      "macros cannot be nested more than 1 levels deep";
  assemble(Action, Input, Depth, 4, AMD_COMGR_STATUS_ERROR, DepthDiagnostic);

  // Parse options and their separate values together.
  const char *SeparateDepth[] = {"-mllvm", "-asm-macro-max-nesting-depth",
                                 "-mllvm", "1",
                                 "-mllvm", "-amdgpu-internalize-symbols"};
  assemble(Action, Input, SeparateDepth, 6, AMD_COMGR_STATUS_ERROR,
           DepthDiagnostic);

  // A later action must start with the default library options.
  assemble(Action, Input, NULL, 0, AMD_COMGR_STATUS_SUCCESS, NULL);

  // Invalid options must return a diagnostic instead of terminating the caller.
  const char *Invalid[] = {"-mllvm", "-comgr-invalid-llvm-option"};
  assemble(Action, Input, Invalid, 2, AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT,
           "Unknown command line argument '-comgr-invalid-llvm-option'");
  assemble(Action, Input, Internalize, 2, AMD_COMGR_STATUS_SUCCESS, NULL);

  amd_comgr_(destroy_action_info(Action));
  amd_comgr_(destroy_data_set(Input));
  return 0;
}
