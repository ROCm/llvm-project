//===- assemble_edge_test.c -----------------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE runs an in-process -cc1as.
/// The existing test assembles one valid file with no options, which leaves
/// most of AssemblerInvocation::createFromArgs and the whole failure path in
/// executeAssembler unexercised. This drives debug-info and -mllvm options and
/// a source that does not assemble.
///
//===----------------------------------------------------------------------===//

#include "amd_comgr.h"
#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *BadAssembly =
    "baz:\n"
    "\tthis_is_not_an_instruction v0, v1\n"
    "\ts_endpgm\n";

// Assemble Source with Options into the caller's DataSetOut, returning the
// action status. Logging is always on so the caller can inspect diagnostics.
// DataSetOut is owned by the caller; everything created here is released here.
static amd_comgr_status_t assemble(const char *Source, size_t SourceSize,
                                  const char **Options, size_t OptionsCount,
                                  amd_comgr_data_set_t DataSetOut) {
  amd_comgr_data_t DataIn;
  amd_comgr_data_set_t DataSetIn;
  amd_comgr_action_info_t ActionInfo;
  amd_comgr_status_t Status;

  checkError(amd_comgr_create_data_set(&DataSetIn),
             "amd_comgr_create_data_set");
  checkError(amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &DataIn),
             "amd_comgr_create_data");
  checkError(amd_comgr_set_data(DataIn, SourceSize, Source),
             "amd_comgr_set_data");
  checkError(amd_comgr_set_data_name(DataIn, "source.s"),
             "amd_comgr_set_data_name");
  checkError(amd_comgr_data_set_add(DataSetIn, DataIn),
             "amd_comgr_data_set_add");

  checkError(amd_comgr_create_action_info(&ActionInfo),
             "amd_comgr_create_action_info");
  checkError(amd_comgr_action_info_set_isa_name(ActionInfo,
                                                "amdgcn-amd-amdhsa--gfx900"),
             "amd_comgr_action_info_set_isa_name");
  checkError(amd_comgr_action_info_set_logging(ActionInfo, true),
             "amd_comgr_action_info_set_logging");
  if (OptionsCount)
    checkError(
        amd_comgr_action_info_set_option_list(ActionInfo, Options, OptionsCount),
        "amd_comgr_action_info_set_option_list");

  Status = amd_comgr_do_action(AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE,
                               ActionInfo, DataSetIn, DataSetOut);

  checkError(amd_comgr_destroy_action_info(ActionInfo),
             "amd_comgr_destroy_action_info");
  checkError(amd_comgr_release_data(DataIn), "amd_comgr_release_data");
  checkError(amd_comgr_destroy_data_set(DataSetIn),
             "amd_comgr_destroy_data_set");
  return Status;
}

// Each of these option sets reaches a different part of
// AssemblerInvocation::createFromArgs. -Wa, options are forwarded verbatim to
// -cc1as, which is the only way a Comgr caller can reach most of them.
static void testAssemblerOptions(const char *Source, size_t Size) {
  const char *Debug[] = {"-g"};
  const char *Dwarf4[] = {"-g", "-gdwarf-4"};
  const char *Dwarf5[] = {"-g", "-gdwarf-5"};
  const char *NoExecStack[] = {"-Wa,--noexecstack"};
  const char *NoCompress[] = {"-g", "-Wa,--compress-debug-sections=none"};
  const char *NoRelax[] = {"-Wa,-mrelax-relocations=no"};
  const char *Mllvm[] = {"-mllvm", "-asm-macro-max-nesting-depth=42"};

  struct {
    const char *Label;
    const char **Options;
    size_t Count;
  } Cases[] = {
      {"-g", Debug, 1},
      {"-gdwarf-4", Dwarf4, 2},
      {"-gdwarf-5", Dwarf5, 2},
      {"--noexecstack", NoExecStack, 1},
      {"--compress-debug-sections=none", NoCompress, 2},
      {"-mrelax-relocations=no", NoRelax, 1},
      {"-mllvm", Mllvm, 2},
  };

  for (size_t I = 0; I < sizeof(Cases) / sizeof(Cases[0]); ++I) {
    amd_comgr_data_set_t DataSetOut;
    checkError(amd_comgr_create_data_set(&DataSetOut),
               "amd_comgr_create_data_set");
    checkError(assemble(Source, Size, Cases[I].Options, Cases[I].Count,
                        DataSetOut),
               Cases[I].Label);
    checkCount(Cases[I].Label, DataSetOut, AMD_COMGR_DATA_KIND_RELOCATABLE, 1);
    checkError(amd_comgr_destroy_data_set(DataSetOut),
               "amd_comgr_destroy_data_set");
  }
}

// A source that does not assemble must fail the action, report the parser
// diagnostic in the log, and produce no relocatable.
static void testAssemblyError(void) {
  amd_comgr_data_set_t DataSetOut;
  amd_comgr_status_t Status;

  checkError(amd_comgr_create_data_set(&DataSetOut),
             "amd_comgr_create_data_set");
  Status = assemble(BadAssembly, strlen(BadAssembly), NULL, 0, DataSetOut);
  checkStatus(Status, AMD_COMGR_STATUS_ERROR,
              "assembling invalid assembly should fail");
  checkCount("assemble invalid", DataSetOut, AMD_COMGR_DATA_KIND_RELOCATABLE,
             0);
  checkLogs("assemble_edge_test", DataSetOut, "error:");

  checkError(amd_comgr_destroy_data_set(DataSetOut),
             "amd_comgr_destroy_data_set");
}

int main(int argc, char *argv[]) {
  char *Buf;
  long Size;

  Size = setBuf(TEST_OBJ_DIR "/source1.s", &Buf);

  testAssemblerOptions(Buf, Size);
  testAssemblyError();

  free(Buf);
  printf("assemble_edge_test passed\n");
  return 0;
}
