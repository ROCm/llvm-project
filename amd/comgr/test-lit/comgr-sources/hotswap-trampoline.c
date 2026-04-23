//===- hotswap-trampoline.c - Trampoline patch hotswap e2e test ------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// End-to-end test binary for trampoline patches: ds_*_2addr_stride64
/// expansion and tensor_load_to_lds multicast fix (dead/live SGPR variants).
///
/// Usage: hotswap-trampoline <asm_file> <source_isa> <target_isa>
///                           [<output_elf>]
///
//===----------------------------------------------------------------------===//

#include "amd_comgr.h"
#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
  if (argc < 4)
    fail("usage: hotswap-trampoline <asm_file> <source_isa> <target_isa> "
         "[<output_elf>]");

  const char *AsmFile = argv[1];
  const char *SourceISA = argv[2];
  const char *TargetISA = argv[3];
  const char *OutputFile = argc > 4 ? argv[4] : NULL;

  char *AsmBuf;
  size_t AsmSize = (size_t)setBuf(AsmFile, &AsmBuf);

  amd_comgr_data_t AsmData;
  amd_comgr_(create_data(AMD_COMGR_DATA_KIND_SOURCE, &AsmData));
  amd_comgr_(set_data(AsmData, AsmSize, AsmBuf));
  amd_comgr_(set_data_name(AsmData, "test_trampoline.s"));

  amd_comgr_data_set_t AsmSet, RelocSet, ExecSet;
  amd_comgr_(create_data_set(&AsmSet));
  amd_comgr_(data_set_add(AsmSet, AsmData));

  amd_comgr_action_info_t ActionInfo;
  amd_comgr_(create_action_info(&ActionInfo));
  amd_comgr_(action_info_set_isa_name(ActionInfo, SourceISA));

  amd_comgr_(create_data_set(&RelocSet));
  amd_comgr_(do_action(AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE,
                        ActionInfo, AsmSet, RelocSet));

  amd_comgr_(create_data_set(&ExecSet));
  amd_comgr_(do_action(AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE,
                        ActionInfo, RelocSet, ExecSet));

  amd_comgr_data_t ExecData;
  amd_comgr_(action_data_get_data(ExecSet, AMD_COMGR_DATA_KIND_EXECUTABLE, 0,
                                  &ExecData));

  size_t InputSize;
  amd_comgr_(get_data(ExecData, &InputSize, NULL));
  printf("INPUT_SIZE: %zu\n", InputSize);

  if (InputSize == 0)
    fail("empty executable");

  amd_comgr_data_t OutputData;
  amd_comgr_status_t Status =
      amd_comgr_hotswap_rewrite(ExecData, SourceISA, TargetISA, &OutputData);

  if (Status != AMD_COMGR_STATUS_SUCCESS)
    fail("hotswap_rewrite failed with status %d", (int)Status);

  printf("REWRITE: SUCCESS\n");

  size_t OutputSize;
  amd_comgr_(get_data(OutputData, &OutputSize, NULL));
  printf("OUTPUT_SIZE: %zu\n", OutputSize);

  if (OutputFile) {
    dumpData(OutputData, OutputFile);
    printf("DUMPED: %s\n", OutputFile);
  }

  amd_comgr_data_t Output2Data;
  Status =
      amd_comgr_hotswap_rewrite(OutputData, SourceISA, TargetISA, &Output2Data);

  if (Status != AMD_COMGR_STATUS_SUCCESS)
    fail("idempotent rewrite failed with status %d", (int)Status);

  size_t Output2Size;
  amd_comgr_(get_data(Output2Data, &Output2Size, NULL));

  if (Output2Size == OutputSize)
    printf("IDEMPOTENT: YES\n");
  else
    printf("IDEMPOTENT: NO (%zu vs %zu)\n", Output2Size, OutputSize);

  amd_comgr_(release_data(AsmData));
  amd_comgr_(release_data(ExecData));
  amd_comgr_(release_data(OutputData));
  amd_comgr_(release_data(Output2Data));
  amd_comgr_(destroy_data_set(AsmSet));
  amd_comgr_(destroy_data_set(RelocSet));
  amd_comgr_(destroy_data_set(ExecSet));
  amd_comgr_(destroy_action_info(ActionInfo));
  free(AsmBuf);

  return 0;
}
