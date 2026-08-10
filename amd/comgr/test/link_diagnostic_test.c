//===- link_diagnostic_test.c ---------------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// AMD_COMGR_ACTION_LINK_BC_TO_BC installs AMDGPUCompilerDiagnosticHandler on
/// the LLVMContext it links in, to forward LLVM's diagnostics into the Comgr
/// log. Nothing exercised that handler, because every existing link test links
/// modules that link cleanly.
///
/// Linking a module against a second copy of itself defines every symbol twice,
/// which LLVM's IR linker reports as an error diagnostic through the context's
/// handler.
///
//===----------------------------------------------------------------------===//

#include "amd_comgr.h"
#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
  char *Buf;
  long Size;
  amd_comgr_data_t DataIn1, DataIn2;
  amd_comgr_data_set_t DataSetIn, DataSetOut;
  amd_comgr_action_info_t ActionInfo;
  amd_comgr_status_t Status;

  Size = setBuf(TEST_OBJ_DIR "/source1.bc", &Buf);

  checkError(amd_comgr_create_data_set(&DataSetIn),
             "amd_comgr_create_data_set");

  // Two data objects with distinct names but identical contents: the set is a
  // set, so the same object added twice would collapse to one entry.
  checkError(amd_comgr_create_data(AMD_COMGR_DATA_KIND_BC, &DataIn1),
             "amd_comgr_create_data");
  checkError(amd_comgr_set_data(DataIn1, Size, Buf), "amd_comgr_set_data");
  checkError(amd_comgr_set_data_name(DataIn1, "source1_copy1.bc"),
             "amd_comgr_set_data_name");
  checkError(amd_comgr_data_set_add(DataSetIn, DataIn1),
             "amd_comgr_data_set_add");

  checkError(amd_comgr_create_data(AMD_COMGR_DATA_KIND_BC, &DataIn2),
             "amd_comgr_create_data");
  checkError(amd_comgr_set_data(DataIn2, Size, Buf), "amd_comgr_set_data");
  checkError(amd_comgr_set_data_name(DataIn2, "source1_copy2.bc"),
             "amd_comgr_set_data_name");
  checkError(amd_comgr_data_set_add(DataSetIn, DataIn2),
             "amd_comgr_data_set_add");

  checkError(amd_comgr_create_action_info(&ActionInfo),
             "amd_comgr_create_action_info");
  checkError(amd_comgr_action_info_set_isa_name(ActionInfo,
                                                "amdgcn-amd-amdhsa--gfx900"),
             "amd_comgr_action_info_set_isa_name");
  // Logging on, so the diagnostic the handler writes is returned in the log
  // data object rather than discarded.
  checkError(amd_comgr_action_info_set_logging(ActionInfo, true),
             "amd_comgr_action_info_set_logging");

  checkError(amd_comgr_create_data_set(&DataSetOut),
             "amd_comgr_create_data_set");

  Status = amd_comgr_do_action(AMD_COMGR_ACTION_LINK_BC_TO_BC, ActionInfo,
                               DataSetIn, DataSetOut);
  checkStatus(Status, AMD_COMGR_STATUS_ERROR,
              "linking a module against a copy of itself should fail");

  // The handler prefixes each diagnostic with its severity, and the IR linker
  // reports a duplicate definition as an error.
  checkLogs("link_diagnostic_test", DataSetOut, "ERROR:");
  checkLogs("link_diagnostic_test", DataSetOut, "multiply defined");

  checkError(amd_comgr_destroy_data_set(DataSetOut),
             "amd_comgr_destroy_data_set");
  checkError(amd_comgr_destroy_action_info(ActionInfo),
             "amd_comgr_destroy_action_info");
  checkError(amd_comgr_release_data(DataIn2), "amd_comgr_release_data");
  checkError(amd_comgr_release_data(DataIn1), "amd_comgr_release_data");
  checkError(amd_comgr_destroy_data_set(DataSetIn),
             "amd_comgr_destroy_data_set");
  free(Buf);

  printf("link_diagnostic_test passed\n");
  return 0;
}
