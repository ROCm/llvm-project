//===- get_data_is_name_test.c --------------------------------------------===//
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

#define MAX_ISA_NAME_SIZE 1024

typedef enum {
  none, // The feature is not supported by V2.
  off,  // The feature is supported in V2 but always disabled.
  on,   // The feature is supported in V2 but always enabled.
  any   // The feature is supported in V2 for both disabled and
        // enabled using different target names.
} feature_mode_t;

typedef struct {
  const char *IsaName;
  bool SupportedV2;
  bool SrameccSupported;
  feature_mode_t SrameccV2;
  bool XnackSupported;
  feature_mode_t XnackV2;
  bool NeedsCOV6;
} isa_features_t;

/* Features supported based on https://llvm.org/docs/AMDGPUUsage.html . */
static isa_features_t IsaFeatures[] = {
    // clang-format off
  //        ISA Name                     V2         ------ SRAMECC ------  ------- XNACK ------- -- NeedsCOV6 --
  //                                     Supported  Supported  V2          Supported  V2
  {"amdgcn-amd-amdhsa--gfx600",          true,      false,     none,       false,     none,       false},
  {"amdgcn-amd-amdhsa--gfx601",          true,      false,     none,       false,     none,       false},
  {"amdgcn-amd-amdhsa--gfx602",          true,      false,     none,       false,     none,       false},
  {"amdgcn-amd-amdhsa--gfx700",          true,      false,     none,       false,     none,       false},
  {"amdgcn-amd-amdhsa--gfx701",          true,      false,     none,       false,     none,       false},
  {"amdgcn-amd-amdhsa--gfx702",          true,      false,     none,       false,     none,       false},
  {"amdgcn-amd-amdhsa--gfx703",          true,      false,     none,       false,     none,       false},
  {"amdgcn-amd-amdhsa--gfx704",          true,      false,     none,       false,     none,       false},
  {"amdgcn-amd-amdhsa--gfx705",          true,      false,     none,       false,     none,       false},
  {"amdgcn-amd-amdhsa--gfx801",          true,      false,     none,       true,      on,         false},
  {"amdgcn-amd-amdhsa--gfx802",          true,      false,     none,       false,     none,       false},
  {"amdgcn-amd-amdhsa--gfx803",          true,      false,     none,       false,     none,       false},
  {"amdgcn-amd-amdhsa--gfx805",          true,      false,     none,       false,     none,       false},
  {"amdgcn-amd-amdhsa--gfx810",          true,      false,     none,       true,      on,         false},
  {"amdgcn-amd-amdhsa--gfx900",          true,      false,     none,       true,      any,        false},
  {"amdgcn-amd-amdhsa--gfx902",          true,      false,     none,       true,      any,        false},
  {"amdgcn-amd-amdhsa--gfx904",          true,      false,     none,       true,      any,        false},
  {"amdgcn-amd-amdhsa--gfx906",          true,      true,      off,        true,      any,        false},
  {"amdgcn-amd-amdhsa--gfx908",          false,     true,      none,       true,      none,       false},
  {"amdgcn-amd-amdhsa--gfx909",          false,     false,     none,       true,      none,       false},
  {"amdgcn-amd-amdhsa--gfx90a",          false,     true,      none,       true,      none,       false},
  {"amdgcn-amd-amdhsa--gfx90c",          true,      false,     none,       true,      off,        false},
  {"amdgcn-amd-amdhsa--gfx942",          false,     true,      none,       true,      none,       false},
  {"amdgcn-amd-amdhsa--gfx950",          false,     true,      none,       true,      none,       false},
  {"amdgcn-amd-amdhsa--gfx1010",         false,     false,     none,       true,      none,       false},
  {"amdgcn-amd-amdhsa--gfx1011",         false,     false,     none,       true,      none,       false},
  {"amdgcn-amd-amdhsa--gfx1012",         false,     false,     none,       true,      none,       false},
  {"amdgcn-amd-amdhsa--gfx1013",         false,     false,     none,       true,      none,       false},
  {"amdgcn-amd-amdhsa--gfx1030",         false,     false,     none,       false,     none,       false},
  {"amdgcn-amd-amdhsa--gfx1031",         false,     false,     none,       false,     none,       false},
  {"amdgcn-amd-amdhsa--gfx1032",         false,     false,     none,       false,     none,       false},
  {"amdgcn-amd-amdhsa--gfx1033",         false,     false,     none,       false,     none,       false},
  {"amdgcn-amd-amdhsa--gfx1034",         false,     false,     none,       false,     none,       false},
  {"amdgcn-amd-amdhsa--gfx1035",         false,     false,     none,       false,     none,       false},
  {"amdgcn-amd-amdhsa--gfx1036",         false,     false,     none,       false,     none,       false},
  {"amdgcn-amd-amdhsa--gfx1100",         false,     false,     none,       false,     none,       false},
  {"amdgcn-amd-amdhsa--gfx1101",         false,     false,     none,       false,     none,       false},
  {"amdgcn-amd-amdhsa--gfx1102",         false,     false,     none,       false,     none,       false},
  {"amdgcn-amd-amdhsa--gfx1103",         false,     false,     none,       false,     none,       false},
  {"amdgcn-amd-amdhsa--gfx1150",         false,     false,     none,       false,     none,       false},
  {"amdgcn-amd-amdhsa--gfx1151",         false,     false,     none,       false,     none,       false},
  {"amdgcn-amd-amdhsa--gfx1152",         false,     false,     none,       false,     none,       false},
  {"amdgcn-amd-amdhsa--gfx1153",         false,     false,     none,       false,     none,       false},
  {"amdgcn-amd-amdhsa--gfx1200",         false,     false,     none,       false,     none,       false},
  {"amdgcn-amd-amdhsa--gfx1201",         false,     false,     none,       false,     none,       false},

  {"amdgcn-amd-amdhsa--gfx9-generic",    true,      false,     none,       true,      any,        true},
  {"amdgcn-amd-amdhsa--gfx9-4-generic",  false,     true,      none,       true,      none,       true},
  {"amdgcn-amd-amdhsa--gfx10-1-generic", false,     false,     none,       true,      none,       true},
  {"amdgcn-amd-amdhsa--gfx10-3-generic", false,     false,     none,       false,     none,       true},
  {"amdgcn-amd-amdhsa--gfx11-generic",   false,     false,     none,       false,     none,       true},
  {"amdgcn-amd-amdhsa--gfx12-generic",   false,     false,     none,       false,     none,       true},
    // clang-format on
};

static size_t IsaFeaturesSize = sizeof(IsaFeatures) / sizeof(IsaFeatures[0]);

bool hasSubString(const char *String, const char *Sub) {
  return !strncmp(String, Sub, strlen(Sub));
}

bool getExpectedIsaName(unsigned CodeObjectVersion, const char *IsaName,
                        char *ExpectedIsaName, bool *NeedsCoV6) {
  char TokenizedIsaName[MAX_ISA_NAME_SIZE];

  strncpy(TokenizedIsaName, IsaName, MAX_ISA_NAME_SIZE);

  char *Token = strtok(TokenizedIsaName, ":");
  isa_features_t *Isa = NULL;
  for (size_t I = 0; I < IsaFeaturesSize; I++) {
    if (strncmp(Token, IsaFeatures[I].IsaName, MAX_ISA_NAME_SIZE) == 0) {
      Isa = &IsaFeatures[I];
      break;
    }
  }
  if (!Isa) {
    printf("The %s target is not supported by the test (update the "
           "isa_features table)\n",
           Token);
    exit(1);
  }

  *NeedsCoV6 = Isa->NeedsCOV6;
  strncpy(ExpectedIsaName, Isa->IsaName, MAX_ISA_NAME_SIZE);

  feature_mode_t Sramecc = any;
  feature_mode_t Xnack = any;

  Token = strtok(NULL, ":");
  while (Token != NULL) {
    if (strncmp(Token, "sramecc", strlen("sramecc")) == 0 &&
        Isa->SrameccSupported) {
      switch (Token[strlen("sramecc")]) {
      case '-':
        Sramecc = off;
        break;
      case '+':
        Sramecc = on;
        break;
      }
    }

    if (strncmp(Token, "xnack", strlen("xnack")) == 0 && Isa->XnackSupported) {
      switch (Token[strlen("xnack")]) {
      case '-':
        Xnack = off;
        break;
      case '+':
        Xnack = on;
        break;
      }
    }

    Token = strtok(NULL, ":");
  }

  switch (CodeObjectVersion) {
  case 2: {
    /* For a V2 ISA string which does not specify a feature, the code object
     * expected ISA string will have a supported feature set to ON. If the
     * feature setting does not match the default then it is not supported.
     */
    if (!Isa->SupportedV2) {
      return false;
    }
    if (Isa->SrameccSupported) {
      if (Sramecc == any) {
        Sramecc = on;
      }
      if ((Sramecc == on) != (Isa->SrameccV2 == on || Isa->SrameccV2 == any)) {
        return false;
      }
    }
    if (Isa->XnackSupported) {
      if (Xnack == any) {
        Xnack = on;
      }
      if ((Xnack == on) != (Isa->XnackV2 == on || Isa->XnackV2 == any)) {
        return false;
      }
    }
    break;
  }

  case 3: {
    /* If a supported feature is not specified in the ISA string then it will
     * be enabled in the expected isa.
     */
    if (Isa->SrameccSupported) {
      if (Sramecc == any) {
        Sramecc = on;
      }
    }
    if (Isa->XnackSupported) {
      if (Xnack == any) {
        Xnack = on;
      }
    }
    break;
  }

  case 4:
  case 5:
  case 6:
    // All ISA strings are valid.
    return true;

  default:
    printf("Code object V%u is not supported by the test (update the "
           "get_expected_isa_name)\n",
           CodeObjectVersion);
    exit(1);
  }

  strncpy(ExpectedIsaName, Isa->IsaName, MAX_ISA_NAME_SIZE);

  if (Isa->SrameccSupported && Sramecc != any) {
    strncat(ExpectedIsaName, Sramecc == on ? ":sramecc+" : ":sramecc-",
            MAX_ISA_NAME_SIZE - strlen(ExpectedIsaName));
  }

  if (Isa->XnackSupported && Xnack != any) {
    strncat(ExpectedIsaName, Xnack == on ? ":xnack+" : ":xnack-",
            MAX_ISA_NAME_SIZE - strlen(ExpectedIsaName));
  }

  return true;
}

void checkIsaName(amd_comgr_data_t Data, const char *InputIsaName,
                  const char *ExpectedIsaName) {
  size_t Size;
  char *IsaName = NULL;
  amd_comgr_status_t Status;

  Status = amd_comgr_get_data_isa_name(Data, &Size, IsaName);
  checkError(Status, "amd_comgr_get_data_isa_name");

  IsaName = malloc(Size);
  if (!IsaName) {
    printf("cannot allocate %zu bytes for isa_name\n", Size);
    exit(1);
  }

  Status = amd_comgr_get_data_isa_name(Data, &Size, IsaName);
  checkError(Status, "amd_comgr_get_data_isa_name");

  if (strcmp(IsaName, ExpectedIsaName)) {
    printf(
        "ISA name match failed: input '%s', expected '%s' but produced '%s'\n",
        InputIsaName, ExpectedIsaName, IsaName);
    exit(1);
  }

  free(IsaName);
}

void compileAndTestIsaName(const char *IsaName, const char *ExpectedIsaName,
                           const char *Options[], size_t OptionsCount) {
  char *BufSource;
  size_t SizeSource;
  amd_comgr_data_t DataSource, DataReloc, DataExec;
  amd_comgr_status_t Status;
  amd_comgr_data_set_t DataSetIn, DataSetBc, DataSetLinked, DataSetReloc,
      DataSetExec;
  amd_comgr_action_info_t DataAction;

  SizeSource = setBuf(TEST_OBJ_DIR "/shared.cl", &BufSource);

  Status = amd_comgr_create_data_set(&DataSetIn);
  checkError(Status, "amd_comgr_create_data_set");

  Status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &DataSource);
  checkError(Status, "amd_comgr_create_data");
  Status = amd_comgr_set_data(DataSource, SizeSource, BufSource);
  checkError(Status, "amd_comgr_set_data");
  Status = amd_comgr_set_data_name(DataSource, "shared.cl");
  checkError(Status, "amd_comgr_set_data_name");
  Status = amd_comgr_data_set_add(DataSetIn, DataSource);
  checkError(Status, "amd_comgr_data_set_add");

  Status = amd_comgr_create_action_info(&DataAction);
  checkError(Status, "amd_comgr_create_action_info");
  Status = amd_comgr_action_info_set_language(DataAction,
                                              AMD_COMGR_LANGUAGE_OPENCL_1_2);
  checkError(Status, "amd_comgr_action_info_set_language");
  Status = amd_comgr_action_info_set_isa_name(DataAction, IsaName);
  checkError(Status, "amd_comgr_action_info_set_isa_name");
  Status =
      amd_comgr_action_info_set_option_list(DataAction, Options, OptionsCount);
  checkError(Status, "amd_comgr_action_info_set_option_list");

  Status = amd_comgr_create_data_set(&DataSetBc);
  checkError(Status, "amd_comgr_create_data_set");
  Status = amd_comgr_do_action(AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC,
                               DataAction, DataSetIn, DataSetBc);
  checkError(Status, "amd_comgr_do_action");

  Status = amd_comgr_create_data_set(&DataSetLinked);
  checkError(Status, "amd_comgr_create_data_set");
  Status = amd_comgr_do_action(AMD_COMGR_ACTION_LINK_BC_TO_BC, DataAction,
                               DataSetBc, DataSetLinked);
  checkError(Status, "amd_comgr_do_action");

  Status = amd_comgr_create_data_set(&DataSetReloc);
  checkError(Status, "amd_comgr_create_data_set");
  Status = amd_comgr_do_action(AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE,
                               DataAction, DataSetLinked, DataSetReloc);
  checkError(Status, "amd_comgr_do_action");

  Status = amd_comgr_action_data_get_data(
      DataSetReloc, AMD_COMGR_DATA_KIND_RELOCATABLE, 0, &DataReloc);
  checkError(Status, "amd_comgr_action_data_get_data");

  Status = amd_comgr_create_data_set(&DataSetExec);
  checkError(Status, "amd_comgr_create_data_set");
  Status = amd_comgr_do_action(AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE,
                               DataAction, DataSetReloc, DataSetExec);
  checkError(Status, "amd_comgr_do_action");

  Status = amd_comgr_action_data_get_data(
      DataSetExec, AMD_COMGR_DATA_KIND_EXECUTABLE, 0, &DataExec);
  checkError(Status, "amd_comgr_action_data_get_data");

  checkIsaName(DataReloc, IsaName, ExpectedIsaName);
  checkIsaName(DataExec, IsaName, ExpectedIsaName);
  printf("ISA name matched %s -> %s\n", IsaName, ExpectedIsaName);

  Status = amd_comgr_release_data(DataSource);
  checkError(Status, "amd_comgr_release_data");
  Status = amd_comgr_release_data(DataReloc);
  checkError(Status, "amd_comgr_release_data");
  Status = amd_comgr_release_data(DataExec);
  checkError(Status, "amd_comgr_release_data");
  Status = amd_comgr_destroy_data_set(DataSetIn);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_destroy_data_set(DataSetBc);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_destroy_data_set(DataSetLinked);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_destroy_data_set(DataSetReloc);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_destroy_data_set(DataSetExec);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_destroy_action_info(DataAction);
  checkError(Status, "amd_comgr_destroy_action_info");
  free(BufSource);
}

void testIsaName(char *Name, const char *Features) {
  char IsaName[MAX_ISA_NAME_SIZE];
  char ExpectedIsaName[MAX_ISA_NAME_SIZE];

  strncpy(IsaName, Name, MAX_ISA_NAME_SIZE);
  strncat(IsaName, Features, MAX_ISA_NAME_SIZE - 1);

  const char *V4Options[] = {"-mcode-object-version=4"};
  size_t V4OptionsCount = sizeof(V4Options) / sizeof(V4Options[0]);

  const char *V6Options[] = {"-mcode-object-version=6"};
  size_t V6OptionsCount = sizeof(V6Options) / sizeof(V6Options[0]);

  // Test object code v6 so generic targets are available.
  bool NeedsCOV6;
  if (getExpectedIsaName(6, IsaName, ExpectedIsaName, &NeedsCOV6)) {
    if (NeedsCOV6) {
      printf("V6 : ");
      compileAndTestIsaName(IsaName, IsaName, V6Options, V6OptionsCount);
    } else {
      printf("V4 : ");
      compileAndTestIsaName(IsaName, IsaName, V4Options, V4OptionsCount);
    }
  }
}

int main(int argc, char *argv[]) {
  size_t IsaCount;
  amd_comgr_status_t Status;

  Status = amd_comgr_get_isa_count(&IsaCount);
  checkError(Status, "amd_comgr_get_isa_count");

  for (size_t I = 0; I < IsaCount; I++) {
    const char *Name;
    char IsaName[MAX_ISA_NAME_SIZE];

    Status = amd_comgr_get_isa_name(I, &Name);
    checkError(Status, "amd_comgr_get_isa_name");

    strncpy(IsaName, Name, MAX_ISA_NAME_SIZE);

    testIsaName(IsaName, "");

    for (size_t I = 0; I < IsaFeaturesSize; I++) {
      if (strncmp(IsaName, IsaFeatures[I].IsaName, MAX_ISA_NAME_SIZE) == 0) {

        if (IsaFeatures[I].SrameccSupported) {
          testIsaName(IsaName, ":sramecc+");
          testIsaName(IsaName, ":sramecc-");
        }

        if (IsaFeatures[I].XnackSupported) {
          testIsaName(IsaName, ":xnack+");
          testIsaName(IsaName, ":xnack-");
        }

        if (IsaFeatures[I].SrameccSupported && IsaFeatures[I].XnackSupported) {
          testIsaName(IsaName, ":sramecc+:xnack+");
          testIsaName(IsaName, ":sramecc+:xnack-");
          testIsaName(IsaName, ":sramecc-:xnack+");
          testIsaName(IsaName, ":sramecc-:xnack-");
        }

        break;
      }
    }
  }

  return 0;
}
