//===- clone_spirv_kernels_test.c -----------------------------------------===//
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

// Test kernel cloning via AMD_COMGR_ACTION_COMPILE_SPIRV_TO_RELOCATABLE
// When block sizes are specified in ActionInfo, this action automatically:
// 1. Translates SPIR-V to bitcode
// 2. Clones kernels for each specified block size
// 3. Compiles to relocatable
// This test:
// 1. Loads a SPIR-V binary
// 2. Sets multiple block sizes in ActionInfo
// 3. Calls AMD_COMGR_ACTION_COMPILE_SPIRV_TO_RELOCATABLE
// 4. Verifies that relocatable objects are produced
// 5. Links to an executable
// 6. Checks that multiple kernel variants exist

int main(int argc, char *argv[]) {
  char *BufSpirv;
  size_t SizeSpirv;
  amd_comgr_data_t DataSpirv;
  amd_comgr_data_set_t DataSetIn, DataSetReloc, DataSetExec;
  amd_comgr_action_info_t DataAction;
  amd_comgr_status_t Status;
  size_t Count;

  // Load SPIR-V binary
  SizeSpirv = setBuf(TEST_OBJ_DIR "/clone_kernels.spv", &BufSpirv);

  // Create input data set
  Status = amd_comgr_create_data_set(&DataSetIn);
  checkError(Status, "amd_comgr_create_data_set");

  // Create SPIR-V data object
  Status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_SPIRV, &DataSpirv);
  checkError(Status, "amd_comgr_create_data");
  Status = amd_comgr_set_data(DataSpirv, SizeSpirv, BufSpirv);
  checkError(Status, "amd_comgr_set_data");
  Status = amd_comgr_set_data_name(DataSpirv, "clone_kernels.spv");
  checkError(Status, "amd_comgr_set_data_name");
  Status = amd_comgr_data_set_add(DataSetIn, DataSpirv);
  checkError(Status, "amd_comgr_data_set_add");

  // Create action info
  Status = amd_comgr_create_action_info(&DataAction);
  checkError(Status, "amd_comgr_create_action_info");

  // Set ISA name (required for compilation actions)
  Status = amd_comgr_action_info_set_isa_name(DataAction,
                                              "amdgcn-amd-amdhsa--gfx900");
  checkError(Status, "amd_comgr_action_info_set_isa_name");

  // Set block sizes to clone: 1024, 512, 256, 64 (simulating warp size)
  size_t block_sizes[] = {1024, 512, 256, 64};
  size_t block_sizes_count = sizeof(block_sizes) / sizeof(block_sizes[0]);

  Status = amd_comgr_action_info_set_block_sizes(DataAction, block_sizes,
                                                 block_sizes_count);
  checkError(Status, "amd_comgr_action_info_set_block_sizes");

  // Enable device lib linking
  Status = amd_comgr_action_info_set_device_lib_linking(DataAction, true);
  checkError(Status, "amd_comgr_action_info_set_device_lib_linking");

  // Create output data set for relocatable objects
  Status = amd_comgr_create_data_set(&DataSetReloc);
  checkError(Status, "amd_comgr_create_data_set");

  // Compile SPIR-V to relocatable with automatic kernel cloning
  Status = amd_comgr_do_action(AMD_COMGR_ACTION_COMPILE_SPIRV_TO_RELOCATABLE,
                               DataAction, DataSetIn, DataSetReloc);
  checkError(Status, "amd_comgr_do_action (compile with cloning)");

  // Verify that relocatable objects were created
  Status = amd_comgr_action_data_count(DataSetReloc,
                                       AMD_COMGR_DATA_KIND_RELOCATABLE, &Count);
  checkError(Status, "amd_comgr_action_data_count");

  if (Count == 0) {
    printf("AMD_COMGR_ACTION_COMPILE_SPIRV_TO_RELOCATABLE with block sizes "
           "Failed: "
           "produced %zu relocatable objects (expected > 0)\n",
           Count);
    exit(1);
  }

  printf("Successfully produced %zu relocatable objects with kernel cloning\n",
         Count);

  // Now link the relocatable objects to an executable
  Status = amd_comgr_create_data_set(&DataSetExec);
  checkError(Status, "amd_comgr_create_data_set");

  Status = amd_comgr_do_action(AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE,
                               DataAction, DataSetReloc, DataSetExec);
  checkError(Status, "amd_comgr_do_action (link)");

  // Verify that an executable was created
  Status = amd_comgr_action_data_count(DataSetExec,
                                       AMD_COMGR_DATA_KIND_EXECUTABLE, &Count);
  checkError(Status, "amd_comgr_action_data_count");

  if (Count != 1) {
    printf("AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE Failed: "
           "produced %zu executable objects (expected 1)\n",
           Count);
    exit(1);
  }

  // Get the executable data
  amd_comgr_data_t DataExec;
  Status = amd_comgr_action_data_get_data(
      DataSetExec, AMD_COMGR_DATA_KIND_EXECUTABLE, 0, &DataExec);
  checkError(Status, "amd_comgr_action_data_get_data");

  // Verify that multiple kernel variants exist by checking symbols
  int kernel_count = 0;
  size_t mangled_count = 0;
  Status = amd_comgr_populate_mangled_names(DataExec, &mangled_count);
  checkError(Status, "amd_comgr_populate_mangled_names");

  // Look for kernel variants with different block size suffixes
  const char *expected_kernels[] = {
      "kernel1",       "kernel1.bs512", "kernel1.bs256", "kernel1.bs64",
      "kernel2",       "kernel2.bs64",  "kernel3",       "kernel3.bs512",
      "kernel3.bs256", "kernel4",       "kernel4.bs512", "kernel4.bs256",
      "kernel5"};
  size_t observed_expected_kernels[sizeof(expected_kernels) /
                                   sizeof(expected_kernels[0])];
  memset(observed_expected_kernels, 0, sizeof(observed_expected_kernels));
  size_t expected_kernel_count =
      sizeof(expected_kernels) / sizeof(expected_kernels[0]);

  // Likewise, some kernels should not be generated for certain block sizes
  // e.g. if they have amdgpu_flat_work_group_size limiting their minimum and
  // maximum block size
  const char *unexpected_kernels[] = {
      // Kernels should not have variants that match their maximum size (default
      // 1024)
      "kernel1.bs1024",
      "kernel3.bs1024",
      "kernel2.bs256",
      "kernel4.bs728",
      "kernel5.bs512",
      // Kernels should not have variants larger than the maximum size
      "kernel2.bs1024",
      "kernel2.bs512",
      "kernel4.bs1024",
      "kernel5.bs1024",
      // Kernels should not have variants smaller than the minimum size
      // FIXME: SPIR-V currently loses the minimum block size limit during
      // translation, so these variants are still generated
      /*"kernel5.bs256", "kernel3.bs64", "kernel4.bs64", "kernel5.bs64"*/
  };
  size_t unexpected_kernel_count = sizeof(unexpected_kernels) /
                                   sizeof(unexpected_kernels[0]);

  for (size_t i = 0; i < mangled_count; i++) {
    size_t name_size;
    Status = amd_comgr_get_mangled_name(DataExec, i, &name_size, NULL);
    checkError(Status, "amd_comgr_get_mangled_name (size)");

    char *name = (char *)malloc(name_size);
    Status = amd_comgr_get_mangled_name(DataExec, i, &name_size, name);
    checkError(Status, "amd_comgr_get_mangled_name");

    // Check for unexpected kernels that should not be generated
    for (size_t j = 0; j < unexpected_kernel_count; j++) {
      if (strcmp(name, unexpected_kernels[j]) == 0) {
        printf("Unexpected kernel '%s' found in executable\n", name);
        exit(1);
      }
    }

    // Check for expected kernels and count their occurrences
    int found = -1;
    for (size_t j = 0; j < expected_kernel_count; j++) {
      if (strcmp(name, expected_kernels[j]) == 0) {
        found = j;
        break;
      }
    }
    if (found != -1)
      observed_expected_kernels[found]++;
    free(name);
  }

  for (size_t j = 0; j < expected_kernel_count; j++) {
    if (observed_expected_kernels[j] == 0) {
      printf("Expected kernel '%s' not found in executable\n",
             expected_kernels[j]);
      exit(1);
    }
  }

  printf("Test PASSED!\n");

  // Cleanup
  Status = amd_comgr_release_data(DataSpirv);
  checkError(Status, "amd_comgr_release_data");
  Status = amd_comgr_release_data(DataExec);
  checkError(Status, "amd_comgr_release_data");
  Status = amd_comgr_destroy_data_set(DataSetIn);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_destroy_data_set(DataSetReloc);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_destroy_data_set(DataSetExec);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_destroy_action_info(DataAction);
  checkError(Status, "amd_comgr_destroy_action_info");

  free(BufSpirv);

  return 0;
}
