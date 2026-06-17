//===- unpackage.c --------------------------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "amd_comgr.h"
#include "common.h"

int main(int argc, char *argv[]) {
  char *PackageData;
  size_t PackageSize;

  if (argc < 5) {
    printf("Usage: %s <bc package> <triple> <arch> <bc output>\n", argv[0]);
    return -1;
  }

  const char *PackagePath = argv[1];
  const char *Triple = argv[2];
  const char *Arch = argv[3];
  const char *BitcodePath = argv[4];

  amd_comgr_data_t OnePackage;
  amd_comgr_data_set_t InputPackages;

  PackageSize = setBuf(PackagePath, &PackageData);

  amd_comgr_(create_data_set(&InputPackages));
  amd_comgr_(create_data(AMD_COMGR_DATA_KIND_PACKAGE, &OnePackage));
  amd_comgr_(set_data(OnePackage, PackageSize, PackageData));
  amd_comgr_(set_data_name(OnePackage, "package.bc"));
  amd_comgr_(data_set_add(InputPackages, OnePackage));

  amd_comgr_data_set_t OutputBitcode;
  amd_comgr_(create_data_set(&OutputBitcode));

  amd_comgr_action_info_t DataAction;
  amd_comgr_(create_action_info(&DataAction));

  amd_comgr_target_id_t PackageEntryIDs[] = {{Triple, Arch}};
  amd_comgr_(action_info_set_package_entry_ids(DataAction, PackageEntryIDs, 1));
  amd_comgr_(do_action(AMD_COMGR_ACTION_UNPACKAGE, DataAction, InputPackages,
                       OutputBitcode));

  size_t Count;
  amd_comgr_(action_data_count(OutputBitcode, AMD_COMGR_DATA_KIND_BC, &Count));

  if (Count != 1) {
    printf("AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC Failed: "
           "produced %zu BC objects (expected 1)\n",
           Count);
    exit(1);
  }

  amd_comgr_data_t OneBitcode;
  amd_comgr_(action_data_get_data(OutputBitcode, AMD_COMGR_DATA_KIND_BC, 0,
                                  &OneBitcode));

  size_t BufferSize;
  amd_comgr_(get_data(OneBitcode, &BufferSize, 0x0));
  char *Buffer = (char *)malloc(BufferSize);
  amd_comgr_(get_data(OneBitcode, &BufferSize, Buffer));

  FILE *BitcodeFile = fopen(BitcodePath, "wb");
  fwrite(Buffer, 1, BufferSize, BitcodeFile);
  fclose(BitcodeFile);

  free(Buffer);
  amd_comgr_(release_data(OneBitcode));
  amd_comgr_(release_data(OnePackage));
  amd_comgr_(destroy_action_info(DataAction));
  amd_comgr_(destroy_data_set(OutputBitcode));
  amd_comgr_(destroy_data_set(InputPackages));
  free(PackageData);

  return 0;

}
