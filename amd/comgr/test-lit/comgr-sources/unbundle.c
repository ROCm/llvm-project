//===- unbundle.c ---------------------------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "amd_comgr.h"
#include "common.h"

int main(int argc, char *argv[]) {
  char *BundleData;
  size_t BundleSize;

  // Optional leading flag: print each output's public data object name. The
  // name is independent of the temporary file names Comgr uses internally.
  int PrintNames = 0;
  int Arg = 1;
  if (Arg < argc && !strcmp(argv[Arg], "--print-names")) {
    PrintNames = 1;
    ++Arg;
  }

  // One bundle plus an <arch> <output> pair per requested entry.
  size_t RemainingArgs = argc - Arg;
  if (RemainingArgs < 3 || RemainingArgs % 2 == 0) {
    printf("Usage: %s [--print-names] <bc bundle> <arch> <bc output> "
           "[<arch> <bc output>]...\n",
           argv[0]);
    return -1;
  }

  const char *BundlePath = argv[Arg];
  size_t NumArch = (RemainingArgs - 1) / 2;
  const char **AllArch = malloc(NumArch * sizeof(*AllArch));
  for (size_t I = 0; I < NumArch; ++I)
    AllArch[I] = argv[Arg + 1 + 2 * I];

  amd_comgr_data_t OneBundle;
  amd_comgr_data_set_t InputBundles;

  BundleSize = setBuf(BundlePath, &BundleData);

  amd_comgr_(create_data_set(&InputBundles));
  amd_comgr_(create_data(AMD_COMGR_DATA_KIND_BC_BUNDLE, &OneBundle));
  amd_comgr_(set_data(OneBundle, BundleSize, BundleData));
  amd_comgr_(set_data_name(OneBundle, "bundle.bc"));
  amd_comgr_(data_set_add(InputBundles, OneBundle));

  amd_comgr_data_set_t OutputBitcode;
  amd_comgr_(create_data_set(&OutputBitcode));

  amd_comgr_action_info_t DataAction;
  amd_comgr_(create_action_info(&DataAction));

  amd_comgr_(action_info_set_bundle_entry_ids(DataAction, AllArch, NumArch));
  amd_comgr_(do_action(AMD_COMGR_ACTION_UNBUNDLE, DataAction, InputBundles,
                       OutputBitcode));

  size_t Count;
  amd_comgr_(action_data_count(OutputBitcode, AMD_COMGR_DATA_KIND_BC, &Count));

  if (Count != NumArch) {
    printf("AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC Failed: "
           "produced %zu BC objects (expected %zu)\n",
           Count, NumArch);
    exit(1);
  }

  // The outputs are in the order of the bundle entry IDs.
  for (size_t I = 0; I < NumArch; ++I) {
    amd_comgr_data_t OneBitcode;
    amd_comgr_(action_data_get_data(OutputBitcode, AMD_COMGR_DATA_KIND_BC, I,
                                    &OneBitcode));

    size_t BufferSize;
    amd_comgr_(get_data(OneBitcode, &BufferSize, 0x0));
    char *Buffer = (char *)malloc(BufferSize);
    amd_comgr_(get_data(OneBitcode, &BufferSize, Buffer));

    FILE *BitcodeFile = fopen(argv[Arg + 2 + 2 * I], "wb");
    fwrite(Buffer, 1, BufferSize, BitcodeFile);
    fclose(BitcodeFile);

    if (PrintNames) {
      size_t NameSize;
      amd_comgr_(get_data_name(OneBitcode, &NameSize, NULL));
      char *Name = (char *)malloc(NameSize);
      amd_comgr_(get_data_name(OneBitcode, &NameSize, Name));
      printf("%s\n", Name);
      free(Name);
    }

    free(Buffer);
    amd_comgr_(release_data(OneBitcode));
  }
  amd_comgr_(release_data(OneBundle));
  amd_comgr_(destroy_action_info(DataAction));
  amd_comgr_(destroy_data_set(OutputBitcode));
  amd_comgr_(destroy_data_set(InputBundles));
  free(AllArch);
  free(BundleData);

  return 0;
}
