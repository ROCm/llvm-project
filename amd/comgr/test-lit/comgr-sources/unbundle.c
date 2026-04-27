//===- unbundle.c ---------------------------------------------------------===//
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

static const char *extensionForKind(amd_comgr_data_kind_t Kind) {
  switch (Kind) {
  case AMD_COMGR_DATA_KIND_BC:
    return "bc";
  case AMD_COMGR_DATA_KIND_SPIRV:
    return "spv";
  case AMD_COMGR_DATA_KIND_EXECUTABLE:
    return "o";
  case AMD_COMGR_DATA_KIND_AR:
    return "a";
  default:
    return "bin";
  }
}

int main(int argc, char *argv[]) {
  if (argc < 4) {
    printf("Usage: %s <bundle> <output-prefix> <entry-id> [<entry-id>...]\n",
           argv[0]);
    return -1;
  }

  const char *BundlePath = argv[1];
  const char *OutputPrefix = argv[2];
  const char **EntryIDs = (const char **)&argv[3];
  size_t EntryCount = (size_t)(argc - 3);

  char *BundleData;
  size_t BundleSize = setBuf(BundlePath, &BundleData);

  amd_comgr_data_t Bundle;
  amd_comgr_data_set_t InputBundles;
  amd_comgr_(create_data_set(&InputBundles));
  amd_comgr_(create_data(AMD_COMGR_DATA_KIND_BUNDLE, &Bundle));
  amd_comgr_(set_data(Bundle, BundleSize, BundleData));
  amd_comgr_(set_data_name(Bundle, "bundle"));
  amd_comgr_(data_set_add(InputBundles, Bundle));

  amd_comgr_data_set_t Outputs;
  amd_comgr_(create_data_set(&Outputs));

  amd_comgr_action_info_t DataAction;
  amd_comgr_(create_action_info(&DataAction));
  amd_comgr_(action_info_set_bundle_entry_ids(DataAction, EntryIDs, EntryCount));

  amd_comgr_(do_action(AMD_COMGR_ACTION_UNBUNDLE, DataAction, InputBundles,
                       Outputs));

  // Walk every kind that the unbundle action may produce and write each
  // output to <prefix>-<entry-id>.<ext>. The entry ID is recovered from the
  // data object's name, which the unbundle action sets to
  // "<input-name>-<entry-id>.<ext>".
  static const amd_comgr_data_kind_t Kinds[] = {
      AMD_COMGR_DATA_KIND_BC,
      AMD_COMGR_DATA_KIND_SPIRV,
      AMD_COMGR_DATA_KIND_EXECUTABLE,
      AMD_COMGR_DATA_KIND_AR,
      AMD_COMGR_DATA_KIND_BYTES,
  };
  size_t KindsCount = sizeof(Kinds) / sizeof(Kinds[0]);

  for (size_t K = 0; K < KindsCount; ++K) {
    size_t Count = 0;
    amd_comgr_(action_data_count(Outputs, Kinds[K], &Count));

    for (size_t I = 0; I < Count; ++I) {
      amd_comgr_data_t Out;
      amd_comgr_(action_data_get_data(Outputs, Kinds[K], I, &Out));

      // Recover entry ID from data name: "bundle-<entry-id>.<ext>".
      size_t NameSize = 0;
      amd_comgr_(get_data_name(Out, &NameSize, NULL));
      char *Name = (char *)malloc(NameSize);
      amd_comgr_(get_data_name(Out, &NameSize, Name));

      const char *EntryStart = strchr(Name, '-');
      if (!EntryStart) {
        printf("Unexpected output name: %s\n", Name);
        exit(1);
      }
      EntryStart += 1; // skip the '-'
      const char *EntryEnd = strrchr(EntryStart, '.');
      if (!EntryEnd) {
        printf("Unexpected output name: %s\n", Name);
        exit(1);
      }
      size_t EntryLen = (size_t)(EntryEnd - EntryStart);

      const char *Ext = extensionForKind(Kinds[K]);
      size_t PathLen = strlen(OutputPrefix) + 1 + EntryLen + 1 + strlen(Ext) + 1;
      char *OutPath = (char *)malloc(PathLen);
      snprintf(OutPath, PathLen, "%s-%.*s.%s", OutputPrefix, (int)EntryLen,
               EntryStart, Ext);

      size_t BufferSize = 0;
      amd_comgr_(get_data(Out, &BufferSize, NULL));
      char *Buffer = (char *)malloc(BufferSize);
      amd_comgr_(get_data(Out, &BufferSize, Buffer));

      FILE *F = fopen(OutPath, "wb");
      fwrite(Buffer, 1, BufferSize, F);
      fclose(F);

      free(Buffer);
      free(OutPath);
      free(Name);
      amd_comgr_(release_data(Out));
    }
  }

  amd_comgr_(release_data(Bundle));
  amd_comgr_(destroy_action_info(DataAction));
  amd_comgr_(destroy_data_set(Outputs));
  amd_comgr_(destroy_data_set(InputBundles));
  free(BundleData);
  return 0;
}
