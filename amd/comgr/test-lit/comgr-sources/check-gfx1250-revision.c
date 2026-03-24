#include "amd_comgr.h"
#include "common.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  if (argc != 2) {
    fprintf(stderr, "Usage: %s <code-object>\n", argv[0]);
    return 1;
  }

  char *Buf;
  size_t Size = setBuf(argv[1], &Buf);

  amd_comgr_data_t DataIn;
  amd_comgr_(create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &DataIn));
  amd_comgr_(set_data(DataIn, Size, Buf));

  amd_comgr_metadata_node_t RootMeta, KernelsList;
  amd_comgr_(get_data_metadata(DataIn, &RootMeta));
  amd_comgr_(metadata_lookup(RootMeta, "amdhsa.kernels", &KernelsList));

  size_t KernelsCount;
  amd_comgr_(get_metadata_list_size(KernelsList, &KernelsCount));

  for (size_t i = 0; i < KernelsCount; i++) {
    amd_comgr_metadata_node_t KernelMeta, RevisionMeta;
    amd_comgr_(index_list_metadata(KernelsList, i, &KernelMeta));

    amd_comgr_status_t Status =
        amd_comgr_metadata_lookup(KernelMeta, ".gfx1250_revision", &RevisionMeta);
    if (Status == AMD_COMGR_STATUS_SUCCESS) {
      char RevisionStr[16];
      size_t StrSize = sizeof(RevisionStr);
      amd_comgr_(get_metadata_string(RevisionMeta, &StrSize, RevisionStr));
      printf("gfx1250_revision: %s\n", RevisionStr);
      amd_comgr_(destroy_metadata(RevisionMeta));
    }

    amd_comgr_(destroy_metadata(KernelMeta));
  }

  amd_comgr_(destroy_metadata(KernelsList));
  amd_comgr_(destroy_metadata(RootMeta));
  amd_comgr_(release_data(DataIn));
  free(Buf);

  return 0;
}
