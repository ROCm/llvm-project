#include "amd_comgr.h"
#include "common.h"

int main(void) {
  amd_comgr_data_t Data, Retained, Invalid = {0};
  amd_comgr_data_set_t Set;
  const char Payload[] = {'a', '\0', 'b', 'c'};
  const char Replacement[] = {'x', 'y'};
  const char *Pointer = NULL, *Again = NULL;
  size_t Size = 123, AgainSize = 0;

  checkStatus(amd_comgr_get_data_pointer(Invalid, &Size, &Pointer),
              AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT, "invalid handle");
  if (Size != 123 || Pointer != NULL)
    fail("error modified outputs");

  checkError(amd_comgr_create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &Data),
             "create data");
  checkStatus(amd_comgr_get_data_pointer(Data, &Size, &Pointer),
              AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT, "unset contents");
  checkError(amd_comgr_set_data(Data, sizeof(Payload), Payload), "set data");
  checkError(amd_comgr_set_data_name(Data, "payload"), "set name");
  checkStatus(amd_comgr_get_data_pointer(Data, NULL, &Pointer),
              AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT, "null size");
  checkStatus(amd_comgr_get_data_pointer(Data, &Size, NULL),
              AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT, "null bytes");
  checkError(amd_comgr_get_data_pointer(Data, &Size, &Pointer), "borrow data");
  if (Size != sizeof(Payload) || memcmp(Pointer, Payload, Size))
    fail("incorrect view");
  checkError(amd_comgr_get_data_pointer(Data, &AgainSize, &Again),
             "borrow again");
  if (Pointer != Again || Size != AgainSize)
    fail("view does not share storage");

  checkError(amd_comgr_create_data_set(&Set), "create set");
  checkError(amd_comgr_data_set_add(Set, Data), "add data");
  checkError(amd_comgr_action_data_get_data(Set, AMD_COMGR_DATA_KIND_EXECUTABLE,
                                            0, &Retained),
             "retain data");
  checkError(amd_comgr_release_data(Data), "release original handle");
  checkError(amd_comgr_destroy_data_set(Set), "release set");
  checkError(amd_comgr_set_data_name(Retained, "renamed"), "rename data");
  checkError(amd_comgr_get_data_pointer(Retained, &AgainSize, &Again),
             "borrow retained data");
  if (Pointer != Again || AgainSize != sizeof(Payload) ||
      memcmp(Again, Payload, AgainSize))
    fail("retained view changed");

  checkError(amd_comgr_set_data(Retained, sizeof(Replacement), Replacement),
             "replace contents");
  checkError(amd_comgr_get_data_pointer(Retained, &Size, &Pointer),
             "borrow replacement");
  if (Size != sizeof(Replacement) || memcmp(Pointer, Replacement, Size))
    fail("replacement view differs");
  checkError(amd_comgr_release_data(Retained), "release retained handle");
  return 0;
}
