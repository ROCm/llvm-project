//===- file_map.c ---------------------------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "amd_comgr.h"
#include "common.h"

int main(int argc, char *argv[]) {
  int Ret;
  amd_comgr_status_t Status;

  const char *FileName = "comgr_map_test_file.txt";

  // Remove any stray file that may exist from before.
  remove(FileName);

#if defined(_WIN32) || defined(_WIN64)
  int FD = _open(FileName, _O_CREAT | _O_RDWR);
#else
  int FD = open(FileName, O_CREAT | O_RDWR, 0755);
#endif
  if (FD < 0) {
    fail("open failed for %s with errno %d", FileName, errno);
  }

  const char *Buffer = "abcdefghi";
  size_t Length = strlen(Buffer);
  size_t Bytes = WriteFileCustom(FD, Buffer, Length);
  if (Bytes != Length) {
    fail("Write failed with ret %d", Bytes);
  }

  amd_comgr_data_t DataObject;
  Status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_RELOCATABLE, &DataObject);
  checkError(Status, "amd_comgr_create_data");

  uint64_t Offset = 2;
  Status = amd_comgr_set_data_from_file_slice(DataObject, FD, Offset, Length);
  checkError(Status, "amd_comgr_get_file_slice");

  char Slice[10];
  size_t SliceLength = Length - 2;
  Status = amd_comgr_get_data(DataObject, &SliceLength, Slice);
  checkError(Status, "amd_comgr_get_data");

  if (SliceLength != Length - Offset) {
    fail("File Slice Length incorrect");
  }

  if (!strncmp(Slice, Buffer, Length - Offset)) {
    fail("File Slice read failed");
  }

#if defined(_WIN32) || defined(_WIN64)
  _close(FD);
#else
  close(FD);
#endif

  if ((Ret = remove(FileName)) != 0) {
#if defined(_WIN32) || defined(_WIN64)
    if ((Ret = remove(FileName)) != 0) {
      fail("remove failed");
    }
#else
    fail("remove failed");
#endif
  }
  return 0;
}
