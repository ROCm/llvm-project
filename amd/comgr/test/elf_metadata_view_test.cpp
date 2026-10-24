#include "amd_comgr.h"
#include "common.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#ifndef _WIN32
#include <sys/mman.h>
#endif

static std::string snapshot(amd_comgr_metadata_node_t Node);

static amd_comgr_status_t collectEntry(amd_comgr_metadata_node_t Key,
                                       amd_comgr_metadata_node_t Value,
                                       void *Context) {
  auto &Entries = *static_cast<std::vector<std::string> *>(Context);
  Entries.push_back(snapshot(Key) + snapshot(Value));
  return AMD_COMGR_STATUS_SUCCESS;
}

static std::string snapshot(amd_comgr_metadata_node_t Node) {
  amd_comgr_metadata_kind_t Kind;
  checkError(amd_comgr_get_metadata_kind(Node, &Kind), "metadata kind");
  std::string Result = std::to_string(Kind) + ":";
  size_t Size = 0;
  switch (Kind) {
  case AMD_COMGR_METADATA_KIND_NULL:
    return Result;
  case AMD_COMGR_METADATA_KIND_STRING: {
    checkError(amd_comgr_get_metadata_string(Node, &Size, nullptr),
               "string size");
    std::string Value(Size, '\0');
    checkError(amd_comgr_get_metadata_string(Node, &Size, Value.data()),
               "string value");
    return Result + std::to_string(Size) + ":" + Value;
  }
  case AMD_COMGR_METADATA_KIND_LIST:
    checkError(amd_comgr_get_metadata_list_size(Node, &Size), "list size");
    Result += std::to_string(Size) + ":";
    for (size_t Index = 0; Index < Size; ++Index) {
      amd_comgr_metadata_node_t Child;
      checkError(amd_comgr_index_list_metadata(Node, Index, &Child),
                 "list child");
      Result += snapshot(Child);
      checkError(amd_comgr_destroy_metadata(Child), "destroy child");
    }
    return Result;
  case AMD_COMGR_METADATA_KIND_MAP: {
    std::vector<std::string> Entries;
    checkError(amd_comgr_iterate_map_metadata(Node, collectEntry, &Entries),
               "map entries");
    std::sort(Entries.begin(), Entries.end());
    Result += std::to_string(Entries.size()) + ":";
    for (const auto &Entry : Entries)
      Result += Entry;
    return Result;
  }
  default:
    fail("unknown metadata kind");
    return {};
  }
}

static void compare(const char *Bytes, size_t Size) {
  amd_comgr_data_t Data;
  checkError(amd_comgr_create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &Data),
             "create data");
  checkError(amd_comgr_set_data(Data, Size, Bytes), "copy data");
  amd_comgr_metadata_node_t Expected{0};
  auto ExpectedStatus = amd_comgr_get_data_metadata(Data, &Expected);
  checkError(amd_comgr_release_data(Data), "release data");
  std::string ExpectedSnapshot;
  if (ExpectedStatus == AMD_COMGR_STATUS_SUCCESS) {
    ExpectedSnapshot = snapshot(Expected);
    checkError(amd_comgr_destroy_metadata(Expected), "destroy expected");
  }

  for (size_t Offset = 0; Offset < alignof(std::max_align_t); ++Offset) {
    auto Storage = std::make_unique<char[]>(Size + Offset);
    char *Input = Storage.get() + Offset;
    memcpy(Input, Bytes, Size);
    amd_comgr_metadata_node_t Actual{123};
    auto Status = amd_comgr_get_elf_metadata(Size, Input, &Actual);
    checkStatus(Status, ExpectedStatus, "view status");
    if (memcmp(Input, Bytes, Size))
      fail("metadata parsing modified input");
    memset(Input, 0xa5, Size);
    Storage.reset();
    if (Status != AMD_COMGR_STATUS_SUCCESS) {
      if (Actual.handle != 123)
        fail("error modified metadata output");
      continue;
    }
    if (snapshot(Actual) != ExpectedSnapshot)
      fail("metadata differs after input release");

    amd_comgr_metadata_kind_t Kind;
    checkError(amd_comgr_get_metadata_kind(Actual, &Kind), "actual kind");
    amd_comgr_metadata_node_t Kernels{0};
    std::string KernelSnapshot;
    if (Kind == AMD_COMGR_METADATA_KIND_MAP &&
        amd_comgr_metadata_lookup(Actual, "amdhsa.kernels", &Kernels) ==
            AMD_COMGR_STATUS_SUCCESS)
      KernelSnapshot = snapshot(Kernels);
    checkError(amd_comgr_destroy_metadata(Actual), "destroy actual");
    if (Kernels.handle) {
      if (snapshot(Kernels) != KernelSnapshot)
        fail("child metadata differs after parent release");
      checkError(amd_comgr_destroy_metadata(Kernels), "destroy kernels");
    }
  }

#ifndef _WIN32
  size_t PageSize = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  size_t MappedSize = ((Size + PageSize - 1) / PageSize + 1) * PageSize;
  void *Mapping = mmap(nullptr, MappedSize, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (Mapping == MAP_FAILED)
    fail("mmap");
  char *Guard = static_cast<char *>(Mapping) + MappedSize - PageSize;
  if (mprotect(Guard, PageSize, PROT_NONE))
    fail("mprotect");
  char *Input = Guard - Size;
  memcpy(Input, Bytes, Size);
  if (mprotect(Mapping, MappedSize - PageSize, PROT_READ))
    fail("mprotect input");
  amd_comgr_metadata_node_t Actual{0};
  auto Status = amd_comgr_get_elf_metadata(Size, Input, &Actual);
  checkStatus(Status, ExpectedStatus, "guarded view status");
  if (munmap(Mapping, MappedSize))
    fail("munmap");
  if (Status == AMD_COMGR_STATUS_SUCCESS) {
    if (snapshot(Actual) != ExpectedSnapshot)
      fail("guarded metadata differs after unmap");
    checkError(amd_comgr_destroy_metadata(Actual), "destroy guarded metadata");
  }
#endif
}

int main() {
  amd_comgr_metadata_node_t Node{123};
  checkStatus(amd_comgr_get_elf_metadata(1, nullptr, &Node),
              AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT, "null input");
  checkStatus(amd_comgr_get_elf_metadata(0, "", &Node),
              AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT, "empty input");
  checkStatus(amd_comgr_get_elf_metadata(1, "x", nullptr),
              AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT, "null output");
  if (Node.handle != 123)
    fail("invalid arguments modified output");
  compare("not an ELF", 10);

  const char *Files[] = {"shared.so",
                         "shared-v2.so",
                         "shared-v3.so",
                         "shared12-v2.so",
                         "shared12-v3.so",
                         "multiple-note-records.out",
                         "multiple-note-records-one-kernel.out"};
  for (const char *File : Files) {
    char *Bytes;
    size_t Size =
        setBuf((std::string(TEST_OBJ_DIR) + "/" + File).c_str(), &Bytes);
    compare(Bytes, Size);
    compare(Bytes, 1);
    compare(Bytes, 16);
    compare(Bytes, 64);
    compare(Bytes, Size / 2);
    free(Bytes);
  }
  return 0;
}
