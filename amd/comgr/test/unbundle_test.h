#ifndef COMGR_UNBUNDLE_TEST_H
#define COMGR_UNBUNDLE_TEST_H

#include "common.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Object/OffloadBundle.h"
#include "llvm/Support/Compression.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/MemoryBuffer.h"
#include <string>
#include <utility>
#include <vector>

namespace UnbundleTest {
constexpr char Magic[] = "__CLANG_OFFLOAD_BUNDLE__";
constexpr char Device[] = "hip-amdgcn-amd-amdhsa-unknown-gfx900";
constexpr char OtherDevice[] = "hip-amdgcn-amd-amdhsa-unknown-gfx906";
constexpr char Host[] = "host-x86_64-unknown-linux-gnu";

inline void appendInteger(std::string &Buffer, uint64_t Value) {
  size_t Offset = Buffer.size();
  Buffer.resize(Offset + sizeof(Value));
  llvm::support::endian::write64le(Buffer.data() + Offset, Value);
}

inline std::string
bundle(llvm::ArrayRef<std::pair<llvm::StringRef, llvm::StringRef>> Entries) {
  uint64_t Offset = sizeof(Magic) - 1 + sizeof(uint64_t);
  for (const auto &Entry : Entries)
    Offset += 3 * sizeof(uint64_t) + Entry.first.size();
  std::string Result = Magic;
  appendInteger(Result, Entries.size());
  for (const auto &Entry : Entries) {
    appendInteger(Result, Offset);
    appendInteger(Result, Entry.second.size());
    appendInteger(Result, Entry.first.size());
    Result += Entry.first;
    Offset += Entry.second.size();
  }
  for (const auto &Entry : Entries)
    Result += Entry.second;
  return Result;
}

inline std::string compress(llvm::StringRef Input,
                            llvm::compression::Format Format,
                            uint16_t Version = 3) {
  auto Buffer = llvm::MemoryBuffer::getMemBuffer(Input, "", false);
  auto Compressed = llvm::object::CompressedOffloadBundle::compress(
      llvm::compression::Params(Format), *Buffer, Version == 1 ? 2 : Version);
  if (!Compressed)
    fail("compress: %s", llvm::toString(Compressed.takeError()).c_str());
  std::string Result = (*Compressed)->getBuffer().str();
  if (Version == 1) {
    Result.erase(8, sizeof(uint32_t));
    llvm::support::endian::write16le(Result.data() + 4, 1);
  }
  return Result;
}

inline llvm::SmallVector<amd_comgr_data_t>
unbundleData(llvm::StringRef Input, std::vector<const char *> Targets,
             bool ExpectSuccess = true, bool Executable = false) {
  amd_comgr_data_set_t Inputs, Outputs;
  amd_comgr_data_t InputData;
  amd_comgr_action_info_t Action;
  auto InputKind = Executable ? AMD_COMGR_DATA_KIND_OBJ_BUNDLE
                              : AMD_COMGR_DATA_KIND_BC_BUNDLE;
  auto OutputKind =
      Executable ? AMD_COMGR_DATA_KIND_EXECUTABLE : AMD_COMGR_DATA_KIND_BC;
  checkError(amd_comgr_create_data_set(&Inputs), "create inputs");
  checkError(amd_comgr_create_data_set(&Outputs), "create outputs");
  checkError(amd_comgr_create_data(InputKind, &InputData), "create input");
  checkError(amd_comgr_set_data(InputData, Input.size(), Input.data()),
             "set input");
  checkError(amd_comgr_set_data_name(InputData, "bundle"), "set name");
  checkError(amd_comgr_data_set_add(Inputs, InputData), "add input");
  checkError(amd_comgr_release_data(InputData), "release input");
  checkError(amd_comgr_create_action_info(&Action), "create action");
  checkError(amd_comgr_action_info_set_language(Action, AMD_COMGR_LANGUAGE_HIP),
             "set language");
  checkError(amd_comgr_action_info_set_bundle_entry_ids(Action, Targets.data(),
                                                        Targets.size()),
             "set targets");
  auto Status =
      amd_comgr_do_action(AMD_COMGR_ACTION_UNBUNDLE, Action, Inputs, Outputs);
  if (ExpectSuccess)
    checkError(Status, "unbundle");
  else if (Status == AMD_COMGR_STATUS_SUCCESS)
    fail("malformed bundle was accepted");
  checkError(amd_comgr_destroy_action_info(Action), "destroy action");
  checkError(amd_comgr_destroy_data_set(Inputs), "destroy inputs");

  llvm::SmallVector<amd_comgr_data_t> Data;
  if (ExpectSuccess) {
    checkCount("outputs", Outputs, OutputKind, Targets.size());
    Data.resize(Targets.size());
    for (size_t Index = 0; Index < Data.size(); ++Index)
      checkError(amd_comgr_action_data_get_data(Outputs, OutputKind, Index,
                                                &Data[Index]),
                 "get output");
  }
  checkError(amd_comgr_destroy_data_set(Outputs), "destroy outputs");
  return Data;
}
} // namespace UnbundleTest

#endif
