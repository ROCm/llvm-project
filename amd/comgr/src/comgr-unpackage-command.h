//===- comgr-package-command.h - UnpackageCommand implementation ----------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef COMGR_PACKAGER_COMMAND_H
#define COMGR_PACKAGER_COMMAND_H

#include "comgr-cache-command.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace COMGR {
// Extracts offload binaries matching a set of requested (triple, arch)
// entries out of an llvm-offload-binary package, via extractOffloadBinaries
class UnpackageCommand final : public CachedCommandAdaptor {
private:
  // The raw contents and name of the AMD_COMGR_DATA_KIND_PACKAGE input. Both
  // are owned by the caller and must outlive this command.
  llvm::StringRef PackageData;
  llvm::StringRef PackageName;

  // The requested (triple, arch) package entry IDs, owned by the caller.
  llvm::ArrayRef<std::pair<std::string, std::string>> PackageEntryIDs;

  // Where output files are written, owned by the caller.
  llvm::StringRef OutputDir;

  // Populated by execute() (cache miss) or writeExecuteOutput() (cache hit):
  // the paths written and the Comgr data kind of each, in matching order.
  llvm::SmallVector<std::string> OutputFileNames;
  llvm::SmallVector<amd_comgr_data_kind_t> DataKinds;

  // To avoid copies, store the output of execute(), such that
  // readExecuteOutput can return a reference.
  llvm::SmallString<64> OutputBuffer;

public:
  UnpackageCommand(
      llvm::StringRef PackageData, llvm::StringRef PackageName,
      llvm::ArrayRef<std::pair<std::string, std::string>> PackageEntryIDs,
      llvm::StringRef OutputDir)
      : PackageData(PackageData), PackageName(PackageName),
        PackageEntryIDs(PackageEntryIDs), OutputDir(OutputDir) {}

  bool canCache() const override;
  llvm::Error writeExecuteOutput(llvm::StringRef CachedBuffer) override;
  llvm::Expected<llvm::StringRef> readExecuteOutput() override;
  amd_comgr_status_t execute(llvm::raw_ostream &LogS) override;

  /// The output file paths written by the most recent execute() or
  /// writeExecuteOutput() call, in order.
  llvm::ArrayRef<std::string> getOutputFileNames() const {
    return OutputFileNames;
  }
  /// The Comgr data kind of each output file, parallel to
  /// getOutputFileNames().
  llvm::ArrayRef<amd_comgr_data_kind_t> getDataKinds() const {
    return DataKinds;
  }

  ~UnpackageCommand() override = default;

protected:
  ActionClass getClass() const override;
  void addOptionsIdentifier(HashAlgorithm &) const override;
  llvm::Error addInputIdentifier(HashAlgorithm &) const override;
};
} // namespace COMGR

#endif
