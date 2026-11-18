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
#include <llvm/Object/OffloadBinary.h>
#include <llvm/Support/MemoryBuffer.h>

#include <memory>

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace COMGR {
// Extracts offload binaries matching a set of requested (triple, arch)
// entries out of an llvm-offload-binary package, via extractOffloadBinaries
class UnpackageCommand final : public CachedCommandAdaptor {
private:
  // The raw contents and name of the AMD_COMGR_DATA_KIND_PACKAGE input, owned
  // by the caller.
  llvm::StringRef PackageData;
  llvm::StringRef PackageName;

  // The requested (triple, arch) package entry IDs, owned by the caller.
  llvm::ArrayRef<std::pair<std::string, std::string>> PackageEntryIDs;

  // Where output files are written, owned by the caller. Unused in the
  // in-place path.
  llvm::StringRef OutputDir;

  // If set, execute() avoids disk and populates InPlaceBuffers instead.
  // Ignored for compressed packages, whose output must go through disk for
  // CommandCache (see writeExecuteOutput()).
  bool PreferInPlace = false;

  // The output paths (or, in-place, logical names) and data kinds, in
  // matching order. Populated by execute() or writeExecuteOutput().
  llvm::SmallVector<std::string> OutputFileNames;
  llvm::SmallVector<amd_comgr_data_kind_t> DataKinds;

  // Whether the last execute() took the in-place path; see isInPlace().
  bool WasInPlace = false;

  // In-place outputs from execute(), parallel to OutputFileNames/DataKinds.
  // Each buffer aliases its OffloadBinary's image and keeps the owning
  // OffloadFile list alive; callers should std::move each entry out.
  llvm::SmallVector<std::unique_ptr<llvm::MemoryBuffer>> InPlaceBuffers;

  // To avoid copies, store the output of execute(), such that
  // readExecuteOutput can return a reference.
  llvm::SmallString<64> OutputBuffer;

public:
  UnpackageCommand(
      llvm::StringRef PackageData, llvm::StringRef PackageName,
      llvm::ArrayRef<std::pair<std::string, std::string>> PackageEntryIDs,
      llvm::StringRef OutputDir, bool PreferInPlace = false)
      : PackageData(PackageData), PackageName(PackageName),
        PackageEntryIDs(PackageEntryIDs), OutputDir(OutputDir),
        PreferInPlace(PreferInPlace) {}

  bool canCache() const override;
  llvm::Error writeExecuteOutput(llvm::StringRef CachedBuffer) override;
  /// Split a cached output into (name, data kind, image) triples, in the
  /// order they were written by readExecuteOutput(), without writing any
  /// file.
  llvm::Error
  splitCachedOutput(llvm::StringRef CachedBuffer,
                    llvm::SmallVectorImpl<std::string> &OutNames,
                    llvm::SmallVectorImpl<amd_comgr_data_kind_t> &OutKinds,
                    llvm::SmallVectorImpl<llvm::StringRef> &OutImages);
  llvm::Expected<llvm::StringRef> readExecuteOutput() override;
  amd_comgr_status_t execute(llvm::raw_ostream &LogS) override;

  /// Output file paths from the most recent execute()/writeExecuteOutput().
  /// Empty when isInPlace(); see getInPlaceBuffers() instead.
  llvm::ArrayRef<std::string> getOutputFileNames() const {
    return OutputFileNames;
  }
  /// Data kind of each output, parallel to getOutputFileNames().
  llvm::ArrayRef<amd_comgr_data_kind_t> getDataKinds() const {
    return DataKinds;
  }

  /// Whether the last execute() produced in-place buffers instead of files.
  bool isInPlace() const { return WasInPlace; }

  /// In-place outputs, parallel to getDataKinds(); only non-empty when
  /// isInPlace(). Callers should std::move each entry out.
  llvm::MutableArrayRef<std::unique_ptr<llvm::MemoryBuffer>>
  getInPlaceBuffers() {
    return InPlaceBuffers;
  }

  ~UnpackageCommand() override = default;

protected:
  ActionClass getClass() const override;
  void addOptionsIdentifier(HashAlgorithm &) const override;
  llvm::Error addInputIdentifier(HashAlgorithm &) const override;
};
} // namespace COMGR

#endif
