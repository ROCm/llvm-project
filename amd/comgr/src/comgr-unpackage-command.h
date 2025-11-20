//===- comgr-package-command.h - UnpackageCommand implementation ----------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef COMGR_PACKAGER_COMMAND_H
#define COMGR_PACKAGER_COMMAND_H

#include <comgr-cache-command.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Object/OffloadBinary.h>

namespace COMGR {
class UnpackageCommand final : public CachedCommandAdaptor {
private:
  const llvm::SmallVector<llvm::object::OffloadFile> &Files;
  const llvm::SmallVector<std::string> &TargetNames;
  const llvm::SmallVector<std::string> &OutputFileNames;

  // To avoid copies, store the output of execute, such that readExecuteOutput
  // can return a reference.
  llvm::SmallString<64> OutputBuffer;

public:
  UnpackageCommand(const llvm::SmallVector<llvm::object::OffloadFile> &Files,
                   const llvm::SmallVector<std::string> &TargetNames,
                   const llvm::SmallVector<std::string> &OutputFileNames)
      : Files(Files), TargetNames(TargetNames),
        OutputFileNames(OutputFileNames) {}

  bool canCache() const override;
  llvm::Error writeExecuteOutput(llvm::StringRef CachedBuffer) override;
  llvm::Expected<llvm::StringRef> readExecuteOutput() override;
  amd_comgr_status_t execute(llvm::raw_ostream &LogS) override;

  ~UnpackageCommand() override = default;

protected:
  ActionClass getClass() const override;
  void addOptionsIdentifier(HashAlgorithm &) const override;
  llvm::Error addInputIdentifier(HashAlgorithm &) const override;
};
} // namespace COMGR

#endif
